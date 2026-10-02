"""Builds and loads the on-disk index.

Layout of an index directory:
    meta.json          doc count, BM25 params, build info
    vocab.json         term -> column id
    docs.jsonl         one line per doc: {"id", "title", "snippet"}
    body.npz/title.npz term counts, CSR (docs x terms)
    links.npz          link graph, CSR (docs x docs), row = source page
"""
from __future__ import annotations

import glob
import gzip
import json
import os
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp

from .text import norm_title, tokenize

K1, B = 1.2, 0.75


def iter_pages(crawl_dir: str):
    """Yields crawled pages, skipping duplicates and unfinished (.part) files."""
    seen = set()
    for path in sorted(glob.glob(os.path.join(crawl_dir, "**", "*.jsonl.gz"), recursive=True)):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                try:
                    page = json.loads(line)
                except json.JSONDecodeError:
                    continue  # truncated last line of a killed crawl
                if page["id"] in seen or not page.get("text"):
                    continue
                seen.add(page["id"])
                yield page


def _analyze(args):
    page, max_tokens = args
    body = tokenize(page["text"], max_tokens)
    return (
        page["id"], page["title"], Counter(tokenize(page["title"])), Counter(body),
        page.get("links", []), page.get("redirects", []), _snippet(page["text"]),
    )


def _snippet(text: str, n: int = 280) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "…"


def build(crawl_dir: str, out_dir: str, max_tokens: int = 3000, min_df: int = 2, workers: int | None = None):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    vocab: dict[str, int] = {}
    cols = {"body": ([], [], [0]), "title": ([], [], [0])}  # indices, data, indptr
    ids, titles, links, alias = [], [], [], {}

    def add(field, counter):
        ind, dat, ptr = cols[field]
        for term, c in counter.items():
            j = vocab.get(term)
            if j is None:
                j = vocab[term] = len(vocab)
            ind.append(j)
            dat.append(c)
        ptr.append(len(ind))

    with open(os.path.join(out_dir, "docs.jsonl"), "w", encoding="utf-8") as docs_fh, \
            Pool(workers) as pool:
        work = ((p, max_tokens) for p in iter_pages(crawl_dir))
        for n, (pid, title, tcount, bcount, outl, redirects, snip) in enumerate(
                pool.imap(_analyze, work, chunksize=256), 1):
            add("title", tcount)
            add("body", bcount)
            ids.append(pid)
            titles.append(title)
            links.append(outl)
            for r in redirects:
                alias.setdefault(r, len(ids) - 1)
            docs_fh.write(json.dumps({"id": pid, "title": title, "snippet": snip}, ensure_ascii=False) + "\n")
            if n % 50_000 == 0:
                print(f"  analyzed {n:,} pages, vocab {len(vocab):,} ({time.time() - t0:.0f}s)")

    n_docs, n_terms = len(ids), len(vocab)
    if n_docs == 0:
        raise SystemExit(f"no pages found under {crawl_dir}")
    mats = {}
    for field, (ind, dat, ptr) in cols.items():
        mats[field] = sp.csr_matrix(
            (np.asarray(dat, np.int32), np.asarray(ind, np.int32), np.asarray(ptr, np.int64)),
            shape=(n_docs, n_terms))
        del ind[:], dat[:]

    # Drop terms seen in fewer than min_df documents (mostly typos and
    # one-off tokens); they cost memory and rarely help ranking.
    df = np.bincount((mats["body"] + mats["title"]).indices, minlength=n_terms)
    keep = np.flatnonzero(df >= min_df)
    remap = {old: new for new, old in enumerate(keep)}
    terms = [None] * n_terms
    for t, j in vocab.items():
        terms[j] = t
    vocab = {terms[old]: new for old, new in remap.items()}
    for field in mats:
        mats[field] = mats[field][:, keep].tocsr()
        mats[field].sort_indices()
        sp.save_npz(os.path.join(out_dir, f"{field}.npz"), mats[field])

    # Resolve link titles (and redirect aliases) to document rows.
    by_title = {t: i for i, t in enumerate(titles)}
    for a, i in alias.items():
        by_title.setdefault(a, i)
    rows, dst = [], []
    for i, outl in enumerate(links):
        for t in set(outl):
            j = by_title.get(t)
            if j is not None and j != i:
                rows.append(i)
                dst.append(j)
    graph = sp.csr_matrix((np.ones(len(rows), np.int8), (rows, dst)), shape=(n_docs, n_docs))
    sp.save_npz(os.path.join(out_dir, "links.npz"), graph)

    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as fh:
        json.dump(vocab, fh, ensure_ascii=False)
    meta = {"docs": n_docs, "terms": len(vocab), "links": int(graph.nnz),
            "k1": K1, "b": B, "max_tokens": max_tokens, "min_df": min_df,
            "build_seconds": round(time.time() - t0, 1)}
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


class Index:
    """Loaded index with BM25 impact matrices for each field.

    BM25 term weights are precomputed per (doc, term), so scoring a query is
    a sum over a few sparse columns. Columns are stored CSC for fast slicing.
    """

    def __init__(self, path: str):
        with open(os.path.join(path, "meta.json")) as fh:
            self.meta = json.load(fh)
        with open(os.path.join(path, "vocab.json"), encoding="utf-8") as fh:
            self.vocab: dict[str, int] = json.load(fh)
        self.titles, self.snippets, self.ids = [], [], []
        with open(os.path.join(path, "docs.jsonl"), encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                self.ids.append(d["id"])
                self.titles.append(d["title"])
                self.snippets.append(d["snippet"])
        self.title_norm = [norm_title(t) for t in self.titles]
        self.title_row = {t: i for i, t in enumerate(self.titles)}

        self.counts = {f: sp.load_npz(os.path.join(path, f"{f}.npz")).tocsr() for f in ("body", "title")}
        self.counts["all"] = (self.counts["body"] + self.counts["title"]).tocsr()
        self.n = self.counts["all"].shape[0]
        self.links = sp.load_npz(os.path.join(path, "links.npz")).tocsr()
        self.inlinks = np.asarray(self.links.sum(axis=0)).ravel()

        df = np.bincount(self.counts["all"].indices, minlength=len(self.vocab))
        self.idf = np.log1p((self.n - df + 0.5) / (df + 0.5)).astype(np.float32)
        self.doc_len = {f: np.asarray(m.sum(axis=1)).ravel().astype(np.float32) for f, m in self.counts.items()}
        self.impact = {f: self._impact(m, self.doc_len[f]).tocsc() for f, m in self.counts.items()}
        self.counts_csc = {f: m.tocsc() for f, m in self.counts.items()}

    def _impact(self, counts, dl):
        k1, b = self.meta["k1"], self.meta["b"]
        avgdl = max(dl.mean(), 1e-9)
        m = counts.astype(np.float32)
        row_len = np.repeat(dl, np.diff(m.indptr))
        tf = m.data
        m.data = self.idf[m.indices] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * row_len / avgdl))
        return m

    def query_terms(self, tokens) -> np.ndarray:
        ids = [self.vocab[t] for t in dict.fromkeys(tokens) if t in self.vocab]
        return np.asarray(ids, dtype=np.int64)

    def bm25(self, terms: np.ndarray, field: str = "all", rows: np.ndarray | None = None) -> np.ndarray:
        if len(terms) == 0:
            return np.zeros(self.n if rows is None else len(rows), np.float32)
        sub = self.impact[field][:, terms]
        if rows is not None:
            sub = sub.tocsr()[rows]
        return np.asarray(sub.sum(axis=1)).ravel()

    def top_k(self, terms: np.ndarray, k: int = 100):
        scores = self.bm25(terms)
        k = min(k, int((scores > 0).sum()))
        if k == 0:
            return np.empty(0, np.int64), np.empty(0, np.float32)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return top, scores[top]
