"""Relevance judgments for training and evaluating the re-ranker.

There are two sources:

1. Real judgments in TREC format (preferred when you have them):
     queries.tsv   qid<TAB>query text
     qrels.txt     qid 0 <article title> <grade 0-2>

2. Synthetic "known-item" queries generated from the crawl. For a sampled
   article we build a short keyword query from its lead paragraph, the way a
   person who half-remembers an article searches for it. The article itself
   is grade 2; articles it links to are grade 1 (topically related). Half the
   queries leave the title words out, so the system has to find the article
   from its description alone.
"""
from __future__ import annotations

import json
import random

import numpy as np

from .index import Index
from .text import tokenize


def synthesize(idx: Index, n: int, seed: int = 13, max_related: int = 20):
    rng = random.Random(seed)
    candidates = [i for i in range(idx.n) if idx.doc_len["body"][i] >= 80]
    rng.shuffle(candidates)
    queries = []
    for row in candidates:
        if len(queries) >= n:
            break
        title_toks = tokenize(idx.titles[row])
        # Lead-paragraph terms come from the stored snippet (first ~280 chars).
        lead = [t for t in tokenize(idx.snippets[row]) if t not in set(title_toks) and t in idx.vocab]
        lead = list(dict.fromkeys(lead))
        if len(lead) < 4:
            continue
        # Prefer distinctive words, with some randomness so queries vary.
        weights = np.array([idx.idf[idx.vocab[t]] for t in lead]) ** 2
        k = rng.randint(2, 4)
        picks = list(np.random.default_rng(rng.randrange(1 << 30)).choice(
            lead, size=min(k, len(lead)), replace=False, p=weights / weights.sum()))
        if rng.random() < 0.5 and title_toks:
            picks.insert(0, rng.choice(title_toks))
        rels = {idx.titles[row]: 2}
        for j in idx.links[row].indices[:max_related]:
            rels.setdefault(idx.titles[j], 1)
        queries.append({"qid": f"s{len(queries)}", "query": " ".join(picks), "rels": rels})
    return queries


def load_trec(idx: Index, queries_tsv: str, qrels_txt: str):
    qs = {}
    with open(queries_tsv, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                qid, text = line.rstrip("\n").split("\t", 1)
                qs[qid] = {"qid": qid, "query": text, "rels": {}}
    with open(qrels_txt, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 4 or parts[0] not in qs:
                continue
            title, grade = " ".join(parts[2:-1]).replace("_", " "), int(parts[-1])
            if grade > 0 and title in idx.title_row:
                qs[parts[0]]["rels"][title] = grade
    return [q for q in qs.values() if q["rels"]]


def save(queries, path):
    with open(path, "w", encoding="utf-8") as fh:
        for q in queries:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")


def load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def split(queries, seed: int = 13, frac=(0.7, 0.15)):
    q = list(queries)
    random.Random(seed).shuffle(q)
    a, b = int(len(q) * frac[0]), int(len(q) * (frac[0] + frac[1]))
    return q[:a], q[a:b], q[b:]
