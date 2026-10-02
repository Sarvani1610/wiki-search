"""Query-document features for the LambdaMART re-ranker."""
from __future__ import annotations

import numpy as np

from .index import Index
from .text import tokenize

FEATURES = [
    "bm25_all", "bm25_body", "bm25_title", "bm25_rank", "bm25_ratio_top",
    "title_term_frac", "body_term_frac", "title_exact", "title_covers_query",
    "query_covers_title", "idf_weighted_body_frac", "min_body_tf", "log_inlinks",
    "local_inlinks", "log_body_len", "title_len", "query_len", "query_idf_sum",
]


def featurize(idx: Index, query: str, k: int = 100):
    """Retrieves BM25 top-k for a query and returns (rows, bm25 scores, X)."""
    toks = tokenize(query)
    terms = idx.query_terms(toks)
    rows, s_all = idx.top_k(terms, k)
    if len(rows) == 0:
        return rows, s_all, np.zeros((0, len(FEATURES)), np.float32)

    s_body = idx.bm25(terms, "body", rows)
    s_title = idx.bm25(terms, "title", rows)
    # Which query terms each candidate contains, per field.
    tf_body = idx.counts_csc["body"][:, terms].tocsr()[rows].toarray()
    tf_title = idx.counts_csc["title"][:, terms].tocsr()[rows].toarray()
    idf = idx.idf[terms]
    nq = max(len(terms), 1)

    qnorm = " ".join(toks)
    qset = set(toks)
    title_exact, title_cov, query_cov, title_len = [], [], [], []
    for r in rows:
        tn = idx.title_norm[r]
        tset = set(tn.split())
        title_exact.append(float(tn == qnorm and tn != ""))
        title_cov.append(float(bool(qset) and qset <= tset))
        query_cov.append(len(tset & qset) / max(len(tset), 1))
        title_len.append(len(tset))

    # Link support from the rest of the candidate list: pages that many of
    # the other top results link to tend to be the central article.
    top = rows[: min(20, len(rows))]
    local_in = np.asarray(idx.links[top][:, rows].sum(axis=0)).ravel()

    X = np.column_stack([
        s_all,
        s_body,
        s_title,
        np.arange(len(rows)),
        s_all / max(s_all[0], 1e-9),
        (tf_title > 0).sum(1) / nq,
        (tf_body > 0).sum(1) / nq,
        title_exact,
        title_cov,
        query_cov,
        ((tf_body > 0) * idf).sum(1) / max(idf.sum(), 1e-9),
        tf_body.min(1) if len(terms) else np.zeros(len(rows)),
        np.log1p(idx.inlinks[rows]),
        local_in,
        np.log1p(idx.doc_len["body"][rows]),
        title_len,
        np.full(len(rows), len(terms)),
        np.full(len(rows), idf.sum()),
    ]).astype(np.float32)
    return rows, s_all, X
