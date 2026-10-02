"""Command line entry point: python -m wikisearch <command> ..."""
import argparse
import json
import os
import sys
import time

from . import judgments
from .index import Index, build
from .rank import Searcher, evaluate, train


def main(argv=None):
    p = argparse.ArgumentParser(prog="wikisearch")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("index", help="build the index from crawler output")
    b.add_argument("--pages", default="data/pages")
    b.add_argument("--out", default="data/index")
    b.add_argument("--max-tokens", type=int, default=3000, help="body tokens kept per article")
    b.add_argument("--min-df", type=int, default=2)
    b.add_argument("--workers", type=int, default=None)

    q = sub.add_parser("queries", help="create train/valid/test judgments")
    q.add_argument("--index", default="data/index")
    q.add_argument("--out", default="data/judgments")
    q.add_argument("--n", type=int, default=5000, help="synthetic queries to generate")
    q.add_argument("--trec-queries", help="queries.tsv (use real judgments instead)")
    q.add_argument("--trec-qrels", help="qrels.txt")
    q.add_argument("--seed", type=int, default=13)

    t = sub.add_parser("train", help="train the LambdaMART re-ranker")
    t.add_argument("--index", default="data/index")
    t.add_argument("--judgments", default="data/judgments")
    t.add_argument("--model", default="data/model")
    t.add_argument("--k", type=int, default=100, help="BM25 candidates per query")

    e = sub.add_parser("eval", help="NDCG@10 of BM25 vs BM25 + LambdaMART on the test split")
    e.add_argument("--index", default="data/index")
    e.add_argument("--judgments", default="data/judgments")
    e.add_argument("--model", default="data/model")
    e.add_argument("--k", type=int, default=100)
    e.add_argument("--split", default="test", choices=["train", "valid", "test"])

    s = sub.add_parser("search", help="run queries (interactive if none given)")
    s.add_argument("query", nargs="*")
    s.add_argument("--index", default="data/index")
    s.add_argument("--model", default="data/model")
    s.add_argument("--n", type=int, default=10)
    s.add_argument("--no-rerank", action="store_true")

    a = p.parse_args(argv)
    if a.cmd == "index":
        print(json.dumps(build(a.pages, a.out, a.max_tokens, a.min_df, a.workers), indent=2))
        return

    t0 = time.time()
    idx = Index(a.index)
    print(f"loaded {idx.n:,} docs, {len(idx.vocab):,} terms in {time.time() - t0:.1f}s", file=sys.stderr)

    if a.cmd == "queries":
        if a.trec_queries:
            qs = judgments.load_trec(idx, a.trec_queries, a.trec_qrels)
        else:
            qs = judgments.synthesize(idx, a.n, a.seed)
        os.makedirs(a.out, exist_ok=True)
        for name, part in zip(("train", "valid", "test"), judgments.split(qs, a.seed)):
            judgments.save(part, os.path.join(a.out, f"{name}.jsonl"))
            print(f"{name}: {len(part)} queries")
    elif a.cmd == "train":
        tr = judgments.load(os.path.join(a.judgments, "train.jsonl"))
        va = judgments.load(os.path.join(a.judgments, "valid.jsonl"))
        print(json.dumps(train(idx, tr, va, a.model, a.k), indent=2))
    elif a.cmd == "eval":
        qs = judgments.load(os.path.join(a.judgments, f"{a.split}.jsonl"))
        print(json.dumps(evaluate(idx, qs, a.model, a.k), indent=2))
    elif a.cmd == "search":
        searcher = Searcher(idx, a.model)
        queries = [" ".join(a.query)] if a.query else None
        while True:
            if queries is None:
                try:
                    text = input("query> ").strip()
                except EOFError:
                    break
                if not text:
                    continue
            else:
                if not queries:
                    break
                text = queries.pop()
            t1 = time.time()
            hits = searcher.search(text, a.n, rerank=not a.no_rerank)
            print(f"\n{len(hits)} results ({(time.time() - t1) * 1000:.0f} ms)")
            for i, (title, score, snip) in enumerate(hits, 1):
                print(f"{i:>2}. {title}  [{score:.3f}]\n    {snip[:160]}")
            print()


if __name__ == "__main__":
    main()
