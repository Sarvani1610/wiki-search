# Wikipedia Search Engine

A concurrent Go crawler that pulls Wikipedia articles into a shared Redis frontier, a BM25 index over the crawl, and a LightGBM LambdaMART re-ranker evaluated with NDCG@10.

```
            ┌──────────── Redis ────────────┐
            │ queue (LIST)   seen (SET)     │
            │ attempts (HASH) failed (LIST) │
            └───────▲───────────────┬───────┘
     new links (Lua │ dedupe)       │ BLPOP title
            ┌───────┴───────────────▼───────┐
            │  Go crawler: N workers / proc │──► MediaWiki API (rate limited)
            └───────────────┬───────────────┘
                            ▼ gzipped JSONL, one file per worker
   index ─► BM25 top-100 ─► 18 features ─► LambdaMART ─► top 10
```

## Layout

| Path | What it is |
|---|---|
| `crawler/` | Go crawler (`go-redis`). Workers share one frontier, so you can run several processes against the same Redis. |
| `search/wikisearch/index.py` | Tokenizes pages and stores term counts as sparse matrices; BM25 weights are precomputed per (doc, term) so a query is a sum over a few columns. |
| `search/wikisearch/features.py` | 18 query-document features: BM25 per field, query-term coverage, title matches, inlinks, links from other top results, lengths. |
| `search/wikisearch/judgments.py` | Relevance labels: real TREC qrels, or synthetic known-item queries built from the crawl. |
| `search/wikisearch/rank.py` | LambdaMART training (early stopping on validation NDCG@10), evaluation, search. |
| `tests/mock_wiki_api.py` | Offline stand-in for the MediaWiki API, serving real article text from WikiText-2. |

## Quick start

```bash
docker compose up -d redis                 # or: redis-server
pip install -r search/requirements.txt

# 1. Crawl (set a real contact in -ua; Wikimedia blocks anonymous bots)
cd crawler && go run . -workers 8 -max 1200000 -out ../data/pages \
  -ua "wiki-search/1.0 (you@school.edu)" && cd ..

# 2-5. Index, labels, train, evaluate
export PYTHONPATH=search
python -m wikisearch index   --pages data/pages --out data/index
python -m wikisearch queries --index data/index --n 5000
python -m wikisearch train
python -m wikisearch eval            # BM25 vs BM25+LambdaMART on the test split
python -m wikisearch search "german battleship world war i"
```

`make` wraps the same steps (`make crawl index queries train eval`).

## Crawler notes

- **Dedupe**: a Lua script does `SADD seen` + `RPUSH queue` atomically, so two workers that find the same link never both queue it.
- **Frontier cap** (`-max-frontier`): links beyond the cap are left unseen, so they can still be discovered later instead of being lost.
- **Retries**: 5xx, 429 and `maxlag` responses back off (honoring `Retry-After`) and requeue; after `-attempts` tries a title goes to `wiki:failed`.
- **Resumable**: Ctrl-C hands in-flight titles back to the queue. Restart without `-reset` to continue. Output files are written as `.part` and renamed when closed, and the indexer drops duplicate page IDs.
- **Budget**: `-max` is checked before each fetch, so a run can overshoot by up to one page per worker.
- **Scale out**: start more processes with the same `-redis` and `-prefix`. Total request rate is `-rps` × processes, so keep it within Wikimedia's API etiquette.

At 20 requests/s, 1.2M pages takes about 17 hours. Index memory grows with `--max-tokens` (body tokens kept per article; default 3000).

## Evaluation

NDCG@10 uses graded gains (2^rel − 1). The re-ranker sees the BM25 top 100 for each query, and queries are split 70/15/15 into train/valid/test.

Without human judgments, `queries` generates **known-item** queries. It picks an article, draws 2–4 distinctive words from its lead paragraph (half the time adding one title word), and labels the article grade 2 and the pages it links to grade 1. With real judgments, pass `--trec-queries queries.tsv --trec-qrels qrels.txt` instead.

`eval` also reports MRR, recall@100 of the BM25 stage, and a bootstrap 95% CI on the NDCG@10 gain.

## Offline test

```bash
cd crawler && REDIS_ADDR=localhost:6379 go test ./...
scripts/e2e_mock.sh path/to/wikitext-2     # full pipeline against the mock API
```

The mock run crawls 686 articles with 8 workers, with 3% of requests failing to exercise retries. Re-ranking lifts NDCG@10 from 0.51 to 0.54 there, with a 95% CI on the gain of +0.02 to +0.05. That corpus is tiny and its link graph is partly synthetic, so treat this as a correctness check. Your numbers from a real crawl will be different.
