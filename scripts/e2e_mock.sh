#!/usr/bin/env bash
# Offline end-to-end run: mock Wikipedia API -> Go crawler (8 workers, Redis)
# -> index -> judgments -> LambdaMART -> NDCG@10. Needs redis-server running.
#   scripts/e2e_mock.sh path/to/wikitext-2
set -euo pipefail
WT=${1:?usage: e2e_mock.sh <wikitext-2 dir>}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK=${WORK:-$ROOT/data/mock}
rm -rf "$WORK" && mkdir -p "$WORK"

python3 "$ROOT/tests/mock_wiki_api.py" --wikitext "$WT" --port 8089 &
MOCK=$!
trap 'kill $MOCK 2>/dev/null' EXIT
sleep 6

(cd "$ROOT/crawler" && go build -o "$WORK/crawler" .)
"$WORK/crawler" -api http://127.0.0.1:8089/w/api.php -prefix mock -reset \
  -workers 8 -rps 400 -seeds "Valkyria Chronicles III,Robert Boulter" -out "$WORK/pages"

export PYTHONPATH=$ROOT/search
python3 -m wikisearch index   --pages "$WORK/pages" --out "$WORK/index"
python3 -m wikisearch queries --index "$WORK/index" --out "$WORK/judgments" --n 5000
python3 -m wikisearch train   --index "$WORK/index" --judgments "$WORK/judgments" --model "$WORK/model"
python3 -m wikisearch eval    --index "$WORK/index" --judgments "$WORK/judgments" --model "$WORK/model"
