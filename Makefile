export PYTHONPATH := search
WORKERS ?= 8
MAX ?= 1200000
UA ?= wiki-search/1.0 (you@example.com)

.PHONY: redis crawl index queries train eval search test

redis:
	docker compose up -d redis

crawl:
	cd crawler && go run . -workers $(WORKERS) -max $(MAX) -out ../data/pages -ua "$(UA)"

index:
	python3 -m wikisearch index --pages data/pages --out data/index

queries:
	python3 -m wikisearch queries --index data/index --n 5000

train:
	python3 -m wikisearch train

eval:
	python3 -m wikisearch eval

search:
	python3 -m wikisearch search

test:
	cd crawler && REDIS_ADDR=localhost:6379 go test ./...
