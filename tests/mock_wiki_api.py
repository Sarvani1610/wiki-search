"""A stand-in for the MediaWiki API, for testing the crawler offline.

Serves real Wikipedia article text from the WikiText-2 corpus (Merity et al.,
~720 articles), mimicking the subset of `action=query` the crawler uses:
extracts, links with `plcontinue` pagination, missing pages, and a 503 on a
configurable fraction of requests so the retry path gets exercised.

WikiText has no hyperlinks, so links are reconstructed: an article links to
every other article whose title appears in its text, plus a few random
articles so the whole corpus is reachable from any seed.

    python tests/mock_wiki_api.py --wikitext path/to/wikitext-2 --port 8089
"""
import argparse
import json
import random
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HEADING = re.compile(r"^ = ([^=].*?) = $")
FIXES = [(" @-@ ", "-"), (" @,@ ", ","), (" @.@ ", "."), (" , ", ", "), (" . ", ". "),
         (" ; ", "; "), (" : ", ": "), (" 's", "'s"), ("( ", "("), (" )", ")"), (" <unk>", "")]


def load_articles(root: Path):
    arts, title, buf = {}, None, []
    for name in ("train.txt", "valid.txt", "test.txt"):
        for line in (root / name).read_text(encoding="utf-8").splitlines():
            m = HEADING.match(line)
            if m:
                if title:
                    arts[title] = clean(" ".join(buf))
                title, buf = clean(m.group(1)).strip(), []
            elif line.strip():
                # Section headings look like " = = History = = "; keep the words.
                buf.append(re.sub(r"^[ =]+|[ =]+$", "", line) if line.startswith(" =") else line.strip())
    if title:
        arts[title] = clean(" ".join(buf))
    # A few WikiText headings were mangled by its <unk> vocabulary cutoff.
    return {t: x for t, x in arts.items() if "<unk>" not in t and len(t) > 3}


def clean(s):
    for a, b in FIXES:
        s = s.replace(a, b)
    return s.strip()


def build_links(arts, seed=7):
    rng = random.Random(seed)
    titles = list(arts)
    links = {}
    for t, text in arts.items():
        found = [o for o in titles if o != t and len(o) > 3 and o in text]
        found += rng.sample(titles, 5)
        links[t] = list(dict.fromkeys(x for x in found if x != t))
    return links


def make_handler(arts, links, ids, fail_rate, page_size):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if random.random() < fail_rate:
                self.send_response(503)
                self.send_header("Retry-After", "0")
                self.end_headers()
                return
            q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
            title = q.get("titles", "")
            if title not in arts:
                page = {"ns": 0, "title": title, "missing": True}
                body = {"batchcomplete": True, "query": {"pages": [page]}}
            else:
                start = int(q.get("plcontinue", "0"))
                ls = links[title][start:start + page_size]
                page = {"pageid": ids[title], "ns": 0, "title": title,
                        "links": [{"ns": 0, "title": l} for l in ls]}
                if "extracts" in q.get("prop", ""):
                    page["extract"] = arts[title]
                body = {"query": {"pages": [page]}}
                if start + page_size < len(links[title]):
                    body["continue"] = {"plcontinue": str(start + page_size), "continue": "||"}
                else:
                    body["batchcomplete"] = True
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    return H


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wikitext", required=True, help="directory with WikiText-2 train/valid/test.txt")
    p.add_argument("--port", type=int, default=8089)
    p.add_argument("--fail-rate", type=float, default=0.03)
    p.add_argument("--page-size", type=int, default=25, help="links per response (forces pagination)")
    a = p.parse_args()
    arts = load_articles(Path(a.wikitext))
    links = build_links(arts)
    ids = {t: 1000 + i for i, t in enumerate(arts)}
    print(f"serving {len(arts)} articles, {sum(map(len, links.values()))} links on :{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(arts, links, ids, a.fail_rate, a.page_size)).serve_forever()


if __name__ == "__main__":
    main()
