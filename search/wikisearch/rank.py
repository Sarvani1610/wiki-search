"""Training, evaluation and search with the LambdaMART re-ranker."""
from __future__ import annotations

import json
import os
import time
import warnings

import lightgbm as lgb
import numpy as np

from .features import FEATURES, featurize
from .index import Index


def ndcg_at_k(ranked_titles, rels: dict, k: int = 10) -> float:
    gains = [2 ** rels.get(t, 0) - 1 for t in ranked_titles[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted((2 ** g - 1 for g in rels.values()), reverse=True)[:k]
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def _rr(ranked_titles, rels):
    for i, t in enumerate(ranked_titles):
        if rels.get(t, 0) >= max(rels.values()):
            return 1.0 / (i + 1)
    return 0.0


def build_matrix(idx: Index, queries, k: int):
    X, y, groups, cache = [], [], [], []
    for q in queries:
        rows, scores, feats = featurize(idx, q["query"], k)
        cache.append((rows, feats))
        if len(rows) == 0:
            continue
        labels = np.array([q["rels"].get(idx.titles[r], 0) for r in rows])
        X.append(feats)
        y.append(labels)
        groups.append(len(rows))
    if not X:
        return None, None, None, cache
    return np.vstack(X), np.concatenate(y), np.array(groups), cache


def train(idx: Index, train_q, valid_q, model_dir: str, k: int = 100, seed: int = 13):
    t0 = time.time()
    Xtr, ytr, gtr, _ = build_matrix(idx, train_q, k)
    Xva, yva, gva, _ = build_matrix(idx, valid_q, k)
    print(f"  features: {len(gtr)} train / {len(gva)} valid queries, {len(Xtr):,} pairs ({time.time() - t0:.0f}s)")
    model = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        lambdarank_truncation_level=20,
        random_state=seed,
        verbose=-1,
    )
    with warnings.catch_warnings():
        # LightGBM 4.6+ renamed eval_set; the old name still works everywhere.
        warnings.filterwarnings("ignore", message=".*eval_set.*")
        model.fit(
            Xtr, ytr, group=gtr,
            eval_set=[(Xva, yva)], eval_group=[gva], eval_at=[10],
            feature_name=FEATURES,
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
    os.makedirs(model_dir, exist_ok=True)
    model.booster_.save_model(os.path.join(model_dir, "lambdamart.txt"))
    imp = dict(zip(FEATURES, model.booster_.feature_importance("gain").round(1).tolist()))
    info = {"best_iteration": model.best_iteration_, "k": k, "train_queries": len(gtr),
            "valid_queries": len(gva), "feature_gain": dict(sorted(imp.items(), key=lambda x: -x[1]))}
    with open(os.path.join(model_dir, "model.json"), "w") as fh:
        json.dump(info, fh, indent=2)
    return info


class Searcher:
    def __init__(self, idx: Index, model_dir: str | None = None, k: int = 100):
        self.idx, self.k = idx, k
        self.booster = None
        if model_dir and os.path.exists(os.path.join(model_dir, "lambdamart.txt")):
            self.booster = lgb.Booster(model_file=os.path.join(model_dir, "lambdamart.txt"))

    def search(self, query: str, n: int = 10, rerank: bool = True):
        rows, bm25, X = featurize(self.idx, query, self.k)
        if rerank and self.booster is not None and len(rows):
            scores = self.booster.predict(X)
            order = np.argsort(-scores, kind="stable")
            rows, scores = rows[order], scores[order]
        else:
            scores = bm25
        return [(self.idx.titles[r], float(s), self.idx.snippets[r]) for r, s in zip(rows[:n], scores[:n])]


def evaluate(idx: Index, queries, model_dir: str, k: int = 100):
    s = Searcher(idx, model_dir, k)
    if s.booster is None:
        raise SystemExit(f"no model in {model_dir}; run `train` first")
    m = {"bm25": {"ndcg@10": [], "mrr": []}, "lambdamart": {"ndcg@10": [], "mrr": []}}
    recall = []
    for q in queries:
        rows, bm25, X = featurize(idx, q["query"], k)
        base = [idx.titles[r] for r in rows]
        rer = [base[i] for i in np.argsort(-s.booster.predict(X), kind="stable")] if len(rows) else []
        for name, ranked in (("bm25", base), ("lambdamart", rer)):
            m[name]["ndcg@10"].append(ndcg_at_k(ranked, q["rels"]))
            m[name]["mrr"].append(_rr(ranked, q["rels"]))
        top = max(q["rels"].values())
        recall.append(float(any(q["rels"].get(t) == top for t in base)))
    out = {name: {metric: round(float(np.mean(v)), 4) for metric, v in d.items()} for name, d in m.items()}
    out["queries"] = len(queries)
    out[f"recall@{k}"] = round(float(np.mean(recall)), 4)
    # Paired bootstrap: how often does re-ranking beat BM25 on resampled query sets?
    diff = np.array(m["lambdamart"]["ndcg@10"]) - np.array(m["bm25"]["ndcg@10"])
    rng = np.random.default_rng(0)
    boots = diff[rng.integers(0, len(diff), (2000, len(diff)))].mean(1)
    out["ndcg@10_gain_95ci"] = [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]
    return out
