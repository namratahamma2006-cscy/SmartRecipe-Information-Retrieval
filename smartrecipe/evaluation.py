"""Retrieval evaluation: Precision@k, Recall@k, F1@k and MAP over a set of
labelled test queries, for several system configurations (ablation)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .engine import SmartRecipeEngine

EVAL_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_queries.json"

CONFIGS = {
    "Baseline TF-IDF": dict(smart_text=False, query_understanding=False),
    "+ Preprocessing & Synonyms": dict(smart_text=True, query_understanding=False),
    "SmartRecipe (full)": dict(smart_text=True, query_understanding=True),
}


def precision(retrieved: list[str], relevant: set[str]) -> float:
    return len(set(retrieved) & relevant) / len(retrieved) if retrieved else 0.0


def recall(retrieved: list[str], relevant: set[str]) -> float:
    return len(set(retrieved) & relevant) / len(relevant) if relevant else 0.0


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) else 0.0


def average_precision(ranked: list[str], relevant: set[str]) -> float:
    hits, total = 0, 0.0
    for i, doc in enumerate(ranked, start=1):
        if doc in relevant:
            hits += 1
            total += hits / i
    return total / len(relevant) if relevant else 0.0


@dataclass
class QueryEval:
    qid: str
    query: str
    ingredients: list[str]
    tests: str
    relevant: list[str]
    retrieved: list[str]
    precision: float
    recall: float
    f1: float
    ap: float


def load_eval_queries(path: Path | str = EVAL_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def evaluate(engine: SmartRecipeEngine, queries: list[dict], k: int = 5) -> list[QueryEval]:
    rows = []
    for q in queries:
        relevant = set(q["relevant"])
        resp = engine.search(q.get("query", ""), q.get("ingredients", []), top_k=k)
        retrieved = [r.id for r in resp.results]
        p, r = precision(retrieved, relevant), recall(retrieved, relevant)
        rows.append(QueryEval(
            qid=q["id"], query=q.get("query", ""), ingredients=q.get("ingredients", []),
            tests=q.get("tests", ""), relevant=sorted(relevant), retrieved=retrieved,
            precision=p, recall=r, f1=f1(p, r), ap=average_precision(retrieved, relevant),
        ))
    return rows


def summarize(rows: list[QueryEval]) -> dict[str, float]:
    n = len(rows) or 1
    return {
        "Precision": sum(r.precision for r in rows) / n,
        "Recall": sum(r.recall for r in rows) / n,
        "F1": sum(r.f1 for r in rows) / n,
        "MAP": sum(r.ap for r in rows) / n,
    }


def compare_configs(recipes: list[dict], queries: list[dict], k: int = 5):
    """Evaluate every configuration; returns {name: (summary, rows)}."""
    out = {}
    for name, flags in CONFIGS.items():
        engine = SmartRecipeEngine(recipes, **flags)
        rows = evaluate(engine, queries, k)
        out[name] = (summarize(rows), rows)
    return out


if __name__ == "__main__":  # python -m smartrecipe.evaluation
    from .engine import load_recipes

    recipes, queries = load_recipes(), load_eval_queries()
    for k in (3, 5, 10):
        print(f"\n=== k = {k} ===")
        for name, (summary, _) in compare_configs(recipes, queries, k).items():
            print(f"{name:30s} " + "  ".join(f"{m}={v:.3f}" for m, v in summary.items()))
