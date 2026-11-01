import pytest

from smartrecipe.engine import load_recipes
from smartrecipe.evaluation import (average_precision, compare_configs, f1,
                                    load_eval_queries, precision, recall)


def test_metric_math():
    ret, rel = ["a", "b", "c", "d"], {"a", "c", "x"}
    assert precision(ret, rel) == 0.5
    assert recall(ret, rel) == pytest.approx(2 / 3)
    assert f1(0.5, 2 / 3) == pytest.approx(2 * 0.5 * (2 / 3) / (0.5 + 2 / 3))
    assert average_precision(ret, rel) == pytest.approx((1 / 1 + 2 / 3) / 3)


def test_metric_edge_cases():
    assert precision([], {"a"}) == 0.0
    assert recall(["a"], set()) == 0.0
    assert f1(0, 0) == 0.0


def test_eval_queries_reference_real_recipes():
    ids = {r["id"] for r in load_recipes()}
    for q in load_eval_queries():
        assert q["relevant"] and set(q["relevant"]) <= ids, q["id"]


def test_smart_system_beats_baseline():
    res = compare_configs(load_recipes(), load_eval_queries(), k=5)
    base = res["Baseline TF-IDF"][0]
    full = res["SmartRecipe (full)"][0]
    assert full["F1"] > base["F1"]
    assert full["MAP"] > base["MAP"]
    assert full["Recall"] >= 0.9
