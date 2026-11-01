import pytest

from smartrecipe.engine import SmartRecipeEngine, ingredients_match, load_recipes


@pytest.fixture(scope="module")
def engine():
    return SmartRecipeEngine(load_recipes())


def ids(resp):
    return [r.id for r in resp.results]


def test_synonym_query_finds_chickpea_recipes(engine):
    top = ids(engine.search("garbanzo bean dishes", top_k=4))
    assert set(top) == {"r05", "r36", "r37", "r44"}


def test_prawn_maps_to_shrimp(engine):
    assert set(ids(engine.search("prawn recipes", top_k=2))) == {"r13", "r46"}


def test_scores_are_sorted_and_bounded(engine):
    resp = engine.search("spicy curry", top_k=20)
    scores = [r.score for r in resp.results]
    assert scores == sorted(scores, reverse=True)
    assert all(0 <= s <= 1 for s in scores)
    assert all(0 <= r.text_score <= 1 + 1e-9 for r in resp.results)


@pytest.mark.parametrize("diet,forbidden", [
    ("Vegetarian", {"meat", "fish", "seafood"}),
    ("Vegan", {"meat", "fish", "seafood", "dairy", "egg", "honey"}),
    ("Gluten-Free", {"gluten"}),
    ("Dairy-Free", {"dairy"}),
    ("Nut-Free", {"nuts"}),
])
def test_diet_filters(engine, diet, forbidden):
    resp = engine.search("dinner", diets=[diet], top_k=0)
    assert resp.results
    for r in resp.results:
        assert diet in r.recipe["diets"]
        assert not forbidden & set(r.recipe["categories"])


def test_diet_from_query_text(engine):
    resp = engine.search("vegan curry", top_k=0)
    assert resp.results
    assert all("Vegan" in r.recipe["diets"] for r in resp.results)


def test_time_filter(engine):
    resp = engine.search("dinner under 20 minutes", top_k=0)
    assert resp.results and all(r.recipe["time_minutes"] <= 20 for r in resp.results)


def test_exclusion_removes_ingredient_and_family(engine):
    resp = engine.search("pasta without spaghetti", top_k=0)
    assert resp.results
    assert all("spaghetti" not in r.recipe["ingredient_tokens"] for r in resp.results)
    resp = engine.search("cheesy food", exclude=["cheese"], top_k=0)
    for r in resp.results:   # family: cheese also excludes cheddar / mozzarella / parmesan
        assert not {"cheese", "cheddar", "mozzarella", "parmesan", "feta", "paneer"} & set(
            r.recipe["ingredient_tokens"])


def test_pantry_matching_coverage_and_missing(engine):
    resp = engine.search("", ["garbanzo beans", "lemon", "garlic", "tahini"], top_k=3)
    assert resp.pantry["garbanzo beans"] == "chickpea"
    best = resp.results[0]
    assert best.id == "r36"   # hummus
    assert set(best.matched) == {"chickpea", "tahini", "lemon", "garlic"}
    assert set(best.missing) == {"olive oil", "cumin"}    # salt is a pantry staple
    assert best.coverage == pytest.approx(4 / 6)
    assert best.usage == 1.0


def test_staples_toggle(engine):
    r = engine.search("", ["chickpea", "tahini", "lemon", "garlic"], top_k=1,
                      assume_staples=False).results[0]
    assert "salt" in r.missing


def test_substitution_flags_owned_substitute(engine):
    resp = engine.search("lemon herb grilled chicken", ["tofu", "lime"], top_k=1)
    subs = resp.results[0].substitutions
    assert ("tofu", True) in subs["chicken"]
    assert ("lime", True) in subs["lemon"]


def test_pantry_typo_recognition(engine):
    assert engine.recognise_ingredients(["tomatos", "avacado"]) == {
        "tomatos": "tomato", "avacado": "avocado"}


def test_shopping_suggestions(engine):
    resp = engine.search("", ["chickpea", "tahini", "lemon", "garlic"])
    names = [n for n, _ in resp.shopping]
    assert "cumin" in names and all(c >= 1 for _, c in resp.shopping)


def test_family_matching():
    assert ingredients_match("pasta", "spaghetti")
    assert ingredients_match("cheddar", "cheese")
    assert not ingredients_match("rice", "pasta")


def test_no_match_returns_empty(engine):
    assert engine.search("xylophone zebra").results == []


def test_filters_only_returns_all_matching(engine):
    resp = engine.search("", diets=["Vegan"], top_k=0)
    assert len(resp.results) == resp.candidates > 0


def test_explanation_terms(engine):
    res = engine.search("mushroom risotto", top_k=1).results[0]
    assert res.id == "r03"
    assert {"mushroom", "risotto"} & {t for t, _ in res.explanation}


def test_baseline_engine_misses_synonyms():
    base = SmartRecipeEngine(load_recipes(), smart_text=False, query_understanding=False)
    assert base.search("garbanzo").results == []
