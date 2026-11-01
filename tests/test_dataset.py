from smartrecipe.engine import load_recipes

REQUIRED = {"id", "name", "cuisine", "course", "time_minutes", "servings", "difficulty",
            "description", "tags", "ingredients", "instructions"}


def test_dataset_schema_and_uniqueness():
    recipes = load_recipes()
    assert len(recipes) >= 50
    assert len({r["id"] for r in recipes}) == len(recipes)
    for r in recipes:
        assert REQUIRED <= set(r), r["id"]
        assert r["difficulty"] in {"Easy", "Medium", "Hard"}
        assert r["time_minutes"] > 0 and r["servings"] > 0
        assert len(r["ingredients"]) >= 3 and len(r["instructions"]) >= 2
        assert all(i["name"] and i["quantity"] for i in r["ingredients"])


def test_derived_diets_are_consistent():
    by_id = {r["id"]: r for r in load_recipes()}
    assert "Vegan" in by_id["r05"]["diets"]               # chana masala
    assert "Vegetarian" not in by_id["r04"]["diets"]      # chicken tikka
    assert "Vegan" not in by_id["r06"]["diets"]           # paneer = dairy
    assert "Gluten-Free" not in by_id["r02"]["diets"]     # spaghetti
    assert "Nut-Free" not in by_id["r13"]["diets"]        # peanuts
    for r in by_id.values():
        if "Vegan" in r["diets"]:
            assert "Vegetarian" in r["diets"]
