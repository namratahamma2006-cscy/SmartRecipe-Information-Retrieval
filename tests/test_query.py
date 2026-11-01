from smartrecipe.query import parse_query


def test_diet_time_course_detection():
    pq = parse_query("quick vegetarian breakfast")
    assert pq.diets == {"Vegetarian"}
    assert pq.max_time == 30
    assert pq.courses == {"Breakfast"}
    assert "breakfast" in pq.terms


def test_explicit_time_limits():
    assert parse_query("pasta under 20 minutes").max_time == 20
    assert parse_query("soup in 1 hour").max_time == 60
    assert parse_query("dinner within 45 mins").max_time == 45


def test_multiword_diets():
    assert parse_query("gluten free cake").diets == {"Gluten-Free"}
    assert parse_query("dairy-free dessert").diets == {"Dairy-Free"}
    assert parse_query("plant based bowl").diets == {"Vegan"}


def test_exclusions_ingredients_and_categories():
    pq = parse_query("curry without onion and no dairy, egg-free")
    assert "onion" in pq.exclude_ingredients
    assert {"dairy", "egg"} <= pq.exclude_categories
    assert "onion" not in pq.terms


def test_two_word_exclusion():
    pq = parse_query("stir fry without bell pepper")
    assert pq.exclude_ingredients == {"bell_pepper"}


def test_concept_expansion():
    pq = parse_query("spicy food")
    assert "chili" in pq.expansions
    assert pq.weighted_terms["spicy"] == 1.0
    assert pq.weighted_terms["chili"] == 0.5


def test_spelling_correction_uses_vocabulary():
    pq = parse_query("choclate desert", vocabulary={"chocolate", "dessert", "cake"})
    assert pq.corrections == {"choclate": "chocolate", "desert": "dessert"}
    assert pq.courses == {"Dessert"}


def test_alias_course_expansion():
    pq = parse_query("dinner")
    assert pq.courses == {"Main"}
    assert "main" in pq.expansions


def test_understanding_off_is_plain_preprocessing():
    pq = parse_query("quick vegan curry", understand=False)
    assert pq.diets == set() and pq.max_time is None
    assert pq.terms == ["quick", "vegan", "curry"]


def test_empty_query():
    pq = parse_query("   ")
    assert pq.terms == [] and not pq.weighted_terms
