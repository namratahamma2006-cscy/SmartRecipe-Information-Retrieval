import pytest

from smartrecipe.preprocessing import (basic_preprocess, canonical_ingredient, lemmatize,
                                       preprocess, tokenize)


@pytest.mark.parametrize("word,lemma", [
    ("tomatoes", "tomato"), ("potatoes", "potato"), ("berries", "berry"),
    ("cookies", "cookie"), ("leaves", "leaf"), ("eggs", "egg"), ("dishes", "dish"),
    ("hummus", "hummus"), ("asparagus", "asparagus"), ("rice", "rice"),
])
def test_lemmatize(word, lemma):
    assert lemmatize(word) == lemma


def test_tokenize_strips_case_accents_punctuation_digits():
    assert tokenize("Jalapeño, 2 TOMATOES & basil!") == ["jalapeno", "tomato", "and", "basil"]


@pytest.mark.parametrize("raw,canonical", [
    ("Garbanzo Beans", "chickpea"), ("aubergine", "eggplant"), ("Prawns", "shrimp"),
    ("capsicum", "bell_pepper"), ("Red bell peppers", "bell_pepper"), ("curd", "yogurt"),
    ("spring onions", "green_onion"), ("coriander leaves", "cilantro"),
    ("extra virgin olive oil", "olive_oil"), ("fresh basil", "basil"),
    ("ground beef", "beef"), ("courgette", "zucchini"), ("chilli", "chili"),
])
def test_canonical_ingredient_synonyms_and_phrases(raw, canonical):
    assert canonical_ingredient(raw) == canonical


def test_preprocess_removes_stopwords_and_keeps_phrases():
    terms = preprocess("I want a recipe with bell peppers and garbanzo beans")
    assert terms == ["bell_pepper", "chickpea"]


def test_baseline_preprocess_has_no_synonyms():
    assert "chickpea" not in basic_preprocess("garbanzo beans")
