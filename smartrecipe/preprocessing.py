"""Text preprocessing: normalisation, tokenisation, lemmatisation, phrase
detection, synonym recognition and stop-word removal.

Pipeline for any piece of text:

    raw text -> lowercase / strip accents & punctuation -> tokens
             -> rule-based lemmatisation ("tomatoes" -> "tomato")
             -> longest-match phrase & synonym mapping
                ("garbanzo beans" -> "chickpea", "bell peppers" -> "bell_pepper")
             -> stop-word removal

Multi-word ingredients become single underscore-joined tokens so that TF-IDF
treats "bell pepper" as one concept instead of "bell" + "pepper".
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from . import knowledge as kb

_IRREGULAR = {
    "leaves": "leaf", "halves": "half", "loaves": "loaf", "knives": "knife",
    "cookies": "cookie", "smoothies": "smoothie", "brownies": "brownie",
    "veggies": "veggie", "pies": "pie", "potatoes": "potato",
    "tomatoes": "tomato", "mangoes": "mango", "chillies": "chili",
    "chilies": "chili", "children": "child", "noodles": "noodles",
    "oats": "oats", "chips": "chip", "greens": "greens", "sprouts": "sprout",
}
_NO_STRIP = {
    "hummus", "asparagus", "couscous", "molasses", "swiss", "brussels", "citrus",
    "grass", "glass", "bass", "gas", "plus", "minus", "less", "tapas",
    "species", "series", "this", "yes",
}


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def lemmatize(token: str) -> str:
    """Very small rule-based English lemmatiser tuned for food vocabulary."""
    if token in _IRREGULAR:
        return _IRREGULAR[token]
    if token in _NO_STRIP or len(token) <= 3:
        return token
    if token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith("oes"):
        return token[:-2]
    if token.endswith(("ches", "shes", "sses", "xes", "zes")):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lowercase, remove accents/punctuation/digits and split into lemmas."""
    text = strip_accents(text.lower())
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z\s]", " ", text)
    return [lemmatize(t) for t in text.split()]


def _phrase_key(text: str) -> tuple[str, ...]:
    return tuple(tokenize(text))


@lru_cache(maxsize=1)
def _phrase_table() -> tuple[dict[tuple[str, ...], str], int]:
    """Map token sequences -> canonical underscore token.

    Includes synonym phrases and every multi-word canonical ingredient name
    known to the knowledge base, so they are kept together as one token.
    """
    table: dict[tuple[str, ...], str] = {}

    def canon_token(name: str) -> str:
        return "_".join(_phrase_key(name))

    canonical_names = set(kb.SYNONYMS.values()) | set(kb.FAMILIES) | set(kb.FAMILIES.values())
    canonical_names |= set(kb.SUBSTITUTES) | set(kb.PANTRY_STAPLES)
    for items in kb.CATEGORIES.values():
        canonical_names |= set(items)
    for terms in kb.CONCEPTS.values():
        canonical_names |= set(terms)
    canonical_names |= _EXTRA_PHRASES

    for name in canonical_names:
        key = _phrase_key(name)
        if key:
            table[key] = canon_token(name)
    # synonyms override (they map onto a *different* canonical form)
    for variant, canonical in kb.SYNONYMS.items():
        key = _phrase_key(variant)
        if key:
            table[key] = canon_token(canonical)
    max_len = max(len(k) for k in table)
    return table, max_len


# Extra multi-word terms that should stay together even though they are not
# referenced in the knowledge tables above.
_EXTRA_PHRASES = {
    "garam masala", "coconut milk", "sesame seed", "chia seed", "baking powder",
    "baking soda", "cocoa powder", "peanut butter", "bean sprout", "rice noodles",
    "egg noodles", "green curry paste", "fish sauce", "soy sauce", "miso paste",
    "balsamic vinegar", "white wine", "nutritional yeast", "burger bun",
    "tomato ketchup", "pizza dough", "sour cream", "maple syrup", "pine nut",
    "corn tortilla", "flour tortilla", "pita bread", "olive oil", "sesame oil",
    "brown sugar", "green onion", "bell pepper", "black bean", "kidney bean",
    "chili powder", "chili flakes", "vegetable broth", "chicken broth",
    "black pepper", "arborio rice", "basmati rice", "gluten free", "dairy free",
    "nut free", "plant based", "lactose free", "middle eastern", "stir fry",
    "fried rice", "chocolate chip", "ice cream",
}


def apply_phrases(tokens: list[str]) -> list[str]:
    """Greedy longest-match replacement of multi-word phrases and synonyms."""
    table, max_len = _phrase_table()
    out: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        for size in range(min(max_len, n - i), 0, -1):
            key = tuple(tokens[i:i + size])
            if key in table:
                out.append(table[key])
                i += size
                break
        else:
            out.append(tokens[i])
            i += 1
    return out


def preprocess(text: str, remove_stopwords: bool = True) -> list[str]:
    """Full pipeline: returns normalised terms ready for indexing/search."""
    terms = apply_phrases(tokenize(text))
    if remove_stopwords:
        terms = [t for t in terms if t not in kb.STOPWORDS and len(t) > 1]
    return terms


def basic_preprocess(text: str) -> list[str]:
    """Baseline pipeline (no lemmatisation, phrases or synonyms) used to
    measure how much the smart pipeline improves retrieval."""
    text = strip_accents(text.lower())
    tokens = re.sub(r"[^a-z\s]", " ", text).split()
    return [t for t in tokens if t not in kb.STOPWORDS and len(t) > 1]


_DESCRIPTORS = {
    "fresh", "chopped", "diced", "sliced", "minced", "large", "small", "medium",
    "ripe", "dried", "grated", "boneless", "skinless", "frozen", "canned",
    "cooked", "raw", "whole", "organic", "extra", "virgin", "finely", "roughly",
    "peeled", "crushed", "toasted", "zest", "handful", "pinch", "cup", "tbsp",
    "tsp", "gram", "kg", "ml", "litre", "piece", "can", "bunch", "sprig",
}


@lru_cache(maxsize=4096)
def canonical_ingredient(name: str) -> str:
    """Normalise a free-text ingredient name to its canonical token.

    >>> canonical_ingredient("Garbanzo Beans")
    'chickpea'
    >>> canonical_ingredient("Red bell peppers")
    'bell_pepper'
    """
    terms = apply_phrases(tokenize(name))
    terms = [t for t in terms if t not in kb.STOPWORDS and t not in _DESCRIPTORS]
    if not terms:
        return ""
    if len(terms) > 1:
        phrases = [t for t in terms if "_" in t]
        if phrases:
            return phrases[-1]
    return "_".join(terms)


def display_name(token: str) -> str:
    return token.replace("_", " ")
