"""SmartRecipe retrieval engine.

Ranking model
-------------
For every candidate recipe that passes the hard filters (diet, time, exclusions,
cuisine/course chosen in the UI) we compute up to three signals in [0, 1]:

* text      - TF-IDF cosine similarity between the query and the recipe
* coverage  - share of the recipe's (non-staple) ingredients the user has
* usage     - share of the user's ingredients the recipe actually uses

The final score is a weighted average over the signals that are available
(text only, pantry only, or both) plus small soft-intent boosts when the
recipe matches a course / cuisine / "easy" preference mentioned in the query.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field
from pathlib import Path

from . import knowledge as kb
from .preprocessing import basic_preprocess, canonical_ingredient, display_name, preprocess
from .query import ParsedQuery, parse_query
from .tfidf import TfidfIndex

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "recipes.json"

DEFAULT_WEIGHTS = {"text": 0.55, "coverage": 0.30, "usage": 0.15}
COURSE_BOOST = 0.08
CUISINE_BOOST = 0.08
EASY_BOOST = 0.03

# field weights used when building the smart document representation
FIELD_WEIGHTS = {"name": 3, "ingredients": 3, "tags": 2, "description": 2,
                 "cuisine": 2, "course": 2, "diet": 1, "instructions": 1}


def _canon(name: str) -> str:
    return canonical_ingredient(name)


FAMILY = {_canon(k): _canon(v) for k, v in kb.FAMILIES.items()}
STAPLES = {_canon(s) for s in kb.PANTRY_STAPLES}
SUBS = {_canon(k): v for k, v in kb.SUBSTITUTES.items()}
CATEGORY_OF: dict[str, set[str]] = {}
for _cat, _items in kb.CATEGORIES.items():
    for _item in _items:
        CATEGORY_OF.setdefault(_canon(_item), set()).add(_cat)


def ingredient_categories(token: str) -> set[str]:
    cats = set(CATEGORY_OF.get(token, set()))
    if token in FAMILY:
        cats |= CATEGORY_OF.get(FAMILY[token], set())
    return cats


def ingredients_match(have: str, need: str) -> bool:
    """True if a user's ingredient satisfies a recipe ingredient.

    Exact canonical match, or related through an ingredient family:
    user "pasta" satisfies recipe "spaghetti"; user "cheddar" satisfies "cheese".
    """
    if have == need:
        return True
    return FAMILY.get(need) == have or FAMILY.get(have) == need


def load_recipes(path: Path | str = DATA_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        recipes = json.load(fh)
    for r in recipes:
        r["ingredient_tokens"] = [_canon(i["name"]) for i in r["ingredients"]]
        cats: set[str] = set()
        for tok in r["ingredient_tokens"]:
            cats |= ingredient_categories(tok)
        r["categories"] = sorted(cats)
        r["diets"] = [d for d, forbidden in kb.DIET_RULES.items()
                      if not cats.intersection(forbidden)]
    return recipes


@dataclass
class RecipeResult:
    recipe: dict
    score: float
    text_score: float | None
    coverage: float | None
    usage: float | None
    boost: float
    matched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    substitutions: dict[str, list[tuple[str, bool]]] = field(default_factory=dict)
    explanation: list[tuple[str, float]] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.recipe["id"]


@dataclass
class SearchResponse:
    results: list[RecipeResult]
    parsed: ParsedQuery
    pantry: dict[str, str]                 # user text -> recognised canonical token
    shopping: list[tuple[str, int]]        # (ingredient, #recipes it would complete)
    candidates: int


class SmartRecipeEngine:
    """Recipe retrieval engine.

    smart_text:          lemmatisation + phrase/synonym handling + field weighting.
    query_understanding: intent parsing, concept expansion, spelling correction,
                         soft boosts. Turning both off gives a plain TF-IDF baseline.
    """

    def __init__(self, recipes: list[dict] | None = None, smart_text: bool = True,
                 query_understanding: bool = True):
        self.recipes = recipes if recipes is not None else load_recipes()
        self.smart_text = smart_text
        self.query_understanding = query_understanding and smart_text
        self.index = TfidfIndex([self._document(r) for r in self.recipes])
        self.vocabulary = set(self.index.vocabulary)
        self.known_ingredients = sorted({t for r in self.recipes for t in r["ingredient_tokens"]})
        self._known_ingredient_set = set(self.known_ingredients) | set(FAMILY.values()) | set(SUBS)
        self._display = {}
        for r in self.recipes:
            for tok, ing in zip(r["ingredient_tokens"], r["ingredients"]):
                self._display.setdefault(tok, ing["name"])

    def display(self, token: str) -> str:
        """Human-friendly name for a canonical token."""
        return self._display.get(token, display_name(token))

    # ------------------------------------------------------------ indexing
    def _document(self, r: dict) -> list[str]:
        if not self.smart_text:
            text = " ".join([
                r["name"], r["description"], " ".join(r["tags"]), r["cuisine"], r["course"],
                " ".join(i["name"] for i in r["ingredients"]), " ".join(r["instructions"]),
            ])
            return basic_preprocess(text)
        w = FIELD_WEIGHTS
        terms: list[str] = []
        terms += preprocess(r["name"]) * w["name"]
        terms += [t for t in r["ingredient_tokens"]] * w["ingredients"]
        terms += preprocess(" ".join(r["tags"])) * w["tags"]
        terms += preprocess(r["description"]) * w["description"]
        terms += preprocess(r["cuisine"]) * w["cuisine"]
        terms += preprocess(r["course"]) * w["course"]
        terms += preprocess(" ".join(r["diets"])) * w["diet"]
        terms += preprocess(" ".join(r["instructions"])) * w["instructions"]
        return terms

    # ------------------------------------------------------- pantry parsing
    def recognise_ingredients(self, items: list[str]) -> dict[str, str]:
        """Map free-text pantry items to canonical ingredient tokens,
        resolving synonyms ("garbanzo beans" -> chickpea) and typos."""
        out: dict[str, str] = {}
        for item in items:
            if not item or not item.strip():
                continue
            tok = _canon(item)
            if tok and tok not in self._known_ingredient_set:
                close = difflib.get_close_matches(tok, self._known_ingredient_set, n=1, cutoff=0.75)
                if close:
                    tok = close[0]
            if tok:
                out[item.strip()] = tok
        return out

    # --------------------------------------------------------------- search
    def search(self, query: str = "", pantry: list[str] | None = None, *,
               diets: list[str] | None = None, max_time: int | None = None,
               cuisines: list[str] | None = None, courses: list[str] | None = None,
               exclude: list[str] | None = None, top_k: int = 10,
               weights: dict[str, float] | None = None,
               assume_staples: bool = True) -> SearchResponse:
        weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        if self.smart_text:
            parsed = parse_query(query, self.vocabulary, understand=self.query_understanding)
        else:
            parsed = ParsedQuery(raw=query, terms=basic_preprocess(query or ""))

        pantry_map = self.recognise_ingredients(pantry or [])
        have = set(pantry_map.values())

        # ---- hard filters (UI + understood from the query)
        diet_filter = set(diets or []) | parsed.diets
        time_limit = min([t for t in (max_time, parsed.max_time) if t] or [0]) or None
        excl_ing = {_canon(e) for e in (exclude or []) if e.strip()} | parsed.exclude_ingredients
        excl_cat = set(parsed.exclude_categories)

        qvec = self.index.transform_query(parsed.weighted_terms)
        # a query that had content terms counts as a text query even when none
        # of its terms are in the vocabulary (then nothing matches textually)
        has_text = bool(parsed.weighted_terms)
        sims = self.index.cosine_similarities(qvec)

        results: list[RecipeResult] = []
        candidates = 0
        filtered_recipes = []
        for i, r in enumerate(self.recipes):
            if diet_filter and not diet_filter.issubset(r["diets"]):
                continue
            if time_limit and r["time_minutes"] > time_limit:
                continue
            if cuisines and r["cuisine"] not in cuisines:
                continue
            if courses and r["course"] not in courses:
                continue
            toks = r["ingredient_tokens"]
            if excl_ing and any(ingredients_match(e, t) or e == t for e in excl_ing for t in toks):
                continue
            if excl_cat and excl_cat.intersection(r["categories"]):
                continue
            candidates += 1
            filtered_recipes.append(r)

            text_score = float(sims[i]) if has_text else None
            coverage = usage = None
            matched: list[str] = []
            missing: list[str] = []
            needed = [t for t in dict.fromkeys(toks) if not (assume_staples and t in STAPLES)]
            if have:
                matched = [t for t in needed if any(ingredients_match(h, t) for h in have)]
                missing = [t for t in needed if t not in matched]
                coverage = len(matched) / len(needed) if needed else 1.0
                used = {h for h in have if any(ingredients_match(h, t) for t in toks)}
                usage = len(used) / len(have)
            else:
                missing = needed

            # relevance gate: must match the text query or use a pantry item
            if has_text and have:
                if (text_score or 0) <= 0 and not usage:
                    continue
            elif has_text and (text_score or 0) <= 0:
                continue
            elif have and not usage:
                continue

            signals = {"text": text_score, "coverage": coverage, "usage": usage}
            avail = {k: v for k, v in signals.items() if v is not None}
            total_w = sum(weights[k] for k in avail)
            base = sum(weights[k] * v for k, v in avail.items()) / total_w if total_w else 0.0

            boost = 0.0
            if self.query_understanding:
                if parsed.courses and r["course"] in parsed.courses:
                    boost += COURSE_BOOST
                if parsed.cuisines and (r["cuisine"] in parsed.cuisines or any(
                        r["cuisine"] in kb.CUISINE_GROUPS.get(c, []) for c in parsed.cuisines)):
                    boost += CUISINE_BOOST
                if parsed.prefer_easy and r["difficulty"] == "Easy":
                    boost += EASY_BOOST

            subs = {}
            for m in missing:
                options = SUBS.get(m, [])
                if options:
                    subs[m] = [(o, _canon(o) in have) for o in options]

            results.append(RecipeResult(
                recipe=r, score=min(1.0, base + boost), text_score=text_score,
                coverage=coverage, usage=usage, boost=boost,
                matched=[self.display(m) for m in matched],
                missing=[self.display(m) for m in missing],
                substitutions={self.display(k): v for k, v in subs.items()},
                explanation=[(self.display(t), s) for t, s in
                             self.index.contributing_terms(qvec, i)] if has_text else [],
            ))

        results.sort(key=lambda x: (-x.score, x.recipe["time_minutes"], x.recipe["name"]))
        shopping = self._shopping_suggestions(filtered_recipes, have, assume_staples) if have else []
        return SearchResponse(results[:top_k] if top_k else results, parsed, pantry_map,
                              shopping, candidates)

    # ------------------------------------------------------- shopping list
    def _shopping_suggestions(self, recipes: list[dict], have: set[str], assume_staples: bool,
                              max_missing: int = 2, limit: int = 5) -> list[tuple[str, int]]:
        """Which single ingredients would unlock the most 'almost complete'
        recipes (at most `max_missing` ingredients missing)?"""
        counts: dict[str, int] = {}
        for r in recipes:
            needed = [t for t in dict.fromkeys(r["ingredient_tokens"])
                      if not (assume_staples and t in STAPLES)]
            missing = [t for t in needed if not any(ingredients_match(h, t) for h in have)]
            if 0 < len(missing) <= max_missing and len(missing) < len(needed):
                for m in missing:
                    counts[m] = counts.get(m, 0) + 1
        ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:limit]
        return [(self.display(k), v) for k, v in ranked]

    def get(self, recipe_id: str) -> dict | None:
        return next((r for r in self.recipes if r["id"] == recipe_id), None)
