"""Natural-language query understanding.

Turns a free-text request such as

    "quick spicy vegan curry without onion under 25 minutes"

into a structured query:

    terms        -> ["spicy", "curry"]            (for TF-IDF)
    expansions   -> {"chili": 0.5, ...}           (concept expansion)
    diets        -> {"Vegan"}                     (hard filter)
    max_time     -> 25                            (hard filter)
    exclude_ingredients -> {"onion"}              (hard filter)
    courses / cuisines   -> soft ranking boosts
    corrections  -> spelling fixes that were applied
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from . import knowledge as kb
from .preprocessing import apply_phrases, preprocess, tokenize

EXPANSION_WEIGHT = 0.5


def _tok(text: str) -> str:
    return "_".join(tokenize(text))


DIET_TOKENS = {_tok(k): v for k, v in kb.DIET_WORDS.items()}
COURSE_TOKENS = {_tok(k): v for k, v in kb.COURSE_WORDS.items()}
CUISINE_TOKENS = {_tok(k): v for k, v in kb.CUISINE_WORDS.items()}
CATEGORY_TOKENS = {_tok(k): v for k, v in kb.CATEGORY_WORDS.items()}
CONCEPT_TOKENS = {
    _tok(k): ["_".join(apply_phrases(tokenize(t))) for t in v]
    for k, v in kb.CONCEPTS.items()
}
QUICK_TOKENS = {_tok(w) for w in kb.QUICK_WORDS}
EASY_TOKENS = {"easy", "simple", "beginner", "basic"}
INTENT_VOCAB = (
    set(DIET_TOKENS) | set(COURSE_TOKENS) | set(CUISINE_TOKENS)
    | set(CONCEPT_TOKENS) | QUICK_TOKENS | EASY_TOKENS
)

_TIME_PATTERNS = [
    (re.compile(r"\b(?:under|less than|below|within|in|max|maximum|up to|upto|at most|<)\s*"
                r"(\d+)\s*(?:minutes?|mins?|m)\b"), 1),
    (re.compile(r"\b(?:under|less than|below|within|in|max|up to|at most)\s*"
                r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)\b"), 60),
    (re.compile(r"\b(\d+)\s*(?:minutes?|mins?)\b"), 1),
    (re.compile(r"\b(?:under|within|in)\s+an?\s+hour\b"), None),
]
_EXCLUDE_RE = re.compile(
    r"\b(?:without|no|not|excluding|exclude|minus|skip|avoid|hold the|free of)\s+"
    r"([a-z]+)(?:\s+([a-z]+))?"
)
_FREE_RE = re.compile(r"\b([a-z]+)[\s-]+free\b")


@dataclass
class ParsedQuery:
    raw: str = ""
    terms: list[str] = field(default_factory=list)
    expansions: dict[str, float] = field(default_factory=dict)
    diets: set[str] = field(default_factory=set)
    max_time: int | None = None
    exclude_ingredients: set[str] = field(default_factory=set)
    exclude_categories: set[str] = field(default_factory=set)
    courses: set[str] = field(default_factory=set)
    cuisines: set[str] = field(default_factory=set)
    prefer_easy: bool = False
    corrections: dict[str, str] = field(default_factory=dict)

    @property
    def weighted_terms(self) -> dict[str, float]:
        weights: dict[str, float] = {}
        for t in self.terms:
            weights[t] = weights.get(t, 0.0) + 1.0
        for t, w in self.expansions.items():
            weights[t] = weights.get(t, 0.0) + w
        return weights

    def summary(self) -> list[str]:
        """Human-readable list of what the parser understood."""
        out = []
        if self.corrections:
            out.append("Spelling: " + ", ".join(f"{a} → {b}" for a, b in self.corrections.items()))
        if self.diets:
            out.append("Diet: " + ", ".join(sorted(self.diets)))
        if self.max_time:
            out.append(f"Max time: {self.max_time} min")
        if self.exclude_ingredients or self.exclude_categories:
            ex = sorted(self.exclude_ingredients | self.exclude_categories)
            out.append("Exclude: " + ", ".join(e.replace("_", " ") for e in ex))
        if self.courses:
            out.append("Course: " + ", ".join(sorted(self.courses)))
        if self.cuisines:
            out.append("Cuisine: " + ", ".join(sorted(self.cuisines)))
        if self.prefer_easy:
            out.append("Prefers easy recipes")
        if self.expansions:
            out.append("Expanded with: " + ", ".join(t.replace("_", " ") for t in self.expansions))
        return out


def _correct(token: str, vocabulary: set[str]) -> str | None:
    if token in vocabulary or len(token) < 4:
        return None
    match = difflib.get_close_matches(token, vocabulary, n=1, cutoff=0.8)
    return match[0] if match else None


def parse_query(text: str, vocabulary: set[str] | None = None,
                understand: bool = True) -> ParsedQuery:
    """Parse a natural-language query.

    vocabulary: index vocabulary used for spelling correction.
    understand: if False, only preprocess (no intents/expansion/correction);
                used for ablation in the evaluation.
    """
    pq = ParsedQuery(raw=text)
    if not text or not text.strip():
        return pq
    if not understand:
        pq.terms = preprocess(text)
        return pq

    work = " " + text.lower().replace("-", " ") + " "

    # 1. cooking time constraints -------------------------------------------
    for pattern, factor in _TIME_PATTERNS:
        m = pattern.search(work)
        if m:
            minutes = 60 if factor is None else int(float(m.group(1)) * factor)
            pq.max_time = minutes if pq.max_time is None else min(pq.max_time, minutes)
            work = work[:m.start()] + " " + work[m.end():]

    # 2. exclusions: "without onion", "no dairy", "egg-free" ------------------
    def _exclude(token: str) -> None:
        if token in CATEGORY_TOKENS:
            pq.exclude_categories.add(CATEGORY_TOKENS[token])
        elif token:
            pq.exclude_ingredients.add(token)

    def _sub_exclusion(m: re.Match) -> str:
        w1, w2 = m.group(1), m.group(2)
        if w2:
            two = apply_phrases(tokenize(f"{w1} {w2}"))
            if len(two) == 1:          # a known two-word ingredient
                _exclude(two[0])
                return " "
        one = apply_phrases(tokenize(w1))
        if one:
            _exclude(one[0])
        return " " + (w2 or "") + " "

    work = _EXCLUDE_RE.sub(_sub_exclusion, work)

    def _sub_free(m: re.Match) -> str:
        tok = _tok(m.group(1))
        if tok in CATEGORY_TOKENS and f"{tok}_free" not in DIET_TOKENS:
            _exclude(tok)
            return " "
        return m.group(0)

    work = _FREE_RE.sub(_sub_free, work)

    # 3. tokenise, correct spelling, extract intents ---------------------------
    vocab = set(vocabulary or ()) | INTENT_VOCAB
    for token in preprocess(work):
        fixed = _correct(token, vocab) if vocabulary else None
        if fixed:
            pq.corrections[token] = fixed
            token = fixed
        if token in DIET_TOKENS:
            pq.diets.add(DIET_TOKENS[token])
            continue
        if token in QUICK_TOKENS:
            pq.max_time = min(pq.max_time or kb.QUICK_MINUTES, kb.QUICK_MINUTES)
            continue
        if token in EASY_TOKENS:
            pq.prefer_easy = True
            continue
        if token == "free":
            continue
        # course / cuisine aliases ("dinner" -> main, "greek" -> mediterranean)
        # are recorded as intents and expanded to the canonical label term
        for table, bucket in ((COURSE_TOKENS, pq.courses), (CUISINE_TOKENS, pq.cuisines)):
            if token in table:
                bucket.add(table[token])
                label = _tok(table[token])
                if label != token:
                    pq.expansions[label] = max(pq.expansions.get(label, 0), EXPANSION_WEIGHT)
        if token in CONCEPT_TOKENS:
            for exp in CONCEPT_TOKENS[token]:
                if exp != token:
                    pq.expansions[exp] = max(pq.expansions.get(exp, 0), EXPANSION_WEIGHT)
        pq.terms.append(token)

    # do not expand into an ingredient the user explicitly excluded
    for ex in pq.exclude_ingredients:
        pq.expansions.pop(ex, None)
    return pq
