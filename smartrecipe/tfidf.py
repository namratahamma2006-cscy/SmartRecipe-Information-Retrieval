"""From-scratch TF-IDF vector space model with cosine similarity (NumPy).

    tf(t, d)  = 1 + log(count(t, d))                    (sublinear term frequency)
    idf(t)    = log((1 + N) / (1 + df(t))) + 1          (smoothed inverse doc frequency)
    w(t, d)   = tf(t, d) * idf(t), then each document vector is L2-normalised
    cos(q, d) = (q . d) / (|q| |d|)  == q . d  because both are unit length
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Mapping

import numpy as np


class TfidfIndex:
    def __init__(self, documents: list[list[str]] | None = None):
        self.vocabulary: dict[str, int] = {}
        self.idf: np.ndarray = np.zeros(0)
        self.matrix: np.ndarray = np.zeros((0, 0))
        if documents is not None:
            self.fit(documents)

    # ------------------------------------------------------------------ fit
    def fit(self, documents: list[list[str]]) -> "TfidfIndex":
        n_docs = len(documents)
        df: Counter[str] = Counter()
        for doc in documents:
            df.update(set(doc))
        self.vocabulary = {term: i for i, term in enumerate(sorted(df))}
        self.idf = np.array(
            [math.log((1 + n_docs) / (1 + df[t])) + 1.0 for t in sorted(df)],
            dtype=float,
        )
        self.matrix = np.vstack([self._vectorize(Counter(doc)) for doc in documents]) \
            if documents else np.zeros((0, len(self.vocabulary)))
        return self

    # ------------------------------------------------------------ transform
    def _vectorize(self, weights: Mapping[str, float]) -> np.ndarray:
        vec = np.zeros(len(self.vocabulary), dtype=float)
        for term, count in weights.items():
            idx = self.vocabulary.get(term)
            if idx is not None and count > 0:
                vec[idx] = (1.0 + math.log(count)) if count >= 1 else count
        vec *= self.idf
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def transform_query(self, terms: Iterable[str] | Mapping[str, float]) -> np.ndarray:
        """Turn query terms (or a term -> weight mapping, used for weighted
        query expansion) into a unit-length TF-IDF vector."""
        weights = terms if isinstance(terms, Mapping) else Counter(terms)
        return self._vectorize(weights)

    def cosine_similarities(self, query_vec: np.ndarray) -> np.ndarray:
        if not query_vec.any():
            return np.zeros(self.matrix.shape[0])
        return self.matrix @ query_vec

    # --------------------------------------------------------------- helpers
    def top_terms(self, doc_index: int, k: int = 8) -> list[tuple[str, float]]:
        row = self.matrix[doc_index]
        inverse = {i: t for t, i in self.vocabulary.items()}
        idx = np.argsort(row)[::-1][:k]
        return [(inverse[i], float(row[i])) for i in idx if row[i] > 0]

    def contributing_terms(self, query_vec: np.ndarray, doc_index: int, k: int = 6):
        """Terms that contribute most to the cosine score (explainability)."""
        contrib = self.matrix[doc_index] * query_vec
        inverse = {i: t for t, i in self.vocabulary.items()}
        idx = np.argsort(contrib)[::-1][:k]
        return [(inverse[i], float(contrib[i])) for i in idx if contrib[i] > 0]
