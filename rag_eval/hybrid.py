"""Hybrid retrieval: fuse the rankings of several retrievers with RRF.

Reciprocal rank fusion (Cormack et al., 2009) scores a document by
`sum(1 / (k + rank))` over the rankings it appears in, with `rank` counted
from 1. It needs only ranks, so rankings from retrievers with incomparable
scores (cosine similarity, BM25) can be fused without calibration.
"""
from collections.abc import Sequence
from typing import Protocol

from .data import Question

DEFAULT_RRF_K = 60
DEFAULT_DEPTH = 100


class Retriever(Protocol):
    def retrieve_many(
            self, questions: Sequence[Question], k: int,
    ) -> list[list[str]]: ...


def rrf_fuse(
        rankings: Sequence[Sequence[str]], k: int = DEFAULT_RRF_K,
) -> list[str]:
    """All documents of the rankings, best fused score first.

    A document repeated within one ranking counts once, at its best rank. Ties
    keep the order in which the documents first appear.
    """
    if k < 0:
        raise ValueError(f'k must not be negative, got {k}')
    scores: dict[str, float] = {}  # insertion order = first appearance
    for ranking in rankings:
        for rank, title in enumerate(dict.fromkeys(ranking), start=1):
            scores[title] = scores.get(title, 0.0) + 1 / (k + rank)
    return sorted(scores, key=lambda title: -scores[title])


class HybridRetriever:
    """Fuse the rankings of several retrievers with `rrf_fuse`.

    Each retriever is asked for `depth` titles per question, and the fused
    ranking is cut to `k`. Same `retrieve` / `retrieve_many` interface as
    `DenseRetriever`, so it can replace it in a run.
    """

    def __init__(
            self, retrievers: Sequence[Retriever], *,
            depth: int = DEFAULT_DEPTH, rrf_k: int = DEFAULT_RRF_K,
    ):
        if not retrievers:
            raise ValueError('at least one retriever is required')
        if depth <= 0:
            raise ValueError(f'depth must be positive, got {depth}')
        self._retrievers = tuple(retrievers)
        self._depth = depth
        self._rrf_k = rrf_k

    def retrieve(self, question: Question, k: int) -> list[str]:
        """Titles of the `k` best paragraphs after fusion, best first."""
        return self.retrieve_many([question], k)[0]

    def retrieve_many(
            self, questions: Sequence[Question], k: int,
    ) -> list[list[str]]:
        """Like `retrieve`, for many questions."""
        if not 0 < k <= self._depth:
            raise ValueError(f'k must be in [1, {self._depth}], got {k}')
        if not questions:
            return []
        per_retriever = [
            retriever.retrieve_many(questions, self._depth)
            for retriever in self._retrievers
        ]
        return [
            rrf_fuse([ranked[i] for ranked in per_retriever], self._rrf_k)[:k]
            for i in range(len(questions))
        ]
