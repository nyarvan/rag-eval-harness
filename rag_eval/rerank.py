"""Cross-encoder reranking of a first-stage retriever's candidates.

The first stage (dense search) returns `candidates_k` titles; a cross-encoder
scores each (question, passage) pair and the best `k` are returned. Scores
are cached on disk, keyed by the pair's text, so a rerun scores only pairs it
has never seen. One cache file per model.
"""
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from .data import Document, Question
from .generate import format_context
from .hybrid import Retriever

DEFAULT_RERANKER = 'BAAI/bge-reranker-v2-m3'
DEFAULT_CANDIDATES_K = 30
DEFAULT_CACHE_DIR = 'data/rerank_cache'
MAX_LENGTH = 512  # tokens per pair; HotpotQA paragraphs are far shorter
BATCH_SIZE = 16


class PairScorer(Protocol):
    """Anything that scores (question, passage) pairs, higher is better."""

    def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> Any: ...


class RerankedRetriever:
    """Reorder a first-stage ranking by cross-encoder score.

    Same `retrieve` / `retrieve_many` interface as `DenseRetriever`, so it
    can replace it in a run. `k` may not exceed `candidates_k`: the
    reranker cannot return what the first stage did not pass on.
    """

    def __init__(
            self, first_stage: Retriever, documents: Sequence[Document], *,
            model_name: str = DEFAULT_RERANKER,
            candidates_k: int = DEFAULT_CANDIDATES_K,
            cache_dir: str | Path = DEFAULT_CACHE_DIR,
            scorer: PairScorer | None = None,
    ):
        if candidates_k <= 0:
            raise ValueError(f'candidates_k must be positive, got {candidates_k}')
        self._first_stage = first_stage
        self._by_title = {document.doc_id: document for document in documents}
        self._model_name = model_name
        self._candidates_k = candidates_k
        self._scorer = scorer
        self._cache_path = Path(cache_dir) / (
            model_name.replace('/', '__') + '.json'
        )
        self._scores = self._read_cache()

    def retrieve(self, question: Question, k: int) -> list[str]:
        """Titles of the `k` best paragraphs after reranking, best first."""
        return self.retrieve_many([question], k)[0]

    def retrieve_many(
            self, questions: Sequence[Question], k: int,
    ) -> list[list[str]]:
        """Like `retrieve`, for many questions."""
        if not 0 < k <= self._candidates_k:
            raise ValueError(
                f'k must be in [1, {self._candidates_k}], got {k}'
            )
        if not questions:
            return []
        candidates = self._first_stage.retrieve_many(
            questions, self._candidates_k
        )
        pairs = [
            (question.question, [self._by_title[title] for title in titles])
            for question, titles in zip(questions, candidates, strict=True)
        ]
        self._score_missing(pairs)
        return [
            self._rerank(question, titles, documents)[:k]
            for (question, documents), titles in zip(
                pairs, candidates, strict=True
            )
        ]

    def _rerank(
            self, question: str, titles: Sequence[str],
            documents: Sequence[Document],
    ) -> list[str]:
        scores = [self._scores[_pair_key(question, d)] for d in documents]
        # sorted() is stable: equal scores keep the first-stage order
        order = sorted(range(len(titles)), key=lambda i: -scores[i])
        return [titles[i] for i in order]

    def _score_missing(
            self, questions: Sequence[tuple[str, Sequence[Document]]],
    ) -> None:
        """Score every uncached pair in one scorer call, then save the cache."""
        missing: dict[str, tuple[str, str]] = {}
        for question, documents in questions:
            for document in documents:
                key = _pair_key(question, document)
                if key not in self._scores and key not in missing:
                    missing[key] = (question, format_context(document))
        if not missing:
            return
        predicted = self._get_scorer().predict(
            list(missing.values()),
            batch_size=BATCH_SIZE, show_progress_bar=False,
        )
        for key, score in zip(missing, predicted, strict=True):
            self._scores[key] = float(score)
        self._write_cache()

    def _get_scorer(self) -> PairScorer:
        if self._scorer is None:
            from sentence_transformers import CrossEncoder
            self._scorer = CrossEncoder(
                self._model_name, max_length=MAX_LENGTH
            )
        return self._scorer

    def _read_cache(self) -> dict[str, float]:
        if not self._cache_path.exists():
            return {}
        return json.loads(self._cache_path.read_text(encoding='utf-8'))

    def _write_cache(self) -> None:
        """Save all scores; the file appears all at once."""
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._cache_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self._scores), encoding='utf-8')
        os.replace(temporary, self._cache_path)


def _pair_key(question: str, document: Document) -> str:
    """SHA-256 of the question and the passage text it is scored against."""
    payload = json.dumps(
        [question, format_context(document)],
        ensure_ascii=False, separators=(',', ':'),
    )
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()
