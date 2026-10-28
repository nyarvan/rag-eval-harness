"""Dense retrieval over the HotpotQA corpus, reusing the week-1 harness.

Paragraph embeddings are cached on disk under the harness cache layout
`<data_dir>/<dataset>-<subset_size>/<model>/documents.npz`. Queries are cheap
to encode, so they are not cached.
"""
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
from harness.cache import DEFAULT_DATA_DIR, documents_path
from harness.embed import Embedding, EmbeddingFile, Encoder, InputMode
from harness.models import get_spec

from .data import Document, Question

DEFAULT_DATASET = 'hotpotqa'
DOCUMENT_MODE = InputMode.TITLE_TEXT


class DenseRetriever:
    """Rank corpus paragraphs by cosine similarity to the question.

    The model is built (and the corpus encoded) on the first retrieval, and
    only when no valid cache file exists. `subset_size` is the number of
    questions the corpus was sampled for; it names the cache directory.
    """

    def __init__(
            self, model_name: str, documents: Sequence[Document], *,
            subset_size: int, dataset: str = DEFAULT_DATASET,
            data_dir: str | Path = DEFAULT_DATA_DIR,
            encoder: Encoder | None = None,
    ):
        self._spec = get_spec(model_name)  # fail fast on an unknown name
        if not documents:
            raise ValueError('documents must not be empty')
        self._documents = tuple(documents)
        self._path = documents_path(
            dataset, subset_size, model_name, data_dir
        )
        self._encoder = encoder
        self._index: EmbeddingFile | None = None

    def retrieve(self, question: Question, k: int) -> list[str]:
        """Titles of the `k` best paragraphs, best first."""
        return self.retrieve_many([question], k)[0]

    def retrieve_many(
            self, questions: Sequence[Question], k: int,
    ) -> list[list[str]]:
        """Like `retrieve`, for many questions encoded in one batch."""
        if k <= 0:
            raise ValueError(f'k must be positive, got {k}')
        if not questions:
            return []
        documents = self._load_index()
        embedding = Embedding(self._get_encoder(), DOCUMENT_MODE)
        # Positions, not question ids, key the results: ids may repeat
        query_ids = np.array([str(i) for i in range(len(questions))])
        queries = {
            'document_ids': query_ids,
            'embeddings': embedding.encode([q.question for q in questions]),
        }
        found = embedding.search(queries, documents, top_k=k)
        return [_ranked_titles(found[query_id]) for query_id in query_ids]

    def _get_encoder(self) -> Encoder:
        if self._encoder is None:
            self._encoder = self._spec.create()
        return self._encoder

    def _load_index(self) -> EmbeddingFile:
        """Document embeddings: from memory, then disk, then the model."""
        if self._index is None:
            self._index = self._read_cache() or self._build_index()
        return self._index

    def _document_ids(self) -> np.ndarray:
        return np.array([document.doc_id for document in self._documents])

    def _read_cache(self) -> EmbeddingFile | None:
        """Cached embeddings, or None if missing or for other documents."""
        if not self._path.exists():
            return None
        with np.load(self._path) as saved:
            if not np.array_equal(saved['document_ids'], self._document_ids()):
                return None
            return {name: saved[name] for name in saved.files}

    def _build_index(self) -> EmbeddingFile:
        pairs = [
            (document.doc_id, asdict(document))
            for document in self._documents
        ]
        embedding = Embedding(self._get_encoder(), DOCUMENT_MODE)
        embeddings = embedding.encode(
            Embedding.build_document_inputs(pairs, DOCUMENT_MODE)
        )
        document_ids = self._document_ids()
        embedding.save_document_embeddings(self._path, document_ids)
        return {'document_ids': document_ids, 'embeddings': embeddings}


def _ranked_titles(scores: dict[str, float]) -> list[str]:
    """Titles sorted by descending score, as plain `str`."""
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [str(title) for title, _ in ranked]
