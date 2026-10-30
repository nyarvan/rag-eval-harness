"""BM25 retrieval over the HotpotQA corpus, with the `bm25s` library.

Words are lowercased, stripped of English stopwords and stemmed. A paragraph
is its title followed by its text. Paragraphs that share no word with the
question are not returned, so a ranking may be shorter than `k`; this keeps
zero-score paragraphs out of a fusion with another retriever.
"""
from collections.abc import Sequence

import bm25s
import Stemmer

from .data import Document, Question

STEMMER_LANGUAGE = 'english'
STOPWORDS = 'en'


class BM25Retriever:
    """Rank corpus paragraphs by BM25 score against the question."""

    def __init__(self, documents: Sequence[Document]):
        if not documents:
            raise ValueError('documents must not be empty')
        self._titles = [document.doc_id for document in documents]
        self._stemmer = Stemmer.Stemmer(STEMMER_LANGUAGE)
        corpus_tokens = self._tokenize(
            [f'{document.title}. {document.text}' for document in documents]
        )
        self._index = bm25s.BM25()
        self._index.index(corpus_tokens, show_progress=False)

    def retrieve(self, question: Question, k: int) -> list[str]:
        """Titles of the `k` best paragraphs, best first."""
        return self.retrieve_many([question], k)[0]

    def retrieve_many(
            self, questions: Sequence[Question], k: int,
    ) -> list[list[str]]:
        """Like `retrieve`, for many questions."""
        if k <= 0:
            raise ValueError(f'k must be positive, got {k}')
        if not questions:
            return []
        query_tokens = self._tokenize([q.question for q in questions])
        indices, scores = self._index.retrieve(
            query_tokens, k=min(k, len(self._titles)), show_progress=False
        )
        return [
            [self._titles[i] for i, score in zip(row, row_scores) if score > 0]
            for row, row_scores in zip(indices.tolist(), scores.tolist())
        ]

    def _tokenize(self, texts: Sequence[str]):
        return bm25s.tokenize(
            list(texts), stopwords=STOPWORDS, stemmer=self._stemmer,
            show_progress=False,
        )
