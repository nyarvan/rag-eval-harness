import re

import numpy as np
import pytest

from rag_eval.data import Document, Question
from rag_eval.retrieve import DenseRetriever

VOCAB = ('alpha', 'beta', 'gamma', 'apple', 'banana', 'cherry', 'delta')


class FakeEncoder:
    """Bag-of-words encoder over VOCAB that remembers what it was given."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def encode(self, inputs: list[str], /, **kwargs) -> np.ndarray:
        self.calls.append(list(inputs))
        vectors = np.zeros((len(inputs), len(VOCAB)), dtype=np.float32)
        for row, text in enumerate(inputs):
            for word in re.findall(r'[a-z]+', text.lower()):
                if word in VOCAB:
                    vectors[row, VOCAB.index(word)] += 1
        return vectors

    @property
    def document_calls(self) -> list[list[str]]:
        return [call for call in self.calls if call[0].startswith('Title:')]


def make_question(text: str, question_id: str = 'q') -> Question:
    return Question(question_id, text, 'a', 'bridge', (), ())


DOCUMENTS = (
    Document('Alpha', 'Alpha', 'alpha apple'),
    Document('Beta', 'Beta', 'beta banana'),
    Document('Gamma', 'Gamma', 'gamma cherry'),
)


def make_retriever(tmp_path, encoder=None, documents=DOCUMENTS):
    return DenseRetriever(
        'bge_m3', documents, subset_size=3, data_dir=tmp_path,
        encoder=encoder or FakeEncoder(),
    )


def test_retrieve_ranks_best_match_first(tmp_path):
    retriever = make_retriever(tmp_path)

    titles = retriever.retrieve(make_question('which banana beta?'), 3)

    assert titles[0] == 'Beta'


def test_retrieve_returns_at_most_k_titles(tmp_path):
    retriever = make_retriever(tmp_path)

    titles = retriever.retrieve(make_question('alpha'), 2)

    assert len(titles) == 2


def test_retrieve_returns_plain_strings(tmp_path):
    retriever = make_retriever(tmp_path)

    titles = retriever.retrieve(make_question('alpha'), 3)

    assert all(type(title) is str for title in titles)


def test_documents_are_encoded_as_title_and_text(tmp_path):
    encoder = FakeEncoder()

    make_retriever(tmp_path, encoder).retrieve(make_question('alpha'), 1)

    assert encoder.document_calls == [[
        'Title: Alpha\nText: alpha apple',
        'Title: Beta\nText: beta banana',
        'Title: Gamma\nText: gamma cherry',
    ]]


def test_document_embeddings_are_reused_from_disk(tmp_path):
    make_retriever(tmp_path).retrieve(make_question('alpha'), 1)
    second_encoder = FakeEncoder()

    titles = make_retriever(tmp_path, second_encoder).retrieve(
        make_question('gamma cherry'), 1
    )

    assert titles == ['Gamma']
    assert second_encoder.document_calls == []


def test_documents_are_encoded_once_per_retriever(tmp_path):
    encoder = FakeEncoder()
    retriever = make_retriever(tmp_path, encoder)

    retriever.retrieve(make_question('alpha'), 1)
    retriever.retrieve(make_question('beta'), 1)

    assert len(encoder.document_calls) == 1


def test_stale_cache_is_rebuilt_when_documents_change(tmp_path):
    make_retriever(tmp_path).retrieve(make_question('alpha'), 1)
    new_documents = (
        Document('Delta', 'Delta', 'delta alpha'),
        Document('Beta', 'Beta', 'beta banana'),
        Document('Gamma', 'Gamma', 'gamma cherry'),
    )
    encoder = FakeEncoder()

    titles = make_retriever(tmp_path, encoder, new_documents).retrieve(
        make_question('delta'), 1
    )

    assert titles == ['Delta']
    assert len(encoder.document_calls) == 1


def test_retrieve_many_matches_retrieve(tmp_path):
    retriever = make_retriever(tmp_path)
    questions = [
        make_question('alpha apple', 'q1'),
        make_question('gamma cherry', 'q2'),
    ]

    batched = retriever.retrieve_many(questions, 2)

    assert batched == [retriever.retrieve(q, 2) for q in questions]


def test_retrieve_many_encodes_all_questions_in_one_call(tmp_path):
    encoder = FakeEncoder()
    retriever = make_retriever(tmp_path, encoder)
    questions = [make_question('alpha', 'q1'), make_question('beta', 'q2')]

    retriever.retrieve_many(questions, 1)

    assert encoder.calls[-1] == ['alpha', 'beta']


def test_retrieve_many_of_nothing_is_empty(tmp_path):
    assert make_retriever(tmp_path).retrieve_many([], 3) == []


@pytest.mark.parametrize('k', [0, -1])
def test_retrieve_rejects_non_positive_k(tmp_path, k):
    with pytest.raises(ValueError):
        make_retriever(tmp_path).retrieve(make_question('alpha'), k)


def test_unknown_model_is_rejected_at_construction(tmp_path):
    with pytest.raises(ValueError):
        DenseRetriever(
            'no_such_model', DOCUMENTS, subset_size=3, data_dir=tmp_path,
            encoder=FakeEncoder(),
        )


def test_empty_corpus_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        make_retriever(tmp_path, documents=())


def test_default_embedding_model_is_bge_m3():
    from rag_eval.retrieve import DEFAULT_EMBEDDING_MODEL
    assert DEFAULT_EMBEDDING_MODEL == 'bge_m3'
