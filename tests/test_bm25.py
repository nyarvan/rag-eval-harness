import pytest

from rag_eval.bm25 import BM25Retriever
from rag_eval.data import Document, Question

DOCUMENTS = (
    Document(
        'Iona Gaels', 'Iona Gaels',
        'The Gaels were led by head coach Tim Cluess in Metro Atlantic play.',
    ),
    Document(
        'Iowa State', 'Iowa State',
        'The Cyclones were coached by Steve Prohm in his first season.',
    ),
    Document(
        'Tosca', 'Tosca',
        'It was made at Cinecittà in Rome and based on an opera by Puccini.',
    ),
    Document('Lush', 'Lush', 'Lush were an English rock band formed in 1987.'),
)


def make_question(text: str, question_id: str = '1') -> Question:
    return Question(question_id, text, 'a', 'bridge', (), ())


@pytest.fixture(scope='module')
def retriever():
    return BM25Retriever(DOCUMENTS)


def test_a_rare_name_finds_the_paragraph_that_contains_it(retriever):
    titles = retriever.retrieve(make_question('Who is Tim Cluess?'), 3)

    assert titles[0] == 'Iona Gaels'


def test_title_words_count_too(retriever):
    titles = retriever.retrieve(make_question('Tell me about Tosca'), 3)

    assert titles[0] == 'Tosca'


def test_results_are_titles_best_first_and_at_most_k(retriever):
    titles = retriever.retrieve(make_question('coach Cluess Gaels'), 1)

    assert titles == ['Iona Gaels']


def test_stemming_matches_word_forms(retriever):
    titles = retriever.retrieve(make_question('Which team coaches Prohm?'), 2)

    assert titles[0] == 'Iowa State'


def test_paragraphs_sharing_no_word_with_the_query_are_not_returned(retriever):
    titles = retriever.retrieve(make_question('Cluess'), 4)

    assert titles == ['Iona Gaels']


def test_a_query_of_unknown_words_returns_nothing(retriever):
    assert retriever.retrieve(make_question('zzzxqy'), 3) == []


def test_a_query_of_only_stopwords_returns_nothing(retriever):
    assert retriever.retrieve(make_question('the of and'), 3) == []


def test_k_larger_than_the_corpus_is_fine(retriever):
    titles = retriever.retrieve(make_question('were'), 50)

    assert len(titles) <= len(DOCUMENTS)


def test_retrieve_many_matches_retrieve(retriever):
    questions = [
        make_question('Tim Cluess', '1'), make_question('Rome opera', '2'),
    ]

    batched = retriever.retrieve_many(questions, 2)

    assert batched == [retriever.retrieve(q, 2) for q in questions]
    assert batched[1][0] == 'Tosca'


def test_no_questions_give_no_results(retriever):
    assert retriever.retrieve_many([], 3) == []


@pytest.mark.parametrize('k', [0, -1])
def test_non_positive_k_is_rejected(retriever, k):
    with pytest.raises(ValueError):
        retriever.retrieve(make_question('Cluess'), k)


def test_an_empty_corpus_is_rejected():
    with pytest.raises(ValueError):
        BM25Retriever(())
