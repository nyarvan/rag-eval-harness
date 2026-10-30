import pytest

from rag_eval.data import Question
from rag_eval.retrieval_eval import evaluate_retrieval, format_table


def make_question(question_id: str, gold=('A', 'B')) -> Question:
    return Question(question_id, f'q{question_id}?', 'a', 'bridge', gold, ())


class FakeRetriever:
    """Returns a fixed ranking per question text and records every call."""

    def __init__(self, rankings: dict[str, list[str]]):
        self.rankings = rankings
        self.calls: list[tuple[int, int]] = []

    def retrieve_many(self, questions, k):
        self.calls.append((len(questions), k))
        return [self.rankings[question.question][:k] for question in questions]


QUESTIONS = [make_question('1'), make_question('2')]
GOOD = FakeRetriever({'q1?': ['A', 'B', 'C'], 'q2?': ['A', 'C', 'B']})


def test_scores_are_means_over_the_questions():
    results = evaluate_retrieval({'good': GOOD}, QUESTIONS, ks=(2, 3))

    assert results['good']['recall@2'] == 0.75
    assert results['good']['all_gold@2'] == 0.5
    assert results['good']['recall@3'] == 1.0
    assert results['good']['all_gold@3'] == 1.0


def test_every_retriever_gets_its_own_row():
    other = FakeRetriever({'q1?': ['C', 'D'], 'q2?': ['C', 'D']})

    results = evaluate_retrieval({'good': GOOD, 'bad': other}, QUESTIONS, (2,))

    assert list(results) == ['good', 'bad']
    assert results['bad']['all_gold@2'] == 0.0


def test_each_retriever_is_asked_once_at_the_largest_k():
    retriever = FakeRetriever({'q1?': ['A', 'B'], 'q2?': ['A', 'B']})

    evaluate_retrieval({'r': retriever}, QUESTIONS, ks=(2, 10, 5))

    assert retriever.calls == [(2, 10)]


def test_no_questions_are_rejected():
    with pytest.raises(ValueError):
        evaluate_retrieval({'good': GOOD}, [], ks=(2,))


def test_no_ks_are_rejected():
    with pytest.raises(ValueError):
        evaluate_retrieval({'good': GOOD}, QUESTIONS, ks=())


def test_table_lists_retrievers_and_scores_per_metric():
    results = evaluate_retrieval({'good': GOOD}, QUESTIONS, ks=(2, 3))

    table = format_table(results, ks=(2, 3))

    assert 'all_gold' in table and 'recall' in table
    assert 'good' in table
    assert '0.500' in table and '1.000' in table
