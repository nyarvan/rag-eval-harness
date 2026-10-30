import pytest

from rag_eval.data import Question
from rag_eval.hybrid import HybridRetriever, rrf_fuse


def make_question(text: str = 'q?', question_id: str = '1') -> Question:
    return Question(question_id, text, 'a', 'bridge', (), ())


class FakeRetriever:
    """Returns a fixed ranking per question text and records every call."""

    def __init__(self, rankings: dict[str, list[str]]):
        self.rankings = rankings
        self.calls: list[tuple[int, int]] = []

    def retrieve_many(self, questions, k):
        self.calls.append((len(questions), k))
        return [self.rankings[question.question][:k] for question in questions]


def test_single_ranking_keeps_its_order():
    assert rrf_fuse([['a', 'b', 'c']]) == ['a', 'b', 'c']


def test_a_document_both_rankings_agree_on_rises():
    # c: 1/63 + 1/61, a: 1/61 only, b: 1/62 only (k = 60)
    assert rrf_fuse([['a', 'b', 'c'], ['c']]) == ['c', 'a', 'b']


def test_ranks_are_one_based():
    # a: 1/2 + 1/3, b: 1/2 (k = 1); with 0-based ranks a and b would tie
    assert rrf_fuse([['a'], ['b', 'a']], k=1) == ['a', 'b']


def test_ties_keep_the_order_of_first_appearance():
    assert rrf_fuse([['a', 'b'], ['b', 'a']]) == ['a', 'b']


def test_documents_from_every_ranking_are_kept():
    fused = rrf_fuse([['a', 'b'], ['c', 'd']])

    assert sorted(fused) == ['a', 'b', 'c', 'd']


def test_a_repeated_document_counts_once_at_its_best_rank():
    assert rrf_fuse([['a', 'b', 'a']]) == ['a', 'b']
    assert rrf_fuse([['a', 'b', 'a'], ['b']]) == ['b', 'a']


def test_no_rankings_give_nothing():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []


def test_negative_k_is_rejected():
    with pytest.raises(ValueError):
        rrf_fuse([['a']], k=-1)


def make_hybrid(dense, bm25, depth=10):
    return HybridRetriever([dense, bm25], depth=depth)


def test_hybrid_fuses_the_rankings_of_its_retrievers():
    dense = FakeRetriever({'q?': ['A', 'B', 'C']})
    bm25 = FakeRetriever({'q?': ['C', 'D', 'A']})

    titles = make_hybrid(dense, bm25).retrieve(make_question(), 4)

    # A and C tie on 1/61 + 1/63, B and D tie on 1/62
    assert titles == ['A', 'C', 'B', 'D']


def test_hybrid_returns_at_most_k_titles():
    dense = FakeRetriever({'q?': ['A', 'B', 'C']})
    bm25 = FakeRetriever({'q?': ['C', 'D', 'A']})

    assert len(make_hybrid(dense, bm25).retrieve(make_question(), 2)) == 2


def test_each_retriever_is_asked_for_depth_not_k():
    dense = FakeRetriever({'q?': ['A']})
    bm25 = FakeRetriever({'q?': ['A']})

    make_hybrid(dense, bm25, depth=7).retrieve_many([make_question()], 3)

    assert dense.calls == [(1, 7)]
    assert bm25.calls == [(1, 7)]


def test_hybrid_fuses_each_question_on_its_own():
    dense = FakeRetriever({'x?': ['A', 'B'], 'y?': ['C', 'D']})
    bm25 = FakeRetriever({'x?': ['B'], 'y?': ['D']})
    questions = [make_question('x?', '1'), make_question('y?', '2')]

    assert make_hybrid(dense, bm25).retrieve_many(questions, 1) == [['B'], ['D']]


def test_a_short_ranking_from_one_retriever_is_fine():
    dense = FakeRetriever({'q?': ['A', 'B']})
    bm25 = FakeRetriever({'q?': []})

    assert make_hybrid(dense, bm25).retrieve(make_question(), 2) == ['A', 'B']


def test_no_questions_give_no_results_and_no_calls():
    dense = FakeRetriever({})
    bm25 = FakeRetriever({})

    assert make_hybrid(dense, bm25).retrieve_many([], 3) == []
    assert dense.calls == [] and bm25.calls == []


@pytest.mark.parametrize('k', [0, -1, 11])
def test_k_outside_one_to_depth_is_rejected(k):
    retriever = make_hybrid(FakeRetriever({}), FakeRetriever({}), depth=10)

    with pytest.raises(ValueError):
        retriever.retrieve(make_question(), k)


def test_at_least_one_retriever_is_required():
    with pytest.raises(ValueError):
        HybridRetriever([], depth=10)


def test_depth_must_be_positive():
    with pytest.raises(ValueError):
        HybridRetriever([FakeRetriever({})], depth=0)
