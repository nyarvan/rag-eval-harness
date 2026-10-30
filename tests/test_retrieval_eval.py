import weakref

import pytest

from rag_eval.data import Question
from rag_eval.retrieval_eval import (
    evaluate_retrieval,
    format_table,
    rank_in_turn,
)


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


class Throwaway(FakeRetriever):
    """A retriever that can be weakly referenced, to see when it is freed."""


def test_rank_in_turn_returns_each_retrievers_rankings():
    factories = {'good': lambda: GOOD, 'other': lambda: FakeRetriever(
        {'q1?': ['C'], 'q2?': ['D']}
    )}

    rankings = rank_in_turn(factories, QUESTIONS, 3)

    assert list(rankings) == ['good', 'other']
    assert rankings['good'] == [['A', 'B', 'C'], ['A', 'C', 'B']]
    assert rankings['other'] == [['C'], ['D']]


def test_rank_in_turn_builds_each_retriever_once_at_the_given_depth():
    built = []

    def make():
        built.append(1)
        return FakeRetriever({'q1?': ['A'], 'q2?': ['A']})

    retriever_calls = rank_in_turn({'r': make}, QUESTIONS, 7)

    assert len(built) == 1
    assert retriever_calls['r'] == [['A'], ['A']]


def test_rank_in_turn_frees_a_retriever_before_building_the_next():
    first_ref = []
    alive_when_second_is_built = []

    def make_first():
        retriever = Throwaway({'q1?': ['A'], 'q2?': ['A']})
        first_ref.append(weakref.ref(retriever))
        return retriever

    def make_second():
        alive_when_second_is_built.append(first_ref[0]() is not None)
        return Throwaway({'q1?': ['B'], 'q2?': ['B']})

    rank_in_turn({'first': make_first, 'second': make_second}, QUESTIONS, 2)

    assert alive_when_second_is_built == [False]


def test_table_lists_retrievers_and_scores_per_metric():
    results = evaluate_retrieval({'good': GOOD}, QUESTIONS, ks=(2, 3))

    table = format_table(results, ks=(2, 3))

    assert 'all_gold' in table and 'recall' in table
    assert 'good' in table
    assert '0.500' in table and '1.000' in table


def parse(*argv):
    from rag_eval.retrieval_eval import build_parser, validate_args
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    return args


@pytest.mark.parametrize('argv', [
    ['--n', '-5'], ['--n', '0'], ['--ks', '0'], ['--depth', '-1'],
    ['--compare-ks', '0'],
])
def test_non_positive_counts_are_rejected(argv):
    with pytest.raises(SystemExit):
        parse(*argv)


def test_ks_above_the_rerank_pool_are_rejected():
    with pytest.raises(SystemExit):
        parse('--ks', '40', '--retrievers', 'hybrid+rerank')


def test_wide_rerank_pool_allows_ks_up_to_fifty():
    args = parse(
        '--ks', '50', '--retrievers', 'hybrid+rerank50', '--compare-ks', '2'
    )
    assert max(args.ks) == 50


def test_wide_rerank_next_to_a_plain_one_is_capped_at_the_plain_pool():
    with pytest.raises(SystemExit):
        parse('--ks', '40', '--retrievers', 'hybrid+rerank50', 'hybrid+rerank')


def test_ks_above_the_depth_are_rejected_for_hybrid():
    with pytest.raises(SystemExit):
        parse('--ks', '20', '--depth', '10', '--retrievers', 'hybrid')


def test_ks_above_the_depth_are_fine_without_a_hybrid():
    args = parse('--ks', '20', '--depth', '10', '--retrievers', 'dense', 'bm25')
    assert max(args.ks) == 20


@pytest.mark.parametrize('retriever', ['hybrid+rerank', 'hybrid+rerank50'])
def test_depth_below_the_rerank_pool_is_rejected_up_front(retriever):
    with pytest.raises(SystemExit):
        parse('--retrievers', 'dense', retriever, '--depth', '20', '--ks', '5')
