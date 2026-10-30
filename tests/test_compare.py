import pytest

from rag_eval.compare import compare_rankings
from rag_eval.data import Question


def make_question(question_id: str, gold=('A', 'B')) -> Question:
    return Question(question_id, f'q{question_id}?', 'a', 'bridge', gold, ())


QUESTIONS = [make_question(str(i)) for i in range(1, 4)]
GOOD = [['A', 'B', 'C']] * 3  # both gold in the top 2 for every question
BAD = [['A', 'C', 'B']] * 3  # only one gold in the top 2 for every question


def only(comparisons, metric):
    [match] = [c for c in comparisons if c.metric == metric]
    return match


def test_a_clear_winner_has_all_the_wins_and_a_positive_difference():
    comparisons = compare_rankings(
        {'bad': BAD, 'good': GOOD}, QUESTIONS, ks=(2,), metrics=('all_gold',)
    )

    [comparison] = comparisons
    assert (comparison.a, comparison.b) == ('good', 'bad')
    assert comparison.metric == 'all_gold@2'
    assert comparison.mean_diff == pytest.approx(1.0)
    assert (comparison.wins, comparison.losses) == (3, 0)
    assert comparison.p_sign == pytest.approx(0.25)


def test_the_interval_contains_the_mean_difference():
    [comparison] = compare_rankings(
        {'bad': BAD, 'good': GOOD}, QUESTIONS, ks=(2,), metrics=('recall',)
    )

    assert comparison.low <= comparison.mean_diff <= comparison.high


def test_recall_counts_partial_credit():
    [comparison] = compare_rankings(
        {'bad': BAD, 'good': GOOD}, QUESTIONS, ks=(2,), metrics=('recall',)
    )

    assert comparison.mean_diff == pytest.approx(0.5)


def test_identical_rankings_do_not_differ():
    [comparison] = compare_rankings(
        {'a': GOOD, 'b': GOOD}, QUESTIONS, ks=(2,), metrics=('all_gold',)
    )

    assert comparison.mean_diff == 0.0
    assert (comparison.wins, comparison.losses) == (0, 0)
    assert comparison.p_sign == 1.0
    assert comparison.changed == ()


def test_changed_lists_the_questions_that_differ():
    mixed = [GOOD[0], BAD[1], GOOD[2]]

    [comparison] = compare_rankings(
        {'bad': BAD, 'mixed': mixed}, QUESTIONS, ks=(2,), metrics=('all_gold',)
    )

    assert comparison.changed == ('1', '3')


def test_losses_are_counted_too():
    mixed = [GOOD[0], BAD[1], BAD[2]]

    [comparison] = compare_rankings(
        {'good': GOOD, 'mixed': mixed}, QUESTIONS, ks=(2,),
        metrics=('all_gold',),
    )

    assert (comparison.wins, comparison.losses) == (0, 2)
    assert comparison.mean_diff < 0


def test_every_retriever_is_compared_with_the_first_by_default():
    comparisons = compare_rankings(
        {'base': BAD, 'x': GOOD, 'y': BAD}, QUESTIONS, ks=(2,),
        metrics=('all_gold',),
    )

    assert [(c.a, c.b) for c in comparisons] == [('x', 'base'), ('y', 'base')]


def test_an_explicit_baseline_can_be_chosen():
    comparisons = compare_rankings(
        {'x': GOOD, 'base': BAD}, QUESTIONS, ks=(2,), metrics=('all_gold',),
        baseline='base',
    )

    assert [(c.a, c.b) for c in comparisons] == [('x', 'base')]


def test_one_comparison_per_metric_and_k():
    comparisons = compare_rankings(
        {'bad': BAD, 'good': GOOD}, QUESTIONS, ks=(1, 2),
        metrics=('all_gold', 'recall'),
    )

    assert sorted(c.metric for c in comparisons) == [
        'all_gold@1', 'all_gold@2', 'recall@1', 'recall@2',
    ]
    assert only(comparisons, 'all_gold@1').mean_diff == 0.0


def test_the_bootstrap_is_reproducible_with_a_seed():
    first = compare_rankings(
        {'bad': BAD, 'good': GOOD}, QUESTIONS, ks=(2,), seed=7
    )
    second = compare_rankings(
        {'bad': BAD, 'good': GOOD}, QUESTIONS, ks=(2,), seed=7
    )

    assert first == second


def test_at_least_two_retrievers_are_needed():
    with pytest.raises(ValueError):
        compare_rankings({'only': GOOD}, QUESTIONS)


def test_an_unknown_baseline_is_rejected():
    with pytest.raises(ValueError):
        compare_rankings(
            {'a': GOOD, 'b': BAD}, QUESTIONS, baseline='missing'
        )


def test_no_questions_are_rejected():
    with pytest.raises(ValueError):
        compare_rankings({'a': [], 'b': []}, [])


def test_a_ranking_per_question_is_required():
    with pytest.raises(ValueError):
        compare_rankings({'a': GOOD, 'b': GOOD[:2]}, QUESTIONS)
