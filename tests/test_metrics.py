import pytest

from rag_eval.metrics import (
    all_gold_at_k,
    exact_match,
    f1,
    normalize_answer,
    recall_at_k,
)


def test_recall_is_one_when_all_gold_in_top_k():
    assert recall_at_k(['A', 'B', 'C'], {'A', 'B'}, 2) == 1.0


def test_recall_is_half_when_one_of_two_gold_found():
    assert recall_at_k(['A', 'C', 'B'], {'A', 'B'}, 2) == 0.5


def test_recall_ignores_gold_ranked_after_k():
    assert recall_at_k(['C', 'A'], {'A'}, 1) == 0.0


def test_recall_accepts_k_larger_than_retrieved():
    assert recall_at_k(['A'], {'A', 'B'}, 10) == 0.5


def test_recall_counts_duplicate_gold_once():
    assert recall_at_k(['A'], ['A', 'A'], 1) == 1.0


def test_recall_counts_duplicate_retrieved_once():
    assert recall_at_k(['A', 'A', 'C'], {'A', 'B'}, 3) == 0.5


def test_recall_rejects_empty_gold():
    with pytest.raises(ValueError):
        recall_at_k(['A'], [], 1)


@pytest.mark.parametrize('k', [0, -1])
def test_recall_rejects_non_positive_k(k):
    with pytest.raises(ValueError):
        recall_at_k(['A'], {'A'}, k)


def test_all_gold_true_when_both_found():
    assert all_gold_at_k(['A', 'B'], {'A', 'B'}, 2) is True


def test_all_gold_false_when_one_missing():
    assert all_gold_at_k(['A', 'C', 'B'], {'A', 'B'}, 2) is False


def test_normalize_answer_strips_case_punctuation_articles():
    assert normalize_answer('The  Beatles.') == 'beatles'


def test_exact_match_ignores_normalized_differences():
    assert exact_match('The Beatles.', 'beatles') == 1.0


def test_exact_match_is_zero_for_different_answers():
    assert exact_match('Beatles', 'Stones') == 0.0


def test_f1_partial_overlap():
    assert f1('New York City', 'New York') == pytest.approx(0.8)


def test_f1_is_one_for_identical_answers():
    assert f1('Paris', 'paris') == 1.0


def test_f1_is_zero_without_overlap():
    assert f1('Paris', 'London') == 0.0


def test_f1_counts_token_multiplicity():
    # common = {red: 1, blue: 1} -> P = 2/3, R = 2/3
    assert f1('red red blue', 'red blue blue') == pytest.approx(2 / 3)


def test_f1_yes_no_mismatch_gets_no_partial_credit():
    assert f1('yes', 'no') == 0.0
    assert f1('yes it is', 'yes') == 0.0


def test_f1_yes_no_match():
    assert f1('Yes.', 'yes') == 1.0
