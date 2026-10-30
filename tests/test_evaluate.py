import pytest

from rag_eval.evaluate import evaluate_records
from rag_eval.records import RunRecord


def make_record(question_id='1', answer='x', reference='x',
                retrieved=('A', 'B'), gold=('A', 'B')) -> RunRecord:
    return RunRecord(
        question_id=question_id, question='q?', contexts=(), answer=answer,
        reference=reference, retrieved=retrieved, gold_titles=gold,
    )


def test_answer_metrics_are_means_over_the_records():
    records = [
        make_record('1', answer='x', reference='x'),
        make_record('2', answer='New York City', reference='New York'),
    ]

    result = evaluate_records(records, ks=(2,))

    assert result['em'] == 0.5
    assert result['f1'] == pytest.approx(0.9)


def test_retrieval_metrics_are_added_when_every_record_has_the_fields():
    records = [
        make_record('1', retrieved=('A', 'B', 'C')),
        make_record('2', retrieved=('C', 'A', 'B')),
    ]

    result = evaluate_records(records, ks=(2, 3))

    assert result['recall@2'] == 0.75
    assert result['all_gold@2'] == 0.5
    assert result['all_gold@3'] == 1.0


def test_only_the_requested_ks_are_reported():
    result = evaluate_records([make_record()], ks=(5,))

    assert 'recall@5' in result and 'recall@2' not in result


def test_records_without_retrieval_fields_get_answer_metrics_only():
    records = [make_record('1', retrieved=(), gold=())]

    assert set(evaluate_records(records)) == {'em', 'f1'}


def test_retrieval_fields_in_only_some_records_are_an_error():
    records = [make_record('1'), make_record('2', retrieved=(), gold=())]

    with pytest.raises(ValueError, match='1 of 2'):
        evaluate_records(records)


def test_no_records_are_rejected():
    with pytest.raises(ValueError):
        evaluate_records([])
