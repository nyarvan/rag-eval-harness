import asyncio
import math
from types import SimpleNamespace

import pytest

from rag_eval.ragas_eval import (
    METRIC_INPUTS,
    RecordScores,
    build_parser,
    read_scores,
    score_record,
    score_records,
    summarize_scores,
    write_scores,
)
from rag_eval.records import RunRecord


def make_record(question_id: str = '1', **overrides) -> RunRecord:
    fields = dict(
        question_id=question_id, question='q?',
        contexts=('Title: A\nText: a', 'Title: B\nText: b'),
        answer='ans', reference='ref', retrieved=('A', 'B'),
        gold_titles=('A', 'B'),
    )
    return RunRecord(**{**fields, **overrides})


class FakeMetric:
    """Metric stand-in: records the keyword arguments it was called with."""

    def __init__(self, value=0.5, error: Exception | None = None, delay=0.0):
        self.value = value
        self.error = error
        self.delay = delay
        self.calls: list[dict] = []

    async def ascore(self, **kwargs):
        self.calls.append(kwargs)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return SimpleNamespace(value=self.value)


def fake_metrics(**overrides):
    return {name: FakeMetric() for name in METRIC_INPUTS} | overrides


def run(coroutine):
    return asyncio.run(coroutine)


def test_each_metric_gets_exactly_the_inputs_it_needs():
    metrics = fake_metrics()

    run(score_record(make_record(), metrics))

    for name, metric in metrics.items():
        assert set(metric.calls[0]) == set(METRIC_INPUTS[name]), name


def test_inputs_map_to_the_record_fields():
    metrics = fake_metrics()
    record = make_record()

    run(score_record(record, metrics))

    call = metrics['faithfulness'].calls[0]
    assert call['user_input'] == 'q?'
    assert call['response'] == 'ans'
    assert call['retrieved_contexts'] == [
        'Title: A\nText: a', 'Title: B\nText: b',
    ]
    assert metrics['factual_correctness'].calls[0]['reference'] == 'ref'


def test_scores_are_the_metric_values_as_floats():
    metrics = fake_metrics(faithfulness=FakeMetric(value=1))

    scores = run(score_record(make_record(), metrics))

    assert scores['faithfulness'] == 1.0
    assert isinstance(scores['faithfulness'], float)
    assert set(scores) == set(METRIC_INPUTS)


def test_a_failing_metric_gives_none_and_does_not_stop_the_others():
    metrics = fake_metrics(context_recall=FakeMetric(error=RuntimeError('x')))

    scores = run(score_record(make_record(), metrics))

    assert scores['context_recall'] is None
    assert scores['faithfulness'] == 0.5


def test_nan_value_is_treated_as_missing():
    metrics = fake_metrics(faithfulness=FakeMetric(value=math.nan))

    assert run(score_record(make_record(), metrics))['faithfulness'] is None


def test_score_records_keeps_order_and_ids():
    records = [make_record('a'), make_record('b'), make_record('c')]

    rows = run(score_records(records, fake_metrics(), concurrency=2))

    assert [row.question_id for row in rows] == ['a', 'b', 'c']
    assert all(row.scores['faithfulness'] == 0.5 for row in rows)


def test_concurrency_limit_is_respected():
    in_flight = 0
    peak = 0

    class Counting(FakeMetric):
        async def ascore(self, **kwargs):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return SimpleNamespace(value=1.0)

    metrics = {name: Counting() for name in METRIC_INPUTS}
    records = [make_record(str(i)) for i in range(6)]

    run(score_records(records, metrics, concurrency=2))

    # two records at a time, each running its four metrics at once
    assert peak <= 2 * len(METRIC_INPUTS)


def test_concurrency_must_be_positive():
    with pytest.raises(ValueError):
        run(score_records([make_record()], fake_metrics(), concurrency=0))


def test_summary_averages_over_scored_rows_and_counts_failures():
    rows = [
        RecordScores('1', {'faithfulness': 1.0, 'context_recall': 0.0}),
        RecordScores('2', {'faithfulness': 0.5, 'context_recall': None}),
    ]

    summary = summarize_scores(rows)

    assert summary['faithfulness'] == pytest.approx(0.75)
    assert summary['context_recall'] == 0.0
    assert summary['failed:context_recall'] == 1
    assert summary['failed:faithfulness'] == 0


def test_summary_of_a_metric_with_no_scores_is_none():
    rows = [RecordScores('1', {'faithfulness': None})]

    assert summarize_scores(rows)['faithfulness'] is None


def test_summary_rejects_no_rows():
    with pytest.raises(ValueError):
        summarize_scores([])


def test_scores_round_trip_through_jsonl(tmp_path):
    path = tmp_path / 'out' / 'scores.jsonl'
    rows = [RecordScores('1', {'faithfulness': 0.25, 'context_recall': None})]

    write_scores(path, rows)

    assert read_scores(path) == rows


def test_non_numeric_value_is_treated_as_missing():
    metrics = fake_metrics(faithfulness=FakeMetric(value='n/a'))

    scores = run(score_record(make_record(), metrics))

    assert scores['faithfulness'] is None
    assert scores['context_recall'] == 0.5


def test_ragas_parser_rejects_non_positive_counts():
    parser = build_parser()
    for flags in (['--n', '-5'], ['--n', '0'], ['--concurrency', '0']):
        with pytest.raises(SystemExit):
            parser.parse_args(['run.jsonl', *flags])
