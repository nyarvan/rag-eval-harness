import pytest

from rag_eval.data import Question
from rag_eval.latency import measure_latency, median_ms


def make_question(question_id: str) -> Question:
    return Question(question_id, f'q{question_id}?', 'a', 'bridge', (), ())


class FakeRetriever:
    def __init__(self):
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, question, k):
        self.calls.append((question.question_id, k))
        return []


def clock(*values):
    ticks = iter(values)
    return lambda: next(ticks)


def test_each_question_is_timed_on_its_own_after_the_warmup():
    retriever = FakeRetriever()
    questions = [make_question(str(i)) for i in range(3)]

    # warm-up on question 0; then (0 -> 2) for question 1, (2 -> 3) for 2
    times = measure_latency(
        retriever, questions, 5, timer=clock(0.0, 2.0, 2.0, 3.0)
    )

    assert times == [2.0, 1.0]
    assert retriever.calls == [('0', 5), ('1', 5), ('2', 5)]


def test_warmup_questions_are_not_timed():
    retriever = FakeRetriever()
    questions = [make_question(str(i)) for i in range(4)]

    times = measure_latency(
        retriever, questions, 3, warmup=2, timer=clock(0.0, 1.0, 1.0, 4.0)
    )

    assert times == [1.0, 3.0]
    assert [call[0] for call in retriever.calls] == ['0', '1', '2', '3']


def test_there_must_be_a_question_left_to_time():
    with pytest.raises(ValueError):
        measure_latency(FakeRetriever(), [make_question('0')], 5)


def test_negative_warmup_is_rejected():
    questions = [make_question('0'), make_question('1')]

    with pytest.raises(ValueError):
        measure_latency(FakeRetriever(), questions, 5, warmup=-1)


def test_median_is_reported_in_milliseconds():
    assert median_ms([0.010, 0.030, 0.020]) == pytest.approx(20.0)


def test_median_of_no_times_is_rejected():
    with pytest.raises(ValueError):
        median_ms([])


def test_latency_parser_rejects_non_positive_counts():
    from rag_eval.latency import build_parser
    parser = build_parser()
    for flag in ('--n', '--rerank-n', '--k'):
        with pytest.raises(SystemExit):
            parser.parse_args([flag, '0'])
