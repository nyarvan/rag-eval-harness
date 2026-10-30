import json

import pytest

from rag_eval.data import Document, Question
from rag_eval.generate import format_context
from rag_eval.records import RunRecord, read_records, write_records
from rag_eval.run import run_dense, summarize

DOCUMENTS = tuple(
    Document(title, title, f'{title.lower()} text')
    for title in ('A', 'B', 'C', 'D')
)


def make_question(question_id: str, gold=('A', 'B'), answer='x') -> Question:
    return Question(
        question_id, f'question {question_id}?', answer, 'bridge', gold, (),
    )


class FakeRetriever:
    def __init__(self, ranking: list[str]):
        self.ranking = ranking
        self.calls: list[tuple[int, int]] = []

    def retrieve_many(self, questions, k):
        self.calls.append((len(questions), k))
        return [self.ranking[:k] for _ in questions]


class FakeGenerator:
    def __init__(self, reply='x'):
        self.reply = reply
        self.calls: list[tuple[str, list[str]]] = []

    def answer(self, question, contexts):
        self.calls.append((question, [doc.title for doc in contexts]))
        return self.reply


def run(questions, ranking=('C', 'A', 'B', 'D'), **kwargs):
    return run_dense(
        questions, DOCUMENTS, FakeRetriever(list(ranking)), FakeGenerator(),
        **kwargs,
    )


def test_contexts_are_the_top_context_k_documents_in_rank_order():
    [record] = run([make_question('1')], retrieve_k=4, context_k=2)

    assert record.contexts == (
        format_context(DOCUMENTS[2]), format_context(DOCUMENTS[0]),
    )


def test_generator_gets_the_same_top_documents():
    generator = FakeGenerator()

    run_dense(
        [make_question('1')], DOCUMENTS, FakeRetriever(['C', 'A', 'B', 'D']),
        generator, retrieve_k=4, context_k=2,
    )

    assert generator.calls == [('question 1?', ['C', 'A'])]


def test_record_keeps_retrieved_titles_gold_and_reference():
    [record] = run([make_question('1', answer='Paris')], retrieve_k=3,
                   context_k=1)

    assert record.retrieved == ('C', 'A', 'B')
    assert record.gold_titles == ('A', 'B')
    assert record.reference == 'Paris'
    assert record.question_id == '1'
    assert record.question == 'question 1?'


def test_one_record_per_question_in_order():
    records = run([make_question('1'), make_question('2')])

    assert [record.question_id for record in records] == ['1', '2']


def test_all_questions_are_retrieved_in_one_batch():
    retriever = FakeRetriever(['A', 'B', 'C', 'D'])

    run_dense(
        [make_question('1'), make_question('2')], DOCUMENTS, retriever,
        FakeGenerator(), retrieve_k=3, context_k=2,
    )

    assert retriever.calls == [(2, 3)]


def test_context_k_above_retrieve_k_is_rejected():
    with pytest.raises(ValueError):
        run([make_question('1')], retrieve_k=2, context_k=3)


def make_record(**overrides) -> RunRecord:
    fields = dict(
        question_id='1', question='q?', contexts=('Title: A\nText: a',),
        answer='x', reference='x', retrieved=('A', 'B'),
        gold_titles=('A', 'B'),
    )
    return RunRecord(**{**fields, **overrides})


def test_records_round_trip_through_jsonl(tmp_path):
    path = tmp_path / 'runs' / 'dense.jsonl'
    records = [make_record(), make_record(question_id='2', answer='é')]

    write_records(path, records)

    assert read_records(path) == records


def test_jsonl_has_the_ragas_fields_one_object_per_line(tmp_path):
    path = tmp_path / 'dense.jsonl'

    write_records(path, [make_record(), make_record()])

    lines = path.read_text(encoding='utf-8').splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert {'question', 'contexts', 'answer', 'reference'} <= set(first)
    assert first['contexts'] == ['Title: A\nText: a']


def test_summarize_averages_retrieval_and_answer_metrics():
    perfect = make_record(retrieved=('A', 'B', 'C'), answer='x', reference='x')
    partial = make_record(
        retrieved=('C', 'A', 'B'), answer='New York City',
        reference='New York',
    )

    summary = summarize([perfect, partial])

    assert summary['recall@2'] == 0.75
    assert summary['all_gold@2'] == 0.5
    assert summary['all_gold@5'] == 1.0
    assert summary['em'] == 0.5
    assert summary['f1'] == pytest.approx(0.9)


def test_summarize_rejects_no_records():
    with pytest.raises(ValueError):
        summarize([])


def test_run_parser_rejects_non_positive_counts():
    from rag_eval.run import build_parser
    parser = build_parser()
    for flag in ('--n', '--retrieve-k', '--context-k', '--candidates-k'):
        for value in ('0', '-5'):
            with pytest.raises(SystemExit):
                parser.parse_args([flag, value])


def test_run_uses_the_shared_retriever_protocol_and_model_name():
    from rag_eval import hybrid, run
    from rag_eval.retrieve import DEFAULT_EMBEDDING_MODEL
    assert run.Retriever is hybrid.Retriever
    assert run.DEFAULT_MODEL == DEFAULT_EMBEDDING_MODEL
