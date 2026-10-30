import json

import pytest

from rag_eval.records import RunRecord, read_records, write_records

VALID = {
    'question_id': '1', 'question': 'q?', 'contexts': ['Title: A\nText: a'],
    'answer': 'x', 'reference': 'x', 'retrieved': ['A'], 'gold_titles': ['A'],
}


def write_lines(path, *objects):
    path.write_text(
        '\n'.join(
            line if isinstance(line, str) else json.dumps(line)
            for line in objects
        ) + '\n',
        encoding='utf-8',
    )


def test_a_valid_line_becomes_a_record(tmp_path):
    path = tmp_path / 'run.jsonl'
    write_lines(path, VALID)

    [record] = read_records(path)

    assert record == RunRecord(
        question_id='1', question='q?', contexts=('Title: A\nText: a',),
        answer='x', reference='x', retrieved=('A',), gold_titles=('A',),
    )


def test_lists_become_tuples():
    record = RunRecord(**VALID)

    assert record.contexts == ('Title: A\nText: a',)
    assert isinstance(record.retrieved, tuple)


def test_retrieved_and_gold_titles_are_optional(tmp_path):
    path = tmp_path / 'run.jsonl'
    minimal = {k: VALID[k] for k in (
        'question_id', 'question', 'contexts', 'answer', 'reference',
    )}
    write_lines(path, minimal)

    [record] = read_records(path)

    assert record.retrieved == () and record.gold_titles == ()


def test_unknown_fields_are_ignored(tmp_path):
    path = tmp_path / 'run.jsonl'
    write_lines(path, {**VALID, 'latency_ms': 12.5})

    [record] = read_records(path)

    assert not hasattr(record, 'latency_ms')


def test_blank_lines_are_skipped(tmp_path):
    path = tmp_path / 'run.jsonl'
    write_lines(path, VALID, '', VALID)

    assert len(read_records(path)) == 2


@pytest.mark.parametrize(
    'missing', ['question_id', 'question', 'contexts', 'answer', 'reference']
)
def test_a_missing_required_field_names_the_file_and_line(tmp_path, missing):
    path = tmp_path / 'run.jsonl'
    broken = {k: v for k, v in VALID.items() if k != missing}
    write_lines(path, VALID, broken)

    with pytest.raises(ValueError) as error:
        read_records(path)

    message = str(error.value)
    assert f'{path}:2' in message
    assert missing in message


def test_contexts_must_be_a_list_not_a_string(tmp_path):
    path = tmp_path / 'run.jsonl'
    write_lines(path, {**VALID, 'contexts': 'Title: A\nText: a'})

    with pytest.raises(ValueError, match='contexts'):
        read_records(path)


def test_a_non_text_answer_is_rejected(tmp_path):
    path = tmp_path / 'run.jsonl'
    write_lines(path, {**VALID, 'answer': 42})

    with pytest.raises(ValueError, match='answer'):
        read_records(path)


@pytest.mark.parametrize('field', ['question_id', 'question'])
def test_an_empty_id_or_question_is_rejected(tmp_path, field):
    path = tmp_path / 'run.jsonl'
    write_lines(path, {**VALID, field: ''})

    with pytest.raises(ValueError, match=field):
        read_records(path)


def test_an_empty_answer_is_allowed(tmp_path):
    path = tmp_path / 'run.jsonl'
    write_lines(path, {**VALID, 'answer': ''})

    assert read_records(path)[0].answer == ''


def test_invalid_json_names_the_line(tmp_path):
    path = tmp_path / 'run.jsonl'
    write_lines(path, VALID, '{not json')

    with pytest.raises(ValueError, match=r'run\.jsonl:2'):
        read_records(path)


def test_records_cannot_be_changed():
    record = RunRecord(**VALID)

    with pytest.raises(ValueError):
        record.answer = 'other'


def test_written_records_read_back_unchanged(tmp_path):
    path = tmp_path / 'out' / 'run.jsonl'
    records = [RunRecord(**VALID), RunRecord(**{**VALID, 'answer': 'é'})]

    write_records(path, records)

    assert read_records(path) == records


def test_the_file_has_one_json_object_per_line_with_the_ragas_fields(tmp_path):
    path = tmp_path / 'run.jsonl'

    write_records(path, [RunRecord(**VALID), RunRecord(**VALID)])

    lines = path.read_text(encoding='utf-8').splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert {'question', 'contexts', 'answer', 'reference'} <= set(first)
    assert first['contexts'] == ['Title: A\nText: a']
