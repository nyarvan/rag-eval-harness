import json

import pytest

from rag_eval.cli import main
from rag_eval.records import RunRecord, write_records


class Spy:
    """Stands in for the paid work and records how it was called."""

    def __init__(self):
        self.calls = []

    def __call__(self, args):
        self.calls.append(args)


def always_yes(question, assume_yes=False):
    return True


def always_no(question, assume_yes=False):
    return False


def only_with_flag(question, assume_yes=False):
    return assume_yes


def make_record(question_id, answer='x', retrieved=('A', 'B'),
                gold=('A', 'B')) -> RunRecord:
    return RunRecord(
        question_id=question_id, question=f'q{question_id}?', contexts=(),
        answer=answer, reference='x', retrieved=retrieved, gold_titles=gold,
    )


@pytest.fixture
def run_file(tmp_path):
    path = tmp_path / 'a.jsonl'
    write_records(path, [make_record(str(i)) for i in range(1, 4)])
    return path


def run_cli(argv, **kwargs):
    kwargs.setdefault('run_fn', Spy())
    kwargs.setdefault('ragas_fn', Spy())
    kwargs.setdefault('corpus_chars_fn', lambda: 400.0)
    kwargs.setdefault('confirm_fn', only_with_flag)
    return main(argv, **kwargs)


def test_run_shows_the_estimate_and_runs_with_yes(capsys):
    spy = Spy()

    code = run_cli(['run', '--n', '10', '--yes'], run_fn=spy)

    assert code == 0
    assert len(spy.calls) == 1 and spy.calls[0].n == 10
    assert 'Generation with gpt-4o-mini' in capsys.readouterr().out


def test_run_without_confirmation_runs_nothing(capsys):
    spy = Spy()

    code = run_cli(['run', '--n', '10'], run_fn=spy)

    assert code == 1
    assert spy.calls == []
    assert 'Cancelled' in capsys.readouterr().out


def test_run_asks_after_the_estimate_is_printed(capsys):
    seen = []

    def ask(question, assume_yes=False):
        seen.append(capsys.readouterr().out)
        return False

    run_cli(['run', '--n', '10'], confirm_fn=ask)

    assert 'requests' in seen[0]


def test_reasoning_runs_cost_more_output_tokens(capsys):
    run_cli(['run', '--n', '10', '--yes'])
    plain = capsys.readouterr().out
    run_cli(['run', '--n', '10', '--cot', '--yes'])
    reasoning = capsys.readouterr().out

    assert '640 output' in plain
    assert '5,120 output' in reasoning


def test_an_unpriced_model_is_reported_not_guessed(capsys):
    run_cli(['run', '--n', '10', '--chat-model', 'new-model', '--yes'])

    assert 'price unknown' in capsys.readouterr().out


def test_eval_prints_the_free_metrics_and_calls_nothing_paid(run_file, capsys):
    ragas = Spy()

    code = run_cli(['eval', str(run_file)], ragas_fn=ragas)

    out = capsys.readouterr().out
    assert code == 0 and ragas.calls == []
    assert 'em: 1.000' in out and 'all_gold@5: 1.000' in out
    assert '3 records' in out


def test_eval_reports_a_bad_file_with_its_line(tmp_path, capsys):
    path = tmp_path / 'bad.jsonl'
    path.write_text(json.dumps({'question_id': '1'}) + '\n', encoding='utf-8')

    code = run_cli(['eval', str(path)])

    err = capsys.readouterr().err
    assert code == 2
    assert f'{path}:1' in err


def test_eval_reports_a_missing_file(tmp_path, capsys):
    code = run_cli(['eval', str(tmp_path / 'missing.jsonl')])

    assert code == 2
    assert 'missing.jsonl' in capsys.readouterr().err


def test_eval_with_ragas_declined_keeps_the_free_metrics(run_file, capsys):
    ragas = Spy()

    code = run_cli(['eval', str(run_file), '--ragas'], ragas_fn=ragas)

    out = capsys.readouterr().out
    assert code == 1 and ragas.calls == []
    assert 'em: 1.000' in out and 'Cancelled' in out


def test_eval_with_ragas_and_yes_runs_the_judge(run_file, capsys):
    ragas = Spy()

    code = run_cli(['eval', str(run_file), '--ragas', '--yes'], ragas_fn=ragas)

    assert code == 0
    assert len(ragas.calls) == 1
    assert ragas.calls[0].run_file == str(run_file)
    assert 'RAGAS judging' in capsys.readouterr().out


def test_the_judge_estimate_counts_only_the_records_that_will_be_scored(
        run_file, capsys):
    run_cli(['eval', str(run_file), '--ragas', '--n', '2', '--yes'])

    # 2 questions x 10.5 requests for the four metrics
    assert '21 requests' in capsys.readouterr().out


def test_compare_prints_a_paired_table(tmp_path, capsys):
    good = tmp_path / 'good.jsonl'
    bad = tmp_path / 'bad.jsonl'
    write_records(good, [make_record(str(i)) for i in range(1, 4)])
    write_records(bad, [make_record(str(i), answer='no') for i in range(1, 4)])

    code = run_cli(['compare', str(bad), str(good)])

    out = capsys.readouterr().out
    assert code == 0
    assert 'good - bad' in out and 'em' in out and 'p (sign)' in out


def test_compare_adds_retrieval_metrics_when_the_files_have_them(
        run_file, tmp_path, capsys):
    other = tmp_path / 'b.jsonl'
    write_records(other, [make_record(str(i)) for i in range(1, 4)])

    run_cli(['compare', str(run_file), str(other)])

    assert 'all_gold@5' in capsys.readouterr().out


def test_compare_skips_retrieval_metrics_without_the_fields(tmp_path, capsys):
    a, b = tmp_path / 'a.jsonl', tmp_path / 'b.jsonl'
    for path in (a, b):
        write_records(path, [
            make_record(str(i), retrieved=(), gold=()) for i in range(1, 4)
        ])

    code = run_cli(['compare', str(a), str(b)])

    assert code == 0
    assert 'all_gold' not in capsys.readouterr().out


def test_compare_of_different_questions_is_an_error(
        run_file, tmp_path, capsys):
    other = tmp_path / 'other.jsonl'
    write_records(other, [make_record(str(i)) for i in (1, 2, 9)])

    code = run_cli(['compare', str(run_file), str(other)])

    assert code == 2
    assert '9' in capsys.readouterr().err


def test_compare_needs_two_files(run_file):
    with pytest.raises(SystemExit) as stop:
        run_cli(['compare', str(run_file)])

    assert stop.value.code == 2


def test_a_command_is_required():
    with pytest.raises(SystemExit) as stop:
        run_cli([])

    assert stop.value.code == 2
