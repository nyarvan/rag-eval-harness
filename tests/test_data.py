import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import rag_eval.data as data

from rag_eval.data import (
    DEFAULT_N,
    Document,
    HotpotSubset,
    Question,
    build_subset,
    load_hotpotqa,
    paragraph_text,
    read_rows,
    sample_rows,
    save_subset,
    to_question,
)


def make_row(
        row_id: str = 'q1',
        titles: tuple[str, ...] = ('A', 'B', 'C'),
        texts: tuple[tuple[str, ...], ...] = (
            ('A one.', ' A two.'),
            ('B one.', ' B two.'),
            ('C one.',),
        ),
        facts: tuple[tuple[str, int], ...] = (('A', 0), ('A', 1), ('B', 0)),
        question_type: str = 'bridge',
) -> dict:
    return {
        'id': row_id,
        'question': f'Question {row_id}?',
        'answer': 'yes',
        'type': question_type,
        'level': 'hard',
        'supporting_facts': {
            'title': [title for title, _ in facts],
            'sent_id': [sent_id for _, sent_id in facts],
        },
        'context': {
            'title': list(titles),
            'sentences': [list(sentences) for sentences in texts],
        },
    }


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text('utf-8').splitlines()
    ]


class TestParagraphText:
    def test_joins_sentences_and_strips_edges(self):
        sentences = ['Ed Wood is a film.', ' The film concerns.', ' Sarah.']

        text = paragraph_text(sentences)

        assert text == 'Ed Wood is a film. The film concerns. Sarah.'

    def test_empty_sentences_give_empty_text(self):
        assert paragraph_text([]) == ''


class TestSampleRows:
    def test_same_seed_gives_same_sample(self):
        rows = [make_row(f'q{i}') for i in range(20)]

        first = sample_rows(rows, 5, seed=1)
        second = sample_rows(rows, 5, seed=1)

        assert first == second

    def test_different_seed_gives_different_sample(self):
        rows = [make_row(f'q{i}') for i in range(50)]

        first = sample_rows(rows, 10, seed=1)
        second = sample_rows(rows, 10, seed=2)

        assert first != second

    def test_returns_n_distinct_rows_and_keeps_input(self):
        rows = [make_row(f'q{i}') for i in range(10)]
        before = list(rows)

        sample = sample_rows(rows, 4, seed=0)

        assert len(sample) == 4
        assert len({row['id'] for row in sample}) == 4
        assert rows == before

    def test_n_equal_to_len_is_allowed(self):
        rows = [make_row(f'q{i}') for i in range(3)]

        assert len(sample_rows(rows, 3, seed=0)) == 3

    @pytest.mark.parametrize('n', [0, -1, 4])
    def test_invalid_n_raises(self, n: int):
        rows = [make_row(f'q{i}') for i in range(3)]

        with pytest.raises(ValueError):
            sample_rows(rows, n, seed=0)


class TestToQuestion:
    def test_maps_basic_fields(self):
        question = to_question(make_row(question_type='comparison'))

        assert question.question_id == 'q1'
        assert question.question == 'Question q1?'
        assert question.answer == 'yes'
        assert question.type == 'comparison'

    def test_gold_titles_are_unique_in_first_appearance_order(self):
        row = make_row(facts=(('B', 0), ('A', 0), ('B', 1), ('A', 1)))

        question = to_question(row)

        assert question.gold_titles == ('B', 'A')

    def test_evidence_is_stripped_in_supporting_facts_order(self):
        row = make_row(facts=(('B', 1), ('A', 1), ('A', 0)))

        question = to_question(row)

        assert question.evidence == ('B two.', 'A two.', 'A one.')

    def test_out_of_range_sent_id_is_skipped(self):
        row = make_row(facts=(('A', 0), ('A', 7), ('B', 0)))

        question = to_question(row)

        assert question.evidence == ('A one.', 'B one.')
        assert question.gold_titles == ('A', 'B')

    def test_fact_with_unknown_title_is_skipped(self):
        row = make_row(facts=(('A', 0), ('Z', 0)))

        question = to_question(row)

        assert question.evidence == ('A one.',)

    def test_duplicate_title_in_row_first_wins(self):
        row = make_row(
            titles=('A', 'A'), texts=(('first.',), ('second.',)),
            facts=(('A', 0),),
        )

        question = to_question(row)

        assert question.evidence == ('first.',)

    def test_evidence_comes_from_given_corpus_map(self):
        row = make_row(facts=(('A', 0), ('A', 1)))
        kept = {'A': ('kept.',)}

        question = to_question(row, kept)

        assert question.evidence == ('kept.',)

    def test_ragged_supporting_facts_raise(self):
        row = make_row()
        row['supporting_facts']['sent_id'] = [0]

        with pytest.raises(ValueError):
            to_question(row)

    def test_does_not_mutate_row(self):
        row = make_row()
        before = json.dumps(row, sort_keys=True)

        to_question(row)

        assert json.dumps(row, sort_keys=True) == before


class TestBuildSubset:
    def test_corpus_is_union_of_context_paragraphs(self):
        rows = [
            make_row('q1'),
            make_row(
                'q2', titles=('C', 'D'),
                texts=(('C one.',), ('D one.',)), facts=(('D', 0),),
            ),
        ]

        subset = build_subset(rows)

        assert [d.doc_id for d in subset.documents] == ['A', 'B', 'C', 'D']
        assert all(d.doc_id == d.title for d in subset.documents)
        assert subset.documents[0].text == 'A one. A two.'
        assert [q.question_id for q in subset.questions] == ['q1', 'q2']
        assert subset.title_conflicts == 0

    def test_first_occurrence_wins_and_conflict_is_counted(self):
        rows = [
            make_row('q1'),
            make_row(
                'q2', titles=('A', 'B'),
                texts=(('A other.',), ('B one.', ' B two.')),
                facts=(('A', 0),),
            ),
        ]

        subset = build_subset(rows)

        texts = {d.title: d.text for d in subset.documents}
        assert texts['A'] == 'A one. A two.'
        assert subset.title_conflicts == 1
        assert len(subset.documents) == 3

    def test_identical_repeat_is_not_a_conflict(self):
        subset = build_subset([make_row('q1'), make_row('q2')])

        assert len(subset.documents) == 3
        assert subset.title_conflicts == 0

    def test_every_gold_title_is_in_corpus(self):
        subset = build_subset([make_row('q1'), make_row('q2')])

        titles = {d.title for d in subset.documents}
        for question in subset.questions:
            assert set(question.gold_titles) <= titles

    def test_empty_rows_give_empty_subset(self):
        subset = build_subset([])

        assert subset == HotpotSubset((), (), 0, 0)

    def test_evidence_of_conflicting_title_comes_from_kept_paragraph(self):
        rows = [
            make_row('q1'),
            make_row(
                'q2', titles=('A', 'B'),
                texts=(('A other.', ' A extra.', ' A third.'), ('B one.',)),
                facts=(('A', 0), ('A', 2), ('B', 0)),
            ),
        ]

        subset = build_subset(rows)

        assert subset.questions[1].evidence == ('A one.', 'B one.')
        assert subset.dropped_facts == 1
        assert subset.documents[0].text == 'A one. A two.'

    def test_dropped_facts_count_unknown_title_and_bad_sent_id(self):
        rows = [
            make_row('q1', facts=(('A', 0), ('A', 9), ('Z', 0))),
            make_row('q2', facts=(('B', 5),)),
        ]

        subset = build_subset(rows)

        assert subset.dropped_facts == 3

    def test_duplicate_title_in_row_keeps_first_without_conflict_loss(self):
        row = make_row(
            titles=('A', 'A'), texts=(('first.',), ('second.',)),
            facts=(('A', 0),),
        )

        subset = build_subset([row])

        assert [d.text for d in subset.documents] == ['first.']
        assert subset.title_conflicts == 1

    def test_ragged_context_raises(self):
        row = make_row()
        row['context']['sentences'] = row['context']['sentences'][:-1]

        with pytest.raises(ValueError):
            build_subset([row])


class TestSaveSubset:
    def test_writes_round_trippable_files(self, tmp_path):
        subset = build_subset([make_row('q1')])
        out_dir = tmp_path / 'nested' / 'out'

        save_subset(subset, out_dir, n=1, seed=7)

        corpus = read_jsonl(out_dir / 'corpus.jsonl')
        questions = read_jsonl(out_dir / 'questions.jsonl')
        meta = json.loads((out_dir / 'meta.json').read_text('utf-8'))
        assert [Document(**d) for d in corpus] == list(subset.documents)
        assert questions[0]['gold_titles'] == ['A', 'B']
        assert questions[0]['evidence'] == ['A one.', 'A two.', 'B one.']
        restored = Question(
            **{
                **questions[0],
                'gold_titles': tuple(questions[0]['gold_titles']),
                'evidence': tuple(questions[0]['evidence']),
            }
        )
        assert restored == subset.questions[0]
        assert meta['dataset'] == 'hotpotqa/hotpot_qa'
        assert meta['config'] == 'distractor'
        assert meta['split'] == 'validation'
        assert len(meta['revision']) == 40
        assert (meta['n'], meta['seed']) == (1, 7)
        assert meta['n_documents'] == 3
        assert meta['title_conflicts'] == 0
        assert meta['dropped_facts'] == 0

    def test_keeps_non_ascii_text_unescaped(self, tmp_path):
        row = make_row(texts=(('Zoë 東京.',), ('B.',), ('C.',)))
        subset = build_subset([row])

        save_subset(subset, tmp_path, n=1, seed=0)

        raw = (tmp_path / 'corpus.jsonl').read_text('utf-8')
        assert 'Zoë 東京.' in raw


class TestIo:
    def test_read_rows_round_trips_parquet(self, tmp_path):
        rows = [make_row('q1'), make_row('q2')]
        path = tmp_path / 'rows.parquet'
        pq.write_table(pa.Table.from_pylist(rows), path)

        loaded = read_rows(path)

        assert loaded == rows

    def test_load_hotpotqa_samples_from_downloaded_file(
            self, tmp_path, monkeypatch,
    ):
        rows = [make_row(f'q{i}') for i in range(6)]
        path = tmp_path / 'rows.parquet'
        pq.write_table(pa.Table.from_pylist(rows), path)
        monkeypatch.setattr(data, 'download_validation', lambda _: path)

        subset = load_hotpotqa(n=3, seed=1)

        assert len(subset.questions) == 3
        assert subset == load_hotpotqa(n=3, seed=1)


# Reproducible snapshot of the default subset (pinned revision, seed 42)
GOLDEN_QUESTION_IDS = (
    '5ae143ed55429920d5234360',
    '5abc19705542993a06baf86e',
    '5ac3e0f7554299194317388b',
)


@pytest.mark.integration
class TestLoadHotpotqa:
    def test_default_subset_is_consistent_and_reproducible(self):
        subset = load_hotpotqa()

        doc_ids = [d.doc_id for d in subset.documents]
        titles = set(doc_ids)
        assert len(subset.questions) == DEFAULT_N
        assert len(titles) == len(doc_ids)
        for question in subset.questions:
            assert len(question.gold_titles) == 2
            assert set(question.gold_titles) <= titles
            assert question.evidence
        assert load_hotpotqa() == subset

    def test_default_subset_matches_snapshot(self):
        subset = load_hotpotqa()

        ids = tuple(q.question_id for q in subset.questions[:3])
        assert ids == GOLDEN_QUESTION_IDS
        assert subset.title_conflicts == 0
        assert subset.dropped_facts == 0
