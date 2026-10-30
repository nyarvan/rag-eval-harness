import os

import pytest

from rag_eval.data import Document, Question
from rag_eval.generate import format_context
from rag_eval.rerank import RerankedRetriever

DOCUMENTS = tuple(
    Document(title, title, f'{title.lower()} text')
    for title in ('A', 'B', 'C', 'D')
)


def make_question(text: str = 'q?', question_id: str = '1') -> Question:
    return Question(question_id, text, 'a', 'bridge', (), ())


class FakeFirstStage:
    """First-stage retriever returning a fixed ranking and recording k."""

    def __init__(self, ranking):
        self.ranking = list(ranking)
        self.calls: list[tuple[int, int]] = []

    def retrieve_many(self, questions, k):
        self.calls.append((len(questions), k))
        return [self.ranking[:k] for _ in questions]


class FakeScorer:
    """Cross-encoder stand-in: scores come from a title -> score table."""

    def __init__(self, table):
        self.table = table
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs, **kwargs):
        self.calls.append(list(pairs))
        return [self.table[passage.split('\n')[0][len('Title: '):]]
                for _, passage in pairs]

    @property
    def scored_pairs(self):
        return [pair for call in self.calls for pair in call]


def make_retriever(tmp_path, scorer, ranking=('A', 'B', 'C', 'D'),
                   candidates_k=4, model_name='model-a'):
    first_stage = FakeFirstStage(ranking)
    retriever = RerankedRetriever(
        first_stage, DOCUMENTS, model_name=model_name,
        candidates_k=candidates_k, cache_dir=tmp_path, scorer=scorer,
    )
    return retriever, first_stage


def test_candidates_are_reordered_by_descending_score(tmp_path):
    scorer = FakeScorer({'A': 0.1, 'B': 0.9, 'C': 0.5, 'D': 0.7})
    retriever, _ = make_retriever(tmp_path, scorer)

    titles = retriever.retrieve(make_question(), 3)

    assert titles == ['B', 'D', 'C']


def test_ties_keep_first_stage_order(tmp_path):
    scorer = FakeScorer({'A': 0.5, 'B': 0.5, 'C': 0.5, 'D': 0.9})
    retriever, _ = make_retriever(tmp_path, scorer)

    titles = retriever.retrieve(make_question(), 4)

    assert titles == ['D', 'A', 'B', 'C']


def test_first_stage_is_asked_for_candidates_not_k(tmp_path):
    scorer = FakeScorer({'A': 1, 'B': 2, 'C': 3, 'D': 4})
    retriever, first_stage = make_retriever(tmp_path, scorer, candidates_k=4)

    retriever.retrieve_many([make_question()], 2)

    assert first_stage.calls == [(1, 4)]


def test_scorer_gets_question_and_formatted_passage(tmp_path):
    scorer = FakeScorer({'A': 1, 'B': 2})
    retriever, _ = make_retriever(
        tmp_path, scorer, ranking=('A', 'B'), candidates_k=2
    )

    retriever.retrieve(make_question('why?'), 2)

    assert scorer.scored_pairs == [
        ('why?', format_context(DOCUMENTS[0])),
        ('why?', format_context(DOCUMENTS[1])),
    ]


def test_scores_are_reused_from_disk(tmp_path):
    first = FakeScorer({'A': 0.1, 'B': 0.9, 'C': 0.5, 'D': 0.7})
    make_retriever(tmp_path, first)[0].retrieve(make_question(), 2)
    second = FakeScorer({})

    titles = make_retriever(tmp_path, second)[0].retrieve(make_question(), 2)

    assert titles == ['B', 'D']
    assert second.calls == []


def test_only_unscored_candidates_are_scored(tmp_path):
    first = FakeScorer({'A': 0.1, 'B': 0.9})
    make_retriever(
        tmp_path, first, ranking=('A', 'B'), candidates_k=2
    )[0].retrieve(make_question(), 2)
    second = FakeScorer({'C': 0.5})

    make_retriever(
        tmp_path, second, ranking=('A', 'B', 'C'), candidates_k=3
    )[0].retrieve(make_question(), 3)

    assert [passage for _, passage in second.scored_pairs] == [
        format_context(DOCUMENTS[2])
    ]


def test_scores_are_per_question(tmp_path):
    scorer = FakeScorer({'A': 0.1, 'B': 0.9, 'C': 0.5, 'D': 0.7})
    retriever, _ = make_retriever(tmp_path, scorer)

    retriever.retrieve(make_question('first?'), 2)
    retriever.retrieve(make_question('second?'), 2)

    assert len(scorer.scored_pairs) == 8


def test_scores_are_per_model(tmp_path):
    first = FakeScorer({'A': 0.1, 'B': 0.9, 'C': 0.5, 'D': 0.7})
    make_retriever(tmp_path, first, model_name='model-a')[0].retrieve(
        make_question(), 2
    )
    other = FakeScorer({'A': 0.9, 'B': 0.1, 'C': 0.2, 'D': 0.3})

    titles = make_retriever(tmp_path, other, model_name='model-b')[0].retrieve(
        make_question(), 2
    )

    assert titles == ['A', 'D']


def test_model_name_with_slash_is_a_valid_cache_file(tmp_path):
    scorer = FakeScorer({'A': 1, 'B': 2, 'C': 3, 'D': 4})
    retriever, _ = make_retriever(tmp_path, scorer, model_name='org/model')

    retriever.retrieve(make_question(), 1)

    assert [path.is_file() for path in tmp_path.iterdir()] == [True]


def test_retrieve_many_keeps_question_order(tmp_path):
    scorer = FakeScorer({'A': 0.1, 'B': 0.9, 'C': 0.5, 'D': 0.7})
    retriever, _ = make_retriever(tmp_path, scorer)
    questions = [make_question('x?', '1'), make_question('y?', '2')]

    assert retriever.retrieve_many(questions, 1) == [['B'], ['B']]


def test_no_questions_give_no_results_and_no_scoring(tmp_path):
    scorer = FakeScorer({})
    retriever, first_stage = make_retriever(tmp_path, scorer)

    assert retriever.retrieve_many([], 3) == []
    assert scorer.calls == []


@pytest.mark.parametrize('k', [0, -1, 5])
def test_k_outside_one_to_candidates_k_is_rejected(tmp_path, k):
    retriever, _ = make_retriever(tmp_path, FakeScorer({}), candidates_k=4)

    with pytest.raises(ValueError):
        retriever.retrieve(make_question(), k)


def test_candidates_k_must_be_positive(tmp_path):
    with pytest.raises(ValueError):
        make_retriever(tmp_path, FakeScorer({}), candidates_k=0)


def test_retrieve_many_scores_all_questions_in_one_call(tmp_path):
    scorer = FakeScorer({'A': 0.1, 'B': 0.9, 'C': 0.5, 'D': 0.7})
    retriever, _ = make_retriever(tmp_path, scorer)
    questions = [make_question('x?', '1'), make_question('y?', '2')]

    retriever.retrieve_many(questions, 2)

    assert len(scorer.calls) == 1
    assert len(scorer.scored_pairs) == 8


def test_cache_is_written_once_per_retrieve_many(tmp_path, monkeypatch):
    scorer = FakeScorer({'A': 0.1, 'B': 0.9, 'C': 0.5, 'D': 0.7})
    retriever, _ = make_retriever(tmp_path, scorer)
    replaced: list[object] = []
    real_replace = os.replace
    monkeypatch.setattr(
        os, 'replace', lambda *args: (replaced.append(args), real_replace(*args))
    )
    questions = [make_question('x?', '1'), make_question('y?', '2')]

    retriever.retrieve_many(questions, 2)

    assert len(replaced) == 1


def test_repeated_pair_is_scored_once(tmp_path):
    scorer = FakeScorer({'A': 0.1, 'B': 0.9, 'C': 0.5, 'D': 0.7})
    retriever, _ = make_retriever(tmp_path, scorer)
    questions = [make_question('same?', '1'), make_question('same?', '2')]

    retriever.retrieve_many(questions, 2)

    assert len(scorer.scored_pairs) == 4
