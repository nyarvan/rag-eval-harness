"""Paired comparison of retrievers and of run files, question by question.

The statistics come from `significance.py` of embedding-benchmark-harness: a
paired bootstrap interval of the mean difference and an exact sign test on the
questions where the two sides differ. Output is a list of its `Comparison`
records, so `significance.format_comparisons` prints it.
"""
from collections.abc import Mapping, Sequence

import numpy as np
from significance import (
    EPS,
    Comparison,
    changed_queries,
    paired_bootstrap,
    sign_test_p,
)

from .data import Question
from .metrics import all_gold_at_k, exact_match, f1, recall_at_k
from .records import RunRecord

METRICS = {'all_gold': all_gold_at_k, 'recall': recall_at_k}
DEFAULT_KS = (5, 10)
DEFAULT_RESAMPLES = 10_000


def compare_rankings(
        rankings: Mapping[str, Sequence[Sequence[str]]],
        questions: Sequence[Question], *,
        ks: Sequence[int] = DEFAULT_KS,
        metrics: Sequence[str] = tuple(METRICS),
        baseline: str | None = None,
        n_resamples: int = DEFAULT_RESAMPLES, seed: int = 0,
) -> list[Comparison]:
    """Every retriever against `baseline` (default: the first one).

    `rankings` maps a retriever's name to its ranked titles, one list per
    question. A comparison is `retriever - baseline`, so a positive mean
    difference means the retriever is better.
    """
    if not questions:
        raise ValueError('questions must not be empty')
    baseline = _choose_baseline(rankings, baseline)
    for name, ranked in rankings.items():
        if len(ranked) != len(questions):
            raise ValueError(
                f'{name!r} has {len(ranked)} rankings for '
                f'{len(questions)} questions'
            )

    ids = [question.question_id for question in questions]
    comparisons = []
    for name in rankings:
        if name == baseline:
            continue
        for metric in metrics:
            for k in ks:
                a = _ranking_scores(rankings[name], questions, metric, k)
                b = _ranking_scores(rankings[baseline], questions, metric, k)
                comparisons.append(_compare(
                    name, baseline, f'{metric}@{k}', a, b, ids,
                    n_resamples, seed,
                ))
    return comparisons


def compare_runs(
        runs: Mapping[str, Sequence[RunRecord]], *,
        ks: Sequence[int] = (), baseline: str | None = None,
        n_resamples: int = DEFAULT_RESAMPLES, seed: int = 0,
) -> list[Comparison]:
    """Every run file against `baseline` (default: the first one).

    Records are matched by `question_id`, so the files may list the questions
    in different orders, but must cover exactly the same questions. Compares
    exact match and F1 of the answers; for each k in `ks` also `all_gold@k`
    and `recall@k`, which need `retrieved` and `gold_titles` in every record.
    """
    baseline = _choose_baseline(runs, baseline)
    indexed = {
        name: _by_question(name, records) for name, records in runs.items()
    }
    ids = list(indexed[baseline])
    for name, by_id in indexed.items():
        if set(by_id) != set(ids):
            raise ValueError(
                f'{name!r} and {baseline!r} cover different questions: '
                f'only in {name!r}: {sorted(set(by_id) - set(ids))[:5]}, '
                f'only in {baseline!r}: {sorted(set(ids) - set(by_id))[:5]}'
            )

    scores = {
        name: _run_scores([by_id[i] for i in ids], ks)
        for name, by_id in indexed.items()
    }
    comparisons = []
    for name in runs:
        if name == baseline:
            continue
        for metric, values in scores[name].items():
            comparisons.append(_compare(
                name, baseline, metric, values, scores[baseline][metric], ids,
                n_resamples, seed,
            ))
    return comparisons


def _choose_baseline(
        named: Mapping[str, object], baseline: str | None,
) -> str:
    """The requested baseline, or the first name; at least two are needed."""
    if len(named) < 2:
        raise ValueError('at least two retrievers or runs are needed')
    baseline = next(iter(named)) if baseline is None else baseline
    if baseline not in named:
        raise ValueError(f'unknown baseline {baseline!r}')
    return baseline


def _compare(
        name: str, baseline: str, metric: str, a: np.ndarray, b: np.ndarray,
        ids: Sequence[str], n_resamples: int, seed: int,
) -> Comparison:
    """`name - baseline` on one metric, from their per-question scores."""
    mean, low, high = paired_bootstrap(a, b, n_resamples, seed)
    diff = a - b
    wins = int((diff > EPS).sum())
    losses = int((diff < -EPS).sum())
    return Comparison(
        name, baseline, metric, mean, low, high, wins, losses,
        sign_test_p(wins, losses), tuple(changed_queries(a, b, list(ids))),
    )


def _ranking_scores(
        rankings: Sequence[Sequence[str]], questions: Sequence[Question],
        metric: str, k: int,
) -> np.ndarray:
    """The metric of every question separately, as floats."""
    score = METRICS[metric]
    return np.array([
        float(score(ranked, question.gold_titles, k))
        for ranked, question in zip(rankings, questions, strict=True)
    ])


def _by_question(
        name: str, records: Sequence[RunRecord],
) -> dict[str, RunRecord]:
    """The records of one run by question id; an id may appear only once."""
    by_id = {}
    for record in records:
        if record.question_id in by_id:
            raise ValueError(
                f'{name!r} lists question {record.question_id!r} twice'
            )
        by_id[record.question_id] = record
    return by_id


def _run_scores(
        records: Sequence[RunRecord], ks: Sequence[int],
) -> dict[str, np.ndarray]:
    """Per-question scores of one run: answer metrics, then retrieval ones."""
    scores = {
        'em': np.array([exact_match(r.answer, r.reference) for r in records]),
        'f1': np.array([f1(r.answer, r.reference) for r in records]),
    }
    if ks and not all(r.gold_titles and r.retrieved for r in records):
        raise ValueError(
            'retrieval metrics need retrieved and gold_titles in every record'
        )
    for k in ks:
        for metric, score in METRICS.items():
            scores[f'{metric}@{k}'] = np.array([
                float(score(r.retrieved, r.gold_titles, k)) for r in records
            ])
    return scores
