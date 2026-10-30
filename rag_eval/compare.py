"""Paired comparison of retrievers, question by question.

The statistics come from `significance.py` of embedding-benchmark-harness: a
paired bootstrap interval of the mean difference and an exact sign test on the
questions where the two retrievers differ. Output is a list of its
`Comparison` records, so `significance.format_comparisons` prints it.
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
from .metrics import all_gold_at_k, recall_at_k

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
    if len(rankings) < 2:
        raise ValueError('at least two retrievers are needed')
    baseline = next(iter(rankings)) if baseline is None else baseline
    if baseline not in rankings:
        raise ValueError(f'unknown baseline {baseline!r}')
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
                a = _scores(rankings[name], questions, metric, k)
                b = _scores(rankings[baseline], questions, metric, k)
                mean, low, high = paired_bootstrap(a, b, n_resamples, seed)
                diff = a - b
                wins = int((diff > EPS).sum())
                losses = int((diff < -EPS).sum())
                comparisons.append(Comparison(
                    name, baseline, f'{metric}@{k}', mean, low, high,
                    wins, losses, sign_test_p(wins, losses),
                    tuple(changed_queries(a, b, ids)),
                ))
    return comparisons


def _scores(
        rankings: Sequence[Sequence[str]], questions: Sequence[Question],
        metric: str, k: int,
) -> np.ndarray:
    """The metric of every question separately, as floats."""
    score = METRICS[metric]
    return np.array([
        float(score(ranked, question.gold_titles, k))
        for ranked, question in zip(rankings, questions, strict=True)
    ])
