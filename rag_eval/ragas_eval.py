"""RAGAS scoring of a run file, with an LLM as the judge.

    python -m rag_eval.ragas_eval runs/dense_rerank_200.jsonl --n 20

Four metrics, each scoring one aspect of a run record:

    factual_correctness  answer against the reference (a fairer yardstick
                         than exact match for short answers)
    faithfulness         is the answer supported by the retrieved contexts
    context_recall       do the contexts contain what the reference says
    context_precision    are the useful contexts ranked first

The judge's replies are cached on disk by RAGAS, so repeating a run is free.
A metric that fails for one record is scored `None` and counted in the
summary; it does not stop the run.
"""
import argparse
import asyncio
import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Protocol

from .records import RunRecord, read_records

logger = logging.getLogger(__name__)

DEFAULT_JUDGE = 'gpt-4o-mini'
DEFAULT_CONCURRENCY = 8
DEFAULT_CACHE_DIR = 'data/ragas_cache'
JUDGE_TEMPERATURE = 0.0

# Keyword arguments that each RAGAS metric's `ascore` takes
METRIC_INPUTS: dict[str, tuple[str, ...]] = {
    'factual_correctness': ('response', 'reference'),
    'faithfulness': ('user_input', 'response', 'retrieved_contexts'),
    'context_recall': ('user_input', 'retrieved_contexts', 'reference'),
    'context_precision': ('user_input', 'reference', 'retrieved_contexts'),
}


class Metric(Protocol):
    async def ascore(self, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class RecordScores:
    """The metric scores of one record; `None` where a metric failed."""

    question_id: str
    scores: Mapping[str, float | None]


async def score_record(
        record: RunRecord, metrics: Mapping[str, Metric],
) -> dict[str, float | None]:
    """Score one record with every metric at once."""
    available = {
        'user_input': record.question,
        'response': record.answer,
        'reference': record.reference,
        'retrieved_contexts': list(record.contexts),
    }
    names = list(metrics)
    values = await asyncio.gather(*(
        _score_one(
            name, metrics[name],
            {key: available[key] for key in METRIC_INPUTS[name]},
        )
        for name in names
    ))
    return dict(zip(names, values, strict=True))


async def score_records(
        records: Sequence[RunRecord], metrics: Mapping[str, Metric], *,
        concurrency: int = DEFAULT_CONCURRENCY,
) -> list[RecordScores]:
    """Score all records, at most `concurrency` of them at a time."""
    if concurrency <= 0:
        raise ValueError(f'concurrency must be positive, got {concurrency}')
    semaphore = asyncio.Semaphore(concurrency)

    async def score(record: RunRecord) -> RecordScores:
        async with semaphore:
            scores = await score_record(record, metrics)
        return RecordScores(record.question_id, scores)

    return list(await asyncio.gather(*(score(r) for r in records)))


def summarize_scores(
        rows: Sequence[RecordScores],
) -> dict[str, float | int | None]:
    """Mean score per metric over the records that got one, plus failures."""
    if not rows:
        raise ValueError('rows must not be empty')
    summary: dict[str, float | int | None] = {}
    for name in rows[0].scores:
        values = [
            row.scores[name] for row in rows if row.scores[name] is not None
        ]
        summary[name] = fmean(values) if values else None
        summary[f'failed:{name}'] = len(rows) - len(values)
    return summary


def write_scores(path: str | Path, rows: Sequence[RecordScores]) -> None:
    """Write one JSON object per record: its id and a field per metric."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as file:
        for row in rows:
            item = {'question_id': row.question_id, **row.scores}
            file.write(json.dumps(item, ensure_ascii=False) + '\n')


def read_scores(path: str | Path) -> list[RecordScores]:
    """Read a file written by `write_scores`."""
    rows = []
    with Path(path).open(encoding='utf-8') as file:
        for line in file:
            if line.strip():
                item = json.loads(line)
                rows.append(RecordScores(item.pop('question_id'), item))
    return rows


def build_metrics(
        judge: str = DEFAULT_JUDGE, cache_dir: str | Path = DEFAULT_CACHE_DIR,
        names: Sequence[str] = tuple(METRIC_INPUTS),
) -> dict[str, Metric]:
    """The named RAGAS metrics (all four by default), judged by `judge`."""
    from openai import AsyncOpenAI
    from ragas.cache import DiskCacheBackend
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        ContextPrecision,
        ContextRecall,
        FactualCorrectness,
        Faithfulness,
    )

    llm = llm_factory(
        judge, client=AsyncOpenAI(),
        cache=DiskCacheBackend(cache_dir=str(cache_dir)),
        temperature=JUDGE_TEMPERATURE,
    )
    classes = {
        'factual_correctness': FactualCorrectness,
        'faithfulness': Faithfulness,
        'context_recall': ContextRecall,
        'context_precision': ContextPrecision,
    }
    return {name: classes[name](llm=llm) for name in names}


async def _score_one(
        name: str, metric: Metric, inputs: Mapping[str, Any],
) -> float | None:
    """One metric on one record; a failure or NaN is `None`, and logged."""
    try:
        result = await metric.ascore(**inputs)
    except Exception as error:  # a single judge failure must not end the run
        logger.warning('%s failed: %s: %s', name, type(error).__name__, error)
        return None
    value = result.value
    if value is None or math.isnan(value):
        return None
    return float(value)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('run_file', help='run file written by rag_eval.run')
    parser.add_argument('--out', help='scores file (default: ragas_<name>)')
    parser.add_argument('--n', type=int, help='score only the first N records')
    parser.add_argument('--judge', default=DEFAULT_JUDGE)
    parser.add_argument(
        '--metrics', nargs='+', choices=list(METRIC_INPUTS),
        default=list(METRIC_INPUTS), help='metrics to compute (default: all)'
    )
    parser.add_argument('--concurrency', type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument('--cache-dir', default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        '--env-file', help='dotenv file with OPENAI_API_KEY (not printed)'
    )
    args = parser.parse_args(argv)

    if args.env_file:
        from dotenv import load_dotenv
        load_dotenv(args.env_file)

    run_path = Path(args.run_file)
    out = args.out or run_path.with_name(f'ragas_{run_path.name}')
    records = read_records(run_path)[:args.n]
    rows = asyncio.run(score_records(
        records, build_metrics(args.judge, args.cache_dir, args.metrics),
        concurrency=args.concurrency,
    ))
    write_scores(out, rows)
    for name, value in summarize_scores(rows).items():
        print(f'{name}: {value if value is None else round(value, 3)}')
    print(f'{len(rows)} records -> {out}')


if __name__ == '__main__':
    main()
