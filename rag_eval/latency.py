"""Latency of retrieval: how long one question takes, asked one at a time.

    python -m rag_eval.latency

Retrievers are timed on single-question calls, the way a live pipeline would
use them, not on the batched calls the evaluation uses. The reranker is timed
with an empty score cache (a temporary directory), because a warm cache would
hide the cost of scoring the candidates.
"""
import argparse
import gc
import tempfile
import time
from collections.abc import Callable, Sequence
from statistics import median
from typing import Protocol

from .args import positive_int
from .bm25 import BM25Retriever
from .data import DEFAULT_N, Question, load_hotpotqa
from .hybrid import DEFAULT_DEPTH, HybridRetriever
from .rerank import RerankedRetriever
from .retrieval_eval import EMBEDDING_MODEL
from .retrieve import DenseRetriever

DEFAULT_QUESTIONS = 20
DEFAULT_RERANK_QUESTIONS = 8  # scoring candidates on a CPU takes seconds each
DEFAULT_K = 5


class SingleRetriever(Protocol):
    def retrieve(self, question: Question, k: int) -> list[str]: ...


def measure_latency(
        retriever: SingleRetriever, questions: Sequence[Question], k: int, *,
        warmup: int = 1, timer: Callable[[], float] = time.perf_counter,
) -> list[float]:
    """Seconds taken by each question after the first `warmup` ones.

    The warm-up questions are run but not timed: the first call pays for
    loading a model or building an index.
    """
    if warmup < 0:
        raise ValueError(f'warmup must not be negative, got {warmup}')
    if len(questions) <= warmup:
        raise ValueError('there must be a question left to time')
    for question in questions[:warmup]:
        retriever.retrieve(question, k)
    times = []
    for question in questions[warmup:]:
        start = timer()
        retriever.retrieve(question, k)
        times.append(timer() - start)
    return times


def median_ms(times: Sequence[float]) -> float:
    """Median of times given in seconds, in milliseconds."""
    if not times:
        raise ValueError('times must not be empty')
    return median(times) * 1000


def build_parser() -> argparse.ArgumentParser:
    """The flags of the latency command."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--n', type=positive_int, default=DEFAULT_QUESTIONS)
    parser.add_argument(
        '--rerank-n', type=positive_int, default=DEFAULT_RERANK_QUESTIONS,
        help='questions timed with the reranker (cold cache, slow)'
    )
    parser.add_argument('--k', type=positive_int, default=DEFAULT_K)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    subset = load_hotpotqa()
    questions = subset.questions
    dense = DenseRetriever(EMBEDDING_MODEL, subset.documents,
                           subset_size=DEFAULT_N)
    bm25 = BM25Retriever(subset.documents)
    hybrid = HybridRetriever([dense, bm25], depth=DEFAULT_DEPTH)

    rows = []
    for name, retriever in (('dense', dense), ('bm25', bm25),
                            ('hybrid', hybrid)):
        times = measure_latency(retriever, questions[:args.n], args.k)
        rows.append((name, median_ms(times), len(times)))
    for name, first_stage in (('dense + rerank', dense),
                              ('hybrid + rerank', hybrid)):
        with tempfile.TemporaryDirectory() as cache_dir:
            reranker = RerankedRetriever(
                first_stage, subset.documents, cache_dir=cache_dir
            )
            times = measure_latency(
                reranker, questions[:args.rerank_n + 1], args.k
            )
            del reranker
            gc.collect()
        rows.append((name + ' (cold cache)', median_ms(times), len(times)))

    width = max(len(name) for name, _, _ in rows)
    print(f'{"retriever":<{width}}  median ms/question  questions timed')
    for name, ms, count in rows:
        print(f'{name:<{width}}  {ms:>18.1f}  {count:>15}')


if __name__ == '__main__':
    main()
