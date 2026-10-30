"""Retrieval-only evaluation: free metrics for several retrievers side by side.

    python -m rag_eval.retrieval_eval --retrievers dense bm25 hybrid

No LLM is called. Prints `all_gold@k` (both supporting paragraphs in the top k,
the key metric for multi-hop questions) and `recall@k` for all questions and
for each question type.
"""
import argparse
from collections.abc import Mapping, Sequence
from statistics import fmean

from .bm25 import BM25Retriever
from .data import DEFAULT_N, Document, Question, load_hotpotqa
from .hybrid import DEFAULT_DEPTH, HybridRetriever, Retriever
from .metrics import all_gold_at_k, recall_at_k
from .rerank import DEFAULT_CANDIDATES_K, RerankedRetriever
from .retrieve import DenseRetriever

DEFAULT_KS = (2, 5, 10, 20, 30)
DEFAULT_RETRIEVERS = ('dense', 'bm25', 'hybrid')
RETRIEVER_CHOICES = (
    'dense', 'bm25', 'hybrid', 'dense+rerank', 'hybrid+rerank',
)
EMBEDDING_MODEL = 'bge_m3'


def score_rankings(
        rankings: Sequence[Sequence[str]], questions: Sequence[Question],
        ks: Sequence[int],
) -> dict[str, float]:
    """Mean `recall@k` and `all_gold@k` of one ranking per question."""
    if not questions:
        raise ValueError('questions must not be empty')
    if not ks:
        raise ValueError('ks must not be empty')
    scores = {}
    for k in sorted(ks):
        pairs = list(zip(rankings, questions, strict=True))
        scores[f'recall@{k}'] = fmean(
            recall_at_k(ranked, q.gold_titles, k) for ranked, q in pairs
        )
        scores[f'all_gold@{k}'] = fmean(
            all_gold_at_k(ranked, q.gold_titles, k) for ranked, q in pairs
        )
    return scores


def evaluate_retrieval(
        retrievers: Mapping[str, Retriever], questions: Sequence[Question],
        ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, dict[str, float]]:
    """Score every retriever; each is asked once, at the largest k."""
    if not ks:
        raise ValueError('ks must not be empty')
    deepest = max(ks)
    return {
        name: score_rankings(
            retriever.retrieve_many(questions, deepest), questions, ks
        )
        for name, retriever in retrievers.items()
    }


def format_table(
        results: Mapping[str, Mapping[str, float]], ks: Sequence[int],
) -> str:
    """One block per metric: a row per retriever, a column per k."""
    ks = sorted(ks)
    width = max(len(name) for name in results)
    blocks = []
    for metric in ('all_gold', 'recall'):
        header = f'{metric:<{width}}  ' + '  '.join(f'@{k:<5}' for k in ks)
        rows = [
            f'{name:<{width}}  '
            + '  '.join(f'{scores[f"{metric}@{k}"]:<6.3f}' for k in ks)
            for name, scores in results.items()
        ]
        blocks.append('\n'.join([header, *rows]))
    return '\n\n'.join(blocks)


def build_retrievers(
        names: Sequence[str], documents: Sequence[Document], depth: int,
) -> dict[str, Retriever]:
    """The named retrievers; the ones they share are built once."""
    dense = DenseRetriever(EMBEDDING_MODEL, documents, subset_size=DEFAULT_N)
    bm25 = BM25Retriever(documents) if (
        'bm25' in names or 'hybrid' in names or 'hybrid+rerank' in names
    ) else None
    hybrid = HybridRetriever([dense, bm25], depth=depth) if bm25 else None
    built = {
        'dense': dense,
        'bm25': bm25,
        'hybrid': hybrid,
        'dense+rerank': RerankedRetriever(dense, documents),
        'hybrid+rerank': hybrid and RerankedRetriever(hybrid, documents),
    }
    return {name: built[name] for name in names}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--n', type=int, help='use only the first N questions')
    parser.add_argument('--ks', type=int, nargs='+', default=DEFAULT_KS)
    parser.add_argument(
        '--retrievers', nargs='+', choices=RETRIEVER_CHOICES,
        default=list(DEFAULT_RETRIEVERS),
    )
    parser.add_argument(
        '--depth', type=int, default=DEFAULT_DEPTH,
        help='how deep each retriever is asked inside the hybrid'
    )
    args = parser.parse_args(argv)
    if max(args.ks) > DEFAULT_CANDIDATES_K and any(
            'rerank' in name for name in args.retrievers):
        parser.error(
            f'--ks above {DEFAULT_CANDIDATES_K} cannot be used with a reranker'
        )

    subset = load_hotpotqa()
    questions = subset.questions[:args.n]
    retrievers = build_retrievers(
        args.retrievers, subset.documents, args.depth
    )
    deepest = max(args.ks)
    rankings = {
        name: retriever.retrieve_many(questions, deepest)
        for name, retriever in retrievers.items()
    }
    groups = {'all': list(range(len(questions)))}
    for kind in sorted({q.type for q in questions}):
        groups[kind] = [i for i, q in enumerate(questions) if q.type == kind]
    for label, indices in groups.items():
        results = {
            name: score_rankings(
                [ranked[i] for i in indices], [questions[i] for i in indices],
                args.ks,
            )
            for name, ranked in rankings.items()
        }
        print(f'\n=== {label} (n={len(indices)})')
        print(format_table(results, args.ks))


if __name__ == '__main__':
    main()
