"""Dense-retrieval run: retrieve, generate, score, and save to a run file.

    python -m rag_eval.run --n 20 --out runs/dense.jsonl

`retrieve_k` is how deep the ranking goes (for recall@k); `context_k` is how
many of those passages the generator actually reads.
"""
import argparse
from collections.abc import Sequence
from statistics import fmean
from typing import Protocol

from .data import DEFAULT_N, Document, Question, load_hotpotqa
from .generate import Generator, OpenAIChat, format_context
from .metrics import all_gold_at_k, exact_match, f1, recall_at_k
from .records import RunRecord, write_records
from .retrieve import DenseRetriever

RETRIEVE_K = 10
CONTEXT_K = 5
RECALL_KS = (2, 5, 10)
DEFAULT_RUN_QUESTIONS = 20
DEFAULT_OUT = 'runs/dense.jsonl'
DEFAULT_MODEL = 'bge_m3'


class Retriever(Protocol):
    def retrieve_many(
            self, questions: Sequence[Question], k: int,
    ) -> list[list[str]]: ...


class Answerer(Protocol):
    def answer(self, question: str, contexts: Sequence[Document]) -> str: ...


def run_dense(
        questions: Sequence[Question], documents: Sequence[Document],
        retriever: Retriever, generator: Answerer, *,
        retrieve_k: int = RETRIEVE_K, context_k: int = CONTEXT_K,
) -> list[RunRecord]:
    """Retrieve for all questions in one batch, then answer each of them."""
    if context_k > retrieve_k:
        raise ValueError(
            f'context_k ({context_k}) must not exceed retrieve_k ({retrieve_k})'
        )
    by_title = {document.doc_id: document for document in documents}
    ranked = retriever.retrieve_many(questions, retrieve_k)
    records = []
    for question, titles in zip(questions, ranked, strict=True):
        contexts = [by_title[title] for title in titles[:context_k]]
        records.append(RunRecord(
            question_id=question.question_id,
            question=question.question,
            contexts=tuple(format_context(document) for document in contexts),
            answer=generator.answer(question.question, contexts),
            reference=question.answer,
            retrieved=tuple(titles),
            gold_titles=question.gold_titles,
        ))
    return records


def summarize(records: Sequence[RunRecord]) -> dict[str, float]:
    """Mean recall@k, all_gold@k, EM and F1 over the records."""
    if not records:
        raise ValueError('records must not be empty')
    summary = {}
    for k in RECALL_KS:
        summary[f'recall@{k}'] = fmean(
            recall_at_k(r.retrieved, r.gold_titles, k) for r in records
        )
        summary[f'all_gold@{k}'] = fmean(
            all_gold_at_k(r.retrieved, r.gold_titles, k) for r in records
        )
    summary['em'] = fmean(exact_match(r.answer, r.reference) for r in records)
    summary['f1'] = fmean(f1(r.answer, r.reference) for r in records)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--n', type=int, default=DEFAULT_RUN_QUESTIONS)
    parser.add_argument('--out', default=DEFAULT_OUT)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--retrieve-k', type=int, default=RETRIEVE_K)
    parser.add_argument('--context-k', type=int, default=CONTEXT_K)
    parser.add_argument(
        '--env-file', help='dotenv file with OPENAI_API_KEY (not printed)'
    )
    args = parser.parse_args(argv)

    if args.env_file:
        from dotenv import load_dotenv
        load_dotenv(args.env_file)

    subset = load_hotpotqa()
    questions = subset.questions[:args.n]
    retriever = DenseRetriever(
        args.model, subset.documents, subset_size=DEFAULT_N
    )
    generator = Generator(OpenAIChat())
    records = run_dense(
        questions, subset.documents, retriever, generator,
        retrieve_k=args.retrieve_k, context_k=args.context_k,
    )
    write_records(args.out, records)
    for name, value in summarize(records).items():
        print(f'{name}: {value:.3f}')
    print(f'{len(records)} records -> {args.out}')


if __name__ == '__main__':
    main()
