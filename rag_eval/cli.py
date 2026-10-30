"""Command line for the evaluation harness.

    python cli.py run  [flags of rag_eval.run]       run a pipeline (paid)
    python cli.py eval RUN_FILE [--ragas]            score a run file
    python cli.py compare RUN_FILE RUN_FILE ...      compare run files

`run` and `eval --ragas` call a paid API, so they print an estimate first and
wait for confirmation (`--yes` skips the question; without a terminal and
without `--yes` they do not start). `eval` and `compare` accept a run file
from any RAG pipeline, as long as it fits `rag_eval.records.RunRecord`.

Exit codes: 0 done, 1 a paid step was not confirmed, 2 invalid input.
"""
import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from statistics import fmean

from significance import format_comparisons

from . import ragas_eval, run
from .args import positive_int
from .compare import compare_runs
from .cost import confirm, estimate_generation, estimate_judging, format_estimate
from .data import DEFAULT_N, load_hotpotqa
from .evaluate import DEFAULT_KS, evaluate_records
from .generate import (
    COT_MAX_TOKENS,
    COT_SYSTEM_PROMPT,
    DEFAULT_MAX_TOKENS,
    SYSTEM_PROMPT,
    format_context,
)
from .records import RunRecord, read_records

EXIT_DONE = 0
EXIT_NOT_CONFIRMED = 1
EXIT_INVALID_INPUT = 2


def build_parser() -> argparse.ArgumentParser:
    """The parser with the three commands; `run` and `eval` reuse the flags
    of `rag_eval.run` and `rag_eval.ragas_eval`."""
    parser = argparse.ArgumentParser(
        prog='cli.py', description=__doc__.split('\n')[0]
    )
    commands = parser.add_subparsers(dest='command', required=True)

    run_cmd = commands.add_parser(
        'run', parents=[run.build_parser(add_help=False)],
        help='run a pipeline on HotpotQA and write a run file (paid)',
    )
    run_cmd.add_argument(
        '--yes', action='store_true', help='do not ask before spending money'
    )

    eval_cmd = commands.add_parser(
        'eval', parents=[ragas_eval.build_parser(add_help=False)],
        help='score a run file: free metrics, and RAGAS with --ragas',
    )
    eval_cmd.add_argument(
        '--ragas', action='store_true',
        help='also score with the RAGAS judge (paid)'
    )
    eval_cmd.add_argument(
        '--ks', type=positive_int, nargs='+', default=list(DEFAULT_KS),
        help='k for recall@k and all_gold@k'
    )
    eval_cmd.add_argument(
        '--yes', action='store_true', help='do not ask before spending money'
    )

    compare_cmd = commands.add_parser(
        'compare', help='paired comparison of run files over the same questions'
    )
    compare_cmd.add_argument(
        'run_files', nargs='+', help='the first one is the baseline'
    )
    compare_cmd.add_argument(
        '--ks', type=positive_int, nargs='+',
        help='k for the retrieval metrics (default: 5 10 when the files have '
             'the fields, none otherwise)'
    )
    return parser


def main(
        argv: Sequence[str] | None = None, *,
        run_fn: Callable[[argparse.Namespace], None] = run.execute,
        ragas_fn: Callable[[argparse.Namespace], None] = ragas_eval.execute,
        corpus_chars_fn: Callable[[], float] | None = None,
        question_count_fn: Callable[[], int] = lambda: DEFAULT_N,
        confirm_fn: Callable[..., bool] = confirm,
) -> int:
    """Run the command in `argv`; the keyword arguments are for tests."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == 'compare' and len(args.run_files) < 2:
        parser.error('compare needs at least two run files')

    if args.command == 'run':
        return _run_command(
            args, run_fn, corpus_chars_fn or _average_paragraph_chars,
            question_count_fn, confirm_fn,
        )
    if args.command == 'eval':
        return _eval_command(args, ragas_fn, confirm_fn)
    return _compare_command(args)


def _run_command(
        args: argparse.Namespace, run_fn: Callable[[argparse.Namespace], None],
        corpus_chars_fn: Callable[[], float],
        question_count_fn: Callable[[], int], confirm_fn: Callable[..., bool],
) -> int:
    try:
        run.check_flags(args)
    except ValueError as error:
        return _invalid(error)
    system = COT_SYSTEM_PROMPT if args.cot else SYSTEM_PROMPT
    estimate = estimate_generation(
        args.chat_model, n_questions=min(args.n, question_count_fn()),
        context_k=args.context_k,
        avg_paragraph_chars=corpus_chars_fn(), system_chars=len(system),
        max_output_tokens=COT_MAX_TOKENS if args.cot else DEFAULT_MAX_TOKENS,
    )
    print(format_estimate(f'Generation with {args.chat_model}', estimate))
    if not confirm_fn('Run it?', assume_yes=args.yes):
        return _not_confirmed()
    run_fn(args)
    return EXIT_DONE


def _eval_command(
        args: argparse.Namespace, ragas_fn: Callable[[argparse.Namespace], None],
        confirm_fn: Callable[..., bool],
) -> int:
    try:
        records = read_records(args.run_file)[:args.n]
        metrics = evaluate_records(records, args.ks)
    except (OSError, ValueError) as error:
        return _invalid(error)
    for name, value in metrics.items():
        print(f'{name}: {value:.3f}')
    print(f'{len(records)} records in {args.run_file}')
    if not args.ragas:
        return EXIT_DONE

    estimate = estimate_judging(args.judge, len(records), args.metrics)
    print(format_estimate(f'RAGAS judging with {args.judge}', estimate))
    if not confirm_fn('Score it?', assume_yes=args.yes):
        return _not_confirmed()
    ragas_fn(args)
    return EXIT_DONE


def _compare_command(args: argparse.Namespace) -> int:
    try:
        runs = _read_runs(args.run_files)
        ks = args.ks if args.ks is not None else _default_ks(runs)
        comparisons = compare_runs(runs, ks=ks)
    except (OSError, ValueError) as error:
        return _invalid(error)
    print(format_comparisons(comparisons))
    return EXIT_DONE


def _read_runs(paths: Sequence[str]) -> dict[str, list[RunRecord]]:
    """Run files by name: the file name without its extension."""
    if len(set(paths)) != len(paths):
        raise ValueError('a run file was given twice')
    names = [Path(path).stem for path in paths]
    if len(set(names)) != len(names):
        names = list(paths)  # same stem in different folders: use the paths
    return {name: read_records(path) for name, path in zip(names, paths)}


def _default_ks(runs: dict[str, list[RunRecord]]) -> tuple[int, ...]:
    """The retrieval ks, if every record of every file has the fields."""
    complete = all(
        r.retrieved is not None and r.gold_titles is not None
        for records in runs.values() for r in records
    )
    return DEFAULT_KS if complete else ()


def _average_paragraph_chars() -> float:
    """Mean length of a paragraph as it appears in a prompt."""
    return fmean(len(format_context(d)) for d in load_hotpotqa().documents)


def _not_confirmed() -> int:
    print('Cancelled: nothing was run. Pass --yes to run without asking.')
    return EXIT_NOT_CONFIRMED


def _invalid(error: Exception) -> int:
    print(f'error: {error}', file=sys.stderr)
    return EXIT_INVALID_INPUT
