"""Rough cost of the paid steps, and the confirmation asked before they run.

Estimates are upper bounds on one fresh run: answers and judge replies that
are already cached cost nothing. Prices are approximate list prices and may be
out of date; a model that is not in `PRICES` gets no dollar figure rather than
a guess.
"""
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

# USD per 1M tokens (input, output)
PRICES: dict[str, tuple[float, float]] = {
    'gpt-4o-mini': (0.15, 0.60),
    'gpt-4o': (2.50, 10.00),
}
CHARS_PER_TOKEN = 4
QUESTION_CHARS = 200  # allowance for the question in a generation prompt

# Judge requests per question for each RAGAS metric. Only the total was
# measured (about 10.5 per question for the four metrics on a real run); the
# split between the metrics is an assumption.
JUDGE_CALLS_PER_QUESTION: dict[str, float] = {
    'factual_correctness': 3.0,
    'faithfulness': 3.0,
    'context_recall': 1.5,
    'context_precision': 3.0,
}
JUDGE_INPUT_TOKENS = 1200  # assumed per judge request
JUDGE_OUTPUT_TOKENS = 150  # assumed per judge request


@dataclass(frozen=True)
class CostEstimate:
    """Requests and tokens of a paid step; `usd` is None for unknown prices."""

    calls: int
    input_tokens: int
    output_tokens: int
    usd: float | None


def estimate_generation(
        model: str, *, n_questions: int, context_k: int,
        avg_paragraph_chars: float, system_chars: int, max_output_tokens: int,
) -> CostEstimate:
    """Answering `n_questions` with `context_k` paragraphs each."""
    chars = system_chars + context_k * avg_paragraph_chars + QUESTION_CHARS
    input_tokens = int(n_questions * chars // CHARS_PER_TOKEN)
    output_tokens = n_questions * max_output_tokens
    return CostEstimate(
        n_questions, input_tokens, output_tokens,
        _usd(model, input_tokens, output_tokens),
    )


def estimate_judging(
        model: str, n_questions: int, metrics: Sequence[str],
) -> CostEstimate:
    """Scoring `n_questions` with the given RAGAS metrics."""
    for metric in metrics:
        if metric not in JUDGE_CALLS_PER_QUESTION:
            raise ValueError(f'unknown metric {metric!r}')
    calls = round(
        n_questions * sum(JUDGE_CALLS_PER_QUESTION[m] for m in metrics)
    )
    input_tokens = calls * JUDGE_INPUT_TOKENS
    output_tokens = calls * JUDGE_OUTPUT_TOKENS
    return CostEstimate(
        calls, input_tokens, output_tokens,
        _usd(model, input_tokens, output_tokens),
    )


def format_estimate(label: str, estimate: CostEstimate) -> str:
    """One line for the terminal, marked as approximate."""
    price = (
        'price unknown for this model' if estimate.usd is None
        else f'about ${estimate.usd:.2f}'
    )
    return (
        f'{label}: about {estimate.calls:,} requests, up to '
        f'{estimate.input_tokens:,} input and {estimate.output_tokens:,} '
        f'output tokens, {price} (approximate; cached replies are free)'
    )


def confirm(
        question: str, *, assume_yes: bool = False,
        input_fn: Callable[[str], str] = input,
        interactive: bool | None = None,
) -> bool:
    """Ask before a paid step; only `y` or `yes` goes ahead.

    Without a terminal nobody can answer, so unless `assume_yes` is set the
    answer is no, instead of a script hanging on a prompt or paying silently.
    """
    if assume_yes:
        return True
    if interactive is None:
        interactive = sys.stdin.isatty()
    if not interactive:
        return False
    return input_fn(f'{question} [y/N] ').strip().lower() in ('y', 'yes')


def _usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    if model not in PRICES:
        return None
    price_in, price_out = PRICES[model]
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000
