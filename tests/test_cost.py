import pytest

from rag_eval.cost import (
    JUDGE_CALLS_PER_QUESTION,
    PRICES,
    CostEstimate,
    confirm,
    estimate_generation,
    estimate_judging,
    format_estimate,
)


def test_generation_input_tokens_follow_the_prompt_size():
    estimate = estimate_generation(
        'gpt-4o-mini', n_questions=10, context_k=5,
        avg_paragraph_chars=400, system_chars=200, max_output_tokens=64,
    )

    # per request: 200 system + 5 * 400 context + 200 question = 2400 chars
    assert estimate.calls == 10
    assert estimate.input_tokens == 10 * 2400 // 4
    assert estimate.output_tokens == 10 * 64


def test_generation_cost_uses_the_model_price():
    estimate = estimate_generation(
        'gpt-4o-mini', n_questions=1000, context_k=5,
        avg_paragraph_chars=400, system_chars=200, max_output_tokens=64,
    )
    price_in, price_out = PRICES['gpt-4o-mini']

    expected = (
        estimate.input_tokens * price_in + estimate.output_tokens * price_out
    ) / 1_000_000
    assert estimate.usd == pytest.approx(expected)


def test_a_model_without_a_known_price_has_no_cost():
    estimate = estimate_generation(
        'some-new-model', n_questions=10, context_k=5,
        avg_paragraph_chars=400, system_chars=200, max_output_tokens=64,
    )

    assert estimate.usd is None
    assert estimate.calls == 10


def test_more_context_costs_more():
    def cost(context_k):
        return estimate_generation(
            'gpt-4o-mini', n_questions=100, context_k=context_k,
            avg_paragraph_chars=400, system_chars=200, max_output_tokens=64,
        ).usd

    assert cost(10) > cost(5) > cost(3)


def test_judging_calls_scale_with_questions_and_metrics():
    all_four = tuple(JUDGE_CALLS_PER_QUESTION)

    full = estimate_judging('gpt-4o-mini', 200, all_four)
    one = estimate_judging('gpt-4o-mini', 200, ('factual_correctness',))

    assert full.calls == round(200 * sum(JUDGE_CALLS_PER_QUESTION.values()))
    assert one.calls == round(
        200 * JUDGE_CALLS_PER_QUESTION['factual_correctness']
    )
    assert one.calls < full.calls


def test_judging_with_an_unknown_metric_is_rejected():
    with pytest.raises(ValueError, match='nonsense'):
        estimate_judging('gpt-4o-mini', 10, ('nonsense',))


def test_the_four_metrics_make_about_ten_and_a_half_calls_per_question():
    # the only figure measured on a real run; the split between metrics is
    # an assumption, so only the total is asserted
    assert sum(JUDGE_CALLS_PER_QUESTION.values()) == pytest.approx(10.5)


def test_the_formatted_estimate_is_marked_as_approximate():
    estimate = CostEstimate(
        calls=200, input_tokens=340_000, output_tokens=12_800, usd=0.0587
    )

    text = format_estimate('Generation with gpt-4o-mini', estimate)

    assert 'Generation with gpt-4o-mini' in text
    assert '200' in text and '$0.06' in text
    assert 'approx' in text.lower()


def test_an_unknown_price_is_said_so_instead_of_guessed():
    estimate = CostEstimate(
        calls=200, input_tokens=1000, output_tokens=100, usd=None
    )

    text = format_estimate('Generation', estimate)

    assert '$' not in text
    assert 'unknown' in text.lower()


@pytest.mark.parametrize('answer', ['y', 'Y', 'yes', 'YES', ' yes '])
def test_confirm_accepts_yes(answer):
    assert confirm('Go on?', input_fn=lambda _: answer, interactive=True)


@pytest.mark.parametrize('answer', ['', 'n', 'no', 'maybe', 'yep'])
def test_confirm_refuses_anything_else(answer):
    assert not confirm('Go on?', input_fn=lambda _: answer, interactive=True)


def test_assume_yes_skips_the_question():
    def fail(_):
        raise AssertionError('must not ask')

    assert confirm('Go on?', assume_yes=True, input_fn=fail, interactive=False)


def test_without_a_terminal_and_without_yes_nothing_runs():
    def fail(_):
        raise AssertionError('must not ask')

    assert not confirm('Go on?', input_fn=fail, interactive=False)
