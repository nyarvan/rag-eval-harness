"""Retrieval and answer metrics: recall@k, all-gold@k, exact match, token F1.

`normalize_answer` and `f1` follow the official HotpotQA evaluation script.
"""
from collections import Counter
from collections.abc import Collection, Sequence
import re
import string


def recall_at_k(
        retrieved: Sequence[str], gold: Collection[str], k: int
) -> float:
    """Share of distinct gold titles found among the first `k` retrieved.

    `retrieved` is ranked best-first. Duplicates on either side count once.
    An empty `gold` is a data bug, so it raises instead of returning a score.
    """
    if k <= 0:
        raise ValueError(f'k must be positive, got {k}')
    gold_titles = set(gold)
    if not gold_titles:
        raise ValueError('gold must not be empty')
    found = gold_titles & set(retrieved[:k])
    return len(found) / len(gold_titles)


def all_gold_at_k(
        retrieved: Sequence[str], gold: Collection[str], k: int
) -> bool:
    """True when every gold title is among the first `k` retrieved."""
    return recall_at_k(retrieved, gold, k) == 1.0


def normalize_answer(s: str) -> str:
    """
    Reference from Github repo: hotpotqa/hotpot/hotpot_evaluate_v1.py
    """
    def remove_articles(text: str) -> str:
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text: str) -> str:
        return ' '.join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(prediction: str, reference: str) -> float:
    """1.0 if the normalized answers are equal, else 0.0."""
    return float(normalize_answer(prediction) == normalize_answer(reference))


def f1(prediction: str, reference: str) -> float:
    """
    Reference from Github repo: hotpotqa/hotpot/hotpot_evaluate_v1.py
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(reference)

    zero = 0.0

    if normalized_prediction in ['yes', 'no', 'noanswer'] \
            and normalized_prediction != normalized_ground_truth:
        return zero
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] \
            and normalized_prediction != normalized_ground_truth:
        return zero

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return zero
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)
