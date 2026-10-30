"""Free metrics for a run file: no LLM is called.

Exact match and F1 of the answers always; `recall@k` and `all_gold@k` too when
the records carry `retrieved` and `gold_titles`.
"""
from collections.abc import Sequence
from statistics import fmean

from .metrics import all_gold_at_k, exact_match, f1, recall_at_k
from .records import RunRecord

DEFAULT_KS = (5, 10)


def evaluate_records(
        records: Sequence[RunRecord], ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, float]:
    """Mean metrics over the records.

    Retrieval fields are optional in a run file, so records without them get
    answer metrics only; but fields present in some records and missing in
    others is a data error and raises.
    """
    if not records:
        raise ValueError('records must not be empty')
    result = {
        'em': fmean(exact_match(r.answer, r.reference) for r in records),
        'f1': fmean(f1(r.answer, r.reference) for r in records),
    }
    with_retrieval = [r for r in records if r.retrieved is not None
                      and r.gold_titles is not None]
    if not with_retrieval:
        return result
    if len(with_retrieval) != len(records):
        raise ValueError(
            'retrieved and gold_titles are present in only '
            f'{len(with_retrieval)} of {len(records)} records'
        )
    for k in ks:
        result[f'recall@{k}'] = fmean(
            recall_at_k(r.retrieved, r.gold_titles, k) for r in records
        )
        result[f'all_gold@{k}'] = fmean(
            all_gold_at_k(r.retrieved, r.gold_titles, k) for r in records
        )
    return result
