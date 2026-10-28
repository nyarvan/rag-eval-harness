"""Small reproducible HotpotQA subset (distractor setting) for RAG evaluation.

    <out_dir>/
        corpus.jsonl     one paragraph per line: doc_id, title, text
        questions.jsonl  question, answer, gold titles and gold sentences
        meta.json        where the subset comes from and how it was sampled

The dataset revision is pinned so the same seed always gives the same subset.
"""
import json
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

DATASET_ID = 'hotpotqa/hotpot_qa'
CONFIG = 'distractor'
SPLIT = 'validation'
SPLIT_FILE = f'{CONFIG}/{SPLIT}-00000-of-00001.parquet'
REVISION = '1908d6afbbead072334abe2965f91bd2709910ab'
DEFAULT_CACHE_DIR = 'data/hf_cache'
DEFAULT_N = 200
DEFAULT_SEED = 42


@dataclass(frozen=True)
class Document:
    """One paragraph of the corpus; the title is unique, so it is the id."""

    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class Question:
    """A question with its gold titles and gold evidence sentences."""

    question_id: str
    question: str
    answer: str
    type: str
    gold_titles: tuple[str, ...]
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class HotpotSubset:
    """Corpus and questions of a subset.

    `title_conflicts` counts titles whose text differs between paragraphs; the
    first text is kept. `dropped_facts` counts supporting facts skipped because
    their title or sentence is missing from the kept paragraph.
    """

    documents: tuple[Document, ...]
    questions: tuple[Question, ...]
    title_conflicts: int
    dropped_facts: int


def paragraph_text(sentences: Sequence[str]) -> str:
    """
    Join sentences into a paragraph; all but the first start with a space.
    """
    return ''.join(sentences).strip()


def sample_rows(
        rows: Sequence[Mapping[str, Any]], n: int, seed: int,
) -> list[Mapping[str, Any]]:
    """Pick `n` distinct rows; the same seed always gives the same rows."""
    if n <= 0 or n > len(rows):
        raise ValueError(f'n must be in [1, {len(rows)}], got {n}')
    return random.Random(seed).sample(list(rows), n)


def to_question(
        row: Mapping[str, Any],
        sentences_by_title: Mapping[str, Sequence[str]] | None = None,
) -> Question:
    """Convert a raw row; facts missing from the paragraphs are skipped.

    Pass the corpus-wide first-wins map so that evidence is always a sentence
    of the stored document; by default the row's own paragraphs are used.
    """
    if sentences_by_title is None:
        sentences_by_title, _ = _first_wins([row])
    return _convert(row, sentences_by_title)[0]


def build_subset(rows: Sequence[Mapping[str, Any]]) -> HotpotSubset:
    """Build the corpus from the rows' paragraphs, deduplicated by title."""
    sentences_by_title, conflicts = _first_wins(rows)
    converted = [_convert(row, sentences_by_title) for row in rows]
    return HotpotSubset(
        documents=tuple(
            Document(title, title, paragraph_text(sentences))
            for title, sentences in sentences_by_title.items()
        ),
        questions=tuple(question for question, _ in converted),
        title_conflicts=conflicts,
        dropped_facts=sum(dropped for _, dropped in converted),
    )


def download_validation(cache_dir: str | Path = DEFAULT_CACHE_DIR) -> Path:
    """Path of the pinned validation parquet; downloaded only if not cached."""
    path = hf_hub_download(
        repo_id=DATASET_ID,
        filename=SPLIT_FILE,
        repo_type='dataset',
        revision=REVISION,
        cache_dir=str(cache_dir),
    )
    return Path(path)


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read all rows of a parquet file as plain dicts."""
    return pq.read_table(path).to_pylist()


def save_subset(
        subset: HotpotSubset, out_dir: str | Path, *, n: int, seed: int,
) -> None:
    """Write the subset and its provenance to `out_dir` (created if needed)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / 'corpus.jsonl', map(asdict, subset.documents))
    _write_jsonl(out / 'questions.jsonl', map(asdict, subset.questions))
    meta = {
        'dataset': DATASET_ID,
        'config': CONFIG,
        'split': SPLIT,
        'revision': REVISION,
        'n': n,
        'seed': seed,
        'n_documents': len(subset.documents),
        'title_conflicts': subset.title_conflicts,
        'dropped_facts': subset.dropped_facts,
    }
    (out / 'meta.json').write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8'
    )


def load_hotpotqa(
        n: int = DEFAULT_N, seed: int = DEFAULT_SEED,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> HotpotSubset:
    """Sample `n` validation questions and build their subset."""
    rows = read_rows(download_validation(cache_dir))
    return build_subset(sample_rows(rows, n, seed))


def _first_wins(
        rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, tuple[str, ...]], int]:
    """Map title -> sentences of its first paragraph, plus the conflict count.

    A conflict is a title whose later paragraph has different text; each such
    title is counted once.
    """
    kept: dict[str, tuple[str, ...]] = {}
    conflicts: set[str] = set()
    for row in rows:
        context = row['context']
        pairs = zip(context['title'], context['sentences'], strict=True)
        for title, sentences in pairs:
            if title not in kept:
                kept[title] = tuple(sentences)
            elif paragraph_text(kept[title]) != paragraph_text(sentences):
                conflicts.add(title)
    return kept, len(conflicts)


def _convert(
        row: Mapping[str, Any],
        sentences_by_title: Mapping[str, Sequence[str]],
) -> tuple[Question, int]:
    """Question of a row and the number of its facts that had to be dropped."""
    facts = row['supporting_facts']
    evidence = []
    dropped = 0
    for title, sent_id in zip(facts['title'], facts['sent_id'], strict=True):
        sentences = sentences_by_title.get(title, ())
        if 0 <= sent_id < len(sentences):
            evidence.append(sentences[sent_id].strip())
        else:
            dropped += 1
    question = Question(
        question_id=row['id'],
        question=row['question'],
        answer=row['answer'],
        type=row['type'],
        gold_titles=tuple(dict.fromkeys(facts['title'])),
        evidence=tuple(evidence),
    )
    return question, dropped


def _write_jsonl(path: Path, items: Iterable[Mapping[str, Any]]) -> None:
    """Write one JSON object per line; asdict turns tuples into lists."""
    with path.open('w', encoding='utf-8') as file:
        for item in items:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    default_subset = load_hotpotqa()
    save_subset(
        default_subset, f'data/hotpotqa-{DEFAULT_N}',
        n=DEFAULT_N, seed=DEFAULT_SEED,
    )
