"""One line of a run file: what was retrieved, asked and answered.

`question`, `contexts`, `answer` and `reference` are the field names RAGAS
expects; the rest lets retrieval metrics be recomputed without re-running.
"""
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunRecord:
    """One question of a run.

    `contexts` are the passages given to the generator, `retrieved` the
    ranked titles the retriever returned (possibly more than were used).
    """

    question_id: str
    question: str
    contexts: tuple[str, ...]
    answer: str
    reference: str
    retrieved: tuple[str, ...]
    gold_titles: tuple[str, ...]


def write_records(path: str | Path, records: Iterable[RunRecord]) -> None:
    """Write one JSON object per line; parent directories are created."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as file:
        for record in records:
            file.write(json.dumps(asdict(record), ensure_ascii=False) + '\n')


def read_records(path: str | Path) -> list[RunRecord]:
    """Read a run file written by `write_records`."""
    with Path(path).open(encoding='utf-8') as file:
        return [_from_json(json.loads(line)) for line in file if line.strip()]


def _from_json(data: dict) -> RunRecord:
    return RunRecord(
        question_id=data['question_id'],
        question=data['question'],
        contexts=tuple(data['contexts']),
        answer=data['answer'],
        reference=data['reference'],
        retrieved=tuple(data['retrieved']),
        gold_titles=tuple(data['gold_titles']),
    )
