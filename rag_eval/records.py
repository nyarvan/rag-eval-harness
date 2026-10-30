"""One line of a run file: what was retrieved, asked and answered.

`question`, `contexts`, `answer` and `reference` are the field names RAGAS
expects; the rest lets retrieval metrics be recomputed without re-running.

The record is also the contract for run files from any RAG pipeline. Reading
validates every line and names the file and line of the first problem; fields
beyond the ones below are ignored.
"""
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class RunRecord(BaseModel):
    """One question of a run.

    `contexts` are the passages given to the generator, `retrieved` the
    ranked titles the retriever returned (possibly more than were used).
    `retrieved` and `gold_titles` are only needed for retrieval metrics;
    None means the field is absent, while an empty `retrieved` means the
    retriever found nothing.
    """

    model_config = ConfigDict(frozen=True)

    question_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    contexts: tuple[str, ...]
    answer: str
    reference: str
    retrieved: tuple[str, ...] | None = None
    gold_titles: tuple[str, ...] | None = None


def write_records(path: str | Path, records: Iterable[RunRecord]) -> None:
    """Write one JSON object per line; parent directories are created."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as file:
        for record in records:
            file.write(record.model_dump_json() + '\n')


def read_records(path: str | Path) -> list[RunRecord]:
    """Read and validate a run file; blank lines are skipped.

    Raises `ValueError` naming `path:line` when a line is not valid JSON or
    does not fit the record.
    """
    records = []
    with Path(path).open(encoding='utf-8') as file:
        for number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(RunRecord.model_validate_json(line))
            except ValidationError as error:
                raise ValueError(f'{path}:{number}: {error}') from error
    return records
