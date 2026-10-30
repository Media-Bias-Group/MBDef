"""The collect step's files: its folder, the CSV ledgers it resumes from, and its run receipts.

Every action appends one row per attempt to its status ledger (an append-only CSV) and rewrites a small snapshot CSV (the current candidates, or the current failures) atomically. The latest row of a paper id is its state, so a run that stops halfway is resumed by running the same command again.
"""

import csv
import io
import json
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path

from mbdef import files

FOLDER = "collect"  # the step's folder under DATA_ROOT
COLLECTION = "citation-papers-to-collect.csv"  # the queue, written by the closure step's corpus check
RECEIPTS = "receipts.jsonl"  # one receipt per run of any action, appended

# The columns of the collection queue.
COLLECTION_FIELDS = [
    "queue_status",
    "title",
    "authors",
    "year",
    "doi",
    "url",
    "formatted_reference",
    "citation_methods",
    "citation_confidences",
    "citing_paper_ids",
    "occurrence_count",
    "review_reason",
    "suggested_corpus_files",
]


def folder(path: str | None) -> Path:
    """The step's folder: `path` if given, else `$DATA_ROOT/collect`."""
    return Path(path) if path else files.data_root() / FOLDER


def collection(path: str | None, step_folder: Path) -> Path:
    """The collection queue: `path` if given, else the one in the step's folder."""
    queue = Path(path) if path else step_folder / COLLECTION
    if not queue.is_file():
        raise SystemExit(
            f"The collection queue {queue} does not exist. The closure step's corpus check "
            "writes it (docs/step-1-2-papers/collect.md); pass another one with --collection."
        )
    return queue


def append_receipt(
    step_folder: Path,
    command: str,
    settings: dict,
    inputs: dict[str, Path],
    outputs: dict[str, Path],
    summary: dict,
) -> None:
    """Append what this run did to `receipts.jsonl`: settings, hashes, code commit, time, counts."""
    receipt = files.receipt(command, settings, inputs, outputs) | {"summary": summary}
    with (step_folder / RECEIPTS).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, ensure_ascii=False, sort_keys=True) + "\n")


def report(summary: dict, file_set: dict[str, Path]) -> None:
    print(json.dumps(summary, sort_keys=True))
    for name, path in file_set.items():
        print(f"{name}: {path}")


def read_csv(
    path: Path,
    fieldnames: list[str],
    *,
    extra: bool = False,
    optional: tuple[str, ...] = (),
) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        actual = reader.fieldnames or []
        if (
            any(name not in actual for name in fieldnames if name not in optional)
            or (not extra and any(name not in fieldnames for name in actual))
            or len(actual) != len(set(actual))
        ):
            raise ValueError(
                f"CSV header mismatch in {path}: expected {fieldnames}, got {actual}"
            )
        rows = []
        for number, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed CSV row {number} in {path}")
            rows.append({**dict.fromkeys(fieldnames, ""), **row})
        return rows


def append_status(path: Path, fieldnames: list[str], row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        with path.open(encoding="utf-8", newline="") as existing:
            previous = next(csv.reader(existing))
        if previous != fieldnames:
            if set(previous) - set(fieldnames):
                raise ValueError(f"Cannot discard ledger columns in {path}")
            rows = read_csv(
                path, fieldnames, optional=tuple(set(fieldnames) - set(previous))
            )
            replace_csv(path, rows, fieldnames)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def latest_by_paperid(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    latest = {}
    for row in rows:
        if not row.get("paperid"):
            raise ValueError("Status CSV is missing required paperid")
        latest[row["paperid"]] = row
    return list(latest.values())


def csv_text(rows: list[dict], fieldnames: list[str]) -> str:
    """ROWS as CSV text with a header (the csv module's format: comma-separated, CRLF lines)."""
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def replace_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    files.write_atomic({path: csv_text(rows, fieldnames)})


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def bounded_results(function, rows, concurrency: int):
    """At most concurrency futures/payloads exist, including completed downloads."""
    rows = iter(rows)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        pending = set()
        while True:
            while len(pending) < concurrency:
                try:
                    row = next(rows)
                except StopIteration:
                    break
                pending.add(pool.submit(function, row))
            if not pending:
                return
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()
