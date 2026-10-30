"""Definition records and the deterministic rules applied to the model's records.

A **definition record** is one term with seven lists of evidence (`FIELDS`); each piece of evidence is a text and the citation markers attached to it (`extractor.py` holds the schema the model fills). The model writes records for one window at a time. After the model, three rules run, and nothing else changes the records:

1. `sanitize`: keep a string only if it occurs in the window (`exact`: contiguous, ignoring case, after runs of white space in the model's string become one space; returned as the window spells it); drop duplicates within a list, and drop a record whose term is not in the window or whose lists are all empty.
2. `merge`: records with the same term (ignoring case) become one; the first spelling wins and the lists are joined without duplicates. A merged term without any definition is dropped.
3. `union`: one paper's windows merged into one list of records, plus one verifier candidate per definition, linked to every window that holds it.

**A predictions file** (`predictions.jsonl`; `write_predictions`, `read_predictions`) is how records leave the pipeline and how any tool's records enter its scorer: one JSON object per line, one record per line, `{"paper_id": "...", "term": "...", "definitions": ["...", ...]}`, optionally with the six other lists (`synonyms`, `description`, ...). An item is a string or `{"text": ..., "citation": [...]}`. Reading groups the lines by paper and merges each paper's records with `merge`, so two files are pooled by concatenating them (`cat a.jsonl b.jsonl`), the way the released variants pool their parts.

Worked example: for the window "Media bias is the bias of journalists." the model's record `{"term": "MEDIA BIAS", "definitions": [{"text": "media  bias is the bias of journalists"}]}` is kept as `{"term": "Media bias", "definitions": [{"text": "Media bias is the bias of journalists", "citation": []}], ...}` (every other list empty), because both strings occur in the window.
"""

import copy
import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path

FIELDS = (
    "definitions",
    "synonyms",
    "description",
    "related_terms",
    "subtypes",
    "supertypes",
    "examples",
)


def clean(text: str) -> str:
    """Runs of white space become one space; the ends are stripped."""
    return re.sub(r"\s+", " ", text or "").strip()


def exact(source: str, value: str) -> str | None:
    """`value` as it is spelled in `source`, if it occurs there (ignoring case and white space runs).

    Case folding can change a string's length (ß becomes ss), so the match is mapped back to the original characters: exact("Straße Bias", "STRASSE") is "Straße".
    """
    value = clean(value)
    if not value:
        return None
    start = source.find(value)
    if start >= 0:
        return source[start : start + len(value)]
    folded = value.casefold()
    start = source.casefold().find(folded)
    if start < 0:
        return None
    positions = [i for i, char in enumerate(source) for _ in char.casefold()]
    result = source[positions[start] : positions[start + len(folded) - 1] + 1]
    return result if result.casefold() == folded else None


def as_dict(value) -> dict:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value if isinstance(value, dict) else dict(value)


def sanitize(records: Iterable | None, source: str) -> list[dict]:
    """Keep only what occurs in `source` (rule 1 of the module docstring)."""
    clean_records = []
    for raw in records or []:
        raw = as_dict(raw)
        term = exact(source, raw.get("term", ""))
        if not term:
            continue
        record = {"term": term}
        for field in FIELDS:
            values = []
            seen = set()
            for item in raw.get(field, []) or []:
                item = as_dict(item)
                text = exact(source, item.get("text", ""))
                if not text or text.casefold() in seen:
                    continue
                citations = []
                for citation in item.get("citation", []) or []:
                    if visible := exact(source, citation):
                        citations.append(visible)
                values.append(
                    {"text": text, "citation": list(dict.fromkeys(citations))}
                )
                seen.add(text.casefold())
            record[field] = values
        if any(record[field] for field in FIELDS):
            clean_records.append(record)
    return clean_records


def merge(records: list[dict], require_definition: bool = True) -> list[dict]:
    """Records with the same term become one (rule 2 of the module docstring)."""
    merged: dict[str, dict] = {}
    order = []
    for record in records:
        term = clean(record.get("term", ""))
        key = term.casefold()
        if not key:
            continue
        if key not in merged:
            merged[key] = {"term": term, **{field: [] for field in FIELDS}}
            order.append(key)
        target = merged[key]
        for field in FIELDS:
            known = {item["text"].casefold(): item for item in target[field]}
            for item in record.get(field, []) or []:
                item_key = item["text"].casefold()
                if item_key in known:
                    known[item_key]["citation"] = list(
                        dict.fromkeys(
                            [*known[item_key]["citation"], *item.get("citation", [])]
                        )
                    )
                else:
                    target[field].append(item)
                    known[item_key] = item
    return [
        merged[key]
        for key in order
        if merged[key]["definitions"] or not require_definition
    ]


def candidate_key(term: str, text: str) -> tuple[str, str]:
    """When two definitions are the same candidate: the merge's rule (term and text, ignoring case)."""
    return clean(term).casefold(), text.casefold()


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def union(paper: dict, windows: list[dict]) -> tuple[dict, list[dict]]:
    """One paper's windows merged (rule 3), and the verifier's contexts.

    `paper` has `paper_id`, `title` and `text`; each window has `source_text`, `text_sha256`, `definition_records` and `raw_definition_records`, in reading order. Returns the union (`union.json`) and one context per window that holds a candidate: the window's text and the candidates found in it (`contexts`, which the verifier also calls on a saved union).
    """
    clean_records, raw_records = [], []
    for window in windows:
        clean_records.extend(copy.deepcopy(window["definition_records"]))
        raw_records.extend(copy.deepcopy(window["raw_definition_records"]))
    records = merge(clean_records)
    candidates = []
    for ri, record in enumerate(records):
        for di, item in enumerate(record["definitions"]):
            candidates.append(
                dict(
                    id=f"d{len(candidates):04d}",
                    term=record["term"],
                    definition=item["text"],
                    record_index=ri,
                    definition_index=di,
                )
            )
    merged = dict(
        paper_id=paper["paper_id"],
        title=paper["title"],
        source_sha256=sha_text(paper["text"]),
        records=records,
        raw_records=merge(raw_records, require_definition=False),
        candidates=candidates,
    )
    return merged, contexts(merged, windows)


def contexts(merged: dict, windows: list[dict]) -> list[dict]:
    """The verifier's contexts of a union: each window that holds a candidate, with its candidates.

    A candidate is in every window whose records hold its term and definition (`candidate_key`), so one candidate can have several contexts. Contexts are ordered by the sha256 of their text, and each lists its candidates in the union's order.
    """
    texts: dict[str, str] = {}
    origins: dict[tuple[str, str], set[str]] = {}
    for window in windows:
        digest = sha_text(window["source_text"])
        if digest != window["text_sha256"]:
            raise ValueError("Window text changed")
        texts[digest] = window["source_text"]
        for record in window["definition_records"]:
            for item in record["definitions"]:
                origins.setdefault(
                    candidate_key(record["term"], item["text"]), set()
                ).add(digest)
    grouped: dict[str, list[dict]] = {}
    for candidate in merged["candidates"]:
        key = candidate_key(candidate["term"], candidate["definition"])
        for digest in sorted(origins[key]):
            if exact(texts[digest], candidate["definition"]) is None:
                raise ValueError("Definition absent from original source context")
            grouped.setdefault(digest, []).append(
                {k: candidate[k] for k in ("id", "term", "definition")}
            )
    return [
        dict(
            paper_id=merged["paper_id"],
            sha256=digest,
            title=merged["title"],
            source_text=texts[digest],
            candidates=grouped[digest],
        )
        for digest in sorted(grouped)
    ]


# Predictions files: records out of the pipeline and into its scorer (module docstring)

PREDICTIONS = "predictions.jsonl"


def write_predictions(path: Path, papers: Iterable[tuple[str, list[dict]]]) -> dict:
    """Write each (paper id, records) as one line per record, in the given order; return the counts."""
    lines, counts = [], {"papers": 0, "records": 0, "definitions": 0}
    for paper_id, records in papers:
        counts["papers"] += bool(records)
        for record in records:
            line = {"paper_id": paper_id, "term": record["term"]}
            line.update({field: record.get(field, []) for field in FIELDS})
            lines.append(json.dumps(line, ensure_ascii=False) + "\n")
            counts["records"] += 1
            counts["definitions"] += len(record.get("definitions", []))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")
    return counts


def read_predictions(path: Path) -> dict[str, list[dict]]:
    """{paper id: records} of a predictions file: its lines grouped by paper, in the order the papers first appear, and each paper's records merged with `merge`. A line that is not a record stops the reading with its line number."""
    grouped: dict[str, list[dict]] = {}
    with Path(path).open(encoding="utf-8") as lines:
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                paper_id, record = predicted_record(json.loads(line))
            except ValueError as error:
                raise ValueError(f"{path}, line {number}: {error}") from None
            grouped.setdefault(paper_id, []).append(record)
    return {paper_id: merge(records) for paper_id, records in grouped.items()}


def predicted_record(line) -> tuple[str, dict]:
    """(paper id, record) of one line of a predictions file, with every item as {"text", "citation"}."""
    if not isinstance(line, dict):
        raise ValueError("a line must be a JSON object")
    paper_id, term = line.get("paper_id"), line.get("term")
    if not isinstance(paper_id, str) or not paper_id.strip():
        raise ValueError('"paper_id" must be a non-empty string')
    if not isinstance(term, str) or not term.strip():
        raise ValueError('"term" must be a non-empty string')
    record: dict = {"term": term}
    for field in FIELDS:
        items = line.get(field, [])
        if not isinstance(items, list):
            raise ValueError(f'"{field}" must be a list')
        record[field] = []
        for item in items:
            if isinstance(item, str):
                item = {"text": item}
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                raise ValueError(
                    f'each item of "{field}" must be a string or {{"text": ...}}'
                )
            if item["text"].strip():
                citations = item.get("citation") or []
                if not isinstance(citations, list):
                    raise ValueError(f'a "citation" in "{field}" must be a list')
                record[field].append(
                    {"text": item["text"], "citation": list(citations)}
                )
    return paper_id, record
