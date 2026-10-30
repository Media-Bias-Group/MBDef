"""Maintain and apply deterministic manual citation links.

`closure resolve` generates the YAML ledger. `closure manual` (`command`) validates completed rows and applies them to the dataset, aggregate report, and review CSV without network access.
"""

import csv
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml

from mbdef import files
from mbdef.closure import RESOLVER, resolver_fingerprint
from mbdef.closure.bibliography import parse_bibliography
from mbdef.closure.citations import reference_from_entry, year_check
from mbdef.closure.tei_sources import Sources
from mbdef.collect.ledger import csv_text
from mbdef.extract.records import FIELDS

MANUAL_LINK_FIELDS = (
    "paperid",
    "tei_file_path",
    "citation",
    "marker",
    "manual_link",
)
ELIGIBLE_METHODS = {"unresolved", "raw_reference_text"}


def _validate_manual_links(payload, path: Path) -> list[dict]:
    if payload is None:
        return []
    if not isinstance(payload, list):
        raise ValueError(f"manual-link ledger must contain a list: {path}")

    rows: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for position, row in enumerate(payload, 1):
        location = f"{path}: row {position}"
        if not isinstance(row, dict) or tuple(row) != MANUAL_LINK_FIELDS:
            raise ValueError(
                f"{location} must contain exactly {MANUAL_LINK_FIELDS} in that order"
            )
        for field in MANUAL_LINK_FIELDS[:-1]:
            if not isinstance(row[field], str):
                raise ValueError(f"{location} field {field!r} must be a string")
        links = row["manual_link"]
        if not isinstance(links, list) or any(
            not isinstance(link, str) for link in links
        ):
            raise ValueError(
                f"{location} field 'manual_link' must be a list of strings"
            )
        key = (row["paperid"], row["citation"], row["marker"])
        if key in seen:
            raise ValueError(f"{location} duplicates manual-link key {key!r}")
        seen.add(key)
        rows.append(row)
    return rows


def load_manual_links(path: Path) -> list[dict]:
    """Load and structurally validate a manual-link YAML ledger."""
    if not path.exists():
        return []
    return _validate_manual_links(
        yaml.safe_load(path.read_text(encoding="utf-8")), path
    )


def dump_manual_links(path: Path, rows: list[dict]) -> None:
    """Write validated manual-link rows as deterministic block-style YAML."""
    validated = _validate_manual_links(rows, path)
    rendered = yaml.safe_dump(
        validated,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered.rstrip("\n") + "\n", encoding="utf-8")


def _cache_file(paper_id: str, sources: Sources) -> Path:
    from mbdef.closure.tei_sources import cache_stem

    return sources.cache / f"{cache_stem(paper_id)}.tei.xml"


def root() -> Path:
    """The folder commands run from, the repository root: the ledger names TEI files relative to it."""
    return Path.cwd()


def _display_path(path: Path) -> str:
    absolute = path.resolve()
    try:
        return absolute.relative_to(root().resolve()).as_posix()
    except ValueError:
        return absolute.as_posix()


def merge_manual_links(
    unresolved: list[dict], sources: Sources, existing_path: Path
) -> list[dict]:
    """Deduplicate failures and preserve compatible completed ledger entries."""
    generated: dict[tuple[str, str, str], dict] = {}
    for failure in unresolved:
        paper_id = failure["paper_id"]
        citation = failure["citation"]
        marker = failure["marker"]
        if not all(isinstance(value, str) for value in (paper_id, citation, marker)):
            raise ValueError(
                f"unresolved citation requires string paper/citation/marker: {failure!r}"
            )
        cache_file = _cache_file(paper_id, sources)
        key = (paper_id, citation, marker)
        generated[key] = {
            "paperid": paper_id,
            "tei_file_path": _display_path(cache_file),
            "citation": citation,
            "marker": marker,
            "manual_link": [],
        }

    previous = load_manual_links(existing_path)
    current_keys = set(generated)
    for row in previous:
        key = (row["paperid"], row["citation"], row["marker"])
        current = generated.get(key)
        path_matches = current and current["tei_file_path"] == row["tei_file_path"]
        if path_matches:
            current["manual_link"] = list(row["manual_link"])
        elif row["manual_link"]:
            reason = "TEI path changed" if key in current_keys else "citation is stale"
            raise ValueError(f"cannot discard completed manual link {key!r}: {reason}")

    return [generated[key] for key in sorted(generated)]


def _ledger_key(row: dict) -> tuple[str, str, str]:
    return row["paperid"], row["citation"], row["marker"]


def _tei_path(row: dict) -> Path:
    path = Path(row["tei_file_path"])
    return path if path.is_absolute() else root() / path


def _url_reference(url: str) -> dict:
    parts = urlsplit(url)
    doi = None
    if parts.hostname and parts.hostname.casefold() == "doi.org":
        doi = unquote(parts.path).strip("/").lower() or None
    return {
        "bibl_id": "manual:url:" + sha256(url.encode("utf-8")).hexdigest()[:16],
        "authors": [],
        "year": None,
        "years_in_text": [],
        "title": None,
        "container": None,
        "volume": None,
        "issue": None,
        "pages": None,
        "publisher": None,
        "pub_place": None,
        "doi": doi,
        "url": url,
        "formatted": url,
        "raw": url,
    }


def _validate_links(row: dict, entries: dict) -> list[dict]:
    references: list[dict] = []
    seen: set[str] = set()
    key = _ledger_key(row)
    for link in row["manual_link"]:
        if not link:
            raise ValueError(f"manual link for {key!r} cannot be empty")
        if link in seen:
            raise ValueError(f"manual link for {key!r} is duplicated: {link!r}")
        seen.add(link)
        if link.startswith("#"):
            bibl_id = link[1:]
            if not bibl_id or bibl_id not in entries:
                raise ValueError(
                    f"manual link for {key!r} has unknown TEI target {link!r}"
                )
            references.append(reference_from_entry(entries[bibl_id]))
            continue
        parts = urlsplit(link)
        if parts.scheme.casefold() not in {"http", "https"} or not parts.netloc:
            raise ValueError(
                f"manual link for {key!r} is not an absolute HTTP URL: {link!r}"
            )
        references.append(_url_reference(link))
    return references


def _paper_index(dataset: dict) -> dict[str, dict]:
    papers = dataset.get("papers")
    if not isinstance(papers, list):
        raise ValueError("citation dataset must contain a papers list")
    indexed: dict[str, dict] = {}
    for paper in papers:
        paper_id = paper.get("paper_id") if isinstance(paper, dict) else None
        if not isinstance(paper_id, str) or paper_id in indexed:
            raise ValueError(
                f"citation dataset has invalid or duplicate paper_id: {paper_id!r}"
            )
        indexed[paper_id] = paper
    return indexed


def _matching_rows(paper: dict, entry: dict) -> list[dict]:
    matches: list[dict] = []
    for record in paper.get("definition_records", []):
        for field in FIELDS:
            for item in record.get(field, []):
                for row in item.get("resolved_citations", []):
                    if (
                        row.get("raw") == entry["citation"]
                        and row.get("marker") == entry["marker"]
                    ):
                        matches.append(row)
    return matches


def _expected_review_fields(
    references: list[dict], checked_year: str
) -> dict[str, str]:
    return {
        "method": "manual_link",
        "confidence": "high",
        "reason": "",
        "year_check": checked_year,
        "bibl_ids": " ".join(reference["bibl_id"] for reference in references),
        "reference": references[0]["formatted"],
        "doi": next(
            (reference["doi"] for reference in references if reference["doi"]), ""
        ),
        "year": next(
            (reference["year"] for reference in references if reference["year"]), ""
        ),
    }


def _move_counter(report: dict, name: str, old: str | None, new: str | None) -> None:
    values = report.get(name)
    if not isinstance(values, dict):
        raise ValueError(f"citation report is missing counter {name!r}")
    if old == new:
        return
    if old is not None:
        count = values.get(old, 0)
        if not isinstance(count, int) or count < 1:
            raise ValueError(
                f"citation report counter {name}.{old} cannot be decremented"
            )
        if count == 1:
            del values[old]
        else:
            values[old] = count - 1
    if new is not None:
        count = values.get(new, 0)
        if not isinstance(count, int):
            raise ValueError(f"citation report counter {name}.{new} is not numeric")
        values[new] = count + 1


def _sort_report_counters(report: dict) -> None:
    for name in ("by_method", "by_confidence", "by_year_check", "by_reason"):
        report[name] = dict(
            sorted(report[name].items(), key=lambda item: (-item[1], item[0]))
        )


def resolve_manual_citations(
    manual_links_path: Path,
    dataset_path: Path,
    report_path: Path,
    review_path: Path,
) -> dict[str, int]:
    """Validate completed ledger entries and atomically stage synchronized outputs."""
    ledger = load_manual_links(manual_links_path)
    filled = [row for row in ledger if row["manual_link"]]
    if not filled:
        return {
            "manual_link_entries": 0,
            "manually_linked_markers": 0,
            "changed_markers": 0,
        }

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("resolver_sha256") != resolver_fingerprint():
        raise ValueError(
            "Citation resolver code changed; regenerate stale outputs before applying manual links"
        )
    with review_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        review_fieldnames = reader.fieldnames
        review = list(reader)
    if not review_fieldnames:
        raise ValueError(f"citation review has no header: {review_path}")
    required_review = {
        "paper_id",
        "citation",
        "marker",
        "method",
        "confidence",
        "reason",
        "year_check",
        "bibl_ids",
        "reference",
        "doi",
        "year",
    }
    if not required_review.issubset(review_fieldnames):
        raise ValueError(f"citation review is missing required columns: {review_path}")

    papers = _paper_index(dataset)
    parsed_by_path: dict[Path, dict] = {}
    path_by_paper: dict[str, Path] = {}
    actions: list[dict] = []
    for entry in filled:
        key = _ledger_key(entry)
        paper = papers.get(entry["paperid"])
        if paper is None:
            raise ValueError(f"manual-link key has no dataset paper: {key!r}")
        if (paper.get("tei") or {}).get("kind") == "missing":
            raise ValueError(
                f"manual-link TEI disagrees with missing dataset TEI: {key!r}"
            )

        tei_path = _tei_path(entry).resolve()
        from mbdef.closure.tei_sources import cache_stem

        expected_name = f"{cache_stem(entry['paperid'])}.tei.xml"
        if tei_path.name != expected_name:
            raise ValueError(
                f"manual-link TEI path for {entry['paperid']} must end in {expected_name}: {tei_path}"
            )
        previous_path = path_by_paper.setdefault(entry["paperid"], tei_path)
        if previous_path != tei_path:
            raise ValueError(
                f"manual-link ledger has conflicting TEI paths for {entry['paperid']}"
            )
        if not tei_path.is_file():
            raise FileNotFoundError(
                f"manual-link TEI missing for paper {entry['paperid']}: {tei_path}"
            )
        if tei_path not in parsed_by_path:
            parsed_by_path[tei_path] = parse_bibliography(tei_path.read_bytes())
        references = _validate_links(entry, parsed_by_path[tei_path]["entries"])

        matches = _matching_rows(paper, entry)
        if not matches:
            raise ValueError(
                f"completed manual-link key matches no dataset row: {key!r}"
            )
        checked_years = []
        states = []
        source_states = []
        for row in matches:
            candidate = dict(row)
            candidate["references"] = references
            checked_year = year_check(candidate)
            checked_years.append(checked_year)
            source_states.append(
                (
                    row.get("method"),
                    row.get("confidence"),
                    row.get("reason"),
                    row.get("year_check"),
                )
            )
            is_exact_manual = (
                row.get("method") == "manual_link"
                and row.get("confidence") == "high"
                and row.get("reason") is None
                and row.get("references") == references
                and row.get("year_check") == checked_year
            )
            is_eligible = row.get("method") in ELIGIBLE_METHODS and not row.get(
                "references"
            )
            if not is_exact_manual and not is_eligible:
                raise ValueError(
                    f"manual-link key conflicts with existing method {row.get('method')!r}: {key!r}"
                )
            states.append("manual" if is_exact_manual else "eligible")
        if len(set(states)) != 1:
            raise ValueError(
                f"manual-link key has a mixture of conflicting states: {key!r}"
            )
        if states[0] == "eligible" and len(set(source_states)) != 1:
            raise ValueError(
                f"manual-link key has inconsistent unresolved states: {key!r}"
            )
        if len(set(checked_years)) != 1:
            raise ValueError(
                f"manual-link key produces inconsistent year checks: {key!r}"
            )

        review_matches = [
            row
            for row in review
            if row["paper_id"] == entry["paperid"]
            and row["citation"] == entry["citation"]
            and row["marker"] == entry["marker"]
        ]
        if len(review_matches) != len(matches):
            raise ValueError(
                f"manual-link key matches {len(matches)} dataset rows but {len(review_matches)} review rows: {key!r}"
            )
        expected_review = _expected_review_fields(references, checked_years[0])
        if states[0] == "manual":
            for row in review_matches:
                if any(row[field] != value for field, value in expected_review.items()):
                    raise ValueError(
                        f"manual-link review row is stale for key: {key!r}"
                    )
        else:
            dataset_states = Counter(
                (
                    row["method"],
                    row["confidence"],
                    row["reason"] or "",
                    row["year_check"],
                )
                for row in matches
            )
            review_states = Counter(
                (row["method"], row["confidence"], row["reason"], row["year_check"])
                for row in review_matches
            )
            if dataset_states != review_states:
                raise ValueError(
                    f"manual-link review state disagrees with dataset: {key!r}"
                )
        actions.append(
            {
                "entry": entry,
                "references": references,
                "rows": matches,
                "review_rows": review_matches,
                "year_check": checked_years[0],
                "state": states[0],
                "expected_review": expected_review,
            }
        )

    existing = [action for action in actions if action["state"] == "manual"]
    changed = [action for action in actions if action["state"] == "eligible"]
    existing_markers = sum(len(action["rows"]) for action in existing)
    if existing:
        if report.get("resolver") != RESOLVER + "+manual":
            raise ValueError(
                "citation report resolver disagrees with existing manual rows"
            )
        if report.get("manual_link_entries") != len(existing):
            raise ValueError(
                "citation report manual_link_entries disagrees with dataset"
            )
        if report.get("manually_linked_markers") != existing_markers:
            raise ValueError(
                "citation report manually_linked_markers disagrees with dataset"
            )
    elif report.get("resolver") != RESOLVER:
        raise ValueError(
            f"manual resolver requires a {RESOLVER} report; regenerate stale outputs"
        )

    if not changed:
        return {
            "manual_link_entries": len(filled),
            "manually_linked_markers": existing_markers,
            "changed_markers": 0,
        }

    detail = {
        row.get("paper_id"): row
        for row in report.get("papers_detail", [])
        if isinstance(row, dict)
    }
    for action in changed:
        references = action["references"]
        checked_year = action["year_check"]
        paper_id = action["entry"]["paperid"]
        paper_detail = detail.get(paper_id)
        if paper_detail is None:
            raise ValueError(f"citation report has no paper detail for {paper_id}")
        for row in action["rows"]:
            old_method = row["method"]
            old_confidence = row["confidence"]
            old_year_check = row["year_check"]
            old_reason = row["reason"]
            _move_counter(report, "by_method", old_method, "manual_link")
            _move_counter(report, "by_confidence", old_confidence, "high")
            _move_counter(report, "by_year_check", old_year_check, checked_year)
            if old_reason:
                _move_counter(report, "by_reason", old_reason, None)
            row.update(
                method="manual_link",
                confidence="high",
                reason=None,
                year_check=checked_year,
                references=references,
            )
            report["linked_markers"] += 1
            report["bibl_links"] += len(references)
            paper_detail["linked_markers"] += 1
            paper_detail["unresolved_markers"] -= 1
            if paper_detail["unresolved_markers"] < 0:
                raise ValueError(
                    f"citation report unresolved count became negative for {paper_id}"
                )
        for review_row in action["review_rows"]:
            review_row.update(action["expected_review"])

    manually_linked_markers = sum(len(action["rows"]) for action in actions)
    report["resolver"] = RESOLVER + "+manual"
    report["manual_link_entries"] = len(filled)
    report["manually_linked_markers"] = manually_linked_markers
    _sort_report_counters(report)
    files.write_atomic(
        {
            dataset_path: files.json_text(dataset),
            report_path: files.json_text(report),
            review_path: csv_text(review, list(review_fieldnames)),
        }
    )
    return {
        "manual_link_entries": len(filled),
        "manually_linked_markers": manually_linked_markers,
        "changed_markers": sum(len(action["rows"]) for action in changed),
    }


def command(manual_links: str, dataset: str, report: str, review: str) -> int:
    """`closure manual`: apply the completed manual links of the ledger to the files of `closure resolve` (rewritten in place, all or none), and print what changed."""
    result = resolve_manual_citations(
        Path(manual_links), Path(dataset), Path(report), Path(review)
    )
    print(json.dumps(result, indent=2))
    receipt = files.receipt(
        "mbdef closure manual",
        {"resolver": RESOLVER},
        {
            "manual_links": Path(manual_links),
            "dataset": Path(dataset),
            "report": Path(report),
            "review": Path(review),
        },
        {},
    )
    receipt["summary"] = result
    files.write_json(Path(report).with_name("manual-receipt.json"), receipt)
    return 0
