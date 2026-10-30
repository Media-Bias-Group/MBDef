"""`collect open-access`: find an open-access PDF for each queued paper with Unpaywall.

For every row of the queue marked `missing`, ask Unpaywall about its DOI and keep the best open-access PDF location; a row without a DOI keeps the web address the reference gave, if any. One row per attempt goes to `open_access_status.csv`; the current downloadable rows go to `open_access_candidates.csv`, which `collect download` reads.

A row without a DOI for which `collect crossref` or `collect semantic-scholar` found one is asked about that DOI, as in the released closure's collection.
"""

import hashlib
import json
import time
from pathlib import Path
from urllib.parse import quote, urlparse

from curl_cffi import requests as curl_requests
from curl_cffi.requests.exceptions import RequestException

from mbdef import files
from mbdef.collect import ledger
from mbdef.collect.identity import (
    collection_paper_id,
    is_arxiv_doi,
    normalize_doi,
)

# Pinned defaults.
UNPAYWALL_URL = "https://api.unpaywall.org/v2"
SLEEP_SECONDS = 1.0  # after each Unpaywall request
TIMEOUT_SECONDS = 20.0

STATUS = "open_access_status.csv"
CANDIDATES = "open_access_candidates.csv"
STATUS_FIELDS = [
    "paperid",
    "queue_status",
    "title",
    "authors",
    "year",
    "doi",
    "source_url",
    "pdf_url",
    "oa_status",
    "oa_host_type",
    "status_code",
    "error",
    "attempted_at",
]
TERMINAL_STATUSES = {
    "has_pdf",
    "not_open_access",
    "open_access_but_no_pdf",
    "not_found",
    "provided_url",
    "no_download_source",
}


def command(
    step_folder: str | None,
    collection: str | None,
    api_url: str,
    sleep_seconds: float,
    timeout_seconds: float,
    limit: int | None,
    reference_urls: bool = False,
) -> int:
    """The `collect open-access` command: resolve, print the counts, append the receipt."""
    email = files.env("UNPAYWALL_EMAIL")
    folder = ledger.folder(step_folder)
    queue = ledger.collection(collection, folder)
    status, candidates = folder / STATUS, folder / CANDIDATES
    # Imported here because both build on this module.
    from mbdef.collect import openalex
    from mbdef.collect.semantic_scholar import found_dois

    rows, skipped_review = queue_rows(queue)
    summary = resolve_open_access_pdfs(
        queries=openalex.unpaywall_queries(
            rows, folder, found_dois(rows, folder), reference_urls
        ),
        skipped_review=skipped_review,
        candidates_path=candidates,
        status_path=status,
        email=email,
        sleep_seconds=sleep_seconds,
        timeout_seconds=timeout_seconds,
        limit=limit,
        api_url=api_url,
    )
    settings = {
        "api_url": api_url,
        "sleep_seconds": sleep_seconds,
        "timeout_seconds": timeout_seconds,
        "limit": limit,
        "reference_urls": reference_urls,
    }
    outputs = {"status": status, "candidates": candidates}
    ledger.append_receipt(
        folder, "collect open-access", settings, {"collection": queue}, outputs, summary
    )
    ledger.report(summary, outputs)
    return 0


def _object(value: object, field: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Malformed provider response: {field} must be an object")
    return value


def _objects(value: object, field: str) -> list[dict]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"Malformed provider response: {field} must be an object list")
    return value


def _fields(payload: dict, **types: type) -> None:
    for field, expected in types.items():
        value = payload.get(field)
        if value is not None and type(value) is not expected:
            raise ValueError(
                f"Malformed provider response: {field} must be {expected.__name__}"
            )


def _best_pdf_url(payload: dict) -> tuple[str, str]:
    _fields(payload, is_oa=bool)
    best = _object(payload.get("best_oa_location"), "best_oa_location")
    locations = _objects(payload.get("oa_locations"), "oa_locations")
    for location in [best, *locations]:
        _fields(location, url_for_pdf=str, host_type=str)
    if not payload.get("is_oa"):
        return "", "not_open_access"

    if best.get("url_for_pdf"):
        return str(best["url_for_pdf"]), str(
            best.get("host_type") or "best_oa_location"
        )

    for location in locations:
        if location.get("url_for_pdf"):
            return str(location["url_for_pdf"]), str(
                location.get("host_type") or "oa_location"
            )
    return "", "open_access_but_no_pdf"


def normalized_queue_doi(value: str) -> str:
    dois = {
        normalized for item in value.split("; ") if (normalized := normalize_doi(item))
    }
    if not dois:
        return ""
    preferred = sorted(doi for doi in dois if not is_arxiv_doi(doi))
    return (preferred or sorted(dois))[0]


def _absolute_http_url(value: str) -> str:
    value = value.strip()
    parsed = urlparse(value)
    return value if parsed.scheme in {"http", "https"} and parsed.netloc else ""


def queue_rows(collection_path: Path) -> tuple[list[dict[str, str]], int]:
    """The queue's `missing` rows as lookups with a stable paper id, and how many rows were skipped."""
    collection = ledger.read_csv(collection_path, ledger.COLLECTION_FIELDS)
    queries: dict[str, dict[str, str]] = {}
    skipped_review = 0
    for source in collection:
        if source["queue_status"] != "missing":
            skipped_review += 1
            continue
        doi = normalized_queue_doi(source["doi"])
        paperid = collection_paper_id(
            doi=doi,
            title=source["title"],
            year=source["year"],
            authors=source["authors"],
        )
        row = {
            "paperid": paperid,
            "queue_status": source["queue_status"],
            "title": source["title"],
            "authors": source["authors"],
            "year": source["year"],
            "doi": doi,
            "source_url": source["url"].strip(),
        }
        # Membership may deliberately retain ambiguous groups. Identical provider queries can share an attempt without merging their membership evidence.
        key = json.dumps(row, sort_keys=True, ensure_ascii=False)
        # Include query evidence unconditionally: IDs must stay stable when other rows disappear from the missing queue after successful acquisition.
        row["paperid"] += "__" + hashlib.sha256(key.encode()).hexdigest()[:16]
        queries[key] = row
    return list(queries.values()), skipped_review


def _read_status(path: Path) -> list[dict[str, str]]:
    return ledger.read_csv(path, STATUS_FIELDS) if path.exists() else []


def latest_status_by_id(path: Path) -> dict[str, dict[str, str]]:
    frame = _read_status(path)
    if not frame:
        return {}
    return {row["paperid"]: row for row in ledger.latest_by_paperid(frame)}


def _matches_current(status: dict[str, str], row: dict[str, str]) -> bool:
    return status["doi"] == row["doi"] and status["source_url"] == row["source_url"]


def _status_row(row: dict[str, str]) -> dict[str, object]:
    return {
        **row,
        "pdf_url": "",
        "oa_status": "",
        "oa_host_type": "",
        "status_code": "",
        "error": "",
        "attempted_at": ledger.utc_now(),
    }


def _resolve_one(
    row: dict[str, str], *, email: str, timeout_seconds: float, api_url: str
) -> dict[str, object]:
    result = _status_row(row)
    provided_url = _absolute_http_url(row["source_url"])
    if not row["doi"]:
        if provided_url:
            result.update(pdf_url=provided_url, oa_status="provided_url")
        else:
            result["oa_status"] = "no_download_source"
        return result

    try:
        response = curl_requests.get(
            f"{api_url.rstrip('/')}/{quote(row['doi'], safe='')}",
            params={"email": email},
            timeout=timeout_seconds,
        )
        result["status_code"] = response.status_code
        if response.status_code == 404:
            provider_status = "not_found"
            pdf_url = ""
            host_type = ""
        else:
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("Unpaywall returned a non-object JSON payload")
            pdf_url, host_type = _best_pdf_url(payload)
            provider_status = "has_pdf" if pdf_url else host_type

        if pdf_url:
            result.update(
                pdf_url=pdf_url,
                oa_status="has_pdf",
                oa_host_type=host_type,
            )
        elif provided_url:
            result.update(
                pdf_url=provided_url,
                oa_status="provided_url",
                oa_host_type=provider_status,
            )
        else:
            result["oa_status"] = provider_status
    except (RequestException, ValueError, TypeError) as exc:
        result.update(oa_status="error", error=str(exc))
    return result


def _write_candidates(
    *,
    candidates_path: Path,
    queue: list[dict[str, str]],
    latest: dict[str, dict[str, str]],
) -> None:
    candidates = []
    for row in queue:
        status = latest.get(row["paperid"])
        if (
            status
            and _matches_current(status, row)
            and status["oa_status"] in {"has_pdf", "provided_url"}
            and status["pdf_url"]
        ):
            candidates.append({name: status[name] for name in STATUS_FIELDS})
    ledger.replace_csv(candidates_path, candidates, STATUS_FIELDS)


def resolve_open_access_pdfs(
    *,
    queries: list[dict[str, str]],
    candidates_path: str | Path,
    status_path: str | Path,
    email: str,
    skipped_review: int = 0,
    sleep_seconds: float = SLEEP_SECONDS,
    timeout_seconds: float = TIMEOUT_SECONDS,
    limit: int | None = None,
    api_url: str = UNPAYWALL_URL,
) -> dict[str, int]:
    """Ask Unpaywall about each of QUERIES (queue rows, each with the DOI to ask about) that has no settled answer, and rewrite the candidates."""
    candidates_path = Path(candidates_path)
    status_path = Path(status_path)
    queue = queries
    latest = latest_status_by_id(status_path)

    pending = [
        row
        for row in queue
        if not (
            (status := latest.get(row["paperid"]))
            and _matches_current(status, row)
            and status["oa_status"] in TERMINAL_STATUSES
        )
    ]
    if limit is not None:
        pending = pending[:limit]

    for row in pending:
        result = _resolve_one(
            row,
            email=email,
            timeout_seconds=timeout_seconds,
            api_url=api_url,
        )
        ledger.append_status(status_path, STATUS_FIELDS, result)
        latest[row["paperid"]] = {
            name: str(result.get(name, "")) for name in STATUS_FIELDS
        }
        if row["doi"] and sleep_seconds > 0:
            time.sleep(sleep_seconds)

    latest = latest_status_by_id(status_path)
    _write_candidates(candidates_path=candidates_path, queue=queue, latest=latest)

    current = [
        status
        for row in queue
        if (status := latest.get(row["paperid"])) and _matches_current(status, row)
    ]
    return {
        "eligible": len(queue),
        "resolved": sum(
            row["oa_status"] in {"has_pdf", "provided_url"} for row in current
        ),
        "unresolved": sum(
            row["oa_status"] in TERMINAL_STATUSES
            and row["oa_status"] not in {"has_pdf", "provided_url"}
            for row in current
        ),
        "errors": sum(row["oa_status"] == "error" for row in current),
        "skipped_review": skipped_review,
    }
