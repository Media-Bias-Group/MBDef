"""`collect crossref`: find the DOI of a queued paper that has none, with a Crossref search.

Each `missing` row without a DOI gets one bibliographic search (its title and up to three author surnames). A returned record counts only under the identity rule that `collect openalex` uses for a search result: the same compared title (a record's title, or its "title: subtitle"), a year the row names (issued, published, print, online or posted) and one shared author surname. The row takes a DOI only if exactly one DOI passes; two or more are `ambiguous`, none is `no_match`, and a row whose title (under 12 characters), year or authors are missing is `insufficient_evidence`. Writes `crossref_status.csv`; `collect semantic-scholar`, `collect openalex` and `collect open-access` read the DOIs it found (docs/step-1-2-papers/collect.md). The search and the rule are those of the released closure's collection.
"""

import os
import random
import time
from pathlib import Path
from typing import TypeIs

from curl_cffi import requests as curl_requests
from curl_cffi.requests.exceptions import RequestException

from mbdef.collect import ledger
from mbdef.collect.identity import (
    normalize_doi,
    queue_surnames,
    unicode_words,
)
from mbdef.collect.open_access import queue_rows
from mbdef.collect.openalex import _years as years_of

# Pinned defaults (those of the released closure's collection).
CROSSREF_URL = "https://api.crossref.org"
ROWS = (
    20  # records per search: the relevance-ranked page is the search's scope (`--rows`)
)
SELECT = "DOI,title,subtitle,author,issued,published,published-print,published-online,posted,type"
MIN_TITLE_CHARS = (
    12  # a shorter compared title is no identity evidence (`--min-title-chars`)
)
SLEEP_SECONDS = 0.2  # between requests: at most 5 a second (Crossref's polite pool)
RATE_FRACTION = (
    0.8  # of the rate Crossref advertises (X-Rate-Limit-Limit per -Interval)
)
TIMEOUT_SECONDS = 30.0
MAX_ATTEMPTS = 9  # per row, for network errors and HTTP 429 and 5xx answers
BACKOFF_SECONDS = (
    1.0  # the wait after a failed attempt, doubling, times a random 0.5 to 1.5
)
BACKOFF_CAP_SECONDS = 120.0
ERROR_ROUNDS = 3  # a row whose lookup failed this many times is not asked again
USER_AGENT = "mbdef/1.0 (citation closure; contact in the mailto parameter)"

STATUS = "crossref_status.csv"
STATUS_FIELDS = [
    "paperid",
    "title",
    "authors",
    "year",
    "first_author",
    "query",
    "cr_status",
    "reason",
    "doi",
    "matched_title",
    "matched_year",
    "matched_first_author",
    "matched_type",
    "n_results",
    "n_title",
    "n_title_year",
    "n_exact",
    "candidate_dois",
    "status_code",
    "error",
    "attempted_at",
]
SETTLED = {"matched", "no_match", "ambiguous", "insufficient_evidence"}
DATE_FIELDS = ("issued", "published", "published-print", "published-online", "posted")


def command(
    step_folder: str | None,
    collection: str | None,
    api_url: str,
    sleep_seconds: float,
    timeout_seconds: float,
    max_attempts: int,
    backoff_seconds: float,
    limit: int | None,
    rows: int = ROWS,
    min_title_chars: int = MIN_TITLE_CHARS,
) -> int:
    """The `collect crossref` command: search, print the counts, append the receipt."""
    mailto = contact()
    folder = ledger.folder(step_folder)
    queue_path = ledger.collection(collection, folder)
    status = folder / STATUS
    queue, skipped_review = queue_rows(queue_path)
    summary = search_all(
        queue,
        status,
        api_url=api_url,
        mailto=mailto,
        sleep_seconds=sleep_seconds,
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
        backoff_seconds=backoff_seconds,
        limit=limit,
        rows=rows,
        min_title_chars=min_title_chars,
    ) | {"skipped_review": skipped_review}
    settings = {
        "rows": rows,
        "min_title_chars": min_title_chars,
        "api_url": api_url,
        "mailto": "set" if mailto else "not set",
        "sleep_seconds": sleep_seconds,
        "timeout_seconds": timeout_seconds,
        "max_attempts": max_attempts,
        "backoff_seconds": backoff_seconds,
        "limit": limit,
    }
    outputs = {"status": status}
    ledger.append_receipt(
        folder,
        "collect crossref",
        settings,
        {"collection": queue_path},
        outputs,
        summary,
    )
    ledger.report(summary, outputs)
    return 0


def contact() -> str:
    """The address Crossref's polite pool asks for: CROSSREF_MAILTO, else UNPAYWALL_EMAIL (both optional)."""
    return (
        os.environ.get("CROSSREF_MAILTO", "").strip()
        or os.environ.get("UNPAYWALL_EMAIL", "").strip()
    )


# The identity rule -------------------------------------------------------------------------------


def first_surname(authors: str) -> str:
    """The first listed author's surname ("Graves, L.; Amazeen, M. A." -> "Graves"), as written."""
    for author in authors.split("; "):
        author = author.strip()
        if author:
            return (
                author.split(",", 1)[0] if "," in author else author.rsplit(" ", 1)[-1]
            )
    return ""


def record_years(item: dict) -> set[int]:
    years = set()
    for key in DATE_FIELDS:
        for parts in (item.get(key) or {}).get("date-parts") or []:
            if parts and isinstance(parts[0], int):
                years.add(parts[0])
    return years


def record_titles(item: dict) -> set[str]:
    """The compared forms of a record's titles: each title, and each "title: subtitle"."""
    titles = [t for t in item.get("title") or [] if isinstance(t, str)]
    subtitles = [s for s in item.get("subtitle") or [] if isinstance(s, str)]
    found = set()
    for title in titles:
        found.add(unicode_words(title))
        for subtitle in subtitles:
            found.add(unicode_words(f"{title}: {subtitle}"))
    return {text for text in found if text}


def record_surnames(item: dict) -> set[str]:
    return {
        name
        for author in item.get("author") or []
        if isinstance(author, dict)
        and (name := unicode_words(str(author.get("family") or "")))
    }


def match(row: dict, items: list[dict], min_title_chars: int = MIN_TITLE_CHARS) -> dict:
    """The rule's verdict on the records of one search, as ledger fields."""
    title = unicode_words(row["title"])
    years, surnames = years_of(row["year"]), queue_surnames(row["authors"])
    found: dict = {"n_results": len(items)}
    missing = [
        name
        for name, present in (
            ("title", len(title) >= min_title_chars),
            ("year", bool(years)),
            ("authors", bool(surnames)),
        )
        if not present
    ]
    if missing:
        return found | {
            "cr_status": "insufficient_evidence",
            "reason": "citation lacks " + "+".join(missing),
        }
    with_title = [item for item in items if title in record_titles(item)]
    with_year = [item for item in with_title if record_years(item) & years]
    exact = [
        item
        for item in with_year
        if record_surnames(item) & surnames and item.get("DOI")
    ]
    dois = sorted({doi for item in exact if (doi := normalize_doi(str(item["DOI"])))})
    found |= {
        "n_title": len(with_title),
        "n_title_year": len(with_year),
        "n_exact": len(exact),
        "candidate_dois": ";".join(dois),
    }
    if len(dois) == 1:
        item = next(i for i in exact if normalize_doi(str(i["DOI"])) == dois[0])
        first = next((a for a in item.get("author") or [] if isinstance(a, dict)), {})
        return found | {
            "cr_status": "matched",
            "reason": "exact title+year+author, unique DOI",
            "doi": dois[0],
            "matched_title": " ".join((item.get("title") or [""])[:1]),
            "matched_year": ";".join(map(str, sorted(record_years(item)))),
            "matched_first_author": str(first.get("family") or ""),
            "matched_type": str(item.get("type") or ""),
        }
    if len(dois) > 1:
        return found | {
            "cr_status": "ambiguous",
            "reason": f"{len(dois)} DOIs pass exact title+year+author",
        }
    reason = (
        "no exact title"
        if not with_title
        else (
            "title but no matching year"
            if not with_year
            else "title+year but no shared author surname"
        )
    )
    return found | {"cr_status": "no_match", "reason": reason}


# Requests ----------------------------------------------------------------------------------------


def backoff(attempt: int, retry_after: str | None, backoff_seconds: float) -> None:
    """Wait after failed attempt ATTEMPT (from 0): doubling from BACKOFF_SECONDS up to the cap, with jitter, and at least what Retry-After asks (up to five minutes)."""
    delay = min(BACKOFF_CAP_SECONDS, backoff_seconds * 2.0**attempt)
    delay *= random.uniform(0.5, 1.5)
    try:
        if retry_after is not None:
            delay = max(delay, min(300.0, float(retry_after)))
    except ValueError:
        pass
    if delay > 0:
        time.sleep(delay)


def follow_advertised_rate(pace: dict, headers, sleep_seconds: float) -> None:
    """Slow down to RATE_FRACTION of the rate Crossref advertises; after a 429 it only slows."""
    if pace["cooled"]:
        return
    try:
        limit = float(headers.get("x-rate-limit-limit"))
        interval = headers.get("x-rate-limit-interval") or "1s"
        seconds = float(interval.rstrip("s")) if interval.endswith("s") else 1.0
        rate = limit / seconds * RATE_FRACTION
        pace["sleep"] = min(5.0, max(sleep_seconds, 1.0 / rate))
    except TypeError, ValueError, ZeroDivisionError:
        pass


def scrub(text: str, secret: str) -> str:
    return text.replace(secret, "<redacted>") if len(secret) >= 4 else text


def search(
    row: dict,
    *,
    api_url: str,
    mailto: str,
    timeout_seconds: float,
    max_attempts: int,
    backoff_seconds: float,
    sleep_seconds: float,
    pace: dict,
    rows: int = ROWS,
    min_title_chars: int = MIN_TITLE_CHARS,
) -> dict:
    """The ledger row of one Crossref search for ROW (a queue row without a DOI)."""
    names = [first_surname(a) for a in row["authors"].split("; ") if a.strip()][:3]
    query = " ".join(part for part in (row["title"].strip(), *names) if part)
    params: dict[str, object] = {
        "query.bibliographic": query,
        "rows": rows,
        "select": SELECT,
    }
    if mailto:
        params["mailto"] = mailto
    base = {
        "paperid": row["paperid"],
        "title": row["title"],
        "authors": row["authors"],
        "year": row["year"],
        "first_author": first_surname(row["authors"]),
        "query": query,
        "attempted_at": ledger.utc_now(),
    }
    if not row["title"].strip():
        return base | {
            "cr_status": "insufficient_evidence",
            "reason": "citation lacks title",
            "n_results": 0,
        }
    last, code = "", ""
    for attempt in range(max_attempts):
        if pace["sleep"] > 0:
            time.sleep(pace["sleep"])
        try:
            response = curl_requests.get(
                f"{api_url.rstrip('/')}/works",
                params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=timeout_seconds,
            )
        except RequestException as exc:
            last = scrub(f"{type(exc).__name__}: {exc}", mailto)
            backoff(attempt, None, backoff_seconds)
            continue
        code = str(response.status_code)
        follow_advertised_rate(pace, response.headers, sleep_seconds)
        if response.status_code == 429 or response.status_code >= 500:
            last = f"HTTP Error {code}"
            if response.status_code == 429:  # half the pace for the rest of the run
                pace["cooled"] = True
                pace["sleep"] = min(5.0, max(pace["sleep"] * 2, sleep_seconds))
            backoff(attempt, response.headers.get("retry-after"), backoff_seconds)
            continue
        if response.status_code != 200:
            return base | {
                "cr_status": "error",
                "status_code": code,
                "error": f"HTTP Error {code}",
            }
        try:
            items = (response.json().get("message") or {}).get("items") or []
            if not isinstance(items, list):
                raise ValueError("Crossref items is not a list")
        except (ValueError, AttributeError) as exc:
            return base | {
                "cr_status": "error",
                "status_code": code,
                "error": scrub(f"malformed Crossref JSON: {exc}", mailto),
            }
        dicts = [item for item in items if isinstance(item, dict)]
        return base | match(row, dicts, min_title_chars) | {"status_code": code}
    return base | {
        "cr_status": "error",
        "status_code": code,
        "error": f"{last} after {max_attempts} attempts with backoff",
    }


# The ledger --------------------------------------------------------------------------------------


def current(status: dict | None, row: dict) -> TypeIs[dict]:
    """A ledger row answers only the queue row with the same citation evidence."""
    return bool(status) and all(
        status[k] == row[k] for k in ("title", "authors", "year")
    )


def read_status(path: Path) -> tuple[dict[str, dict], dict[str, int]]:
    """The latest ledger row of each paper id, and how many of its rows were errors."""
    latest: dict[str, dict] = {}
    errors: dict[str, int] = {}
    if path.exists():
        for row in ledger.read_csv(path, STATUS_FIELDS):
            latest[row["paperid"]] = row
            if row["cr_status"] == "error":
                errors[row["paperid"]] = errors.get(row["paperid"], 0) + 1
    return latest, errors


def search_all(
    queue: list[dict],
    status_path: Path,
    *,
    api_url: str = CROSSREF_URL,
    mailto: str = "",
    sleep_seconds: float = SLEEP_SECONDS,
    timeout_seconds: float = TIMEOUT_SECONDS,
    max_attempts: int = MAX_ATTEMPTS,
    backoff_seconds: float = BACKOFF_SECONDS,
    limit: int | None = None,
    rows: int = ROWS,
    min_title_chars: int = MIN_TITLE_CHARS,
) -> dict[str, int]:
    """Search every row without a DOI that has no settled answer yet; return the counts."""
    eligible = [row for row in queue if not row["doi"]]
    latest, errors = read_status(status_path)
    pending = [
        row
        for row in eligible
        if not (
            current(latest.get(row["paperid"]), row)
            and latest[row["paperid"]]["cr_status"] in SETTLED
        )
        and errors.get(row["paperid"], 0) < ERROR_ROUNDS
    ]
    if limit is not None:
        pending = pending[:limit]
    pace = {"sleep": sleep_seconds, "cooled": False}
    for row in pending:
        result = search(
            row,
            api_url=api_url,
            mailto=mailto,
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
            sleep_seconds=sleep_seconds,
            pace=pace,
            rows=rows,
            min_title_chars=min_title_chars,
        )
        ledger.append_status(status_path, STATUS_FIELDS, result)
    latest, _ = read_status(status_path)
    answered = [
        status for row in eligible if current(status := latest.get(row["paperid"]), row)
    ]
    by_status = {
        name: sum(s["cr_status"] == name for s in answered)
        for name in ("matched", "no_match", "ambiguous", "insufficient_evidence")
    }
    return {
        "eligible": len(eligible),
        **by_status,
        "errors": sum(s["cr_status"] == "error" for s in answered),
        "with_doi": len(queue) - len(eligible),
        "unattempted": len(eligible) - len(answered),
    }
