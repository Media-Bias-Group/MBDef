"""`collect openalex`: find PDFs of the queued papers with OpenAlex, by DOI or by title search.

A row with a DOI is looked up by that DOI. A row without one is searched by its title (filtered to its year); a result counts only if title, year and one author surname agree exactly, and a search with two such results, or with more results than one page holds, is `ambiguous` and selects nothing. Every PDF location of the matched work is kept, repository copies first, so `collect download` can try the others when the first fails. Writes `openalex_status.csv` and `openalex_candidates.csv`.

A row without a DOI for which `collect crossref` or `collect semantic-scholar` found one is looked up by that DOI instead, and recorded under the original row (`query_doi` stays empty; `matched_by` says who found the DOI, for example `crossref_doi`). If OpenAlex lacks that DOI the row is `not_found` (`crossref_doi_not_in_openalex`) with the DOI kept for Unpaywall, and is not searched by its title: the search already found the DOI. This is the chaining the released closure's collection used.
"""

import json
import os
import re
import time
import unicodedata
from pathlib import Path
from urllib.parse import quote, urlparse

from curl_cffi import requests as curl_requests
from curl_cffi.requests.exceptions import RequestException

from mbdef.collect import ledger
from mbdef.collect.identity import (
    normalize_doi,
    queue_surnames,
    unicode_words,
)
from mbdef.collect.open_access import STATUS_FIELDS as BASE_FIELDS
from mbdef.collect.open_access import _fields, _object, _objects, queue_rows

# Pinned defaults.
OPENALEX_URL = "https://api.openalex.org"
SLEEP_SECONDS = 0.1  # after every request; retries back off exponentially from it
TIMEOUT_SECONDS = 20.0
MAX_RETRIES = 3  # for HTTP 429 and 5xx answers

STATUS = "openalex_status.csv"
CANDIDATES = "openalex_candidates.csv"
STATUS_FIELDS = BASE_FIELDS + [
    "alternate_pdf_urls",
    "query_doi",
    "openalex_id",
    "matched_by",
    "matched_title",
    "matched_year",
]
TERMINAL_STATUSES = {
    "has_pdf",
    "not_open_access",
    "open_access_but_no_pdf",
    "not_found",
    "ambiguous",
}
SELECT_FIELDS = ",".join(
    (
        "id",
        "doi",
        "title",
        "publication_year",
        "authorships",
        "open_access",
        "best_oa_location",
        "locations",
    )
)
YEAR = re.compile(r"(?<!\d)(?:18|19|20)\d{2}(?!\d)")
VERSION_RANK = {"publishedVersion": 0, "acceptedVersion": 1, "submittedVersion": 2}
# `search` is a Lucene query: stray quotes, wildcards and operators in a plain bibliographic title are rejected by OpenAlex with HTTP 400.
LUCENE_SYNTAX = re.compile(r'["?*~^:\\/(){}\[\]+\-|!]')
# The title search for a row without a DOI: two rules, each named by its `--title-rule` value.
# `2026-09-28`: a full-text `search` filtered to the row's years (`matched_by` "title"). `2026-09-29`:
# OpenAlex's title filter, years one around the row's; a record whose first author is not among the cited authors is ignored (a book review lists its reviewer first); a record with the row's year counts, and a year one off only when none has it (`title_filter`, `title_filter_year_off_by_one`).
# Several records that count, or more records than one page shows, make the row ambiguous.
TITLE_RULES = ("2026-09-28", "2026-09-29")
TITLE_RULE = "2026-09-29"  # the rule the released closure found its papers with (7,817 of its rows)
TITLE_FILTER = "title_filter"
TITLE_FILTER_OFF_BY_ONE = "title_filter_year_off_by_one"


def command(
    step_folder: str | None,
    collection: str | None,
    api_url: str,
    sleep_seconds: float,
    timeout_seconds: float,
    max_retries: int,
    limit: int | None,
    title_rule: str = TITLE_RULE,
) -> int:
    """The `collect openalex` command: resolve, print the counts, append the receipt."""
    api_key = os.environ.get(
        "OPENALEX_API_KEY", ""
    ).strip()  # optional: a larger daily budget
    folder = ledger.folder(step_folder)
    queue = ledger.collection(collection, folder)
    status, candidates = folder / STATUS, folder / CANDIDATES
    # Imported here because `semantic_scholar` builds on this module.
    from mbdef.collect.semantic_scholar import found_dois, pdf_pending

    rows = queue_rows(queue)[0]
    summary = resolve_openalex_pdfs(
        collection_path=queue,
        candidates_path=candidates,
        status_path=status,
        found=found_dois(rows, folder),
        waiting=pdf_pending(rows, folder),
        api_key=api_key,
        sleep_seconds=sleep_seconds,
        timeout_seconds=timeout_seconds,
        limit=limit,
        max_retries=max_retries,
        api_url=api_url,
        title_rule=title_rule,
    )
    settings = {
        "api_url": api_url,
        "api_key": "set" if api_key else "not set",
        "sleep_seconds": sleep_seconds,
        "timeout_seconds": timeout_seconds,
        "max_retries": max_retries,
        "limit": limit,
        "title_rule": title_rule,
    }
    outputs = {"status": status, "candidates": candidates}
    ledger.append_receipt(
        folder, "collect openalex", settings, {"collection": queue}, outputs, summary
    )
    ledger.report(summary, outputs)
    if summary["rate_limited"]:
        raise SystemExit(
            "OpenAlex rate limit exhausted; progress is saved. "
            "Set OPENALEX_API_KEY or rerun later to resume."
        )
    return 0


def _absolute_http_url(value: object) -> str:
    text = str(value or "").strip()
    parsed = urlparse(text)
    return text if parsed.scheme in {"http", "https"} and parsed.netloc else ""


def _work_surnames(work: dict) -> set[str]:
    surnames = set()
    for authorship in work.get("authorships") or []:
        author = authorship.get("author") or {}
        name = str(author.get("display_name") or "").strip()
        if name and (normalized := unicode_words(name.rsplit(" ", 1)[-1])):
            surnames.add(normalized)
    return surnames


def _validate_work(work: dict) -> None:
    """Validate consumed JSON fields so schema errors remain retryable row failures."""
    _fields(work, id=str, doi=str, title=str, publication_year=int)
    for authorship in _objects(work.get("authorships"), "authorships"):
        _fields(_object(authorship.get("author"), "author"), display_name=str)
    _fields(_object(work.get("open_access"), "open_access"), is_oa=bool)
    locations = _objects(work.get("locations"), "locations")
    best = _object(work.get("best_oa_location"), "best_oa_location")
    for location in [best, *locations]:
        _fields(location, pdf_url=str, version=str, is_oa=bool)
        _fields(_object(location.get("source"), "source"), type=str)


def _years(value: str) -> set[int]:
    return {int(match.group()) for match in YEAR.finditer(value)}


def matches_metadata(row: dict[str, str], work: dict) -> bool:
    """The identity rule for a search result: same DOI (if the row has one), the same compared title, a year the row names, and at least one shared author surname."""
    if row["doi"] and normalize_doi(str(work.get("doi") or "")) != row["doi"]:
        return False
    if unicode_words(row["title"]) != unicode_words(str(work.get("title") or "")):
        return False
    expected_years = _years(row["year"])
    work_year = work.get("publication_year")
    if not expected_years or work_year not in expected_years:
        return False
    expected_surnames = queue_surnames(row["authors"])
    if not expected_surnames or not expected_surnames.intersection(
        _work_surnames(work)
    ):
        return False
    return True


def _matching_search_results(row: dict[str, str], payload: dict) -> list[dict]:
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("OpenAlex search returned no results list")
    for work in _objects(results, "results"):
        _validate_work(work)
    return [
        work
        for work in results
        if isinstance(work, dict) and matches_metadata(row, work)
    ]


def _location_rank(location: dict) -> tuple[int, int, int]:
    source = location.get("source") or {}
    source_type = str(source.get("type") or "")
    host_rank = 0 if source_type == "repository" else 1 if source_type else 2
    return (
        host_rank,
        VERSION_RANK.get(str(location.get("version") or ""), 3),
        0 if location.get("is_oa") else 1,
    )


def _pdf_locations(work: dict) -> list[tuple[str, str]]:
    locations = [
        location
        for location in work.get("locations") or []
        if isinstance(location, dict) and _absolute_http_url(location.get("pdf_url"))
    ]
    best = work.get("best_oa_location")
    if isinstance(best, dict) and _absolute_http_url(best.get("pdf_url")):
        locations.append(best)
    locations.sort(key=_location_rank)

    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for location in locations:
        url = _absolute_http_url(location.get("pdf_url"))
        if not url or url in seen:
            continue
        seen.add(url)
        source = location.get("source") or {}
        host_type = str(source.get("type") or "openalex_location")
        found.append((url, host_type))
    return found


class RateLimitExceeded(RuntimeError):
    """OpenAlex rejected the request with 429 after exhausting retries."""


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _search_text(title: str) -> str:
    """Strip Lucene query syntax so an ordinary title never becomes a malformed query."""
    return re.sub(r"\s+", " ", LUCENE_SYNTAX.sub(" ", title)).strip()


def get_json(
    url: str,
    *,
    params: dict[str, object],
    api_key: str,
    timeout_seconds: float,
    sleep_seconds: float,
    max_retries: int,
) -> tuple[int, dict]:
    for attempt in range(max_retries + 1):
        try:
            response = curl_requests.get(
                url,
                params=params,
                headers=_headers(api_key),
                timeout=timeout_seconds,
            )
            if response.status_code == 404:
                return 404, {}
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < max_retries:
                    continue
                if response.status_code == 429:
                    raise RateLimitExceeded(
                        "OpenAlex returned 429 after "
                        f"{max_retries + 1} attempts; the daily budget is likely spent. "
                        "Set OPENALEX_API_KEY or rerun later to resume."
                    )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("OpenAlex returned a non-object JSON payload")
            return response.status_code, payload
        finally:
            delay = sleep_seconds if attempt == 0 else sleep_seconds * 2**attempt
            if delay > 0:
                time.sleep(delay)
    raise AssertionError("unreachable: the retry loop always returns or raises")


def _lookup_work(
    row: dict[str, str],
    *,
    api_url: str,
    api_key: str,
    timeout_seconds: float,
    sleep_seconds: float,
    max_retries: int,
    title_rule: str = TITLE_RULE,
) -> tuple[dict | None, str, int, bool]:
    """(work, matched_by, status code, ambiguous) for ROW: by its DOI if it has one (a DOI OpenAlex lacks is `doi_not_in_openalex`, without a title search: Unpaywall gets the DOI), else by its title under TITLE_RULE."""
    if row["doi"]:
        status_code, work = get_json(
            f"{api_url.rstrip('/')}/works/doi:{quote(row['doi'], safe='')}",
            params={"select": SELECT_FIELDS},
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            sleep_seconds=sleep_seconds,
            max_retries=max_retries,
        )
        if work:
            _validate_work(work)
            returned_doi = str(work.get("doi") or "")
            if normalize_doi(returned_doi) != row["doi"]:
                raise ValueError(
                    f"OpenAlex DOI lookup returned mismatched DOI: {returned_doi}"
                )
            return work, "doi", status_code, False
        return None, "doi_not_in_openalex", status_code, False
    search = dict(
        api_url=api_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        sleep_seconds=sleep_seconds,
        max_retries=max_retries,
    )
    if title_rule == "2026-09-28":
        return _full_text_search(row, **search)
    return _title_filter(row, **search)


def _full_text_search(
    row: dict[str, str], **request
) -> tuple[dict | None, str, int, bool]:
    """The title rule `2026-09-28`: a full-text search filtered to the row's years."""
    search = _search_text(row["title"])
    if not search:
        return None, "title", 0, False
    filters = []
    years = sorted(_years(row["year"]))
    if years:
        filters.append("publication_year:" + "|".join(str(year) for year in years))
    params: dict[str, object] = {
        "search": search,
        "per-page": 200,
        "select": SELECT_FIELDS,
    }
    if filters:
        params["filter"] = ",".join(filters)
    status_code, payload = get_json(
        f"{request.pop('api_url').rstrip('/')}/works", params=params, **request
    )
    # A truncated result page does not establish uniqueness. Fail closed so the operator can refine the query; never silently select the first-page winner.
    matches = _matching_search_results(row, payload)
    meta = _object(payload.get("meta"), "meta")
    _fields(meta, count=int)
    total = meta.get("count")
    if total is not None and total < 0:
        raise ValueError("Malformed provider response: negative result count")
    if total is not None and total > len(payload["results"]):
        return None, "title", status_code, True
    if len(matches) == 1:
        return matches[0], "title", status_code, False
    return None, "title", status_code, len(matches) > 1


def first_author_cited(work: dict, authors: str) -> bool:
    """Whether the record's first author is one of the cited authors (accents and case aside)."""
    names = [
        str((a.get("author") or {}).get("display_name") or "")
        for a in work.get("authorships") or []
    ]
    first = re.findall(r"\w+", _fold(names[0]))[-1:] if names and names[0] else []
    return bool(first) and first[0] in set(re.findall(r"\w+", _fold(authors)))


def _fold(text: str) -> str:
    return "".join(
        c
        for c in unicodedata.normalize("NFKD", text or "")
        if not unicodedata.combining(c)
    ).casefold()


def _title_filter(row: dict[str, str], **request) -> tuple[dict | None, str, int, bool]:
    """The title rule `2026-09-29` (TITLE_RULES): OpenAlex's title filter."""
    text = _search_text(row["title"]).replace(",", " ")
    if not text:
        return None, TITLE_FILTER, 0, False
    years = sorted(_years(row["year"]))
    near = sorted({year + step for year in years for step in (-1, 0, 1)})
    filters = [f"title.search:{text}"]
    if near:
        filters.append("publication_year:" + "|".join(map(str, near)))
    status_code, payload = get_json(
        f"{request.pop('api_url').rstrip('/')}/works",
        params={"filter": ",".join(filters), "per-page": 200, "select": SELECT_FIELDS},
        **request,
    )
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("OpenAlex search returned no results list")
    for work in _objects(results, "results"):
        _validate_work(work)
    meta = _object(payload.get("meta"), "meta")
    _fields(meta, count=int)
    if (meta.get("count") or 0) > len(results):
        return None, TITLE_FILTER, status_code, True  # more than one page: not unique
    results = [
        work
        for work in results
        if isinstance(work, dict) and first_author_cited(work, row["authors"])
    ]
    exact = [work for work in results if matches_metadata({**row, "doi": ""}, work)]
    if exact:
        if len(exact) == 1:
            return exact[0], TITLE_FILTER, status_code, False
        return None, TITLE_FILTER, status_code, True
    if not years:
        return None, TITLE_FILTER, status_code, False
    near_row = {**row, "doi": "", "year": "; ".join(map(str, near))}
    shifted = [work for work in results if matches_metadata(near_row, work)]
    if len(shifted) == 1:
        return shifted[0], TITLE_FILTER_OFF_BY_ONE, status_code, False
    matched_by = TITLE_FILTER_OFF_BY_ONE if shifted else TITLE_FILTER
    return None, matched_by, status_code, len(shifted) > 1


def _status_row(row: dict[str, str]) -> dict[str, object]:
    return {
        "paperid": row["paperid"],
        "queue_status": row["queue_status"],
        "title": row["title"],
        "authors": row["authors"],
        "year": row["year"],
        "doi": row["doi"],
        "source_url": row["source_url"],
        "pdf_url": "",
        "oa_status": "",
        "oa_host_type": "",
        "status_code": "",
        "error": "",
        "attempted_at": ledger.utc_now(),
        "alternate_pdf_urls": "[]",
        "query_doi": row["doi"],
        "openalex_id": "",
        "matched_by": "",
        "matched_title": "",
        "matched_year": "",
    }


def resolve_one(
    row: dict[str, str],
    *,
    api_url: str,
    api_key: str,
    timeout_seconds: float,
    sleep_seconds: float,
    max_retries: int,
    title_rule: str = TITLE_RULE,
) -> dict[str, object]:
    """The status row of one OpenAlex lookup; raises `RateLimitExceeded` when the budget is spent."""
    result = _status_row(row)
    try:
        work, matched_by, status_code, ambiguous = _lookup_work(
            row,
            api_url=api_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            sleep_seconds=sleep_seconds,
            max_retries=max_retries,
            title_rule=title_rule,
        )
        result["status_code"] = status_code
        result["matched_by"] = matched_by
        if ambiguous:
            result["oa_status"] = "ambiguous"
            return result
        if work is None:
            result["oa_status"] = "not_found"
            return result

        result["openalex_id"] = str(work.get("id") or "")
        result["matched_title"] = str(work.get("title") or "")
        result["matched_year"] = str(work.get("publication_year") or "")
        returned_doi = str(work.get("doi") or "")
        if returned_doi:
            result["doi"] = normalize_doi(returned_doi)
        locations = _pdf_locations(work)
        if locations:
            result["pdf_url"], result["oa_host_type"] = locations[0]
            result["alternate_pdf_urls"] = json.dumps(
                [url for url, _host_type in locations[1:]], separators=(",", ":")
            )
            result["oa_status"] = "has_pdf"
        else:
            open_access = work.get("open_access") or {}
            result["oa_status"] = (
                "open_access_but_no_pdf"
                if open_access.get("is_oa")
                else "not_open_access"
            )
    except (RequestException, ValueError, TypeError) as exc:
        result.update(oa_status="error", error=str(exc))
    return result


def resolve_found(
    row: dict[str, str],
    doi: str,
    source: str,
    *,
    api_url: str,
    api_key: str,
    timeout_seconds: float,
    sleep_seconds: float,
    max_retries: int,
    title_rule: str = TITLE_RULE,
) -> dict[str, object]:
    """The status row of looking ROW up: by its own DOI or title (`resolve_one`), or, if SOURCE (`crossref` or `s2`) found DOI for a row without one, by that DOI, recorded under the row with the DOI and who found it."""
    result = resolve_one(
        row | {"doi": doi} if doi else row,
        api_url=api_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        sleep_seconds=sleep_seconds,
        max_retries=max_retries,
        title_rule=title_rule,
    )
    if not doi:
        return result
    result["query_doi"] = row["doi"]
    result["matched_by"] = {
        "doi": f"{source}_doi",
        "doi_not_in_openalex": f"{source}_doi_not_in_openalex",
    }.get(str(result["matched_by"]), f"{source}_doi+{result['matched_by']}")
    return result


def _matches_current(
    status: dict[str, str],
    row: dict[str, str],
    found: dict[str, tuple[str, str]] | None = None,
) -> bool:
    """A status answers ROW if the row's evidence is the same and, for a row with a found DOI, if it is the lookup of that DOI."""
    source = (found or {}).get(row["paperid"], ("", ""))[1]
    return all(
        (
            status["query_doi"] == row["doi"],
            status["source_url"] == row["source_url"],
            status["title"] == row["title"],
            status["authors"] == row["authors"],
            status["year"] == row["year"],
            not source or status["matched_by"].startswith(f"{source}_doi"),
        )
    )


# What OpenAlex left without a PDF, for Unpaywall.
NO_PDF = {"not_open_access", "open_access_but_no_pdf", "not_found"}


def unpaywall_queries(
    queue: list[dict[str, str]],
    folder: Path,
    found: dict[str, tuple[str, str]],
    reference_urls: bool = False,
) -> list[dict[str, str]]:
    """The rows `collect open-access` asks Unpaywall about, each with its DOI, as in the released closure: a row OpenAlex settled without a PDF by the DOI OpenAlex names (the queue's, a found one, or its title match's); a row OpenAlex has not settled by the queue's or a found DOI; no row OpenAlex found a PDF for. Unpaywall is asked by DOI only (`source_url` empty). With REFERENCE_URLS, the reference's own web address is kept too: a fallback when Unpaywall has no PDF, and the only source of a row with no DOI at all."""
    latest = latest_status(folder / STATUS)
    queries = []
    for row in queue:
        status = latest.get(row["paperid"])
        settled = (
            status
            and _matches_current(status, row, found)
            and status["oa_status"] in TERMINAL_STATUSES
        )
        own = row["doi"] or found.get(row["paperid"], ("", ""))[0]
        if settled:
            doi = status["doi"] if status["oa_status"] in NO_PDF else ""
        else:
            doi = own
        url = row["source_url"] if reference_urls else ""
        if doi:
            queries.append(row | {"doi": doi, "source_url": url})
        elif (
            reference_urls
            and not own
            and not (settled and status["oa_status"] == "has_pdf")
        ):
            queries.append(row | {"doi": ""})
    return queries


def latest_status(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    frame = ledger.latest_by_paperid(ledger.read_csv(path, STATUS_FIELDS))
    return {row["paperid"]: row for row in frame}


def _write_candidates(
    path: Path,
    queue: list[dict[str, str]],
    latest: dict[str, dict[str, str]],
    found: dict[str, tuple[str, str]] | None = None,
) -> None:
    rows = []
    for queue_row in queue:
        status = latest.get(queue_row["paperid"])
        if (
            status
            and _matches_current(status, queue_row, found)
            and status["oa_status"] == "has_pdf"
            and status["pdf_url"]
        ):
            rows.append({name: status[name] for name in STATUS_FIELDS})
    ledger.replace_csv(path, rows, STATUS_FIELDS)


def settled(
    status: dict | None,
    row: dict,
    found: dict[str, tuple[str, str]] | None,
    title_rule: str = TITLE_RULE,
) -> bool:
    """Whether STATUS settles ROW: a final answer to the row as it stands. Under the title rule `2026-09-29`, a row the full-text title search left ambiguous or not found is asked again (as in the released closure); one it settled with a work keeps that work."""
    return bool(
        status
        and _matches_current(status, row, found)
        and status["oa_status"] in TERMINAL_STATUSES
        and not (
            title_rule != TITLE_RULES[0]
            and status["matched_by"] == "title"
            and status["oa_status"] in ("ambiguous", "not_found")
        )
    )


def resolve_openalex_pdfs(
    *,
    collection_path: str | Path,
    candidates_path: str | Path,
    status_path: str | Path,
    found: dict[str, tuple[str, str]] | None = None,
    api_key: str = "",
    sleep_seconds: float = SLEEP_SECONDS,
    timeout_seconds: float = TIMEOUT_SECONDS,
    limit: int | None = None,
    max_retries: int = MAX_RETRIES,
    api_url: str = OPENALEX_URL,
    title_rule: str = TITLE_RULE,
    waiting: frozenset[str] | set[str] = frozenset(),
) -> dict[str, int]:
    """Look up each pending queue row (not the WAITING ones: a Semantic Scholar PDF may still come for them) and rewrite the candidates."""
    collection_path = Path(collection_path)
    candidates_path = Path(candidates_path)
    status_path = Path(status_path)
    queue, skipped_review = queue_rows(collection_path)
    latest = latest_status(status_path)
    pending = [
        row
        for row in queue
        if not settled(latest.get(row["paperid"]), row, found, title_rule)
    ]
    waiting_now = sum(row["paperid"] in waiting for row in pending)
    pending = [row for row in pending if row["paperid"] not in waiting]
    if (
        not status_path.exists()
    ):  # every row may be waiting: the ledger exists all the same
        ledger.replace_csv(status_path, [], STATUS_FIELDS)
    if limit is not None:
        pending = pending[:limit]

    rate_limited = False
    found = found or {}
    for row in pending:
        try:
            resolved = resolve_found(
                row,
                *found.get(row["paperid"], ("", "")),
                api_url=api_url,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                sleep_seconds=sleep_seconds,
                max_retries=max_retries,
                title_rule=title_rule,
            )
        except RateLimitExceeded:
            # The budget is global, so every remaining row would fail the same way.
            # Stop, keep the ledger intact, and let the next run resume.
            rate_limited = True
            break
        ledger.append_status(status_path, STATUS_FIELDS, resolved)

    latest = latest_status(status_path)
    _write_candidates(candidates_path, queue, latest, found)
    current = [
        status
        for row in queue
        if (status := latest.get(row["paperid"]))
        and _matches_current(status, row, found)
    ]
    return {
        "eligible": len(queue),
        "resolved": sum(row["oa_status"] == "has_pdf" for row in current),
        "unresolved": sum(
            row["oa_status"] in TERMINAL_STATUSES
            and row["oa_status"] not in {"has_pdf", "ambiguous"}
            for row in current
        ),
        "ambiguous": sum(row["oa_status"] == "ambiguous" for row in current),
        "errors": sum(row["oa_status"] == "error" for row in current),
        "rate_limited": int(rate_limited),
        "skipped_review": skipped_review,
        "waiting_for_a_download": waiting_now,
        "unattempted": len(queue) - len(current),
    }
