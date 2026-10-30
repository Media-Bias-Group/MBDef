"""`collect semantic-scholar` and `collect semantic-scholar-doi`: more ways to a paper's DOI and PDF.

`semantic-scholar` asks Semantic Scholar's title match about each paper Crossref found no DOI for (`no_match`, or three failed searches). The one best match counts only under the identity rule of `collect openalex` (the same compared title, a year the row names, a shared author surname). Semantic Scholar sometimes names a DOI that is not the paper's own (a book review's, say), so its DOI is kept only if Crossref's record of that DOI passes `collect crossref`'s rule too, or if it is the arXiv DOI of the matched arXiv id. A kept DOI goes to OpenAlex and Unpaywall like a Crossref one; the match's open PDF and arXiv copy are candidates for `collect download`. Writes `semantic_scholar_status.csv` and `semantic_scholar_candidates.csv`.

`semantic-scholar-doi` asks Semantic Scholar about the DOIs whose PDF is still missing after OpenAlex, 500 at a time, for the open copies it knows: its open-access PDF, the arXiv copy, and, for a paper in PubMed Central, Europe PMC's rendering of it as a PDF. Writes `semantic_scholar_doi_status.csv` and `semantic_scholar_doi_candidates.csv`.

Both make the requests and apply the rules of the released closure's collection.
"""

import json
import os
import random
import re
import time
from pathlib import Path
from urllib.parse import quote, urlparse

from curl_cffi import requests as curl_requests
from curl_cffi.requests.exceptions import RequestException

from mbdef.collect import crossref, ledger
from mbdef.collect.download import FIELDNAMES as DOWNLOAD_FIELDS
from mbdef.collect.download import STATUS as DOWNLOAD_STATUS
from mbdef.collect.identity import (
    normalize_doi,
    queue_surnames,
    unicode_words,
)
from mbdef.collect.open_access import queue_rows
from mbdef.collect.openalex import STATUS as OPENALEX_STATUS
from mbdef.collect.openalex import STATUS_FIELDS as CANDIDATE_FIELDS
from mbdef.collect.openalex import _years as years_of
from mbdef.collect.openalex import matches_metadata

# Pinned defaults (those of the released closure's collection).
SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1"
MATCH_FIELDS = "title,year,authors,externalIds,openAccessPdf"
SLEEP_SECONDS = (
    1.111  # between title matches: 0.9 a second, under an API key's one a second
)
CHECK_SLEEP_SECONDS = (
    2.5  # between Crossref checks of a DOI Semantic Scholar names: 0.4 a second
)
TIMEOUT_SECONDS = 30.0
MAX_ATTEMPTS = 6  # per paper, for network errors and HTTP 429 and 5xx answers
CHECK_ATTEMPTS = 4  # per Crossref check
BACKOFF_SECONDS = 5.0  # the wait after a failed attempt grows by this much each time
MIN_TITLE_CHARS = crossref.MIN_TITLE_CHARS

STATUS = "semantic_scholar_status.csv"
CANDIDATES = "semantic_scholar_candidates.csv"
STATUS_FIELDS = [
    "paperid",
    "title",
    "authors",
    "year",
    "s2_status",
    "reason",
    "s2_paper_id",
    "doi",
    "doi_check",
    "arxiv",
    "pdf_url",
    "matched_title",
    "matched_year",
    "match_score",
    "status_code",
    "error",
    "attempted_at",
]
SETTLED = {"matched", "no_match", "insufficient_evidence"}
ARXIV_PDF = "https://arxiv.org/pdf/{arxiv}"


def api_key() -> str:
    """SEMANTIC_SCHOLAR_API_KEY (optional): without one, requests share the public rate limit."""
    return os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()


def command(
    step_folder: str | None,
    collection: str | None,
    api_url: str,
    crossref_url: str,
    sleep_seconds: float,
    check_sleep_seconds: float,
    timeout_seconds: float,
    max_attempts: int,
    backoff_seconds: float,
    limit: int | None,
) -> int:
    """The `collect semantic-scholar` command: match, print the counts, append the receipt."""
    key, mailto = api_key(), crossref.contact()
    folder = ledger.folder(step_folder)
    queue_path = ledger.collection(collection, folder)
    status, candidates = folder / STATUS, folder / CANDIDATES
    queue, skipped_review = queue_rows(queue_path)
    summary = match_all(
        queue,
        folder,
        api_url=api_url,
        crossref_url=crossref_url,
        key=key,
        mailto=mailto,
        sleep_seconds=sleep_seconds,
        check_sleep_seconds=check_sleep_seconds,
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
        backoff_seconds=backoff_seconds,
        limit=limit,
    ) | {"skipped_review": skipped_review}
    settings = {
        "api_url": api_url,
        "crossref_url": crossref_url,
        "api_key": "set" if key else "not set",
        "mailto": "set" if mailto else "not set",
        "sleep_seconds": sleep_seconds,
        "check_sleep_seconds": check_sleep_seconds,
        "timeout_seconds": timeout_seconds,
        "max_attempts": max_attempts,
        "backoff_seconds": backoff_seconds,
        "limit": limit,
    }
    outputs = {"status": status, "candidates": candidates}
    inputs = {"collection": queue_path, "crossref": folder / crossref.STATUS}
    ledger.append_receipt(
        folder, "collect semantic-scholar", settings, inputs, outputs, summary
    )
    ledger.report(summary, outputs)
    return 0


# The title match ---------------------------------------------------------------------------------


def match(row: dict, paper: dict) -> dict:
    """The rule's verdict on Semantic Scholar's best title match, as ledger fields: the paper is reshaped as an OpenAlex work and judged by `openalex.matches_metadata`."""
    work = {
        "title": str(paper.get("title") or ""),
        "publication_year": paper.get("year"),
        "doi": "",
        "authorships": [
            {"author": {"display_name": str(a.get("name") or "")}}
            for a in paper.get("authors") or []
            if isinstance(a, dict)
        ],
    }
    ids = paper.get("externalIds") or {}
    pdf = str((paper.get("openAccessPdf") or {}).get("url") or "")
    found = {
        "s2_paper_id": str(paper.get("paperId") or ""),
        "matched_title": work["title"],
        "matched_year": str(work["publication_year"] or ""),
        "match_score": str(paper.get("matchScore") or ""),
    }
    if unicode_words(row["title"]) != unicode_words(work["title"]):
        return found | {
            "s2_status": "no_match",
            "reason": "best title match is a different title",
        }
    if not matches_metadata(row | {"doi": ""}, work):
        return found | {
            "s2_status": "no_match",
            "reason": "title matches but year or author surname does not",
        }
    return found | {
        "s2_status": "matched",
        "reason": "exact title+year+author (repo rule)",
        "doi": normalize_doi(str(ids.get("DOI") or "")),
        "arxiv": str(ids.get("ArXiv") or ""),
        "pdf_url": pdf if pdf.startswith(("http://", "https://")) else "",
    }


def check_doi(
    row: dict,
    result: dict,
    *,
    crossref_url: str,
    mailto: str,
    timeout_seconds: float,
    check_sleep_seconds: float,
    backoff_seconds: float,
) -> dict:
    """Keep a matched paper's DOI only if Crossref's record of it passes `crossref.match` for ROW, or if it is the arXiv DOI of the matched arXiv id; otherwise drop it (its PDFs stay)."""
    doi = result.get("doi") or ""
    if not doi or result.get("s2_status") != "matched":
        return result
    if result.get("arxiv") and doi == f"10.48550/arxiv.{result['arxiv'].lower()}":
        return result | {"doi_check": "arxiv_doi"}
    for attempt in range(CHECK_ATTEMPTS):
        if check_sleep_seconds > 0:
            time.sleep(check_sleep_seconds)
        try:
            response = curl_requests.get(
                f"{crossref_url.rstrip('/')}/works/{quote(doi, safe='/')}",
                params={"mailto": mailto} if mailto else {},
                headers={"User-Agent": crossref.USER_AGENT},
                timeout=timeout_seconds,
            )
        except RequestException:
            response = None
        if response is not None and response.status_code == 200:
            try:
                item = response.json().get("message") or {}
            except ValueError, AttributeError:
                item = {}
            verdict = crossref.match(row, [item] if isinstance(item, dict) else [])
            if verdict["cr_status"] == "matched" and verdict.get("doi") == doi:
                return result | {"doi_check": "crossref_record_passes"}
            return result | {
                "doi": "",
                "doi_check": f"dropped {doi}: Crossref record fails the rule "
                f"({verdict.get('reason', '')})",
            }
        if response is not None and response.status_code == 404:
            return result | {
                "doi": "",
                "doi_check": f"dropped {doi}: not a Crossref DOI",
            }
        retry_after = (
            response.headers.get("retry-after") if response is not None else None
        )
        # A fifth of the title match's wait, doubling: one second at first with the defaults.
        crossref.backoff(attempt, retry_after, backoff_seconds / 5)
    return result | {
        "s2_status": "error",
        "error": f"Crossref check of the S2 DOI {doi} failed",
    }


def lookup(
    row: dict,
    *,
    api_url: str,
    key: str,
    timeout_seconds: float,
    max_attempts: int,
    backoff_seconds: float,
    sleep_seconds: float,
    pace: dict,
) -> dict:
    """The ledger row of one title match for ROW, before the DOI check."""
    base = {
        "paperid": row["paperid"],
        "title": row["title"],
        "authors": row["authors"],
        "year": row["year"],
        "attempted_at": ledger.utc_now(),
    }
    missing = [
        name
        for name, present in (
            ("title", len(unicode_words(row["title"])) >= MIN_TITLE_CHARS),
            ("year", bool(years_of(row["year"]))),
            ("authors", bool(queue_surnames(row["authors"]))),
        )
        if not present
    ]
    if missing:
        return base | {
            "s2_status": "insufficient_evidence",
            "reason": "citation lacks " + "+".join(missing),
        }
    last, code = "", ""
    for attempt in range(max_attempts):
        if pace["sleep"] > 0:
            time.sleep(pace["sleep"])
        try:
            response = curl_requests.get(
                f"{api_url.rstrip('/')}/paper/search/match",
                params={"query": row["title"].strip(), "fields": MATCH_FIELDS},
                headers={"x-api-key": key} if key else {},
                timeout=timeout_seconds,
            )
        except RequestException as exc:
            last = crossref.scrub(f"{type(exc).__name__}: {exc}", key)
            wait(attempt, None, backoff_seconds)
            continue
        code = str(response.status_code)
        if response.status_code == 429 or response.status_code >= 500:
            # Half the pace after a 429 (at least one request in ten seconds); every answered request raises it again by a tenth, up to the starting pace.
            last = f"HTTP Error {code}"
            if response.status_code == 429:
                pace["sleep"] = min(10.0, max(pace["sleep"] * 2, sleep_seconds))
            wait(attempt, response.headers.get("retry-after"), backoff_seconds)
            continue
        pace["sleep"] = max(sleep_seconds, pace["sleep"] / 1.1)
        if response.status_code == 404:
            return base | {
                "s2_status": "no_match",
                "reason": "no Semantic Scholar title match",
                "status_code": code,
            }
        if response.status_code != 200:
            return base | {
                "s2_status": "error",
                "status_code": code,
                "error": f"HTTP Error {code}",
            }
        try:
            data = response.json().get("data") or []
            paper = data[0] if data and isinstance(data[0], dict) else None
        except (ValueError, AttributeError) as exc:
            return base | {
                "s2_status": "error",
                "status_code": code,
                "error": crossref.scrub(f"malformed Semantic Scholar JSON: {exc}", key),
            }
        if paper is None:
            return base | {
                "s2_status": "no_match",
                "reason": "no Semantic Scholar title match",
                "status_code": code,
            }
        return base | match(row, paper) | {"status_code": code}
    return base | {
        "s2_status": "error",
        "status_code": code,
        "error": f"{last} after {max_attempts} attempts with backoff",
    }


def wait(attempt: int, retry_after: str | None, backoff_seconds: float) -> None:
    """After failed attempt ATTEMPT (from 0): Retry-After, else BACKOFF_SECONDS more each time; at most a minute, with jitter."""
    try:
        delay = float(retry_after or "")
    except TypeError, ValueError:
        delay = backoff_seconds * (attempt + 1)
    delay = min(60.0, delay) * random.uniform(0.8, 1.2)
    if delay > 0:
        time.sleep(delay)


# The ledger --------------------------------------------------------------------------------------


def read_status(path: Path) -> tuple[dict[str, dict], dict[str, int]]:
    latest: dict[str, dict] = {}
    errors: dict[str, int] = {}
    if path.exists():
        for row in ledger.read_csv(path, STATUS_FIELDS):
            latest[row["paperid"]] = row
            if row["s2_status"] == "error":
                errors[row["paperid"]] = errors.get(row["paperid"], 0) + 1
    return latest, errors


def wanted(queue: list[dict], folder: Path) -> list[dict]:
    """The rows without a DOI that Crossref settled as `no_match` or gave up on."""
    found, errors = crossref.read_status(folder / crossref.STATUS)
    return [
        row
        for row in queue
        if not row["doi"]
        and (
            (
                crossref.current(c := found.get(row["paperid"]), row)
                and c["cr_status"] == "no_match"
            )
            or errors.get(row["paperid"], 0) >= crossref.ERROR_ROUNDS
        )
    ]


def pdf_urls(status: dict) -> list[str]:
    """A title match's copies: its open-access PDF, then its arXiv copy."""
    urls = [status["pdf_url"]] if status.get("pdf_url") else []
    if status.get("arxiv"):
        urls.append(ARXIV_PDF.format(arxiv=status["arxiv"]))
    return list(dict.fromkeys(urls))


def candidate(row: dict, status: dict, urls: list[str], host_type: str) -> dict:
    """A row of a candidates file that `collect download` reads (the columns of OpenAlex's)."""
    return dict.fromkeys(CANDIDATE_FIELDS, "") | {
        "paperid": row["paperid"],
        "queue_status": row["queue_status"],
        "title": row["title"],
        "authors": row["authors"],
        "year": row["year"],
        "source_url": row["source_url"],
        "query_doi": row["doi"],
        "doi": status.get("doi") or row["doi"],
        "pdf_url": urls[0],
        "alternate_pdf_urls": json.dumps(urls[1:], separators=(",", ":")),
        "oa_status": "has_pdf",
        "oa_host_type": host_type,
        "status_code": status.get("status_code", ""),
        "attempted_at": status.get("attempted_at", ""),
        "matched_by": host_type,
        "matched_title": status.get("matched_title", ""),
        "matched_year": status.get("matched_year", ""),
    }


def match_all(
    queue: list[dict],
    folder: Path,
    *,
    api_url: str = SEMANTIC_SCHOLAR_URL,
    crossref_url: str = crossref.CROSSREF_URL,
    key: str = "",
    mailto: str = "",
    sleep_seconds: float = SLEEP_SECONDS,
    check_sleep_seconds: float = CHECK_SLEEP_SECONDS,
    timeout_seconds: float = TIMEOUT_SECONDS,
    max_attempts: int = MAX_ATTEMPTS,
    backoff_seconds: float = BACKOFF_SECONDS,
    limit: int | None = None,
) -> dict[str, int]:
    """Match every wanted row that has no settled answer yet; rewrite the candidates; the counts."""
    status_path = folder / STATUS
    eligible = wanted(queue, folder)
    latest, errors = read_status(status_path)
    pending = [
        row
        for row in eligible
        if not (
            crossref.current(latest.get(row["paperid"]), row)
            and latest[row["paperid"]]["s2_status"] in SETTLED
        )
        and errors.get(row["paperid"], 0) < crossref.ERROR_ROUNDS
    ]
    if limit is not None:
        pending = pending[:limit]
    pace = {"sleep": sleep_seconds}
    for row in pending:
        result = lookup(
            row,
            api_url=api_url,
            key=key,
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
            sleep_seconds=sleep_seconds,
            pace=pace,
        )
        result = check_doi(
            row,
            result,
            crossref_url=crossref_url,
            mailto=mailto,
            timeout_seconds=timeout_seconds,
            check_sleep_seconds=check_sleep_seconds,
            backoff_seconds=backoff_seconds,
        )
        ledger.append_status(
            status_path,
            STATUS_FIELDS,
            {k: crossref.scrub(str(v), key) for k, v in result.items()},
        )
    latest, _ = read_status(status_path)
    answered = [
        (row, status)
        for row in eligible
        if crossref.current(status := latest.get(row["paperid"]), row)
    ]
    matched = [(row, s) for row, s in answered if s["s2_status"] == "matched"]
    found = [
        candidate(row, s, urls, "semantic_scholar")
        for row, s in matched
        if (urls := pdf_urls(s))
    ]
    ledger.replace_csv(folder / CANDIDATES, found, CANDIDATE_FIELDS)
    return {
        "eligible": len(eligible),
        "matched": len(matched),
        "with_doi": sum(bool(s["doi"]) for _, s in matched),
        "with_pdf": len(found),
        "no_match": sum(s["s2_status"] == "no_match" for _, s in answered),
        "insufficient_evidence": sum(
            s["s2_status"] == "insufficient_evidence" for _, s in answered
        ),
        "errors": sum(s["s2_status"] == "error" for _, s in answered),
        "unattempted": len(eligible) - len(answered),
    }


# DOIs Crossref or Semantic Scholar found -------------------------------------------------------


def found_dois(queue: list[dict], folder: Path) -> dict[str, tuple[str, str]]:
    """paper id -> (DOI, who found it) for the rows without a DOI: Crossref's match, else Semantic Scholar's (a DOI that passed its check). `collect openalex` and `collect open-access` look these DOIs up as if the queue had them."""
    by_crossref, _ = crossref.read_status(folder / crossref.STATUS)
    by_s2, _ = read_status(folder / STATUS)
    found = {}
    for row in queue:
        if row["doi"]:
            continue
        c, s = by_crossref.get(row["paperid"]), by_s2.get(row["paperid"])
        if crossref.current(c, row) and c["cr_status"] == "matched" and c["doi"]:
            found[row["paperid"]] = (c["doi"], "crossref")
        elif crossref.current(s, row) and s["s2_status"] == "matched" and s["doi"]:
            found[row["paperid"]] = (s["doi"], "s2")
    return found


def pdf_pending(queue: list[dict], folder: Path) -> set[str]:
    """The paper ids of rows without a DOI (own or found) whose title match has a PDF that may still come: not yet tried, downloaded, or failed in a way a later run may undo. `collect openalex` does not search their titles (the released closure's order: the match's own copy first)."""
    from mbdef.collect import download

    by_s2, _ = read_status(folder / STATUS)
    found = found_dois(queue, folder)
    tried = download.latest_status(folder / download.STATUS)
    return {
        row["paperid"]
        for row in queue
        if not row["doi"]
        and row["paperid"] not in found
        and crossref.current(status := by_s2.get(row["paperid"]), row)
        and status["s2_status"] == "matched"
        and pdf_urls(status)
        and not download.failed_for_good(tried.get(row["paperid"]))
    }


# Open copies of a DOI ----------------------------------------------------------------------------

BATCH_FIELDS = "openAccessPdf,externalIds,isOpenAccess"
BATCH_SIZE = 500  # DOIs per request, Semantic Scholar's maximum
BATCH_SLEEP_SECONDS = 1.2  # after each batch
BATCH_ATTEMPTS = (
    6  # per batch; a batch that fails all of them is asked again on the next run
)
BATCH_TIMEOUT_SECONDS = 120.0
EUROPEPMC_PDF = "https://europepmc.org/articles/{pmcid}?pdf=render"
# Hosts that copy papers without the rights holder's permission: never a source.
NOT_AUTHORISED_HOST = (
    r"(^|\.)(communicationcache\.com|sci-hub\.[a-z]+|scihub\.[a-z]+|libgen\.[a-z]+|z-lib\.[a-z]+"
    r"|zlibrary\.[a-z]+|annas-archive\.[a-z]+|booksc\.[a-z]+|pdfdrive\.[a-z]+)$"
)
SETTLED_WITHOUT_DOWNLOAD = {"not_open_access", "open_access_but_no_pdf", "has_pdf"}

DOI_STATUS = "semantic_scholar_doi_status.csv"
DOI_CANDIDATES = "semantic_scholar_doi_candidates.csv"
DOI_STATUS_FIELDS = [
    "paperid",
    "doi",
    "s2_paper_id",
    "is_oa",
    "oa_status",
    "license",
    "pdf_url",
    "arxiv",
    "pmc",
    "status_code",
    "attempted_at",
]


def europepmc_pdf(pmc: str) -> str:
    """Europe PMC's PDF rendering of a PubMed Central paper ("6543210" or "PMC6543210")."""
    pmcid = pmc if str(pmc).upper().startswith("PMC") else f"PMC{pmc}"
    return EUROPEPMC_PDF.format(pmcid=pmcid)


def open_copies(status: dict | None) -> list[str]:
    """The copies Semantic Scholar names for a DOI: its open-access PDF, the arXiv copy and Europe PMC's rendering, without hosts that are not authorised."""
    if not status:
        return []
    urls = [status.get("pdf_url") or ""]
    if status.get("arxiv"):
        urls.append(ARXIV_PDF.format(arxiv=status["arxiv"]))
    if status.get("pmc"):
        urls.append(europepmc_pdf(status["pmc"]))
    return [
        url
        for url in dict.fromkeys(urls)
        if url.startswith(("http://", "https://"))
        and not re.search(NOT_AUTHORISED_HOST, urlparse(url).netloc, re.IGNORECASE)
    ]


def doi_command(
    step_folder: str | None,
    collection: str | None,
    api_url: str,
    sleep_seconds: float,
    timeout_seconds: float,
    backoff_seconds: float,
) -> int:
    """The `collect semantic-scholar-doi` command: look up, print the counts, append the receipt."""
    key = api_key()
    folder = ledger.folder(step_folder)
    queue_path = ledger.collection(collection, folder)
    queue, _ = queue_rows(queue_path)
    summary = lookup_dois(
        queue,
        folder,
        api_url=api_url,
        key=key,
        sleep_seconds=sleep_seconds,
        timeout_seconds=timeout_seconds,
        backoff_seconds=backoff_seconds,
    )
    settings = {
        "api_url": api_url,
        "api_key": "set" if key else "not set",
        "sleep_seconds": sleep_seconds,
        "timeout_seconds": timeout_seconds,
        "backoff_seconds": backoff_seconds,
    }
    outputs = {"status": folder / DOI_STATUS, "candidates": folder / DOI_CANDIDATES}
    inputs = {
        "collection": queue_path,
        "openalex": folder / OPENALEX_STATUS,
        "downloads": folder / DOWNLOAD_STATUS,
    }
    ledger.append_receipt(
        folder, "collect semantic-scholar-doi", settings, inputs, outputs, summary
    )
    ledger.report(summary, outputs)
    return 0


def dois_wanted(queue: list[dict], folder: Path) -> dict[str, list[str]]:
    """DOI -> paper ids: the papers OpenAlex settled (with or without a PDF address) that have no downloaded PDF and no answer in the ledger yet. The DOI is OpenAlex's, else the queue's, else the one Crossref or Semantic Scholar found."""
    openalex = {}
    if (folder / OPENALEX_STATUS).exists():
        openalex = {
            row["paperid"]: row
            for row in ledger.read_csv(folder / OPENALEX_STATUS, CANDIDATE_FIELDS)
        }
    downloaded = set()
    if (folder / DOWNLOAD_STATUS).exists():
        for row in ledger.read_csv(
            folder / DOWNLOAD_STATUS,
            DOWNLOAD_FIELDS,
            optional=("title", "authors", "year"),
        ):
            if ledger.truthy(row["downloaded"]):
                downloaded.add(row["paperid"])
            else:
                downloaded.discard(row["paperid"])
    have = {row["paperid"] for row in read_doi_status(folder / DOI_STATUS)}
    found = found_dois(queue, folder)
    want: dict[str, list[str]] = {}
    for row in queue:
        paperid = row["paperid"]
        status = openalex.get(paperid) or {}
        doi = status.get("doi") or row["doi"] or found.get(paperid, ("", ""))[0]
        if not doi or paperid in have or paperid in downloaded:
            continue
        if status.get("oa_status") in SETTLED_WITHOUT_DOWNLOAD:
            want.setdefault(doi, []).append(paperid)
    return want


def read_doi_status(path: Path) -> list[dict]:
    return ledger.read_csv(path, DOI_STATUS_FIELDS) if path.exists() else []


def lookup_dois(
    queue: list[dict],
    folder: Path,
    *,
    api_url: str = SEMANTIC_SCHOLAR_URL,
    key: str = "",
    sleep_seconds: float = BATCH_SLEEP_SECONDS,
    timeout_seconds: float = BATCH_TIMEOUT_SECONDS,
    backoff_seconds: float = BACKOFF_SECONDS,
) -> dict[str, int]:
    """Ask Semantic Scholar about every wanted DOI, BATCH_SIZE at a time; rewrite the candidates."""
    status_path = folder / DOI_STATUS
    want = dois_wanted(queue, folder)
    dois, recorded, failed = sorted(want), 0, 0
    for start in range(0, len(dois), BATCH_SIZE):
        chunk = dois[start : start + BATCH_SIZE]
        response = None
        for attempt in range(BATCH_ATTEMPTS):
            try:
                response = curl_requests.post(
                    f"{api_url.rstrip('/')}/paper/batch",
                    params={"fields": BATCH_FIELDS},
                    json={"ids": [f"DOI:{doi}" for doi in chunk]},
                    headers={"x-api-key": key} if key else {},
                    timeout=timeout_seconds,
                )
            except RequestException:
                response = None
            if response is not None and response.status_code == 200:
                break
            if backoff_seconds > 0:
                time.sleep(backoff_seconds * (attempt + 1))
        papers = None
        if response is not None and response.status_code == 200:
            try:
                papers = response.json()
            except ValueError:
                papers = None
        if not isinstance(papers, list) or len(papers) != len(chunk):
            failed += sum(len(want[doi]) for doi in chunk)
            continue
        for doi, paper in zip(chunk, papers, strict=True):
            paper = paper if isinstance(paper, dict) else {}
            pdf = paper.get("openAccessPdf") or {}
            ids = paper.get("externalIds") or {}
            for paperid in want[doi]:
                ledger.append_status(
                    status_path,
                    DOI_STATUS_FIELDS,
                    {
                        "paperid": paperid,
                        "doi": doi,
                        "s2_paper_id": paper.get("paperId") or "",
                        "is_oa": paper.get("isOpenAccess") or "",
                        "oa_status": pdf.get("status") or "",
                        "license": pdf.get("license") or "",
                        "pdf_url": pdf.get("url") or "",
                        "arxiv": ids.get("ArXiv") or "",
                        "pmc": ids.get("PubMedCentral") or "",
                        "status_code": 200 if paper else 404,
                        "attempted_at": ledger.utc_now(),
                    },
                )
                recorded += 1
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    latest = {row["paperid"]: row for row in read_doi_status(status_path)}
    rows = []
    for row in queue:
        status = latest.get(row["paperid"])
        if status and (urls := open_copies(status)):
            rows.append(candidate(row, status, urls, "semantic_scholar_doi"))
    ledger.replace_csv(folder / DOI_CANDIDATES, rows, CANDIDATE_FIELDS)
    return {
        "dois_asked": len(dois),
        "rows_recorded": recorded,
        "rows_failed": failed,
        "with_copies": len(rows),
    }
