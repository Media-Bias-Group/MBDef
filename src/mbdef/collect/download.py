"""`collect download`: download the PDFs that a resolver found into zip batches.

Each candidate's URLs (the first, then any alternates OpenAlex listed) are tried in order with a browser's TLS fingerprint, until one answers with a PDF (the bytes `%PDF-` within the first kilobyte) no larger than the size cap. Each PDF is stored, uncompressed, as `<xx>/<paper id>.pdf` in `pdf_batches/pdf_batch_<n>.zip`, with its paper id and DOI in the member's comment, so the archive is its own record if the ledger write is interrupted. Writes `pdf_download_status.csv` and `failed_pdf_downloads.csv`.
"""

import html
import json
import re
import threading
import time
import zipfile
from pathlib import Path
from urllib.parse import urljoin, urlparse

from curl_cffi import CurlOpt
from curl_cffi import requests as curl_requests
from curl_cffi.requests.exceptions import RequestException

from mbdef.collect import ledger
from mbdef.collect.identity import filename_safe
from mbdef.collect.open_access import CANDIDATES as OPEN_ACCESS_CANDIDATES
from mbdef.collect.open_access import STATUS_FIELDS as CANDIDATE_FIELDS

# Pinned defaults.
CONCURRENCY = 4  # parallel downloads
TIMEOUT_SECONDS = 45.0  # per URL, including connection, redirects and body
MAX_PDF_MB = 80.0  # MiB
PDF_BATCH_SIZE = 5000  # PDFs per zip
IMPERSONATE = "chrome"  # the browser whose TLS fingerprint curl_cffi sends

# Hosts asked at most once every so many seconds, one request at a time (Europe PMC's rendering refused a burst of requests with HTTP 429 in the released closure's collection, while spaced ones succeed), and how long to wait after a 429 or 503 that names no Retry-After.
PACED_HOSTS = {"europepmc.org": 3.0, "www.ebi.ac.uk": 3.0}
PACE_COOLDOWN_SECONDS = 60.0
# After a failed download: Europe PMC's rendering of a PubMed Central paper, and the PDF a landing page declares for indexers (the released closure's recovery); at most this many more.
EUROPE_PMC_URL = "https://europepmc.org"
RECOVERY_URLS = 4
PMC_ID = re.compile(r"\b(PMC\d{4,9})\b")
PDF_LINKS = (
    re.compile(
        r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)', re.I
    ),
    re.compile(
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']citation_pdf_url', re.I
    ),
    re.compile(
        r'<link[^>]+type=["\']application/pdf["\'][^>]+href=["\']([^"\']+)', re.I
    ),
)
LANDING_TIMEOUT_SECONDS = 30.0

# The class of a failed download (`failure_class`). A later run may succeed after these, so a failure of one of them is not final: `collect openalex` keeps a Semantic Scholar PDF's row waiting.
TRANSIENT = ("http_429", "http_5xx", "timeout", "network")
RETRY_LATER = TRANSIENT + ("http_202", "http_203", "other")

STATUS = "pdf_download_status.csv"
FAILED = "failed_pdf_downloads.csv"
ARCHIVES = "pdf_batches"
FIELDNAMES = [
    "paperid",
    "doi",
    "title",
    "authors",
    "year",
    "pdf_url",
    "pdf_archive_path",
    "pdf_archive_member",
    "downloaded",
    "status_code",
    "error",
    "pdf_bytes",
    "elapsed_seconds",
    "attempted_at",
]


def command(
    step_folder: str | None,
    candidates: str | None,
    concurrency: int,
    timeout_seconds: float,
    max_pdf_mb: float,
    pdf_batch_size: int,
    limit: int | None,
    europe_pmc_url: str = EUROPE_PMC_URL,
) -> int:
    """The `collect download` command: download, print the counts, append the receipt."""
    folder = ledger.folder(step_folder)
    candidates_path = (
        Path(candidates) if candidates else folder / OPEN_ACCESS_CANDIDATES
    )
    if not candidates_path.is_file():
        raise SystemExit(
            f"The candidates file {candidates_path} does not exist. "
            "`collect open-access` or `collect openalex` writes it (docs/step-1-2-papers/collect.md)."
        )
    status, failed, archives = folder / STATUS, folder / FAILED, folder / ARCHIVES
    summary = download_open_access_pdfs_to_archives(
        candidates_path=candidates_path,
        pdf_archive_dir=archives,
        status_path=status,
        failed_path=failed,
        concurrency=concurrency,
        timeout_seconds=timeout_seconds,
        max_pdf_mb=max_pdf_mb,
        pdf_batch_size=pdf_batch_size,
        limit=limit,
        europe_pmc_url=europe_pmc_url,
    )
    settings = {
        "concurrency": concurrency,
        "timeout_seconds": timeout_seconds,
        "max_pdf_mb": max_pdf_mb,
        "pdf_batch_size": pdf_batch_size,
        "impersonate": IMPERSONATE,
        "limit": limit,
        "europe_pmc_url": europe_pmc_url,
        "paced_hosts": PACED_HOSTS,
    }
    outputs = {"status": status, "failed": failed, "archives": archives}
    ledger.append_receipt(
        folder,
        "collect download",
        settings,
        {"candidates": candidates_path},
        outputs,
        summary,
    )
    ledger.report(summary, outputs)
    return 0


def member_name(paperid: str) -> str:
    safe = filename_safe(paperid)
    prefix = safe[:2].lower() if len(safe) >= 2 else "xx"
    return f"{prefix}/{safe}.pdf"


def _paperid_from_member(member: str) -> str:
    return Path(member).stem


def _archive_member_exists(archive_path: str | Path, member: str) -> bool:
    if not archive_path or not member:
        return False
    try:
        with zipfile.ZipFile(archive_path) as archive:
            return member in archive.namelist()
    except FileNotFoundError, OSError, zipfile.BadZipFile:
        return False


def _load_downloaded(
    status_path: Path, archive_dir: Path
) -> tuple[dict[str, tuple[str, str, int]], int]:
    downloaded: dict[str, tuple[str, str, int]] = {}
    archived_count = 0
    for archive_path in sorted(archive_dir.glob("pdf_batch_*.zip")):
        try:
            with zipfile.ZipFile(archive_path) as archive:
                for info in archive.infolist():
                    if not info.filename.endswith(".pdf"):
                        continue
                    comment = info.comment.decode("utf-8")
                    identifier = (
                        json.loads(comment)["paperid"]
                        if comment.startswith("{")
                        else comment or _paperid_from_member(info.filename)
                    )
                    downloaded[identifier] = (
                        str(archive_path),
                        info.filename,
                        info.file_size,
                    )
                    archived_count += 1
        except OSError, ValueError, KeyError, zipfile.BadZipFile:
            continue

    if status_path.exists():
        latest = ledger.latest_by_paperid(
            ledger.read_csv(
                status_path, FIELDNAMES, optional=("title", "authors", "year")
            )
        )
        for row in latest:
            if (
                ledger.truthy(row["downloaded"])
                and row["paperid"] not in downloaded
                and _archive_member_exists(
                    row["pdf_archive_path"], row["pdf_archive_member"]
                )
            ):
                try:
                    with zipfile.ZipFile(row["pdf_archive_path"]) as archive:
                        size = archive.getinfo(row["pdf_archive_member"]).file_size
                except FileNotFoundError, KeyError, OSError, zipfile.BadZipFile:
                    continue
                downloaded[row["paperid"]] = (
                    row["pdf_archive_path"],
                    row["pdf_archive_member"],
                    size,
                )
    return downloaded, archived_count


def _read_candidates(path: Path) -> list[dict]:
    frame = ledger.read_csv(path, CANDIDATE_FIELDS, extra=True)
    candidates: list[dict] = []
    seen: set[str] = set()
    for row in frame:
        paperid = row["paperid"]
        if not paperid or not row["pdf_url"]:
            continue
        if paperid in seen:
            raise ValueError(f"Duplicate paperid in {path}: {paperid}")
        seen.add(paperid)
        urls = [row["pdf_url"]]
        encoded_alternates = row.get("alternate_pdf_urls", "")
        if encoded_alternates:
            try:
                alternates = json.loads(encoded_alternates)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid alternate_pdf_urls for {paperid} in {path}"
                ) from exc
            if not isinstance(alternates, list) or not all(
                isinstance(url, str) for url in alternates
            ):
                raise ValueError(
                    f"alternate_pdf_urls must be a JSON string list for {paperid}"
                )
            urls.extend(alternates)
        candidates.append(
            {
                "paperid": paperid,
                "doi": row["doi"],
                "title": row["title"],
                "authors": row["authors"],
                "year": row["year"],
                "pdf_url": row["pdf_url"],
                "pdf_urls": list(dict.fromkeys(urls)),
            }
        )
    return candidates


def download_one(
    *,
    paperid: str,
    doi: str,
    pdf_urls: list[str],
    timeout_seconds: float,
    max_pdf_mb: float,
    title: str = "",
    authors: str = "",
    year: str = "",
    europe_pmc_url: str | None = EUROPE_PMC_URL,
) -> dict[str, object]:
    """The status row of one candidate; a PDF found is in its `_pdf_bytes_data` field. If no address gives a PDF, the recovery addresses are tried (`recovery_urls`; not with EUROPE_PMC_URL None), and the error names them when they fail too."""
    started = time.monotonic()
    row: dict[str, object] = {
        "paperid": paperid,
        "doi": doi,
        "title": title,
        "authors": authors,
        "year": year,
        "pdf_url": pdf_urls[0],
        "pdf_archive_path": "",
        "pdf_archive_member": "",
        "downloaded": False,
        "status_code": "",
        "error": "",
        "pdf_bytes": "",
        "elapsed_seconds": "",
        "attempted_at": ledger.utc_now(),
    }
    failures = []
    for pdf_url in pdf_urls:
        row["pdf_url"] = pdf_url
        response = None
        host = urlparse(pdf_url).netloc.lower()
        try:
            PACING.wait(host)
            response = curl_requests.get(
                pdf_url,
                impersonate=IMPERSONATE,
                headers={"Accept": "application/pdf,*/*"},
                timeout=(10, timeout_seconds),
                # Streaming mode otherwise sets only a low-speed timeout. This libcurl deadline also bounds a server that keeps trickling bytes.
                curl_options={CurlOpt.TIMEOUT_MS: max(1, int(timeout_seconds * 1000))},
                allow_redirects=True,
                stream=True,
            )
            row["status_code"] = response.status_code
            PACING.after(
                host, response.status_code, response.headers.get("retry-after", "")
            )
            max_bytes = int(max_pdf_mb * 1024 * 1024)
            if response.status_code != 200:
                raise ValueError(f"download_status_{response.status_code}")
            content = bytearray()
            for chunk in response.iter_content():
                if len(content) + len(chunk) > max_bytes:
                    raise ValueError(f"pdf_too_large max_mb={max_pdf_mb:g}")
                content.extend(chunk)
            pdf_bytes = bytes(content)
            row["pdf_bytes"] = len(pdf_bytes)
            if b"%PDF-" not in pdf_bytes[:1024]:
                content_type = response.headers.get("content-type", "").lower()
                error = f"not_pdf content_type={content_type}"
            else:
                row["_pdf_bytes_data"] = pdf_bytes
                row["error"] = ""
                break
        except Exception as exc:
            error = str(exc)
        finally:
            if response is not None:
                response.close()
        failures.append(f"{pdf_url}: {error}")
        row["error"] = error
    if "_pdf_bytes_data" not in row and len(pdf_urls) > 1:
        row["error"] = "all_sources_failed: " + " | ".join(failures)
    row["elapsed_seconds"] = round(time.monotonic() - started, 3)
    if "_pdf_bytes_data" in row or europe_pmc_url is None:
        return row
    extra = recovery_urls(pdf_urls, str(row["error"]), europe_pmc_url)
    if not extra:
        return row
    again = download_one(
        paperid=paperid,
        doi=doi,
        pdf_urls=extra,
        timeout_seconds=timeout_seconds,
        max_pdf_mb=max_pdf_mb,
        title=title,
        authors=authors,
        year=year,
        europe_pmc_url=None,
    )
    if "_pdf_bytes_data" in again:
        return again
    tried = again["error"] if len(extra) > 1 else f"{extra[0]}: {again['error']}"
    # The recovery addresses stay named in the error, so they count as tried.
    row["error"] = f"{row['error']} | recovery: {tried}"[:4000]
    return row


class Pacing:
    """When each paced host may be asked next: `wait` books a slot, `after` adds a cool-down after a 429 or 503 (its Retry-After, else PACE_COOLDOWN_SECONDS, at most 600 s)."""

    def __init__(self, hosts: dict[str, float], clock=time.monotonic, sleep=time.sleep):
        self.hosts, self.clock, self.sleep = hosts, clock, sleep
        self.next: dict[str, float] = {}
        self.lock = threading.Lock()

    def wait(self, host: str) -> None:
        gap = self.hosts.get(host)
        if gap is None:
            return
        with self.lock:
            start = max(self.clock(), self.next.get(host, 0.0))
            self.next[host] = start + gap
        self.sleep(max(0.0, start - self.clock()))

    def after(self, host: str, status: int, retry_after: str = "") -> None:
        if host not in self.hosts or status not in (429, 503):
            return
        text = str(retry_after).strip()
        wait = float(text) if text.isdigit() else PACE_COOLDOWN_SECONDS
        with self.lock:
            self.next[host] = max(
                self.next.get(host, 0.0), self.clock() + min(wait, 600.0)
            )


PACING = Pacing(PACED_HOSTS)


def landing_pdf_urls(page_url: str, text: str) -> list[str]:
    """The PDF links a landing page declares for indexers (citation_pdf_url, a PDF <link>)."""
    found = [
        urljoin(page_url, html.unescape(url.strip()))
        for pattern in PDF_LINKS
        for url in pattern.findall(text[:500000])
    ]
    return [u for u in dict.fromkeys(found) if u.startswith(("http://", "https://"))]


def recovery_urls(
    urls: list[str], error: str, europe_pmc_url: str = EUROPE_PMC_URL
) -> list[str]:
    """More sources for a failed download: Europe PMC's rendering of each PubMed Central paper the addresses name, and, when an address answered with HTML, the PDF its page declares."""
    base = europe_pmc_url.rstrip("/")
    extra = [
        f"{base}/articles/{pmc}?pdf=render" for u in urls for pmc in PMC_ID.findall(u)
    ]
    if "text/html" in (error or ""):
        for url in urls[:2]:
            try:
                PACING.wait(urlparse(url).netloc.lower())
                page = curl_requests.get(
                    url,
                    impersonate=IMPERSONATE,
                    timeout=LANDING_TIMEOUT_SECONDS,
                    allow_redirects=True,
                )
                PACING.after(urlparse(url).netloc.lower(), page.status_code)
                extra += landing_pdf_urls(
                    str(page.url), page.text if page.status_code == 200 else ""
                )
            except RequestException:
                continue
    return [u for u in dict.fromkeys(extra) if u not in urls][:RECOVERY_URLS]


def failure_class(error: str, status_code: str = "") -> str:
    """The class of a failed download, from its recorded error (the released closure's classes): for several addresses the first transient one, else the first; then `too_large`, `html_not_pdf`, `http_404`, `http_403`, `http_429`, `http_5xx`, `timeout`, `tls_certificate`, `network`, `http_<code>` or `other`."""
    text = (error or "").lower()
    if text.startswith("all_sources_failed:"):
        kinds = [failure_class(part.rsplit(": ", 1)[-1]) for part in text.split(" | ")]
        return next(
            (k for k in kinds if k in TRANSIENT), kinds[0] if kinds else "other"
        )
    found = re.search(r"download_status_(\d{3})", text)
    code = str(status_code)
    status = int(found.group(1)) if found else (int(code) if code.isdigit() else 0)
    if "pdf_too_large" in text:
        return "too_large"
    if "not_pdf" in text:
        return "html_not_pdf"
    if status in (404, 410):
        return "http_404"
    if status in (401, 403, 451):
        return "http_403"
    if status == 429:
        return "http_429"
    if status >= 500:
        return "http_5xx"
    if "curl: (28)" in text or "timed out" in text or "timeout" in text:
        return "timeout"
    if (
        "curl: (60)" in text
        or "ssl certificate" in text
        or ("curl: (35)" in text and "certificate" in text)
    ):
        return "tls_certificate"
    if re.search(r"curl: \((6|7|16|18|35|52|55|56|92)\)", text):
        return "network"
    return f"http_{status}" if status else "other"


def failed_for_good(status: dict | None) -> bool:
    """Whether a download ledger row is a failure that a later run would not undo."""
    return (
        bool(status)
        and not ledger.truthy(status.get("downloaded", ""))
        and failure_class(status.get("error") or "", status.get("status_code") or "")
        not in RETRY_LATER
    )


def _archive_pdf(
    *,
    archive_dir: Path,
    paperid: str,
    pdf_bytes: bytes,
    archive_index: int,
    batch_size: int,
    doi: str = "",
) -> tuple[str, str]:
    archive_dir.mkdir(parents=True, exist_ok=True)
    batch_number = archive_index // batch_size + 1
    archive_path = archive_dir / f"pdf_batch_{batch_number:06d}.zip"
    member = member_name(paperid)
    with zipfile.ZipFile(
        archive_path, "a", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as archive:
        info = zipfile.ZipInfo(member)
        info.comment = json.dumps({"paperid": paperid, "doi": doi}).encode("utf-8")
        archive.writestr(info, pdf_bytes)
    return str(archive_path), member


def _recovery_row(candidate: dict, archive: tuple[str, str, int]) -> dict[str, object]:
    archive_path, member, size = archive
    return {
        "paperid": candidate["paperid"],
        "doi": candidate["doi"],
        "title": candidate["title"],
        "authors": candidate["authors"],
        "year": candidate["year"],
        "pdf_url": candidate["pdf_url"],
        "pdf_archive_path": archive_path,
        "pdf_archive_member": member,
        "downloaded": True,
        "status_code": "",
        "error": "",
        "pdf_bytes": size,
        "elapsed_seconds": 0,
        "attempted_at": ledger.utc_now(),
    }


def latest_status(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    frame = ledger.latest_by_paperid(
        ledger.read_csv(path, FIELDNAMES, optional=("title", "authors", "year"))
    )
    return {row["paperid"]: row for row in frame}


def _refresh_failed(
    *,
    failed_path: Path,
    candidates: list[dict],
    latest: dict[str, dict[str, str]],
    downloaded: dict[str, tuple[str, str, int]],
) -> None:
    rows = []
    for candidate in candidates:
        if candidate["paperid"] in downloaded:
            continue
        status = latest.get(candidate["paperid"])
        if status and not ledger.truthy(status["downloaded"]):
            rows.append({name: status[name] for name in FIELDNAMES})
    ledger.replace_csv(failed_path, rows, FIELDNAMES)


def download_open_access_pdfs_to_archives(
    *,
    candidates_path: str | Path,
    pdf_archive_dir: str | Path,
    status_path: str | Path,
    failed_path: str | Path,
    concurrency: int = CONCURRENCY,
    timeout_seconds: float = TIMEOUT_SECONDS,
    max_pdf_mb: float = MAX_PDF_MB,
    pdf_batch_size: int = PDF_BATCH_SIZE,
    limit: int | None = None,
    europe_pmc_url: str = EUROPE_PMC_URL,
) -> dict[str, int]:
    if (
        concurrency <= 0
        or timeout_seconds <= 0
        or max_pdf_mb <= 0
        or pdf_batch_size <= 0
    ):
        raise ValueError(
            "Download concurrency, timeouts, size, and batch size must be positive"
        )
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    candidates_path = Path(candidates_path)
    pdf_archive_dir = Path(pdf_archive_dir)
    status_path = Path(status_path)
    failed_path = Path(failed_path)
    pdf_archive_dir.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)

    candidates = _read_candidates(candidates_path)
    downloaded, archived_count = _load_downloaded(status_path, pdf_archive_dir)
    latest = latest_status(status_path)

    for candidate in candidates:
        archived = downloaded.get(candidate["paperid"])
        if not archived:
            continue
        status = latest.get(candidate["paperid"])
        # Expected identity is immutable for already archived bytes. A new URL is allowed, but must never silently rewrite which DOI those bytes belong to.
        with zipfile.ZipFile(archived[0]) as archive:
            comment = archive.getinfo(archived[1]).comment.decode("utf-8")
            recorded = json.loads(comment) if comment.startswith("{") else status or {}
        if recorded.get("doi") and recorded["doi"] != candidate["doi"]:
            raise ValueError(f"Archived DOI changed for {candidate['paperid']}")
        if not (
            status
            and ledger.truthy(status["downloaded"])
            and status["doi"] == candidate["doi"]
            and status["pdf_url"] in candidate["pdf_urls"]
            and _archive_member_exists(
                status["pdf_archive_path"], status["pdf_archive_member"]
            )
        ):
            recovered = _recovery_row(candidate, archived)
            ledger.append_status(status_path, FIELDNAMES, recovered)
            latest[candidate["paperid"]] = {
                name: str(recovered.get(name, "")) for name in FIELDNAMES
            }

    pending = [
        candidate for candidate in candidates if candidate["paperid"] not in downloaded
    ]
    already_downloaded = len(candidates) - len(pending)
    pending_count = len(pending)
    if limit is not None:
        pending = pending[:limit]

    newly_downloaded = 0
    failed = 0

    def run(row):
        return download_one(
            paperid=row["paperid"],
            doi=row["doi"],
            title=row["title"],
            authors=row["authors"],
            year=row["year"],
            pdf_urls=row["pdf_urls"],
            timeout_seconds=timeout_seconds,
            max_pdf_mb=max_pdf_mb,
            europe_pmc_url=europe_pmc_url,
        )

    for row in ledger.bounded_results(run, pending, concurrency):
        pdf_bytes = row.pop("_pdf_bytes_data", None)
        if pdf_bytes is not None:
            try:
                archive_path, member = _archive_pdf(
                    archive_dir=pdf_archive_dir,
                    paperid=str(row["paperid"]),
                    doi=str(row["doi"]),
                    pdf_bytes=pdf_bytes,
                    archive_index=archived_count,
                    batch_size=pdf_batch_size,
                )
            except Exception as exc:
                row["error"] = f"archive_error: {exc}"
            else:
                archived_count += 1
                row["pdf_archive_path"] = archive_path
                row["pdf_archive_member"] = member
                row["downloaded"] = True
                downloaded[str(row["paperid"])] = (
                    archive_path,
                    member,
                    len(pdf_bytes),
                )
                newly_downloaded += 1
        if not ledger.truthy(row["downloaded"]):
            failed += 1
        ledger.append_status(status_path, FIELDNAMES, row)

    latest = latest_status(status_path)
    _refresh_failed(
        failed_path=failed_path,
        candidates=candidates,
        latest=latest,
        downloaded=downloaded,
    )
    return {
        "candidates": len(candidates),
        "downloaded": newly_downloaded,
        "failed": failed,
        "skipped": already_downloaded,
        "deferred": pending_count - len(pending),
    }
