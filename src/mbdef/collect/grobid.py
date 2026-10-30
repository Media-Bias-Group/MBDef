"""`collect grobid`: convert the downloaded PDFs to TEI with GROBID, keeping only the paper asked for.

A running GROBID service converts each downloaded PDF (the command never starts one; docs/step-1-2-papers/collect.md says how). The TEI is kept only if it has body text and its title page identifies the requested work (`identity_match`, below): the same DOI; or, if the PDF shows no usable DOI, the same title, a shared author surname and the year (or no year, or one year off); or the requested title and an author opening the text when GROBID read no title; or a reviewed verdict of the closure step's PDF identity check bound to these exact PDF bytes. A different DOI on the title page rejects the PDF. Each kept TEI is written as `grobid/<xx>/work_<sha256>.grobid.tei.xml` with a receipt `….closure.json` beside it that binds the queue row, the PDF's and the TEI's sha256 and the rule that matched. Writes `grobid_status.csv` and `failed_grobid.csv`.
"""

import hashlib
import io
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote, unquote

from curl_cffi import CurlMime
from curl_cffi import requests as curl_requests
from curl_cffi.requests.exceptions import RequestException

from mbdef import files
from mbdef.collect import ledger
from mbdef.collect.download import FIELDNAMES as DOWNLOAD_FIELDS
from mbdef.collect.download import STATUS as DOWNLOADS
from mbdef.collect.identity import (
    RECEIPT_SCHEMA,
    TEI,
    normalize_doi,
    parse_paper_header,
    queue_surnames,
    read_receipt,
    receipt_path,
    unicode_words,
    valid_fulltext,
)

# Pinned defaults. The released closure ran GROBID 0.9.1 with 8 and 10 requests at a time (on two machines).
GROBID_URL = "http://localhost:8070"
CONCURRENCY = (
    10  # parallel GROBID requests, as the run that wrote the released TEI folder
)
TIMEOUT_SECONDS = 180.0  # per PDF
CONSOLIDATE_HEADER = (
    False  # GROBID's header consolidation looks the header up online; off
)
GROBID_IMAGES = ("grobid/grobid:0.9.1-full", "grobid/grobid:0.9.1-crf")  # GPU, CPU-only

STATUS = "grobid_status.csv"
FAILED = "failed_grobid.csv"
TEI_FOLDER = "grobid"
FIELDNAMES = [
    "paperid",
    "doi",
    "pdf_archive_path",
    "pdf_archive_member",
    "grobid_xml_path",
    "converted",
    "status_code",
    "error",
    "elapsed_seconds",
    "attempted_at",
]


def command(
    step_folder: str | None,
    downloads: str | None,
    identity_verdicts: str | None,
    grobid_url: str,
    concurrency: int,
    timeout_seconds: float,
    consolidate_header: bool,
    limit: int | None,
) -> int:
    """The `collect grobid` command: convert, print the counts, append the receipt."""
    folder = ledger.folder(step_folder)
    downloads_path = Path(downloads) if downloads else folder / DOWNLOADS
    status, failed, tei = folder / STATUS, folder / FAILED, folder / TEI_FOLDER
    verdicts = (
        {
            accepted["paperid"]: accepted
            for accepted in json.loads(Path(identity_verdicts).read_text())["accepted"]
        }
        if identity_verdicts
        else None
    )
    summary = run_grobid_on_pdf_archives(
        pdf_download_status_path=downloads_path,
        xml_dir=tei,
        status_path=status,
        failed_path=failed,
        grobid_url=grobid_url,
        concurrency=concurrency,
        grobid_timeout_seconds=timeout_seconds,
        consolidate_header=consolidate_header,
        limit=limit,
        identity_verdicts=verdicts,
    )
    settings = {
        "grobid_url": grobid_url,
        "grobid_version": grobid_version(grobid_url)
        if summary["converted"] + summary["failed"]
        else None,
        "concurrency": concurrency,
        "timeout_seconds": timeout_seconds,
        "consolidate_header": consolidate_header,
        "limit": limit,
    }
    inputs = {"downloads": downloads_path}
    if identity_verdicts:
        inputs["identity_verdicts"] = Path(identity_verdicts)
    outputs = {"status": status, "failed": failed, "tei": tei}
    ledger.append_receipt(folder, "collect grobid", settings, inputs, outputs, summary)
    ledger.report(summary, outputs)
    if summary["failed"]:
        raise SystemExit(
            f"{summary['failed']} PDFs were not converted; {failed} says why. "
            "Rerun to retry them."
        )
    return 0


def _grobid_alive(grobid_url: str, timeout_seconds: float = 5.0) -> bool:
    try:
        response = curl_requests.get(
            grobid_url.rstrip("/") + "/api/isalive", timeout=timeout_seconds
        )
    except RequestException:
        return False
    return response.status_code == 200 and response.text.strip().lower() in {
        "true",
        "1",
        "ok",
    }


def grobid_version(grobid_url: str) -> str:
    """The version the GROBID service reports, for the receipt ("unknown" if it does not say)."""
    try:
        response = curl_requests.get(grobid_url.rstrip("/") + "/api/version", timeout=5)
    except RequestException:
        return "unknown"
    return response.text.strip() if response.status_code == 200 else "unknown"


def tei_path(xml_dir: Path, *, paperid: str, doi: str) -> Path:
    """Where the TEI of one queue row goes: named by the hash of its paper id and DOI."""
    # Provider discovery can map several queue rows to one DOI. Each receipt owns its queue row and PDF bytes; sharing a DOI path makes those owners oscillate.
    key = json.dumps([paperid, normalize_doi(doi)], ensure_ascii=False)
    identity = "work_" + hashlib.sha256(key.encode()).hexdigest()
    return xml_dir / identity[-2:] / f"{identity}.grobid.tei.xml"


# DOIs on a PDF's first page that say nothing about which work the PDF is (a probe of all 138 DOI conflicts of the released closure's collection: 79 had the cited title and a cited author).
_TEMPLATE_DOI = re.compile(
    r"n{4,}|x{3,}|y{3,}|z{4,}|^0+\.0+$|^1122445\.1122456$|\.doi$|^[a-z]+1010000$|vxix|^v1/p\d\d$"
)
_DATA_DOI_PREFIXES = (
    "10.5281/",
    "10.6084/",
    "10.7910/",
    "10.21979/",
    "10.17605/",
    "10.17632/",
    "10.5061/",
    "10.3886/",
)
_PREPRINT_DOI = re.compile(
    r"^10\.(?:48550/arxiv\.|2139/ssrn\.|13140/rg\.|31234/|31235/|31219/|21203/rs\.|20944/preprints"
    r"|1590/scielopreprints|1101/\d{4}\.\d\d\.\d\d)"
)


def doi_evidence(expected: str, observed: str) -> str:
    """Why an observed DOI that differs from the requested one is no evidence either way ("" if it is evidence)."""
    suffix = observed.split("/", 1)[1] if "/" in observed else ""
    if (
        expected.startswith(observed)
        or observed.endswith(("-", ".", "/"))
        or "://" in observed
        or not re.fullmatch(r"10\.\d{4,9}/[^\s,()]+", observed)
    ):
        # 10.1002/asi; 10.1051/,00063(2017); 10.1007/978-3-; a web address read into it
        return "cut short or garbled"
    if observed.startswith(expected):
        rest = observed[len(expected) :]
        if re.fullmatch(r"[._-](?:n?\d+|ch\d+)", rest):
            # a chapter or part of the requested work (book 10.4324/9781003163237, PDF ...-11) is a different work
            return ""
        # 10.1177/1464884915610988jou.sagepub.com, 10.1073/pnas.2209384120 1of5
        return "requested DOI with text glued on"
    if re.sub(r"[^a-z0-9]", "", observed) == re.sub(r"[^a-z0-9]", "", expected):
        return "same DOI, other punctuation"  # 10.11591/ijeecs.v22i3 vs v22.i3
    if _TEMPLATE_DOI.search(suffix):
        # 10.1145/nnnnnnn.nnnnnnn, ACM's sample 10.1145/1122445.1122456, 10.1109/access.2017.doi
        return "template DOI"
    if observed.startswith(_DATA_DOI_PREFIXES) and not expected.startswith(
        _DATA_DOI_PREFIXES
    ):
        # the paper cites its data on page 1 (Zenodo, figshare, Dataverse, OSF, ...)
        return "dataset DOI"
    if _PREPRINT_DOI.match(observed) or _PREPRINT_DOI.match(expected):
        # arXiv / SSRN / ... version of the same work (the closure's corpus check treats arXiv DOIs the same way)
        return "preprint DOI"
    return ""


def identity_match(
    payload: bytes,
    doi: str,
    title: str,
    authors: str,
    year: str,
    llm_verdict: dict | None = None,
    pdf_sha256: str = "",
) -> str:
    """The rule by which a converted TEI is the requested work; raises ValueError if none holds."""
    header = parse_paper_header(io.BytesIO(payload))
    expected, observed = normalize_doi(doi), normalize_doi(header["doi"])
    if observed and observed.rstrip(").,;:]([>") == expected:
        observed = expected  # trailing punctuation GROBID picked up with the DOI
    if (
        expected
        and observed
        and observed != expected
        and doi_evidence(expected, observed)
    ):
        observed = (
            ""  # no evidence either way: the title, year and author checks below decide
        )
    if expected and observed:
        if expected != observed:
            raise ValueError(
                "grobid_identity_conflict: DOI differs from requested work"
            )
        return "doi"
    shared_author = queue_surnames(authors) & queue_surnames(
        "; ".join(header["authors"])
    )
    if (
        title
        and unicode_words(title) == unicode_words(header["title"] or "")
        and shared_author
    ):
        if header["year"] and header["year"] in year.split("; "):
            return "title_year_author"
        if not header["year"]:
            return "title_author_year_missing"
        years = {int(y) for y in year.split("; ") if y.strip().isdigit()}
        if str(header["year"]).isdigit() and any(
            abs(int(header["year"]) - y) == 1 for y in years
        ):
            return "title_author_year_off_by_one"
    if title and not (header["title"] or "").strip():
        # No title parsed on page 1: the requested title and an author surname must open the text.
        opening = f" {unicode_words(_opening_text(payload))} "
        if (
            unicode_words(title)
            and f" {unicode_words(title)} " in opening
            and any(f" {s} " in opening for s in queue_surnames(authors))
        ):
            return "title_in_text"
    if (
        llm_verdict
        and llm_verdict.get("verdict") == "same_work"
        and llm_verdict.get("pdf_sha256") == pdf_sha256
    ):
        # A reviewed verdict of the PDF identity check (the closure step) bound to these exact PDF bytes; a DOI conflict raised above.
        return "llm_verified"
    raise ValueError("grobid_identity_unverified: no matching DOI or title/year/author")


def _opening_text(payload: bytes, chars: int = 4000) -> str:
    root = ET.fromstring(payload)
    parts = []
    for tag in ("teiHeader", "text"):
        node = root.find(f"{TEI}{tag}")
        if node is not None:
            parts.append(" ".join("".join(node.itertext()).split()))
    return " ".join(parts)[:chars]


def _read_download_status(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    frame = ledger.latest_by_paperid(
        ledger.read_csv(path, DOWNLOAD_FIELDS, optional=("title", "authors", "year"))
    )
    rows: list[dict[str, str]] = []
    for row in frame:
        if not ledger.truthy(row["downloaded"]):
            continue
        rows.append(
            {
                "paperid": row["paperid"],
                "doi": row["doi"],
                "title": row["title"],
                "authors": row["authors"],
                "year": row["year"],
                "pdf_archive_path": row["pdf_archive_path"],
                "pdf_archive_member": row["pdf_archive_member"],
            }
        )
    return rows


def load_converted(status_path: Path) -> dict[str, dict[str, str]]:
    if not status_path.exists():
        return {}
    frame = ledger.latest_by_paperid(ledger.read_csv(status_path, FIELDNAMES))
    return {row["paperid"]: row for row in frame}


def _is_complete(
    row: dict[str, str], latest: dict[str, dict[str, str]], xml_dir: Path
) -> bool:
    path = tei_path(xml_dir, paperid=row["paperid"], doi=row["doi"])
    if not path.exists():
        # Reuse pre-upgrade conversions only after the same receipt/hash checks.
        previous = (latest.get(row["paperid"]) or {}).get("grobid_xml_path")
        if previous and Path(previous).resolve().is_relative_to(xml_dir.resolve()):
            path = Path(previous)
    try:
        receipt = read_receipt(path)
        if (
            not receipt
            or receipt["paperid"] != row["paperid"]
            or receipt["requested_doi"] != normalize_doi(row["doi"])
        ):
            return False
        if any(
            receipt.get(key, "") != row.get(key, "")
            for key in ("title", "authors", "year")
        ):
            return False
        with zipfile.ZipFile(row["pdf_archive_path"]) as archive:
            return (
                hashlib.sha256(archive.read(row["pdf_archive_member"])).hexdigest()
                == receipt["pdf_sha256"]
            )
    except OSError, ValueError, KeyError, zipfile.BadZipFile:
        return False


def process_fulltext(
    pdf_bytes: bytes,
    filename: str,
    grobid_url: str,
    timeout_seconds: float,
    consolidate_header: bool,
):
    """GROBID's answer to one PDF (its TEI in `.content` when the status is 200); citations are never consolidated."""
    multipart = CurlMime()
    multipart.addpart(
        name="input",
        filename=filename,
        content_type="application/pdf",
        data=pdf_bytes,
    )
    try:
        return curl_requests.post(
            grobid_url.rstrip("/") + "/api/processFulltextDocument",
            multipart=multipart,
            data={
                "consolidateHeader": "1" if consolidate_header else "0",
                "consolidateCitations": "0",
            },
            timeout=(10, timeout_seconds),
        )
    finally:
        multipart.close()


def convert_folder(
    pdfs: Path,
    out: Path,
    grobid_url: str = GROBID_URL,
    paper_ids: list[str] | None = None,
    concurrency: int = CONCURRENCY,
    timeout_seconds: float = TIMEOUT_SECONDS,
) -> dict:
    """Convert every `<paper id>.pdf` of the folder PDFS (or those of PAPER_IDS) with a running GROBID into `OUT/<paper id>.tei.xml`, for `extract run --pdfs`. Unlike `collect grobid` there is no queue and no identity check: the file name is the paper id. A PDF that GROBID cannot convert is reported and left out; a requested paper that fails, or a folder where nothing converts, stops the run. Returns what the receipt records."""
    if not _grobid_alive(grobid_url):
        raise SystemExit(
            f"No GROBID answers at {grobid_url}. Start one, for example "
            f"`docker run --rm --init -p 8070:8070 {GROBID_IMAGES[1]}` (docs/step-1-2-papers/collect.md), "
            "or give the URL of yours with --grobid-url."
        )
    found = {unquote(p.name.removesuffix(".pdf")): p for p in Path(pdfs).glob("*.pdf")}
    wanted = paper_ids or sorted(found)
    missing = [p for p in wanted if p not in found]
    if missing or not wanted:
        raise SystemExit(
            f"{pdfs} has no <paper id>.pdf for: {', '.join(missing) or 'any paper'}"
        )
    out.mkdir(parents=True, exist_ok=True)

    def convert(paper_id: str) -> str | None:
        """None if converted, else why not."""
        try:
            response = process_fulltext(
                found[paper_id].read_bytes(),
                found[paper_id].name,
                grobid_url,
                timeout_seconds,
                CONSOLIDATE_HEADER,
            )
        except RequestException as error:
            return str(error)
        if response.status_code != 200:
            return f"GROBID status {response.status_code}: {response.text[:200]}"
        if not valid_fulltext(response.content, require_title=False):
            return "GROBID returned no body text"
        (out / f"{quote(paper_id, safe='')}.tei.xml").write_bytes(response.content)
        return None

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        failed = {
            paper_id: problem
            for paper_id, problem in zip(wanted, pool.map(convert, wanted), strict=True)
            if problem
        }
    for paper_id, problem in failed.items():
        print(
            f"GROBID did not convert {found[paper_id].name}: {problem}", file=sys.stderr
        )
    if failed and (paper_ids or len(failed) == len(wanted)):
        raise SystemExit(f"{len(failed)} of {len(wanted)} PDFs were not converted.")
    return {
        "url": grobid_url,
        "version": grobid_version(grobid_url),
        "consolidate_header": CONSOLIDATE_HEADER,
        "pdfs": len(wanted),
        "converted": len(wanted) - len(failed),
        "failed": failed,
    }


def convert_one(
    *,
    paperid: str,
    doi: str,
    pdf_archive_path: str,
    pdf_archive_member: str,
    xml_dir: Path,
    grobid_url: str,
    grobid_timeout_seconds: float,
    consolidate_header: bool,
    title: str = "",
    authors: str = "",
    year: str = "",
    llm_verdict: dict | None = None,
) -> dict[str, object]:
    """Convert one archived PDF and write its TEI and receipt if it is the requested work."""
    started = time.monotonic()
    xml_path = tei_path(xml_dir, paperid=paperid, doi=doi)
    row: dict[str, object] = {
        "paperid": paperid,
        "doi": doi,
        "pdf_archive_path": pdf_archive_path,
        "pdf_archive_member": pdf_archive_member,
        "grobid_xml_path": str(xml_path),
        "converted": False,
        "status_code": "",
        "error": "",
        "elapsed_seconds": "",
        "attempted_at": ledger.utc_now(),
    }
    try:
        with zipfile.ZipFile(pdf_archive_path) as archive:
            pdf_bytes = archive.read(pdf_archive_member)
        response = process_fulltext(
            pdf_bytes,
            f"{paperid}.pdf",
            grobid_url,
            grobid_timeout_seconds,
            consolidate_header,
        )
        row["status_code"] = response.status_code
        payload = response.content
        if response.status_code != 200:
            row["error"] = (
                f"grobid_status_{response.status_code}: {response.text[:500]}"
            )
        elif not valid_fulltext(payload, require_title=False):
            row["error"] = "invalid_tei_xml"
        else:
            pdf_sha256 = hashlib.sha256(pdf_bytes).hexdigest()
            match = identity_match(
                payload, doi, title, authors, year, llm_verdict, pdf_sha256
            )
            header = parse_paper_header(io.BytesIO(payload))
            receipt = {
                "schema": RECEIPT_SCHEMA,
                "depth": 1,
                "paperid": paperid,
                "requested_doi": normalize_doi(doi),
                "observed_doi": normalize_doi(header["doi"]),
                "title": title,
                "authors": authors,
                "year": year,
                "identity_match": match,
                "observed_title": header["title"] or "",
                "observed_year": header["year"] or "",
                "pdf_sha256": pdf_sha256,
                "tei_sha256": hashlib.sha256(payload).hexdigest(),
                "pdf_archive_path": pdf_archive_path,
                "pdf_archive_member": pdf_archive_member,
                "grobid_url": grobid_url,
            }
            if match == "llm_verified" and llm_verdict:
                receipt["identity_llm"] = {
                    k: llm_verdict.get(k)
                    for k in (
                        "model",
                        "revision",
                        "prompt_version",
                        "verdict",
                        "pdf_title",
                        "evidence",
                        "reasoning",
                        "checks",
                    )
                }
            files.write_atomic(
                {
                    xml_path: payload,
                    receipt_path(xml_path): json.dumps(receipt, sort_keys=True) + "\n",
                }
            )
            row["converted"] = True
    except Exception as exc:
        row["error"] = str(exc)
    finally:
        row["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return row


def _refresh_failed(
    *,
    failed_path: Path,
    downloaded: list[dict[str, str]],
    latest: dict[str, dict[str, str]],
    xml_dir: Path,
) -> None:
    rows = []
    for download in downloaded:
        if _is_complete(download, latest, xml_dir):
            continue
        status = latest.get(download["paperid"])
        if status and not ledger.truthy(status["converted"]):
            rows.append({name: status[name] for name in FIELDNAMES})
    ledger.replace_csv(failed_path, rows, FIELDNAMES)


def run_grobid_on_pdf_archives(
    *,
    pdf_download_status_path: str | Path,
    xml_dir: str | Path,
    status_path: str | Path,
    failed_path: str | Path,
    grobid_url: str = GROBID_URL,
    concurrency: int = CONCURRENCY,
    grobid_timeout_seconds: float = TIMEOUT_SECONDS,
    consolidate_header: bool = CONSOLIDATE_HEADER,
    limit: int | None = None,
    identity_verdicts: dict[str, dict] | None = None,
) -> dict[str, int]:
    if concurrency <= 0 or grobid_timeout_seconds <= 0:
        raise ValueError("GROBID concurrency and timeout must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    pdf_download_status_path = Path(pdf_download_status_path)
    xml_dir = Path(xml_dir)
    status_path = Path(status_path)
    failed_path = Path(failed_path)
    xml_dir.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)

    downloaded = _read_download_status(pdf_download_status_path)
    latest = load_converted(status_path)
    pending = [row for row in downloaded if not _is_complete(row, latest, xml_dir)]
    already_converted = len(downloaded) - len(pending)
    pending_count = len(pending)
    if limit is not None:
        pending = pending[:limit]

    if not pending:
        _refresh_failed(
            failed_path=failed_path,
            downloaded=downloaded,
            latest=latest,
            xml_dir=xml_dir,
        )
        return {
            "downloaded": len(downloaded),
            "converted": 0,
            "failed": 0,
            "skipped": len(downloaded),
            "deferred": 0,
        }
    if not _grobid_alive(grobid_url):
        raise SystemExit(
            f"GROBID is not alive at {grobid_url}. Start it first, for example with "
            f"`docker run --rm --init --ulimit core=0 -p 8070:8070 {GROBID_IMAGES[1]}` "
            "(docs/step-1-2-papers/collect.md), or pass its address with --grobid-url."
        )

    converted = 0
    failed = 0

    def run(row):
        return convert_one(
            paperid=row["paperid"],
            doi=row["doi"],
            title=row["title"],
            authors=row["authors"],
            year=row["year"],
            pdf_archive_path=row["pdf_archive_path"],
            pdf_archive_member=row["pdf_archive_member"],
            xml_dir=xml_dir,
            grobid_url=grobid_url,
            grobid_timeout_seconds=grobid_timeout_seconds,
            consolidate_header=consolidate_header,
            llm_verdict=(identity_verdicts or {}).get(row["paperid"]),
        )

    for row in ledger.bounded_results(run, pending, concurrency):
        ledger.append_status(status_path, FIELDNAMES, row)
        if ledger.truthy(row["converted"]):
            converted += 1
        else:
            failed += 1

    latest = load_converted(status_path)
    _refresh_failed(
        failed_path=failed_path,
        downloaded=downloaded,
        latest=latest,
        xml_dir=xml_dir,
    )
    return {
        "downloaded": len(downloaded),
        "converted": converted,
        "failed": failed,
        "skipped": already_converted,
        "deferred": pending_count - len(pending),
    }
