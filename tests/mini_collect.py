"""Tiny inputs for the collect tests: a collection queue, PDF stubs and GROBID TEI files.

Everything here is written out literally (the queue's columns, the TEI layout GROBID returns), so the tests do not borrow the code's own definitions of them.
"""

import csv
import hashlib
import json
import subprocess
import zipfile
from email import policy
from email.parser import BytesParser
from pathlib import Path
from xml.sax.saxutils import escape

from cli_helpers import write_env
from http_replay import NO_NETWORK, Answer

REPO = Path(__file__).resolve().parents[1]

# The header of the collection queue that the closure step's corpus check writes.
QUEUE_FIELDS = [
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
# The header of the download ledger that `collect download` writes and `collect grobid` reads.
DOWNLOAD_FIELDS = [
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
EMAIL = "research@example.org"


def write_rows(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{name: "" for name in fields} | row for row in rows])


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def head() -> str | None:
    """The commit checked out in this repository, which receipts must name; None outside a checkout (a release unpacked as plain files), where receipts record no commit."""
    done = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
    )
    return done.stdout.strip() if done.returncode == 0 else None


def read_receipts(folder: Path) -> list[dict]:
    lines = (folder / "receipts.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def setup(tmp_path: Path, queue: list[dict] | None = None, **env: str) -> tuple:
    """A work folder with `.env` and a data root whose collect folder holds `queue`.

    Returns (work folder, the data root's collect folder, extra environment for `run_cli`).
    """
    work, root = tmp_path / "work", tmp_path / "data-root"
    work.mkdir()
    write_env(work, DATA_ROOT=str(root), **({"UNPAYWALL_EMAIL": EMAIL} | env))
    folder = root / "collect"
    if queue is not None:
        write_rows(folder / "citation-papers-to-collect.csv", queue, QUEUE_FIELDS)
    return work, folder, dict(NO_NETWORK)


def tei(
    *,
    title: str = "",
    surnames: tuple[str, ...] = (),
    year: str = "",
    doi: str = "",
    body: str = "Full document text.",
    header_title: str | None = None,
) -> bytes:
    """A TEI file as GROBID writes one: the title page in the header, then the body."""
    authors = "".join(
        f"<author><persName><surname>{escape(s)}</surname></persName></author>"
        for s in surnames
    )
    date = f'<monogr><imprint><date when="{year}"/></imprint></monogr>' if year else ""
    idno = f'<idno type="DOI">{escape(doi)}</idno>' if doi else ""
    stated = title if header_title is None else header_title
    analytic = (
        f'<title level="a" type="main">{escape(stated)}</title>' if stated else ""
    )
    return (
        '<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc>'
        f"<titleStmt><title>{escape(stated)}</title></titleStmt><sourceDesc>"
        f"<biblStruct><analytic>{analytic}{authors}</analytic>{date}{idno}"
        "</biblStruct></sourceDesc></fileDesc></teiHeader>"
        f"<text><body><p>{escape(body)}</p></body></text></TEI>"
    ).encode()


def form_parts(request) -> dict[str, tuple[str | None, str, bytes]]:
    """The parts of a multipart form request: name -> (file name, content type, bytes)."""
    message = BytesParser(policy=policy.HTTP).parsebytes(
        f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode()
        + request.body
    )
    parts = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        payload = part.get_payload(decode=True)
        assert isinstance(name, str) and isinstance(payload, bytes)
        parts[name] = (part.get_filename(), part.get_content_type(), payload)
    return parts


def fake_grobid(teis: dict[str, bytes]):
    """A GROBID that answers each uploaded `<paper id>.pdf` with that paper's TEI from `teis`."""

    def convert(request) -> Answer:
        filename, _, _ = form_parts(request)["input"]
        assert filename is not None
        return Answer(body=teis[filename.removesuffix(".pdf")])

    return convert


def downloads(folder: Path, papers: dict[str, dict], pdfs: dict[str, bytes]) -> Path:
    """A download ledger and one zip holding the PDF of each paper id (`papers` gives its row)."""
    archive = pdf_archive(
        folder / "pdf_batches/pdf_batch_000001.zip",
        {f"{paperid[:2].lower()}/{paperid}.pdf": pdfs[paperid] for paperid in papers},
    )
    rows = [
        {
            "paperid": paperid,
            "pdf_archive_path": str(archive),
            "pdf_archive_member": f"{paperid[:2].lower()}/{paperid}.pdf",
            "downloaded": "True",
        }
        | row
        for paperid, row in papers.items()
    ]
    path = folder / "pdf_download_status.csv"
    write_rows(path, rows, DOWNLOAD_FIELDS)
    return path


def pdf_archive(path: Path, members: dict[str, bytes]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return path
