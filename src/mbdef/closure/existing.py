"""Cited works found only in the project's existing corpus become closure papers.

The corpus check marks a cited work "present" when a TEI with the same DOI, or the same title and author, is in the seed corpus or in the existing corpus (`existing-corpus-tei.zip`). Works present only in the existing corpus were therefore never fetched, and never extracted either: they are not seed papers. Their TEIs already exist, so no download is needed. For each such work `add_existing`

  * takes the matched TEI from the zip and checks it against the zip's SHA-256 list,
  * names it like any closure paper: the paper id of the work's queue row (as `before-queue.csv` would list it, with queue_status "present") and the TEI path `collect.grobid.tei_path` gives,
  * writes the TEI and a receipt (schema corpus-closure/1, identity_match "existing_corpus", the corpus check's method and the zip member), never overwriting a file with other content.
"""

import hashlib
import io
import json
import zipfile
from pathlib import Path

from mbdef.collect.grobid import tei_path
from mbdef.collect.identity import (
    RECEIPT_SCHEMA,
    collection_paper_id,
    parse_paper_header,
    receipt_path,
)
from mbdef.collect.open_access import normalized_queue_doi


def joined(values) -> str:
    return "; ".join(sorted({v for v in values or [] if v}))


def queue_row(reference: dict) -> dict:
    """The work's row as the corpus check writes queue rows (corpus_check._collection_row), status "present"."""
    return {
        "title": reference.get("title") or "",
        "authors": joined(reference.get("authors")),
        "year": joined(reference.get("years")),
        "doi": joined(reference.get("dois")),
        "url": reference.get("url") or "",
    }


def paper_id(row: dict) -> str:
    """Same construction as the driver's queue ids: the work's paper id, then 16 hex digits of its row's sha256."""
    doi = normalized_queue_doi(row["doi"])
    base = {
        "paperid": collection_paper_id(
            doi=doi, title=row["title"], year=row["year"], authors=row["authors"]
        ),
        "queue_status": "present",
        "title": row["title"],
        "authors": row["authors"],
        "year": row["year"],
        "doi": doi,
        "source_url": row["url"].strip(),
    }
    return (
        base["paperid"]
        + "__"
        + hashlib.sha256(
            json.dumps(base, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()[:16]
    )


def existing_only(before: dict) -> list[tuple[int, dict]]:
    """The cited works whose only TEI is in the existing corpus, not among the seed papers' TEIs."""
    out = []
    for i, p in enumerate(before["papers"]):
        m = p["corpus_match"]
        files = m.get("files") or []
        if (
            m["status"] == "present"
            and files
            and not any("seed-tei" in f["corpus_path"] for f in files)
        ):
            out.append((i, p))
    return out


def add_existing(
    before: dict,
    zip_path: Path,
    sha256_list: Path,
    grobid_dir: Path,
    exclude: set[str],
) -> list[dict]:
    """Write the TEI and receipt of each existing-corpus-only work into GROBID_DIR; return the list of what was added."""
    listed = {}
    for line in Path(sha256_list).read_text().splitlines():
        if line.strip():
            digest, member = line.split("  ", 1)
            listed[member] = digest
    archive = zipfile.ZipFile(zip_path)
    added = []
    for index, p in existing_only(before):
        ref, m = p["reference"], p["corpus_match"]
        member = sorted(f["file"] for f in m["files"])[0]
        tei = archive.read(member)
        digest = hashlib.sha256(tei).hexdigest()
        if listed.get(member) != digest:
            raise SystemExit(f"{member}: SHA-256 differs from the existing corpus list")
        row = queue_row(ref)
        pid = paper_id(row)
        if pid in exclude:
            continue
        doi = normalized_queue_doi(row["doi"])
        path = tei_path(Path(grobid_dir), paperid=pid, doi=doi)
        receipt_file = receipt_path(path)
        header = parse_paper_header(io.BytesIO(tei))
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "depth": 1,
            "paperid": pid,
            "requested_doi": doi,
            "title": row["title"],
            "authors": row["authors"],
            "year": row["year"],
            "identity_match": "existing_corpus",
            "corpus_match_method": m["method"],
            "tei_source": f"existing-corpus-tei.zip:{member}",
            "observed_doi": header.get("doi") or "",
            "observed_title": header.get("title") or "",
            "observed_year": header.get("year") or "",
            "tei_sha256": digest,
            "pdf_sha256": "",
            "pdf_archive_path": "",
            "pdf_archive_member": "",
            "grobid_url": "",
        }
        if path.exists() or receipt_file.exists():
            if not (
                path.read_bytes() == tei
                and json.loads(receipt_file.read_text()) == receipt
            ):
                raise SystemExit(f"{path} exists with other content")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(tei)
            receipt_file.write_text(
                json.dumps(receipt, ensure_ascii=False, indent=1, sort_keys=True) + "\n"
            )
        added.append(
            {
                "before_index": index,
                "paper_id": pid,
                "tei": str(path),
                "member": member,
                "method": m["method"],
                "cited_title": row["title"],
                "tei_title": header.get("title") or "",
                "citing_seed_papers": len({o["paper_id"] for o in p["occurrences"]}),
            }
        )
    return added


def write_added(path: Path, added: list[dict]) -> None:
    """The released byte format: indent 1, non-ASCII kept, a final newline."""
    Path(path).write_text(json.dumps(added, ensure_ascii=False, indent=1) + "\n")
