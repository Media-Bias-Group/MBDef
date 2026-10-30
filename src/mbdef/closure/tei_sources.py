"""Locate and cache one GROBID TEI per MBDef-Gold paper.

Priority: `<archive>/paper_<paper_id>.grobid.tei.xml`, then `<grobid_files>/doi_<doi with '/'->'_'>.grobid.tei.xml`, then the `<source_zip>` member `<quote(paper_id)>.tei.xml`. Every folder is caller-supplied (a folder can be a read-only network mount that is offline, so all reads are exception-guarded) and the winner is cached in the caller's `cache` folder.
"""

import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from urllib.parse import quote

from mbdef.collect.identity import TEI, read_receipt

_ZIPS: dict[tuple, zipfile.ZipFile] = {}


def cache_stem(paper_id: str) -> str:
    stem = quote(paper_id, safe="")
    return (
        stem
        if len(stem) <= 200
        else stem[:140] + "__" + hashlib.sha256(paper_id.encode()).hexdigest()
    )


def _validate_seed(data: bytes, paper: dict, kind: str) -> str:
    """Refuse a TEI whose DOI names another work; return why a DOI was set aside ("" if none was)."""
    import io

    from mbdef.collect.grobid import doi_evidence
    from mbdef.collect.identity import normalize_doi, parse_paper_header

    expected = normalize_doi(
        (paper.get("metadata") or {}).get("doi")
        or (paper["paper_id"][4:] if paper["paper_id"].startswith("doi:") else "")
    )
    observed = normalize_doi(parse_paper_header(io.BytesIO(data))["doi"])
    note = ""
    if expected and observed and observed != expected:
        # The rule collected PDFs use: a template, cut-off or glued DOI says nothing about the work.
        why = doi_evidence(expected, observed)
        if why:
            note = f"DOI {observed} in the source is no evidence ({why}); catalogue DOI {expected}"
            observed = ""
    if expected and (
        (observed and expected != observed) or (kind == "grobid_files" and not observed)
    ):
        raise ValueError(
            f"source identity not verified: expected DOI {expected}, observed {observed}"
        )
    return note


@dataclass(frozen=True)
class Sources:
    archive: Path
    grobid_files: Path
    source_zip: Path
    cache: Path
    directories: tuple[Path, ...] = ()

    @cached_property
    def files(self) -> dict[str, list[Path]]:
        found: dict[str, list[Path]] = {}
        for root in self.directories:
            for path in sorted(root.rglob("*.xml")):
                found.setdefault(path.name, []).append(path)
        return found


def _zip(path: Path) -> zipfile.ZipFile:
    """One open handle per archive path; 20 papers come from the same zip."""
    stat = path.stat()
    key = (path.resolve(), stat.st_size, stat.st_mtime_ns)
    handle = _ZIPS.get(key)
    if handle is None:
        for previous in list(_ZIPS):
            if previous[0] == key[0]:
                _ZIPS.pop(previous).close()
        handle = _ZIPS[key] = zipfile.ZipFile(path)
    return handle


def candidates(paper: dict, sources: Sources) -> list[tuple[str, str]]:
    paper_id = paper["paper_id"]
    stem = quote(paper_id, safe="")
    found = [("archive", str(sources.archive / f"paper_{stem}.grobid.tei.xml"))]
    names = {f"{stem}.tei.xml", f"{stem}.xml", f"{stem}.grobid.tei.xml"}
    if "/" not in paper_id and "\\" not in paper_id:
        names |= {f"{paper_id}.tei.xml", f"{paper_id}.xml"}
    for name in sorted(names):
        paths = sources.files.get(name, [])
        if len(paths) > 1:
            raise ValueError(f"ambiguous seed source filename: {name}")
        found.extend(("directory", str(path)) for path in paths)
    doi = (paper.get("metadata") or {}).get("doi") or (
        paper_id[4:] if paper_id.startswith("doi:") else ""
    )
    for value in dict.fromkeys(v for v in (doi, doi.lower()) if v):
        name = f"doi_{value.replace('/', '_')}.grobid.tei.xml"
        found.append(("grobid_files", str(sources.grobid_files / name)))
    if sources.source_zip.is_file():
        found.append(("paper_sources_zip", f"{quote(paper_id, safe='')}.tei.xml"))
    return found


def load_tei(
    paper: dict, sources: Sources, refresh: bool = False
) -> tuple[bytes, dict]:
    """Return (TEI bytes, provenance); provenance carries a bare file name, never a path."""
    stem = cache_stem(paper["paper_id"])
    cached, sidecar = sources.cache / f"{stem}.tei.xml", sources.cache / f"{stem}.json"
    source_key = [
        str(path.resolve())
        for path in (
            sources.archive,
            sources.grobid_files,
            sources.source_zip,
            *sources.directories,
        )
    ]
    if not refresh and cached.is_file() and sidecar.is_file():
        try:
            data = cached.read_bytes()
            provenance = json.loads(sidecar.read_text(encoding="utf-8"))
            path = Path(provenance["source_path"])
            stat = path.stat()
            stamp = [stat.st_size, stat.st_mtime_ns]
            if (
                provenance.get("source_key") == source_key
                and provenance.get("source_stamp") == stamp
                and provenance.get("sha256") == hashlib.sha256(data).hexdigest()
                and (
                    provenance.get("kind") == "paper_sources_zip"
                    or not read_receipt(path)
                )
            ):
                provenance["doi_note"] = _validate_seed(data, paper, provenance["kind"])
                return data, provenance
        except OSError, ValueError, KeyError:
            pass
    errors: list[str] = []
    for kind, locator in candidates(paper, sources):
        try:
            if kind == "paper_sources_zip":
                source_path = sources.source_zip
                data = _zip(sources.source_zip).read(locator)
                name = f"{sources.source_zip.name}:{locator}"
            else:
                source_path = Path(locator)
                if read_receipt(source_path):
                    raise ValueError(
                        "acquired paper cannot be used as a seed (depth limit 1)"
                    )
                data = source_path.read_bytes()
                name = Path(locator).name
            parsed = ET.fromstring(data)
            if parsed.tag != f"{TEI}TEI" or parsed.find(f"{TEI}teiHeader") is None:
                raise ValueError("invalid source TEI")
            doi_note = _validate_seed(data, paper, kind)
            stat = source_path.stat()
        except (
            OSError,
            ValueError,
            KeyError,
            ET.ParseError,
            zipfile.BadZipFile,
        ) as exc:
            errors.append(f"{kind}: {exc!r}")
            continue
        provenance = {
            "kind": kind,
            "file": name,
            "errors": errors,
            "doi_note": doi_note,
            "source_path": str(source_path.resolve()),
            "source_key": source_key,
            "source_stamp": [stat.st_size, stat.st_mtime_ns],
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        try:
            sources.cache.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(data)
            sidecar.write_text(
                json.dumps(provenance, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        except OSError as exc:  # a read-only cache must not fail the run
            errors.append(f"cache: {exc!r}")
        return data, provenance
    raise FileNotFoundError(f"No TEI for {paper['paper_id']}: {errors}")
