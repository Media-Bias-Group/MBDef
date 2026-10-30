"""Check every resolved citation of `data/validated_gt_citations.json` against the local corpus.

The citation-closure dataset is read-only input: membership annotations land in a separate copy (`data/validated_gt_citations_corpus.json` by default) so the closure step and the corpus-membership step stay independently validatable.

`scan_corpus` indexes the configured corpus roots from their `teiHeader` only — a directory of GROBID TEI files or a zip of TEI members (such as the MBDef-Gold source papers themselves) — collapses duplicate files into publication identities, and `annotate_corpus_membership` decides membership per *cited publication* through exact DOI, corroborated exact title and arXiv variant evidence. Fuzzy titles are optional review suggestions only. Conflicting metadata, several qualifying identities and an incomplete scan all produce `review` instead of a match, and every unmatched citation lands in a collection CSV.
"""

import json
import os
import re
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import unquote
from xml.etree.ElementTree import ParseError

from mbdef import files
from mbdef.closure.manual_citations import _display_path
from mbdef.collect import ledger
from mbdef.collect.identity import (
    clean,
    is_arxiv_doi,
    normalize_doi,
    parse_paper_header,
    read_receipt,
    unicode_words,
)
from mbdef.collect.ledger import COLLECTION_FIELDS
from mbdef.extract.records import FIELDS

CHECKER = "taxonomy_pipeline.closure.corpus_check/1"  # a recorded id: it keeps its name (files.py)
# The title threshold of the project's earlier paper matching.
TITLE_SIMILARITY = 0.92
EMPTY_ROOT = "no TEI files found"

# `data/grobid` (a read-only mount of the GROBID corpus) names every file after its DOI with *every* `/` replaced by `_`, so `10.18653/v1/2023.acl-long.486` arrives as `doi_10.18653_v1_2023.acl-long.486.grobid.tei.xml` and an underscore is ambiguous: it is either a restored separator or a literal DOI character (`10.48009/4_iis_2025_126`).
FILENAME_DOI = re.compile(r"^doi_(10\.\d{3,9})_(.+?)(?:\.grobid)?\.tei\.xml$")
# A DOI suffix rarely nests deeper than this; the cap bounds the readings per file.
FILENAME_DOI_DEPTH = 6
TEI_SUFFIX = re.compile(r"(?:\.grobid)?\.tei\.xml$|\.xml$", re.IGNORECASE)
ARXIV_REGISTRANT = "10.48550"
ARXIV_TOKEN = re.compile(r"(?:^|[^a-z0-9])arxiv(?:[^a-z0-9]|$)")
ARXIV_ID = re.compile(
    r"arxiv(?:\.org)?[ :/]*(?:abs|pdf)?/?\s*"
    r"((?:[a-z-]+(?:\.[a-z]{2})?/\d{7})|(?:\d{4}\.\d{4,5}))",
    re.IGNORECASE,
)


def arxiv_ids(*values: str | None) -> frozenset[str]:
    """arXiv identity evidence from `arxiv.org/abs/<id>` URLs and `arXiv:<id>` text."""
    found = set()
    for value in values:
        for match in ARXIV_ID.finditer(clean(value or "")):
            found.add(match.group(1).casefold())
    return frozenset(found)


def _surnames(authors) -> frozenset[str]:
    """Author evidence is the surname portion of the existing `"Surname, I."` values."""
    found = set()
    for author in authors or []:
        if not isinstance(author, str):
            continue
        surname = unicode_words(author.split(",", 1)[0])
        if surname:
            found.add(surname)
    return frozenset(found)


def _filename_dois(name: str) -> tuple[str, ...]:
    """Every plausible DOI reading of a `doi_<doi>` file name, separators restored left to right.

    The first reading restores only the registrant separator (the documented convention); each following one restores one more `_`. All of them are exact DOI strings, so a reading only ever matches a cited DOI that is spelled identically.
    """
    match = FILENAME_DOI.match(name)
    if not match:
        return ()
    registrant, remainder = match.group(1), match.group(2)
    parts = remainder.split("_", FILENAME_DOI_DEPTH - 1)
    readings: list[str] = []
    for restored in range(len(parts)):
        suffix = "/".join(parts[: restored + 1])
        if restored + 1 < len(parts):
            suffix += "_" + "_".join(parts[restored + 1 :])
        doi = normalize_doi(f"{registrant}/{suffix}")
        if doi and doi not in readings:
            readings.append(doi)
    return tuple(readings)


def _member_identity(name: str) -> tuple[str, tuple[str, ...]]:
    """`(decoded paper identifier, DOI readings)` of a zip member name.

    `data/paper_sources.zip` names every member after `quote(paper_id, safe="")`, so `doi%3A10.1145%2F3677092.tei.xml` decodes to that exact DOI while a URL or content-hash identifier decodes to something that is not a DOI and contributes no DOI reading; the decoded identifier itself stays arXiv/URL evidence.
    """
    stem = unquote(TEI_SUFFIX.sub("", Path(name).name))
    doi = normalize_doi(stem)
    return stem, (doi,) if doi else ()


def _resolved(path: Path) -> str:
    """Identity of a file/directory for dedup; falls back when the path cannot be stat'd."""
    try:
        return str(path.resolve())
    except OSError:
        return os.path.abspath(path)


def _tei_names(root: Path) -> list[str]:
    """Root-relative names of every TEI file in a directory root or zip-archive root."""
    if root.is_file():
        with zipfile.ZipFile(root) as archive:
            return sorted(
                info.filename
                for info in archive.infolist()
                if not info.is_dir() and info.filename.casefold().endswith(".xml")
            )
    if not root.is_dir():
        return []
    paths = sorted({*root.rglob("*.tei.xml"), *root.rglob("*.xml")})
    return [path.relative_to(root).as_posix() for path in paths]


def _unique_roots(corpus_paths) -> list[Path]:
    """Dedup by resolved path, caller order kept: the first root owns a shared file."""
    roots, seen = [], set()
    for value in corpus_paths:
        root = Path(value)
        key = _resolved(root)
        if key in seen:
            continue
        seen.add(key)
        roots.append(root)
    return roots


def scan_corpus(
    corpus_paths: list[Path], workers: int = 16, allow_empty_root: bool = False
) -> tuple[list[dict], list[dict]]:
    """Index every configured corpus root from its TEI headers.

    A root is either a directory of TEI files or a zip archive whose members are TEI files. Returns `(papers, scan_errors)`, both ordered by configured-root order then file. A paper row carries `corpus_path`, root-relative `file`, the header metadata, the primary `doi`, its alternative filename separator readings in `doi_readings`, the decoded archive-member identifier in `source` (`None` for a directory root), and the private `_source_order` of its root. A `scan_errors` row with `error: None` is an accepted note (an empty root under `--allow-empty-root`) that keeps the source visible with zero files; any row with a non-null `error` makes the scan incomplete, so unmatched citations become `review` rather than a definitive miss. Raises if no root holds any TEI file.
    """
    if workers < 1:
        raise ValueError(f"workers must be positive: {workers}")
    roots = _unique_roots(corpus_paths)
    if not roots:
        raise ValueError("no corpus paths configured")
    jobs: list[tuple[int, Path, str, bool]] = []
    errors: list[dict] = []
    seen_files: set[str] = set()
    for source_order, root in enumerate(roots):
        display = _display_path(root)
        member = False
        scan_failed = False
        try:
            member = root.is_file()
            found = _tei_names(root)
        except (OSError, zipfile.BadZipFile) as exc:
            errors.append(
                {
                    "_source_order": source_order,
                    "corpus_path": display,
                    "file": None,
                    "error": repr(exc),
                }
            )
            found = []
            scan_failed = True
        fresh = []
        root_key = _resolved(root) if member else ""
        for name in found:
            key = f"{root_key}:{name}" if member else _resolved(root / name)
            if key in seen_files:
                continue
            seen_files.add(key)
            fresh.append(name)
        if not found and not scan_failed:
            errors.append(
                {
                    "_source_order": source_order,
                    "corpus_path": display,
                    "file": None,
                    "error": None if allow_empty_root else EMPTY_ROOT,
                }
            )
        jobs.extend((source_order, root, name, member) for name in fresh)
    if not jobs:
        raise FileNotFoundError(
            f"no TEI files under any configured corpus root: {[_display_path(r) for r in roots]}"
        )

    worker_state = threading.local()
    archives = []
    archives_lock = threading.Lock()

    def index(job: tuple[int, Path, str, bool]) -> tuple[dict | None, dict | None]:
        source_order, root, name, member = job
        location = {
            "_source_order": source_order,
            "corpus_path": _display_path(root),
            "file": name,
        }
        try:
            if member:
                # Parse the ZIP directory once per worker, not once per member.
                # Each thread owns its handles; no mutable seek position is shared.
                if not hasattr(worker_state, "archives"):
                    worker_state.archives = {}
                archive = worker_state.archives.get(root)
                if archive is None:
                    archive = worker_state.archives[root] = zipfile.ZipFile(root)
                    with archives_lock:
                        archives.append(archive)
                with archive.open(name) as stream:
                    header = parse_paper_header(stream)
            else:
                header = parse_paper_header(root / name)
                receipt = read_receipt(root / name)
                if receipt and not header["doi"]:
                    header["doi"] = receipt["requested_doi"]
        except (OSError, ValueError, ParseError, KeyError, zipfile.BadZipFile) as exc:
            return None, {**location, "error": repr(exc)}
        header_doi = normalize_doi(header["doi"])
        source, readings = (
            _member_identity(name)
            if member
            else (None, _filename_dois(Path(name).name))
        )
        if header_doi:
            readings = ()
        return {
            **location,
            "title": header["title"],
            "authors": header["authors"],
            "year": header["year"],
            "doi": header_doi or (readings[0] if readings else None),
            # Alternative separator readings are match evidence only, never the identity.
            "doi_readings": list(readings[1:]),
            "doi_unverified": bool(readings) and not member,
            "source": source,
            "raw": header["raw"],
        }, None

    try:
        with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            results = list(pool.map(index, jobs))
    finally:
        for archive in archives:
            archive.close()
    papers = [paper for paper, _ in results if paper is not None]
    errors.extend(error for _, error in results if error is not None)
    papers.sort(key=lambda row: (row["_source_order"], row["file"]))
    errors.sort(key=lambda row: (row["_source_order"], row["file"] or ""))
    return papers, errors


def _evidence(doi: str | None, title, year, authors, raw=None, url=None) -> dict:
    """Normalized matching evidence shared by corpus identities and cited papers."""
    normalized = normalize_doi(doi)
    return {
        "dois": frozenset({normalized} if normalized else ()),
        "titles": frozenset({unicode_words(title or "")} - {""}),
        "years": frozenset({clean(year or "")} - {""}),
        "authors": _surnames(authors),
        "arxiv": arxiv_ids(normalized, raw, url),
    }


def _absorb(group: dict, evidence: dict) -> None:
    for key in ("dois", "titles", "years", "authors", "arxiv"):
        group[key] |= evidence[key]


def _conflicts(left: dict, right: dict, arxiv_exempt: bool = False) -> bool:
    """Two nonempty-but-disjoint DOI, year or author sets never auto-accept."""
    for key in ("dois", "years"):
        if left[key] and right[key] and not left[key] & right[key] and not arxiv_exempt:
            return True
    return bool(
        left["authors"] and right["authors"] and not left["authors"] & right["authors"]
    )


def _is_arxiv(group: dict) -> bool:
    return bool(group["arxiv"]) or any(is_arxiv_doi(doi) for doi in group["dois"])


def _canonical_doi(dois) -> str | None:
    published = sorted(doi for doi in dois if not is_arxiv_doi(doi))
    return (published or sorted(dois) or [None])[0]


def _new_group(**extra) -> dict:
    return {
        "dois": frozenset(),
        "titles": frozenset(),
        "years": frozenset(),
        "authors": frozenset(),
        "arxiv": frozenset(),
        **extra,
    }


def _attach(titles: dict[str, list[dict]], evidence: dict) -> dict | None:
    """A DOI-less record joins an existing group only when exactly one is compatible."""
    title = next(iter(evidence["titles"]), "")
    if not title:
        return None
    pool = [group for group in titles.get(title, []) if not _conflicts(group, evidence)]
    return pool[0] if len(pool) == 1 else None


def _index_titles(titles: dict[str, list[dict]], group: dict) -> None:
    for title in group["titles"]:
        bucket = titles.setdefault(title, [])
        if not any(known is group for known in bucket):
            bucket.append(group)


def _ambiguous_dois(corpus: list[dict]) -> frozenset[str]:
    """DOIs that several distinct corpus titles claim, so they identify no publication.

    GROBID truncates a header DOI at a line break (`10.17159/1727-`, `10.31577/cai`) and a preprint often keeps its LaTeX template's sample DOI (`10.1145/nnnnnnn.nnnnnnn`, `10.1145/1122445.1122456`). Both make one DOI key cover unrelated papers: 35 of them cover up to 22 titles each in the live corpus. Those files keep their titles and stay matchable, but the DOI is dropped instead of merging the papers into one identity.
    """
    claimed: dict[str, set[str]] = {}
    for row in corpus:
        if row.get("doi_unverified"):
            continue
        doi = normalize_doi(row["doi"])
        title = unicode_words(row["title"] or "")
        if doi and title:
            claimed.setdefault(doi, set()).add(title)
    return frozenset(doi for doi, titles in claimed.items() if len(titles) > 1)


def corpus_identities(corpus: list[dict]) -> list[dict]:
    """Collapse duplicate corpus files into publication identities, keeping every location."""
    ambiguous = _ambiguous_dois(corpus)
    identities: list[dict] = []
    by_doi: dict[str, dict] = {}
    titles: dict[str, list[dict]] = {}
    for row in sorted(
        corpus,
        key=lambda row: (
            not bool(normalize_doi(row["doi"])),
            bool(row.get("doi_unverified")),
            row["corpus_path"],
            row["file"],
        ),
    ):
        evidence = _evidence(
            row["doi"],
            row["title"],
            row["year"],
            row["authors"],
            row["raw"],
            # A zip member named after its paper id carries the arXiv/publisher URL.
            row.get("source"),
        )
        doi = normalize_doi(row["doi"])
        readings = frozenset(
            reading
            for value in row.get("doi_readings") or ()
            if (reading := normalize_doi(value))
        )
        evidence = {**evidence, "dois": (evidence["dois"] | readings) - ambiguous}
        if doi in ambiguous:
            doi = ""
        if doi and not row.get("doi_unverified"):
            identity = by_doi.get(doi)
            if identity is None:
                identity = by_doi[doi] = _new_group(
                    files=[],
                    display=[],
                    primary_dois=frozenset(),
                    unverified_dois=frozenset(),
                    verified_dois=frozenset(),
                )
                identities.append(identity)
        else:
            identity = _attach(titles, evidence)
            if identity is None:
                identity = _new_group(
                    files=[],
                    display=[],
                    primary_dois=frozenset(),
                    unverified_dois=frozenset(),
                    verified_dois=frozenset(),
                )
                identities.append(identity)
        _absorb(identity, evidence)
        identity["unverified_dois"] |= readings | (
            evidence["dois"] if row.get("doi_unverified") else frozenset()
        )
        if doi:
            identity["primary_dois"] |= {doi}
            if not row.get("doi_unverified"):
                identity["verified_dois"] |= {doi}
        _index_titles(titles, identity)
        identity["files"].append(
            {"corpus_path": row["corpus_path"], "file": row["file"]}
        )
        identity["display"].append(row)
    for identity in identities:
        identity["unverified_dois"] -= identity["verified_dois"]
        identity["files"].sort(key=lambda row: (row["corpus_path"], row["file"]))
        rows = identity.pop("display")
        identity["title"] = next((row["title"] for row in rows if row["title"]), None)
        identity["year"] = next((row["year"] for row in rows if row["year"]), None)
        # `authors` stays surname evidence; `author_names` is the reportable rendering.
        identity["author_names"] = sorted(
            {author for row in rows for author in row["authors"]}
        )
        identity["doi"] = _canonical_doi(identity["primary_dois"] or identity["dois"])
    return identities


def _validate_dataset(dataset: dict) -> list[dict]:
    """Structural validation with the offending JSON location; no partial output is written."""
    papers = dataset.get("papers")
    if not isinstance(papers, list):
        raise ValueError("dataset.papers must be a list")
    if "paper_count" in dataset and dataset["paper_count"] != len(papers):
        raise ValueError(
            f"dataset.paper_count {dataset.get('paper_count')!r} disagrees with papers[{len(papers)}]"
        )
    seen: set[str] = set()
    for index, paper in enumerate(papers):
        location = f"papers[{index}]"
        if not isinstance(paper, dict):
            raise ValueError(f"{location} must be an object")
        paper_id = paper.get("paper_id")
        if not isinstance(paper_id, str) or not paper_id:
            raise ValueError(f"{location}.paper_id must be a non-empty string")
        if paper_id in seen:
            raise ValueError(f"{location}.paper_id is duplicated: {paper_id!r}")
        seen.add(paper_id)
        if paper.get("metadata") is not None and not isinstance(
            paper["metadata"], dict
        ):
            raise ValueError(f"{location}.metadata must be an object")
        records = paper.get("definition_records")
        if not isinstance(records, list):
            raise ValueError(f"{location}.definition_records must be a list")
        for record_index, record in enumerate(records):
            record_location = f"{location}.definition_records[{record_index}]"
            if not isinstance(record, dict):
                raise ValueError(f"{record_location} must be an object")
            for field in FIELDS:
                items = record.get(field, [])
                if not isinstance(items, list):
                    raise ValueError(f"{record_location}.{field} must be a list")
                for item_index, item in enumerate(items):
                    item_location = f"{record_location}.{field}[{item_index}]"
                    if not isinstance(item, dict):
                        raise ValueError(f"{item_location} must be an object")
                    raw = item.get("citation", [])
                    if not isinstance(raw, list) or not all(
                        isinstance(value, str) for value in raw
                    ):
                        raise ValueError(
                            f"{item_location}.citation must be a list of strings"
                        )
                    citations = item.get("resolved_citations")
                    if citations is None:
                        continue
                    if not isinstance(citations, list):
                        raise ValueError(
                            f"{item_location}.resolved_citations must be a list"
                        )
                    for citation_index, citation in enumerate(citations):
                        citation_location = (
                            f"{item_location}.resolved_citations[{citation_index}]"
                        )
                        if not isinstance(citation, dict):
                            raise ValueError(f"{citation_location} must be an object")
                        references = citation.get("references")
                        if not isinstance(references, list):
                            raise ValueError(
                                f"{citation_location}.references must be a list"
                            )
                        for reference_index, reference in enumerate(references):
                            if not isinstance(reference, dict):
                                raise ValueError(
                                    f"{citation_location}.references[{reference_index}] must be an object"
                                )
    return papers


def _occurrences(dataset: dict) -> tuple[list[dict], int]:
    """Every reference occurrence, including all candidates of an ambiguous marker."""
    papers = _validate_dataset(dataset)
    occurrences: list[dict] = []
    markers_without_references = 0
    for paper in papers:
        paper_id = paper["paper_id"]
        paper_title = (paper.get("metadata") or {}).get("title") or paper.get("title")
        for record_index, record in enumerate(paper["definition_records"]):
            term = record.get("term")
            for field in FIELDS:
                for item_index, item in enumerate(record.get(field, [])):
                    for citation_index, citation in enumerate(
                        item.get("resolved_citations") or []
                    ):
                        references = citation["references"]
                        if not references:
                            markers_without_references += 1
                            continue
                        for reference_index, reference in enumerate(references):
                            occurrences.append(
                                {
                                    "paper_id": paper_id,
                                    "paper_title": paper_title,
                                    "record_index": record_index,
                                    "term": term,
                                    "field": field,
                                    "item_index": item_index,
                                    "citation_index": citation_index,
                                    "reference_index": reference_index,
                                    "raw": citation.get("raw"),
                                    "marker": citation.get("marker"),
                                    "method": citation.get("method"),
                                    "confidence": citation.get("confidence"),
                                    "year_check": citation.get("year_check"),
                                    "bibl_id": reference.get("bibl_id"),
                                    "reference": reference,
                                }
                            )
    occurrences.sort(key=_occurrence_key)
    return occurrences, markers_without_references


def _occurrence_key(occurrence: dict) -> tuple:
    return (
        occurrence["paper_id"],
        occurrence["record_index"],
        occurrence["field"],
        occurrence["item_index"],
        occurrence["citation_index"],
        occurrence["reference_index"],
    )


def cited_papers(occurrences: list[dict]) -> list[dict]:
    """Deduplicate reference occurrences into canonical cited publications."""
    papers: list[dict] = []
    by_doi: dict[str, dict] = {}
    by_text: dict[str, dict] = {}
    titles: dict[str, list[dict]] = {}
    for occurrence in sorted(
        occurrences,
        key=lambda row: (
            not bool(normalize_doi(row["reference"].get("doi"))),
            _occurrence_key(row),
        ),
    ):
        reference = occurrence["reference"]
        evidence = _evidence(
            reference.get("doi"),
            reference.get("title"),
            reference.get("year"),
            reference.get("authors"),
            reference.get("raw"),
            reference.get("url"),
        )
        doi = next(iter(evidence["dois"]), "")
        title = next(iter(evidence["titles"]), "")
        if doi:
            paper = by_doi.get(doi)
            if paper is None:
                paper = by_doi[doi] = _new_group(occurrences=[])
                papers.append(paper)
        else:
            paper = _attach(titles, evidence)
            if paper is None and title:
                paper = _new_group(occurrences=[])
                papers.append(paper)
            elif paper is None:
                # No title: URL/raw is the only identity evidence. With neither, each occurrence must remain separate instead of sharing the empty key.
                key = (reference.get("url") or "").strip() or unicode_words(
                    reference.get("raw") or ""
                )
                paper = by_text.get(key) if key else None
                if paper is None:
                    paper = _new_group(occurrences=[])
                    papers.append(paper)
                    if key:
                        by_text[key] = paper
        _absorb(paper, evidence)
        _index_titles(titles, paper)
        paper["occurrences"].append(occurrence)
    return _merge_arxiv_variants(papers)


def _merge_arxiv_variants(papers: list[dict]) -> list[dict]:
    """A preprint citation and a version-of-record citation are one canonical paper."""
    published: dict[str, list[dict]] = {}
    for paper in papers:
        if any(not is_arxiv_doi(doi) for doi in paper["dois"]):
            for title in paper["titles"]:
                published.setdefault(title, []).append(paper)
    merged: list[dict] = []
    for paper in papers:
        if not paper["dois"] or not all(is_arxiv_doi(doi) for doi in paper["dois"]):
            merged.append(paper)
            continue
        related = {
            id(target): target
            for title in paper["titles"]
            for target in published.get(title, [])
        }
        pool = [
            target
            for target in related.values()
            if target is not paper
            and target["titles"] & paper["titles"]
            and target["authors"] & paper["authors"]
        ]
        if len(pool) != 1:
            merged.append(paper)
            continue
        target = pool[0]
        _absorb(target, paper)
        target["occurrences"].extend(paper["occurrences"])
    for paper in merged:
        paper["occurrences"].sort(key=_occurrence_key)
    return merged


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    matcher = SequenceMatcher(None, left, right)
    if (
        matcher.real_quick_ratio() < TITLE_SIMILARITY
        or matcher.quick_ratio() < TITLE_SIMILARITY
    ):
        return 0.0
    return matcher.ratio()


def _best_ratio(titles, identity: dict) -> float:
    return max(
        (_similarity(title, other) for title in titles for other in identity["titles"]),
        default=0.0,
    )


def _candidate_pool(paper: dict, index: dict) -> list[tuple[dict, list[str], float]]:
    """Identities related by DOI, exact title, or a fuzzy title over the 0.92 threshold.

    Tier 4 needs exact year and overlapping surname evidence. Sweeping the union of those indexes also retains near-title conflicts that fail one guard as review suggestions.
    """
    found: dict[int, tuple[dict, set[str]]] = {}

    def note(identity: dict, evidence: str) -> None:
        found.setdefault(id(identity), (identity, set()))[1].add(evidence)

    for doi in sorted(paper["dois"]):
        for identity in index["by_doi"].get(doi, []):
            note(
                identity,
                "doi_filename" if doi in identity["unverified_dois"] else "doi",
            )
    titles = sorted(paper["titles"])
    for title in titles:
        for identity in index["by_title"].get(title, []):
            note(identity, "title_exact")
    exact_pool = [
        (identity, sorted(evidence), _best_ratio(titles, identity))
        for identity, evidence in found.values()
    ]
    if index.get("fuzzy_suggestions") and not _accept(paper, exact_pool)[0] and titles:
        related = {
            id(identity): identity
            for key in ("authors", "years")
            for value in sorted(paper[key])
            for identity in index[f"by_{key[:-1]}"].get(value, [])
        }
        for identity in related.values():
            if (
                id(identity) not in found
                and _best_ratio(titles, identity) >= TITLE_SIMILARITY
            ):
                note(identity, "title_fuzzy")
    pool = [
        (identity, sorted(evidence), _best_ratio(titles, identity))
        for identity, evidence in found.values()
    ]
    pool.sort(
        key=lambda row: (
            -row[2],
            row[0]["files"][0]["corpus_path"],
            row[0]["files"][0]["file"],
        )
    )
    return pool


def _tier2_method(paper: dict, identity: dict) -> str | None:
    year = bool(paper["years"] & identity["years"])
    author = bool(paper["authors"] & identity["authors"])
    if year and author:
        return "title_year_author_exact"
    if year:
        return "title_year_exact"
    if author:
        return "title_author_exact"
    return None


def _accept(
    paper: dict, pool: list[tuple[dict, list[str], float]]
) -> tuple[list[dict], str | None]:
    """First tier with any qualifying identity decides; several qualifying ones are ambiguous."""
    tier = [(identity, "doi") for identity, evidence, _ in pool if "doi" in evidence]
    if tier:
        return [identity for identity, _ in tier], "doi"
    tier = [
        identity
        for identity, evidence, _ in pool
        if "doi_filename" in evidence
        and paper["titles"] & identity["titles"]
        and _tier2_method(paper, identity)
        and not _conflicts(paper, identity)
    ]
    if tier:
        return tier, "doi_filename_corroborated"
    tier = [
        (identity, method)
        for identity, evidence, _ in pool
        if "title_exact" in evidence
        and not _conflicts(paper, identity)
        and (method := _tier2_method(paper, identity))
    ]
    if tier:
        methods = sorted({method for _, method in tier})
        return [identity for identity, _ in tier], methods[0]
    tier = [
        identity
        for identity, evidence, _ in pool
        if "title_exact" in evidence
        and (_is_arxiv(paper) or _is_arxiv(identity))
        and paper["authors"] & identity["authors"]
        and not _non_arxiv_doi_conflict(paper, identity)
    ]
    if tier:
        return tier, "title_author_exact_arxiv_variant"
    return [], None


def _non_arxiv_doi_conflict(paper: dict, identity: dict) -> bool:
    if not paper["dois"] or not identity["dois"] or paper["dois"] & identity["dois"]:
        return False
    return not any(is_arxiv_doi(doi) for doi in paper["dois"] | identity["dois"])


def _identifiable(paper: dict) -> bool:
    return bool(
        paper["dois"] or (paper["titles"] and (paper["years"] or paper["authors"]))
    )


def _candidate_row(
    identity: dict, evidence: list[str], ratio: float, conflict: bool
) -> dict:
    return {
        "doi": identity["doi"],
        "title": identity["title"],
        "year": identity["year"],
        "authors": identity["author_names"],
        "similarity": round(ratio, 4),
        "evidence": evidence,
        "conflict": conflict,
        "files": identity["files"],
    }


def _match(paper: dict, index: dict, incomplete: bool) -> tuple[dict, list[dict]]:
    pool = _candidate_pool(paper, index)
    accepted, method = _accept(paper, pool)
    candidates = [
        _candidate_row(
            identity,
            evidence,
            ratio,
            _conflicts(
                paper,
                identity,
                arxiv_exempt=(
                    "title_exact" in evidence
                    and (_is_arxiv(paper) or _is_arxiv(identity))
                    and not _non_arxiv_doi_conflict(paper, identity)
                ),
            ),
        )
        for identity, evidence, ratio in pool
    ]
    if len(accepted) == 1:
        file_set = accepted[0]["files"]
        keep = method == "title_author_exact_arxiv_variant"
        return (
            {
                "in_corpus": True,
                "status": "present",
                "method": method,
                "files": file_set,
                "reason": None,
            },
            candidates if keep else [],
        )
    if not _identifiable(paper):
        reason = "insufficient_reference_metadata"
    elif accepted:
        reason = "ambiguous_corpus_candidates"
    elif paper["dois"] & index.get("ambiguous_dois", frozenset()):
        reason = "ambiguous_corpus_doi"
    elif any(row["conflict"] for row in candidates):
        reason = "metadata_conflict"
    elif candidates:
        reason = "unverified_corpus_candidates"
    elif incomplete:
        reason = "corpus_scan_incomplete"
    else:
        return (
            {
                "in_corpus": False,
                "status": "missing",
                "method": None,
                "files": [],
                "reason": None,
            },
            candidates,
        )
    return (
        {
            "in_corpus": None,
            "status": "review",
            "method": None,
            "files": [],
            "reason": reason,
        },
        candidates,
    )


def _corpus_index(identities: list[dict], fuzzy_suggestions: bool = False) -> dict:
    by_doi: dict[str, list[dict]] = {}
    by_title: dict[str, list[dict]] = {}
    by_author: dict[str, list[dict]] = {}
    by_year: dict[str, list[dict]] = {}
    for identity in identities:
        for doi in sorted(identity["dois"]):
            by_doi.setdefault(doi, []).append(identity)
    for identity in identities:
        for title in identity["titles"]:
            by_title.setdefault(title, []).append(identity)
        for surname in identity["authors"]:
            by_author.setdefault(surname, []).append(identity)
        for year in identity["years"]:
            by_year.setdefault(year, []).append(identity)
    return {
        "fuzzy_suggestions": fuzzy_suggestions,
        "by_doi": by_doi,
        "by_title": by_title,
        "by_author": by_author,
        "by_year": by_year,
    }


def _sources(corpus: list[dict], scan_errors: list[dict]) -> list[dict]:
    rows: dict[str, dict] = {}
    for paper in corpus:
        row = rows.setdefault(
            paper["corpus_path"],
            {
                "_source_order": paper.get("_source_order", len(rows)),
                "path": paper["corpus_path"],
                "file_count": 0,
                "parsed_count": 0,
                "errors": [],
            },
        )
        row["file_count"] += 1
        row["parsed_count"] += 1
    for error in scan_errors:
        row = rows.setdefault(
            error["corpus_path"],
            {
                "_source_order": error.get("_source_order", len(rows)),
                "path": error["corpus_path"],
                "file_count": 0,
                "parsed_count": 0,
                "errors": [],
            },
        )
        if error["file"] is not None:
            row["file_count"] += 1
        if error["error"] is not None:
            row["errors"].append({"file": error["file"], "error": error["error"]})
    ordered = []
    for row in sorted(
        rows.values(), key=lambda item: (item["_source_order"], item["path"])
    ):
        row["errors"].sort(key=lambda entry: (entry["file"] or "", entry["error"]))
        ordered.append(
            {
                "path": row["path"],
                "file_count": row["file_count"],
                "parsed_count": row["parsed_count"],
                "error_count": len(row["errors"]),
                "errors": row["errors"],
            }
        )
    return ordered


def _reference_view(paper: dict) -> dict:
    """Canonical reference metadata: first occurrence in sort order, gaps filled behind it."""
    references = [occurrence["reference"] for occurrence in paper["occurrences"]]

    def first(field: str):
        return next(
            (reference.get(field) for reference in references if reference.get(field)),
            None,
        )

    return {
        "title": first("title"),
        "authors": list(first("authors") or []),
        "year": first("year"),
        # An arXiv DOI is a manifestation: the version of record is the canonical identity.
        "doi": _canonical_doi(paper["dois"]),
        "url": first("url"),
        "formatted": first("formatted"),
        "raw": first("raw"),
        "dois": sorted(paper["dois"]),
        "years": sorted(paper["years"]),
    }


def _paper_sort_key(paper: dict) -> tuple:
    reference = paper["view"]
    return (
        unicode_words(reference["title"] or ""),
        reference["doi"] or "",
        unicode_words(reference["url"] or ""),
        unicode_words(reference["raw"] or ""),
        _occurrence_key(paper["occurrences"][0]),
    )


def _joined(values) -> str:
    return "; ".join(sorted({value for value in values if value}))


def _collection_row(paper: dict) -> dict:
    reference = paper["view"]
    match = paper["corpus_match"]
    return {
        "queue_status": "review"
        if paper.get("acquisition_review")
        else match["status"],
        "title": reference["title"] or "",
        "authors": _joined(reference["authors"]),
        "year": _joined(reference["years"]),
        "doi": _joined(reference["dois"]),
        "url": reference["url"] or "",
        "formatted_reference": reference["formatted"] or reference["raw"] or "",
        "citation_methods": _joined(paper["citation_methods"]),
        "citation_confidences": _joined(paper["citation_confidences"]),
        "citing_paper_ids": _joined(
            occurrence["paper_id"] for occurrence in paper["occurrences"]
        ),
        "occurrence_count": len(paper["occurrences"]),
        "review_reason": "citation_link_requires_review"
        if paper.get("acquisition_review")
        else match["reason"] or "",
        "suggested_corpus_files": _joined(
            f"{row['corpus_path']}:{row['file']}"
            for candidate in paper["candidates"]
            for row in candidate["files"]
        ),
    }


def _report_occurrence(occurrence: dict) -> dict:
    return {
        key: occurrence[key]
        for key in (
            "paper_id",
            "paper_title",
            "record_index",
            "term",
            "field",
            "item_index",
            "citation_index",
            "reference_index",
            "raw",
            "marker",
            "method",
            "confidence",
            "year_check",
            "bibl_id",
        )
    }


def annotate_corpus_membership(
    dataset: dict,
    corpus: list[dict],
    scan_errors: list[dict],
    *,
    fuzzy_suggestions: bool = False,
) -> tuple[dict, dict, list[dict]]:
    """Annotate the loaded dataset copy and derive the coverage report and collection queue."""
    occurrences, markers_without_references = _occurrences(dataset)
    incomplete = any(error["error"] is not None for error in scan_errors)
    index = _corpus_index(corpus_identities(corpus), fuzzy_suggestions)
    index["ambiguous_dois"] = _ambiguous_dois(corpus)
    papers = cited_papers(occurrences)
    for paper in papers:
        paper["view"] = _reference_view(paper)
        paper["citation_methods"] = sorted(
            {
                occurrence["method"]
                for occurrence in paper["occurrences"]
                if occurrence["method"]
            }
        )
        paper["citation_confidences"] = sorted(
            {
                occurrence["confidence"]
                for occurrence in paper["occurrences"]
                if occurrence["confidence"]
            }
        )
        match, candidates = _match(paper, index, incomplete)
        # Corpus membership and citation correctness are separate facts. Never acquire a work on ambiguous/contradictory evidence alone, even if its DOI is known.
        supported = any(
            occurrence["confidence"] in {"high", "medium"}
            and "ambiguous" not in (occurrence["method"] or "")
            and occurrence.get("year_check") not in {"mismatch", "year_in_entry_text"}
            for occurrence in paper["occurrences"]
        )
        paper["acquisition_review"] = not supported
        paper["corpus_match"] = match
        paper["candidates"] = candidates
        for occurrence in paper["occurrences"]:
            occurrence["reference"]["corpus_match"] = dict(match)
    papers.sort(key=_paper_sort_key)
    summary = {
        "reference_occurrences": len(occurrences),
        "resolved_markers_without_references": markers_without_references,
        "unique_cited_papers": len(papers),
        "present": sum(
            1 for paper in papers if paper["corpus_match"]["status"] == "present"
        ),
        "missing": sum(
            1 for paper in papers if paper["corpus_match"]["status"] == "missing"
        ),
        "review": sum(
            1 for paper in papers if paper["corpus_match"]["status"] == "review"
        ),
    }
    report = {
        "checker": CHECKER,
        "dataset": None,
        "annotated_dataset": None,
        "sources": _sources(corpus, scan_errors),
        "summary": summary,
        "papers": [
            {
                "reference": paper["view"],
                "occurrences": [
                    _report_occurrence(occurrence)
                    for occurrence in paper["occurrences"]
                ],
                "citation_methods": paper["citation_methods"],
                "citation_confidences": paper["citation_confidences"],
                "acquisition_review": paper["acquisition_review"],
                "corpus_match": {
                    **paper["corpus_match"],
                    "candidates": paper["candidates"],
                },
            }
            for paper in papers
        ],
    }
    rows = [
        _collection_row(paper)
        for paper in papers
        if paper["corpus_match"]["status"] in ("missing", "review")
    ]
    return dataset, report, rows


def check_citation_corpus(
    dataset_path: Path,
    corpus_paths: list[Path],
    report_path: Path,
    collection_csv_path: Path,
    annotated_dataset_path: Path,
    workers: int = 16,
    allow_empty_root: bool = False,
    fuzzy_suggestions: bool = False,
) -> dict:
    """Scan and atomically replace the annotated dataset copy, the report and the queue CSV.

    `dataset_path` is never written: the citation-closure artifact stays byte-identical so that step and this membership step can be validated separately.
    """
    paths = [dataset_path, report_path, collection_csv_path, annotated_dataset_path]
    if len({_resolved(path) for path in paths}) != len(paths):
        raise ValueError(
            "output paths must be distinct and must not overwrite the closure dataset: "
            f"{_display_path(dataset_path)}"
        )
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    corpus, scan_errors = scan_corpus(
        corpus_paths, workers=workers, allow_empty_root=allow_empty_root
    )
    dataset, report, rows = annotate_corpus_membership(
        dataset, corpus, scan_errors, fuzzy_suggestions=fuzzy_suggestions
    )
    report["dataset"] = dataset_path.name
    report["annotated_dataset"] = annotated_dataset_path.name
    files.write_atomic(
        {
            annotated_dataset_path: files.json_text(dataset),
            report_path: files.json_text(report),
            collection_csv_path: ledger.csv_text(rows, COLLECTION_FIELDS),
        }
    )
    return report


ANNOTATED = "annotated-dataset.json"
COVERAGE = "coverage.json"
QUEUE = "citation-papers-to-collect.csv"  # the name `collect` looks for


def command(
    dataset: str,
    corpus_paths: list[str],
    out: str,
    workers: int,
    fuzzy_suggestions: bool,
    allow_empty_root: bool,
) -> int:
    """`closure check`: check every cited work of DATASET against the corpus roots, into OUT.

    Writes the dataset copy with each work's corpus match, the coverage report, the queue of works to collect (the collect step's input) and a receipt.
    """
    folder = Path(out)
    if folder.exists() and any(folder.iterdir()):
        raise SystemExit(f"{folder} is not empty; give a new or empty folder.")
    annotated, coverage, queue = folder / ANNOTATED, folder / COVERAGE, folder / QUEUE
    report = check_citation_corpus(
        Path(dataset),
        [Path(p) for p in corpus_paths],
        coverage,
        queue,
        annotated,
        workers=workers,
        allow_empty_root=allow_empty_root,
        fuzzy_suggestions=fuzzy_suggestions,
    )
    settings = {
        "corpus_paths": corpus_paths,
        "workers": workers,
        "fuzzy_suggestions": fuzzy_suggestions,
        "allow_empty_root": allow_empty_root,
        "checker": CHECKER,
    }
    inputs = {"dataset": Path(dataset)} | {
        f"corpus_{i}": Path(p) for i, p in enumerate(corpus_paths)
    }
    receipt = files.receipt(
        "mbdef closure check",
        settings,
        inputs,
        {"annotated": annotated, "coverage": coverage, "queue": queue},
    )
    receipt["summary"] = report["summary"]
    files.write_json(folder / files.RECEIPT, receipt)
    print(json.dumps(report["summary"], sort_keys=True))
    for name, path in (
        ("annotated", annotated),
        ("coverage", coverage),
        ("queue", queue),
    ):
        print(f"{name}: {path}")
    return 0
