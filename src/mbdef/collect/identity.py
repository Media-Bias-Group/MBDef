"""Which work a queue row, a PDF or a TEI file is: DOIs, compared text, paper ids, TEI headers, receipts.

The collect step uses these to name what it downloads and to check that a converted PDF is the paper that was asked for. The closure step uses the same rules to match references to papers, so they live here once (the closure step imports them from collect).
"""

import hashlib
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import IO
from urllib.parse import unquote

TEI = "{http://www.tei-c.org/ns/1.0}"
XMLID = "{http://www.w3.org/XML/1998/namespace}id"


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def unicode_words(text: str) -> str:
    """Text for comparing titles and names: case-folded, punctuation dropped, Unicode letters, numbers, marks and mathematical symbols kept ("Fake News—A Survey" -> "fake news a survey").

    One of four normalisers with different rules, each kept because released outputs depend on it: this one keeps every script, because collect and the closure compare titles and names of papers in any language (the others: `groundtruth.ascii_words`, `closure.definition_citations.nfkc_words`, `closure.pdf_identity.lower_ascii_words`)."""
    text = unicodedata.normalize("NFC", text).casefold()
    return " ".join(
        "".join(
            char if unicodedata.category(char)[0] in "LNMS" else " " for char in text
        ).split()
    )


# DOIs ---------------------------------------------------------------------------------------------

DOI_SHAPE = re.compile(r"10\.\d{3,9}/\S+")
DOI_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
    "doi.org/",
    "dx.doi.org/",
    "doi:",
    "doi ",
)


def normalize_doi(value: str | None) -> str:
    """Lower case without resolver prefixes ("https://doi.org/10.1234/AB" -> "10.1234/ab"), or "" if it is not shaped like a DOI. An opaque DOI suffix is never rewritten."""
    text = clean(value or "").casefold()
    changed = True
    while changed:
        changed = False
        for prefix in DOI_PREFIXES:
            if text.startswith(prefix):
                text, changed = text[len(prefix) :].strip(), True
                if "doi.org/" in prefix:
                    text = unquote(text)
    return text if DOI_SHAPE.fullmatch(text) else ""


def is_arxiv_doi(doi: str) -> bool:
    return bool(
        re.fullmatch(
            r"10\.48550/arxiv\.(?:[a-z-]+(?:\.[a-z]{2})?/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?",
            normalize_doi(doi),
        )
    )


# Paper ids ----------------------------------------------------------------------------------------


def clean_identifier(value: object) -> str:
    text = str(value or "").strip()
    return re.sub(r"\s+", " ", text)


def filename_safe(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    safe = safe.strip("._-")
    return (
        safe
        if len(safe) <= 210
        else safe[:169] + "_" + hashlib.sha1(value.encode()).hexdigest()
    ) or "unknown"


def generated_hash(*parts: object) -> str:
    joined = "\n".join(
        clean_identifier(part).lower() for part in parts if clean_identifier(part)
    )
    return "hash_" + hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]


def collection_paper_id(*, doi: str, title: str, year: str, authors: str) -> str:
    """`doi_<safe DOI>__<sha1 of the DOI>`, or `hash_<sha1 of title, year, authors>` without a DOI."""
    normalized_doi = normalize_doi(doi)
    if not normalized_doi:
        return generated_hash(title, year, authors)
    digest = hashlib.sha1(normalized_doi.encode("utf-8")).hexdigest()
    return f"{filename_safe('doi_' + normalized_doi)[:120]}__{digest}"


def queue_surnames(authors: str) -> set[str]:
    """The compared surnames of a queue row's authors ("Lovelace, A.; Grace Hopper" -> lovelace, hopper)."""
    surnames = set()
    for author in authors.split("; "):
        author = author.strip()
        if not author:
            continue
        surname = (
            author.split(",", 1)[0] if "," in author else author.rsplit(" ", 1)[-1]
        )
        if normalized := unicode_words(surname):
            surnames.add(normalized)
    return surnames


# TEI headers --------------------------------------------------------------------------------------

YEAR = re.compile(r"(?:19|20)\d{2}")


def tei_text(node, path: str) -> str | None:
    found = node.find(path)
    return clean("".join(found.itertext())) or None if found is not None else None


def _title(entry) -> tuple[str | None, str]:
    for path, level in (
        (f'.//{TEI}title[@level="a"][@type="main"]', "a"),
        (f'.//{TEI}title[@level="a"]', "a"),
        (f'.//{TEI}title[@level="m"]', "m"),
    ):
        value = tei_text(entry, path)
        if value:
            return value, level
    return None, ""


def _container(entry, level: str) -> str | None:
    for node in entry.iter(f"{TEI}title"):
        if node.get("level") == "j" and node.get("type") != "abbrev":
            value = clean("".join(node.itertext()))
            if value:
                return value
    return tei_text(entry, f'.//{TEI}title[@level="m"]') if level == "a" else None


def _authors(entry) -> list[dict]:
    authors = []
    for person in entry.iter(f"{TEI}persName"):
        surname = tei_text(person, f"{TEI}surname") or ""
        forenames = [
            clean("".join(node.itertext()))
            for node in person.iter(f"{TEI}forename")
            if clean("".join(node.itertext()))
        ]
        if surname or forenames:
            authors.append({"forenames": forenames, "surname": surname})
    return authors


def _year(entry) -> str | None:
    for node in entry.iter(f"{TEI}date"):
        for value in (node.get("when") or "", clean("".join(node.itertext()))):
            found = YEAR.search(value)
            if found:
                return found.group(0)
    return None


def _pages(entry) -> str | None:
    node = entry.find(f'.//{TEI}biblScope[@unit="page"]')
    if node is None:
        return None
    start, end = node.get("from"), node.get("to")
    if start and end:
        return f"{start}-{end}"
    return start or clean("".join(node.itertext())) or None


def author_short(author: dict) -> str:
    initials = " ".join(f"{name[0]}." for name in author["forenames"] if name)
    surname = author["surname"]
    if surname and initials:
        return f"{surname}, {initials}"
    return surname or initials


def format_entry(entry: dict) -> str:
    """Deterministic APA-like rendering; falls back to the verbatim TEI text."""
    shorts = [value for value in map(author_short, entry["authors"]) if value]
    year = entry["year"] or "n.d."
    if len(shorts) > 1:
        authors = ", ".join(shorts[:-1]) + f" & {shorts[-1]}"
    else:
        authors = shorts[0] if shorts else ""
    title = entry["title"]
    parts: list[str] = []
    if authors:
        parts.append(f"{authors} ({year}).")
        if title:
            parts.append(title if title[-1] in ".?!" else f"{title}.")
    elif title:
        parts.append(title if title[-1] in ".?!" else f"{title}.")
        parts.append(f"({year}).")
    tail: list[str] = []
    if entry["container"]:
        container = entry["container"]
        if entry["volume"]:
            container += f" {entry['volume']}"
            if entry["issue"]:
                container += f"({entry['issue']})"
        if entry["pages"]:
            container += f", {entry['pages']}"
        tail.append(container if container[-1] in ".?!" else f"{container}.")
    elif entry["publisher"]:
        place = f"{entry['pub_place']}: " if entry["pub_place"] else ""
        tail.append(f"{place}{entry['publisher']}.")
    if entry["doi"]:
        tail.append(f"https://doi.org/{entry['doi']}")
    elif entry["url"]:
        tail.append(entry["url"])
    if not title and not tail:
        return entry["raw"]
    return clean(" ".join(parts + tail))


def tei_entry(node) -> dict:
    """One TEI `biblStruct` (a reference, or the paper's own title page) as a plain record."""
    title, level = _title(node)
    doi = tei_text(node, f'.//{TEI}idno[@type="DOI"]')
    pointer = node.find(f".//{TEI}ptr")
    entry = {
        "bibl_id": node.get(XMLID),
        "authors": _authors(node),
        "year": _year(node),
        "title": title,
        "container": _container(node, level),
        "volume": tei_text(node, f'.//{TEI}biblScope[@unit="volume"]'),
        "issue": tei_text(node, f'.//{TEI}biblScope[@unit="issue"]'),
        "pages": _pages(node),
        "publisher": tei_text(node, f".//{TEI}publisher"),
        "pub_place": tei_text(node, f".//{TEI}pubPlace"),
        "doi": doi.lower() if doi else None,
        "url": (pointer.get("target") or None) if pointer is not None else None,
        "raw": clean("".join(node.itertext())),
        "ref_count": 0,
    }
    # GROBID merges records and mis-parses dates ("… 22 2 1984. 1987 …"), so keep every year present in the entry text: it is what makes a marker/entry year clash visible.
    entry["years_in_text"] = sorted(set(YEAR.findall(entry["raw"])))
    entry["formatted"] = format_entry(entry)
    return entry


def header_doi(header) -> str | None:
    for node in header.iter(f"{TEI}idno"):
        if (node.get("type") or "").casefold() == "doi":
            value = clean("".join(node.itertext()))
            if value:
                return value
    return None


def _header_arxiv_markers(header) -> list[str]:
    markers = []
    for node in header.iter(f"{TEI}idno"):
        if "arxiv" not in (node.get("type") or "").casefold():
            continue
        value = clean("".join(node.itertext()))
        if value:
            markers.append(value if "arxiv" in value.casefold() else f"arXiv:{value}")
    return markers


def _header_doi_value(value: str | None) -> str | None:
    """Case and resolver-prefix handling as the extraction step's TEI reader does it."""
    if not value:
        return None
    return value.casefold().removeprefix("https://doi.org/") or None


def parse_paper_header(source: Path | IO[bytes]) -> dict:
    """Title, short author names, year, DOI and raw text of a paper, read from its TEI header.

    `source` is a filesystem path or an open binary stream, so a TEI stored as an archive member is indexed without being written out. Parsing stops as soon as `teiHeader` closes, so indexing a 7,000+ TEI corpus never materializes full body text. `OSError` and `ElementTree.ParseError` propagate: a corpus file that cannot be read must stay a visible scan failure, not empty metadata.
    """
    header = None
    for _event, node in ET.iterparse(source, events=("end",)):
        if node.tag == f"{TEI}teiHeader":
            header = node
            break
    if header is None:
        raise ValueError("missing TEI header")
    if not clean("".join(header.itertext())):
        raise ValueError("empty TEI header")
    bibl = header.find(f"{TEI}fileDesc/{TEI}sourceDesc/{TEI}biblStruct")
    entry = tei_entry(bibl) if bibl is not None else None
    raw = (entry["raw"] if entry else None) or tei_text(header, f"{TEI}fileDesc") or ""
    raw = clean(" ".join([raw, *_header_arxiv_markers(header)]))
    return {
        "title": (entry["title"] if entry else None)
        or tei_text(header, f"{TEI}fileDesc/{TEI}titleStmt/{TEI}title"),
        "authors": [
            value
            for value in map(author_short, entry["authors"] if entry else [])
            if value
        ],
        "year": entry["year"] if entry else None,
        "doi": _header_doi_value(
            (entry["doi"] if entry else None) or header_doi(header)
        ),
        "raw": raw,
    }


# Full text and conversion receipts ----------------------------------------------------------------

# How a converted PDF was bound to the requested work (see grobid.identity_match).
# "title_author_year_missing", "title_author_year_off_by_one" and "title_in_text" accept correct PDFs whose header lacks a year, carries a preprint year, or lacks a parsed title (a trial showed the strict rule "title_year_author" rejecting them).
# "existing_corpus": a TEI taken from the project's existing corpus, matched to the cited work by the corpus check (its method is in the receipt); such works were "present" but had never been extracted.
IDENTITY_MATCHES = {
    "doi",
    "title_year_author",
    "title_author_year_missing",
    "title_author_year_off_by_one",
    "title_in_text",
    "llm_verified",
    "existing_corpus",
}
RECEIPT_SCHEMA = "corpus-closure/1"


def valid_fulltext(payload: bytes, require_title: bool = True) -> bool:
    """A TEI document with body text (and, if required, a title)."""
    try:
        root = ET.fromstring(payload)
    except ET.ParseError:
        return False
    title = root.find(f"{TEI}teiHeader/{TEI}fileDesc/{TEI}titleStmt/{TEI}title")
    body = root.find(f"{TEI}text/{TEI}body")
    return bool(
        root.tag == f"{TEI}TEI"
        and body is not None
        and (
            not require_title
            or (title is not None and "".join(title.itertext()).strip())
        )
        and "".join(body.itertext()).strip()
    )


# The conversion receipt beside each TEI file that `collect grobid` writes, and the files it matches in a folder
RECEIPT_SUFFIX = ".closure.json"
CONVERSION_RECEIPTS = f"*.grobid.tei{RECEIPT_SUFFIX}"


def receipt_path(path: Path) -> Path:
    """The conversion receipt next to a TEI file: `x.grobid.tei.xml` -> `x.grobid.tei.closure.json`."""
    return path.with_suffix(RECEIPT_SUFFIX)


def converted_tei(receipt: Path) -> Path:
    """The TEI file a conversion receipt describes (`receipt_path` the other way round)."""
    return receipt.with_name(receipt.name.removesuffix(RECEIPT_SUFFIX) + ".xml")


def read_receipt(path: Path) -> dict | None:
    """The conversion receipt of a TEI file (None if it has none); fails if the TEI changed since."""
    receipt_file = receipt_path(path)
    if not receipt_file.exists():
        return None
    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    payload = path.read_bytes()
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("depth") != 1
        or receipt.get("tei_sha256") != hashlib.sha256(payload).hexdigest()
        or receipt.get("identity_match") not in IDENTITY_MATCHES
        # as at conversion: body text required, parsed title optional
        or not valid_fulltext(payload, require_title=False)
    ):
        raise ValueError(f"invalid acquisition receipt or changed TEI: {path}")
    return receipt
