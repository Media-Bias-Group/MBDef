"""Parse a GROBID TEI into bibliography records, in-text refs and raw back-matter text.

`parse_bibliography` keeps only `back//listBibl/biblStruct` entries carrying an `xml:id` (the `teiHeader` biblStruct describes the paper itself), and records every `<ref type="bibr">` with its containing paragraph so a resolved citation is auditable without reopening the TEI. A reference entry is read by `collect.identity.tei_entry`, the same reader that reads a paper's own title page.
"""

import difflib
import re
import xml.etree.ElementTree as ET
from typing import Any

from mbdef.collect.identity import (
    TEI,
    XMLID,
    clean,
    format_entry,
    header_doi,
    tei_entry,
    tei_text,
    unicode_words,
)

CONTEXT = 800


def _doi_key(value: str | None) -> str:
    """A DOI for comparison only: case, resolver prefix, %-escapes and trailing punctuation removed."""
    value = re.sub(
        r"%[0-9a-f]{2}", "", (value or "").casefold().removeprefix("https://doi.org/")
    )
    return re.sub(r"[^0-9a-z]+$", "", value)


def _different_works(a: str, b: str) -> bool:
    """Two entry titles that cannot be the same work (both present, neither contains the other, little in common)."""
    a, b = unicode_words(a or ""), unicode_words(b or "")
    if not a or not b or a in b or b in a:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() < 0.6


def _ignore_misplaced_dois(entries: dict[str, dict], root) -> None:
    """Drop DOIs GROBID attached to the wrong reference-list entries.

    GROBID copies the citing paper's own DOI, printed in its header or page footer, into reference entries, or one entry's DOI into the next. The corpus check groups references by DOI first, so such a DOI merged unrelated works into one (53 groups, 393 citations; 117 works were never looked up). A DOI on several entries that are clearly different works, or the paper's own DOI on an entry that is not the paper itself, is no evidence: it is dropped, kept in `doi_ignored`, and the entries are identified by their title like any reference without a DOI."""
    header = root.find(f"{TEI}teiHeader")
    own_entry = (
        header.find(f".//{TEI}sourceDesc//{TEI}biblStruct")
        if header is not None
        else None
    )
    own_doi = _doi_key(
        (
            tei_text(own_entry, f'.//{TEI}idno[@type="DOI"]')
            if own_entry is not None
            else None
        )
        or (header_doi(header) if header is not None else None)
    )
    own_title = (
        tei_text(
            root,
            f'{TEI}teiHeader//{TEI}titleStmt/{TEI}title[@level="a"][@type="main"]',
        )
        or ""
    )
    by_doi: dict[str, list[dict]] = {}
    for entry in entries.values():
        if key := _doi_key(entry["doi"]):
            by_doi.setdefault(key, []).append(entry)
    for key, group in by_doi.items():
        shared = any(
            _different_works(a["title"], b["title"])
            for i, a in enumerate(group)
            for b in group[i + 1 :]
        )
        for entry in group:
            own = (
                bool(own_doi)
                and key == own_doi
                and _different_works(entry["title"], own_title)
            )
            if shared or own:
                entry["doi_ignored"] = {
                    "doi": entry["doi"],
                    "reason": "shared by different entries"
                    if shared
                    else "the citing paper's own DOI",
                }
                entry["doi"] = None
                entry["formatted"] = format_entry(entry)


def parse_bibliography(tei: bytes) -> dict[str, Any]:
    root = ET.fromstring(tei)
    application = root.find(
        f"{TEI}teiHeader//{TEI}encodingDesc//{TEI}appInfo/{TEI}application"
    )
    entries: dict[str, dict] = {}
    for node in root.findall(f".//{TEI}back//{TEI}listBibl/{TEI}biblStruct"):
        if node.get(XMLID):
            entry = tei_entry(node)
            entries[entry["bibl_id"]] = entry
    refs: list[dict] = []
    paragraph = -1
    for block in root.iter(f"{TEI}p"):
        found = [node for node in block.iter(f"{TEI}ref") if node.get("type") == "bibr"]
        if not found:
            continue
        paragraph += 1
        context = clean("".join(block.itertext()))
        if len(context) > CONTEXT:
            context = context[:CONTEXT] + "…"
        for node in found:
            refs.append(
                {
                    "text": clean("".join(node.itertext())),
                    "target": (node.get("target") or "").lstrip("#") or None,
                    "paragraph": paragraph,
                    "context": context,
                }
            )
    seen = {node for block in root.iter(f"{TEI}p") for node in block.iter(f"{TEI}ref")}
    for node in root.iter(f"{TEI}ref"):
        if node.get("type") == "bibr" and node not in seen:
            refs.append(
                {
                    "text": clean("".join(node.itertext())),
                    "target": (node.get("target") or "").lstrip("#") or None,
                    "paragraph": -1,
                    "context": None,
                }
            )
    for ref in refs:
        if ref["target"] in entries:
            entries[ref["target"]]["ref_count"] += 1
    _ignore_misplaced_dois(entries, root)
    back = root.find(f".//{TEI}back")
    return {
        "grobid_version": (
            application.get("version") if application is not None else None
        ),
        "title": tei_text(
            root,
            f'{TEI}teiHeader//{TEI}titleStmt/{TEI}title[@level="a"][@type="main"]',
        ),
        "entries": entries,
        "refs": refs,
        "back_text": clean("".join(back.itertext())) if back is not None else "",
    }
