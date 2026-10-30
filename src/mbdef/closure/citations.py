"""Split MBDef-Gold citation markers apart and resolve each one against a TEI bibliography.

Ladder, first hit wins: the GROBID `<ref type="bibr" target="#bN">` surface text (exact, then partial), author/year against `back//listBibl/biblStruct`, then raw reference-list text. Nothing is invented: endnote numbers, dataset codes and works absent from the bibliography keep an empty `references` list plus a reason code.
"""

import re

from mbdef.collect.identity import author_short, clean, unicode_words

PREFIX = re.compile(
    r"^(?:(?:e\.g\.|i\.e\.|see also|also see|see|cf\.|for example|"
    r"for reviews|reviewed in|for a similar approach,\s*see)[,:]?\s*)+",
    re.I,
)
PAGE = re.compile(r"\b(pp?\.\s*[\dixvlc]+(?:\s*[-–]\s*[\dixvlc]+)?)", re.I)
PAGE_ONLY = re.compile(
    r"^(?:pp?\.|pages?)\s*[\dixvlc]+(?:\s*[-–]\s*[\dixvlc]+)?$", re.I
)
NUMERIC = re.compile(r"^[\d\s,\-–]+$")
YEAR = re.compile(r"(?<!\d)(?:18|19|20)\d{2}(?!\d)")
NAME = re.compile(r"\b([^\W\d_][^\W\d_’'-]*(?:[’'-][^\W\d_]+)*)", re.UNICODE)
STOPWORDS = {
    "And",
    "Et",
    "Al",
    "See",
    "Also",
    "The",
    "In",
    "Pp",
    "P",
    "Eg",
    "Cf",
    "Emphasis",
    "Added",
}
PUNCT = re.compile(r"[()\[\];,]")
BRACKETS = {"(": ")", "[": "]"}


def _key(text: str) -> str:
    return unicode_words(PUNCT.sub(" ", text))


def _unwrap(text: str) -> str:
    """Drop outer brackets only when the opener's partner is the final character.

    Stripping each side on its own turns `"Wilbanks (1987)"` into `"Wilbanks (1987"`.
    """
    closer = BRACKETS.get(text[:1])
    if not closer or not text.endswith(closer):
        return text
    depth = 0
    for position, char in enumerate(text):
        if char in BRACKETS:
            depth += 1
        elif char in (")", "]"):
            depth -= 1
            if depth == 0:
                # `"(Smith 1990) and (Jones 1991)"` closes early: it is not a wrapper.
                return text[1:-1] if position == len(text) - 1 else text
    return text


def _split_comma_works(piece: str) -> list[str]:
    """Split a comma group only after one work's year and before another author."""
    units: list[str] = []
    start = 0
    for separator in re.finditer(",", piece):
        current = piece[start : separator.start()]
        following = piece[separator.end() :].lstrip()
        candidate = PREFIX.sub("", following).lstrip()
        if YEAR.search(current) and NAME.match(candidate) and YEAR.search(candidate):
            units.append(current.strip())
            start = separator.end()
    units.append(piece[start:].strip())
    return [unit for unit in units if unit]


def split_units(citation: str) -> list[dict]:
    text = _unwrap((citation or "").strip()).strip()
    units = []
    for group in text.split(";"):
        for raw_piece in _split_comma_works(group):
            piece = PREFIX.sub("", raw_piece).strip()
            if not piece:
                continue
            found = PAGE.search(piece)
            core = clean(re.sub(r",?\s*" + PAGE.pattern, "", piece, flags=re.I)).strip(
                " ,"
            )
            units.append(
                {"raw": piece, "text": core, "page": found.group(1) if found else None}
            )
    return units


def surname_year(core: str) -> tuple[list[str], str | None]:
    found = YEAR.search(core)
    year = found.group(0) if found else None
    author_text = core[: found.start()] if found else core
    names = [
        name
        for name in NAME.findall(author_text)
        if name.casefold() not in {word.casefold() for word in STOPWORDS}
    ]
    if re.search(r"\bet\s+al\.?", author_text, re.I) and names:
        names = names[-1:]
    return names, year


def build_index(parsed: dict) -> dict:
    refs = parsed["refs"]
    exact: dict[str, list[int]] = {}
    order: list[tuple[str, int]] = []
    for position, ref in enumerate(refs):
        if not ref["target"]:
            continue
        key = _key(ref["text"])
        if not key:
            continue
        order.append((key, position))
        exact.setdefault(key, []).append(position)
    by_year: dict[str, list[tuple[str, list[str]]]] = {}
    by_surname: dict[str, set[str]] = {}
    for bibl_id, entry in parsed["entries"].items():
        surnames = [
            unicode_words(author["surname"])
            for author in entry["authors"]
            if author["surname"]
        ]
        if entry["year"]:
            by_year.setdefault(entry["year"], []).append((bibl_id, surnames))
        for surname in surnames:
            by_surname.setdefault(surname, set()).add(bibl_id)
    return {
        "ref_exact": exact,
        "ref_order": order,
        "refs": refs,
        "by_year": by_year,
        "by_surname": by_surname,
        "entry_text": {
            bibl_id: entry["raw"] for bibl_id, entry in parsed["entries"].items()
        },
        "bibl_count": len(parsed["entries"]),
        "back_text": parsed["back_text"],
        "has_tei": True,
    }


def empty_index() -> dict:
    return {
        "ref_exact": {},
        "ref_order": [],
        "refs": [],
        "by_year": {},
        "by_surname": {},
        "entry_text": {},
        "bibl_count": 0,
        "back_text": "",
        "has_tei": False,
    }


def _sort_ids(bibl_ids) -> list[str]:
    def order(value: str) -> tuple[int, str]:
        digits = re.search(r"\d+", value)
        return (int(digits.group(0)) if digits else 1 << 30, value)

    return sorted(dict.fromkeys(bibl_ids), key=order)


def _year_candidates(index: dict, year: str, names: list[str]) -> list[str]:
    wanted = {unicode_words(name) for name in names}
    return [
        bibl_id
        for bibl_id, surnames in index["by_year"].get(year, [])
        if wanted and wanted.issubset(set(surnames))
    ]


def _year_only_targets(index: dict, years: list[str]) -> dict[str, list[int]]:
    """Ref positions per year for markers that are nothing but digits (`"1974"`)."""
    found: dict[str, list[int]] = {}
    for year in years:
        positions = [position for key, position in index["ref_order"] if year in key]
        found[year] = positions
    return found


def _reason(core: str, index: dict, year: str | None) -> str:
    if not index["has_tei"]:
        return "tei_not_found"
    if index["bibl_count"] == 0:
        return "no_bibliography_in_tei"
    if NUMERIC.match(core):
        return (
            "note_marker_no_bibliography"
            if index["bibl_count"] < 2
            else "numeric_marker_unmatched"
        )
    if year:
        return "year_surname_not_in_bibliography"
    return "no_year_no_surname_match"


def resolve_unit(core: str, index: dict) -> dict:
    key = _key(core)
    years = YEAR.findall(core)
    names, year = surname_year(core)
    out = {
        "method": "unresolved",
        "confidence": "none",
        "bibl_ids": [],
        "surnames": names,
        "years": years,
        "ref_text": None,
        "ref_context": None,
        "ref_paragraph": None,
        "raw_reference": None,
        "reason": None,
    }

    def from_ref(position: int, method: str, confidence: str) -> None:
        ref = index["refs"][position]
        out.update(
            method=method,
            confidence=confidence,
            bibl_ids=[target.lstrip("#") for target in ref["target"].split()],
            ref_text=ref["text"],
            ref_context=ref["context"],
            ref_paragraph=ref["paragraph"],
        )

    positions = index["ref_exact"].get(key, []) if key else []
    if positions:
        signatures = {tuple(index["refs"][p]["target"].split()) for p in positions}
        from_ref(positions[0], "ref_target", "high")
        if len(signatures) > 1:
            out.update(
                method="ref_target_ambiguous",
                confidence="low",
                bibl_ids=_sort_ids(
                    t.lstrip("#") for signature in signatures for t in signature
                ),
            )
    elif NUMERIC.match(core) and years:
        # A digits-only marker ("1974") may be a bare year. Substring matching would pick the first same-year reference out of up to 36, so resolve only when unique.
        found = _year_only_targets(index, years)
        unique = all(
            len({index["refs"][position]["target"] for position in positions}) == 1
            for positions in found.values()
        )
        if found and unique and all(found.values()):
            first = found[years[0]][0]
            from_ref(first, "ref_target_partial", "medium")
            out["bibl_ids"] = _sort_ids(
                index["refs"][position]["target"]
                for positions in found.values()
                for position in positions
            )
        elif any(
            len({index["refs"][p]["target"] for p in positions}) > 1
            for positions in found.values()
        ):
            out["reason"] = "year_only_marker_ambiguous"
    elif len(key) >= 4 and not NUMERIC.match(core):
        # Endnote markers ("7", "3,9,14,123,155") substring-match the digits of unrelated ref text ("[4,"), so partial matching needs a real author/year surface string.
        matches = [
            position
            for candidate_key, position in index["ref_order"]
            if (
                f" {key} " in f" {candidate_key} " or f" {candidate_key} " in f" {key} "
            )
            and (not years or years[0] in candidate_key)
        ]
        if matches:
            signatures = {tuple(index["refs"][p]["target"].split()) for p in matches}
            from_ref(matches[0], "ref_target_partial", "medium")
            if len(signatures) > 1:
                out.update(
                    method="ref_target_partial_ambiguous",
                    confidence="low",
                    bibl_ids=_sort_ids(
                        t.lstrip("#") for signature in signatures for t in signature
                    ),
                )
    if not out["bibl_ids"] and year and names:
        candidates = _year_candidates(index, year, names)
        if len(candidates) == 1:
            out.update(
                method="bibl_author_year", confidence="medium", bibl_ids=candidates
            )
        elif candidates:
            out.update(
                method="bibl_author_year_ambiguous",
                confidence="low",
                bibl_ids=_sort_ids(candidates),
            )
    if not out["bibl_ids"] and year and names:
        wanted = {unicode_words(name) for name in names}
        candidates = [
            bibl_id
            for surname in wanted
            for bibl_id in index["by_surname"].get(surname, ())
            if year in index["entry_text"].get(bibl_id, "")
        ]
        candidates = _sort_ids(candidates)
        if len(candidates) == 1:
            out.update(method="bibl_text_year", confidence="low", bibl_ids=candidates)
        elif candidates:
            out.update(
                method="bibl_text_year_ambiguous", confidence="low", bibl_ids=candidates
            )
    if not out["bibl_ids"] and not year and names:
        candidates = [
            next(iter(index["by_surname"][unicode_words(name)]))
            for name in names
            if len(index["by_surname"].get(unicode_words(name), ())) == 1
        ]
        if candidates:
            out.update(
                method="bibl_surname", confidence="low", bibl_ids=_sort_ids(candidates)
            )
    if out["bibl_ids"] and len(years) > 1:
        added = {other: _year_candidates(index, other, names) for other in years[1:]}
        extra = [bibl_id for ids in added.values() for bibl_id in ids]
        if extra:
            out["bibl_ids"] = _sort_ids(out["bibl_ids"] + extra)
            # "(Reyna & Brainerd, 1992, 1995)" can hit two 1995 works by the same authors:
            # keep both candidates, but stop calling the link medium confidence.
            if any(len(ids) > 1 for ids in added.values()):
                out["confidence"] = "low"
    if not out["bibl_ids"] and names and year and index["back_text"]:
        for name in names:
            # References carry internal periods ("Bobo, L.; Kluegel, JR. …, 1997."), so the window may cross them; it stays a review-only snippet with no bibl_ids.
            found = re.search(
                re.escape(name) + r".{0,400}?" + year, index["back_text"], re.I | re.S
            )
            if found:
                out.update(
                    method="raw_reference_text",
                    confidence="low",
                    raw_reference=clean(found.group(0)),
                    # No bibl_ids, so this unit lands in the unresolved report and needs a reason code there: the work is only present as reference-list text.
                    reason="raw_reference_only",
                )
                break
    if out["method"] == "unresolved" and not out["reason"]:
        out["reason"] = _reason(core, index, year)
    return out


def year_check(row: dict) -> str:
    """Does the marker year agree with the linked entry, its text, or neither?"""
    if not row["references"] or not row["years"]:
        return "none"
    years = set(row["years"])
    if years & {ref["year"] for ref in row["references"] if ref["year"]}:
        return "match"
    if years & {
        year for ref in row["references"] for year in ref.get("years_in_text", [])
    }:
        return "year_in_entry_text"
    return "mismatch"


def reference_from_entry(entry: dict) -> dict:
    """A bibliography entry inlined beside the marker that names it.

    Authors collapse to `Surname, I.` strings (the full forename list lives in the TEI); `ref_count` is a per-paper statistic and belongs to the report, not here.
    """
    return {
        "bibl_id": entry["bibl_id"],
        "authors": [value for value in map(author_short, entry["authors"]) if value],
        "year": entry["year"],
        "years_in_text": entry["years_in_text"],
        "title": entry["title"],
        "container": entry["container"],
        "volume": entry["volume"],
        "issue": entry["issue"],
        "pages": entry["pages"],
        "publisher": entry["publisher"],
        "pub_place": entry["pub_place"],
        "doi": entry["doi"],
        "url": entry["url"],
        "formatted": entry["formatted"],
        "raw": entry["raw"],
    }


def resolve_citations(item: dict, index: dict, entries: dict) -> list[dict]:
    """One flat record per citation marker, carrying the work(s) it names inline.

    `raw` is the annotated citation string the marker came from, so a record joins back onto `item["citation"]` by value. A page locator folds into the record it qualifies, so `"(p. 517)"` only keeps a record of its own when no marker precedes it in the same item. `references` is empty exactly when nothing resolved.
    """
    records: list[dict] = []
    for citation in item.get("citation", []):
        for unit in split_units(citation):
            if PAGE_ONLY.match(unit["raw"].strip()) or not unit["text"]:
                page = unit["page"] or unit["raw"].strip()
                if records:
                    records[-1]["page"] = records[-1]["page"] or page
                    continue
                records.append(
                    {
                        "raw": citation,
                        "marker": None,
                        "page": page,
                        "method": "page_locator",
                        "confidence": "none",
                        "year_check": "none",
                        "reason": "page_locator",
                        "surnames": [],
                        "years": [],
                        "ref_text": None,
                        "ref_context": None,
                        "ref_paragraph": None,
                        "raw_reference": None,
                        "references": [],
                    }
                )
                continue
            resolved = resolve_unit(unit["text"], index)
            record = {
                "raw": citation,
                "marker": unit["text"],
                "page": unit["page"],
                "method": resolved["method"],
                "confidence": resolved["confidence"],
                "year_check": "none",
                "reason": resolved["reason"],
                "surnames": resolved["surnames"],
                "years": resolved["years"],
                "ref_text": resolved["ref_text"],
                "ref_context": resolved["ref_context"],
                "ref_paragraph": resolved["ref_paragraph"],
                "raw_reference": resolved["raw_reference"],
                "references": [
                    reference_from_entry(entries[bibl_id])
                    for bibl_id in resolved["bibl_ids"]
                    if bibl_id in entries
                ],
            }
            record["year_check"] = year_check(record)
            if any(bibl_id not in entries for bibl_id in resolved["bibl_ids"]):
                record["confidence"] = "low"
                record["reason"] = "bibliography_target_missing"
            if record["year_check"] in ("year_in_entry_text", "mismatch"):
                # The link comes from GROBID, but its rendered year contradicts the marker ("Zatz 1987" -> an entry dated 1984): never call that high.
                record["confidence"] = "low"
            records.append(record)
    return records
