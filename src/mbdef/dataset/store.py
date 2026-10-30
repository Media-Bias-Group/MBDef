"""What the dataset build reads from the corpus store besides the P≥28 view (`postprocess.corpus.load_view`).

The store keeps every extracted item once (`occurrences`, one row per item of a term in a paper) with every extraction run that found it (`occurrence_sources`: the source, i.e. corpus and variant such as "seeds:P28", and the record and item index there, with that run's own citation list). An item belongs to the dataset when at least one of its sources is a P≥28 extraction, the rule the view applies to definitions. Only the papers, terms, extracted items and their citation lists are read (tables `papers`, `mentions`, `occurrences`, `occurrence_sources`): no paragraph context and no other text of the papers, so a store without them serves. The store is opened read-only.
"""

import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from mbdef.postprocess.corpus import P28_SOURCES


def _open(store: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{store}?immutable=1", uri=True)


def p28_rows(db: sqlite3.Connection) -> dict[str, list[tuple]]:
    """{item: [((source, record, item index), its citation list as written)]}, every P≥28 copy of every item."""
    rows: dict[str, list[tuple]] = defaultdict(list)
    query = (
        "SELECT occ_id, source_id, record_index, item_index, raw_citations FROM occurrence_sources "
        f"WHERE source_id IN ({','.join('?' * len(P28_SOURCES))})"
    )
    for occ, source, record, item, raw in db.execute(query, P28_SOURCES):
        rows[occ].append(((source, record, item), raw))
    return rows


def p28_places_and_citations(
    db: sqlite3.Connection,
) -> tuple[dict[str, tuple], dict[str, list[str]]]:
    """For every item with a P≥28 source: its first P≥28 place (source, record, item) and its P≥28 citation list.

    The store keeps one `occurrences.citations` per item, the first-ingested copy's, which is often another extraction run's (the P≥20 run reads with more passes and adds markers). The dataset takes P≥28's own copies instead: the markers of every P≥28 copy of the item, in the order of their places, each once."""
    rows = p28_rows(db)
    place, cites = {}, {}
    for occ, copies in rows.items():
        copies.sort()
        place[occ] = copies[0][0]
        seen: dict[str, None] = {}
        for _, raw in copies:
            for marker in json.loads(raw) if raw else []:
                seen.setdefault(str(marker), None)
        cites[occ] = list(seen)
    return place, cites


def extracted_items(
    store: Path, mention_ids: set[str], fields: tuple[str, ...]
) -> dict[str, dict[str, list[dict]]]:
    """mention -> field -> its items {text, citations, occ_id}: the items with a P≥28 source, in the order the P≥28 extraction lists them (record, then item; an item found in several records takes its first place), each with P≥28's own citation list."""
    db = _open(store)
    try:
        place, cites = p28_places_and_citations(db)
        found: dict[str, dict[str, list[tuple]]] = defaultdict(
            lambda: defaultdict(list)
        )
        query = (
            "SELECT occ_id, mention_id, field, text FROM occurrences "
            f"WHERE field IN ({','.join('?' * len(fields))})"
        )
        for occ, mention, field, text in db.execute(query, fields):
            if mention in mention_ids and occ in place:
                item = dict(text=text, citations=cites[occ], occ_id=occ)
                found[mention][field].append((place[occ], occ, item))
        return {
            m: {f: [item for *_, item in sorted(v)] for f, v in by.items()}
            for m, by in found.items()
        }
    finally:
        db.close()


def corpus_of(store: Path) -> dict[str, str]:
    """{paper id: its corpus, "seeds" or "closure"} of every paper in the store."""
    db = _open(store)
    try:
        return dict(db.execute("SELECT paper_id, corpus FROM papers"))
    finally:
        db.close()


def p28_copies(store: Path, occ_ids: set[str]) -> int:
    """How many items of the P≥28 extraction these items are: the store keeps an item that the extraction wrote twice for one term in one paper once, with both places."""
    db = _open(store)
    try:
        rows = p28_rows(db)
        return sum(len(rows[occ]) for occ in occ_ids)
    finally:
        db.close()


def definition_citations(store: Path, def_ids: set[str]) -> dict[str, list[str]]:
    """Each definition's citation list as P≥28 extracted it."""
    db = _open(store)
    try:
        _, cites = p28_places_and_citations(db)
        return {occ: cites[occ] for occ in def_ids}
    finally:
        db.close()


def first_copy_citation_changes(store: Path, items: dict) -> Counter:
    """How many of the released items, per field, have a P≥28 citation list that differs from the store's first-ingested copy (`occurrences.citations`)."""
    db = _open(store)
    try:
        first = {
            occ: [str(x) for x in json.loads(raw)] if raw else []
            for occ, raw in db.execute("SELECT occ_id, citations FROM occurrences")
        }
    finally:
        db.close()
    return Counter(
        field
        for by in items.values()
        for field, found in by.items()
        for item in found
        if item["citations"] != first[item["occ_id"]]
    )


def field_counts(store: Path, mention_ids: set[str]) -> Counter:
    """Items per field of these terms, counted in SQL apart from `extracted_items`: a P≥28 source joined to its item."""
    db = _open(store)
    try:
        query = (
            "SELECT DISTINCT o.occ_id, o.mention_id, o.field FROM occurrences o "
            "JOIN occurrence_sources s ON s.occ_id = o.occ_id "
            f"WHERE s.source_id IN ({','.join('?' * len(P28_SOURCES))})"
        )
        return Counter(
            f for _, m, f in db.execute(query, P28_SOURCES) if m in mention_ids
        )
    finally:
        db.close()
