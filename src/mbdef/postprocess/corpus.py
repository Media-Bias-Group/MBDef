"""The Step 5 corpus: every Extracted Concept of the chosen extractor's view (P≥28), read from the corpus store.

The view is every seed paper plus the closure papers that the links file marks `in_P28`; a Definition belongs to it when at least one of its sources is a P≥28 extraction. On the v1.1 corpus that is 67,394 Extracted Concepts with 110,399 Definitions from 6,241 of the view's 6,479 papers. Both inputs are only read.
"""

import csv
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from mbdef import files

P28_SOURCES = ("seeds:P28", "closure:P28")
# Both inputs are read from the data root: the corpus store (`postprocess build-store` makes its four tables from the corpus release v1.1), and the release's definition links, which mark the closure papers of the P≥28 view.
# Neither is in data/data-root.tsv: both hold the papers' text, so they are not public.
STORE = "frozen/artifacts/experiments/taxonomy-curation-v2/library.sqlite3"
LINKS = "frozen/closure/definition-corpus-v1.1/links/definition-links.csv"


def default_store() -> Path:
    return files.data_root() / STORE


def default_links() -> Path:
    return files.data_root() / LINKS


@dataclass(frozen=True)
class Definition:
    def_id: str  # the store's occurrence id
    text: str


@dataclass(frozen=True)
class ExtractedConcept:
    """One term as one paper defines it: the store's mention, with its P≥28 Definitions in id order."""

    mention_id: str
    paper_id: str
    corpus: str  # "seeds" or "closure"
    label: str
    definitions: tuple[Definition, ...]


@dataclass(frozen=True)
class CorpusView:
    papers: frozenset[str]  # every paper of the view, with or without Definitions
    concepts: list[
        ExtractedConcept
    ]  # the Extracted Concepts with at least one Definition, sorted by mention id


def load_view(store: Path | None = None, links: Path | None = None) -> CorpusView:
    """The P≥28 view of the corpus store: its papers and its Extracted Concepts."""
    store, links = store or default_store(), links or default_links()
    csv.field_size_limit(1 << 30)
    with links.open(newline="") as f:
        view = {
            r["closure_paper_id"]
            for r in csv.DictReader(f)
            if r["in_P28"] == "1" and r["closure_paper_id"]
        }
    db = sqlite3.connect(f"file:{store}?immutable=1", uri=True)
    try:
        corpus = dict(db.execute("SELECT paper_id, corpus FROM papers"))
        keep = {p for p, c in corpus.items() if c == "seeds" or p in view}
        mentions = {
            m: (p, label)
            for m, p, label in db.execute(
                "SELECT mention_id, paper_id, display_term FROM mentions"
            )
        }
        p28 = {
            o
            for (o,) in db.execute(
                f"SELECT occ_id FROM occurrence_sources WHERE source_id IN ({','.join('?' * len(P28_SOURCES))})",
                P28_SOURCES,
            )
        }
        definitions: dict[str, list[Definition]] = {}
        for occ, m, text in db.execute(
            "SELECT occ_id, mention_id, text FROM occurrences WHERE field = 'definitions' ORDER BY occ_id"
        ):
            if occ in p28 and m in mentions and mentions[m][0] in keep:
                definitions.setdefault(m, []).append(Definition(occ, text))
    finally:
        db.close()
    return CorpusView(
        frozenset(keep),
        [
            ExtractedConcept(
                m,
                mentions[m][0],
                corpus[mentions[m][0]],
                mentions[m][1],
                tuple(definitions[m]),
            )
            for m in sorted(definitions)
        ],
    )
