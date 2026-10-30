"""The corpus store that Step 5 reads: the extracted terms and definitions of release v1.1 in one SQLite file (docs/step-5-post-processing/postprocess.md#the-corpus-store).

Release v1.1 holds, for each corpus, the four output variants of `extract derive` (`<corpus>/derived/DONE.json` and `<corpus>/derived/<variant>/predictions.json`, docs/step-3-extraction/extract.md). `build` adds them to a new file, the seed papers' first and then the closure papers', each corpus's variants in the order of its DONE.json, into the four tables that `corpus.load_view` reads:

- `papers`: one row per paper: its id, title, DOI and corpus.
- `mentions`: one row per term as one paper names it, over all variants. Spellings that give the same key in one paper (`base_key`: case, punctuation, a leading article and an acronym in brackets set aside) are one mention; it keeps the first spelling seen (`display_term`) and the variant that first had it (`first_seen`).
- `occurrences`: one row per distinct text (`exact`: whitespace collapsed) in one field of one mention, with the citations of its first item.
- `occurrence_sources`: one row per item that gave that text: its variant, record and item number, the text as written and its citations.

A row's id is a hash of what identifies it (`digest`), so the same input always gives the same ids, and a row is written only once (the first time it is seen). It follows the ingest code that first read the corpus release into a database, reduced to these four tables: the stored corpus store (`corpus.STORE` in the data root) holds the same rows in the same order (tests/test_postprocess_store.py; `build-store --compare`). Its other tables are left out, among them the paragraph of the paper each definition was found in (`contexts`), which is paper text.
"""

import hashlib
import itertools
import json
import re
import sqlite3
import unicodedata
from pathlib import Path

from mbdef import files
from mbdef.extract.derive import tag

# Release v1.1's folder in the data root (data/data-root.tsv), and its two corpora in the order the store adds them.
RELEASE = "frozen/closure/definition-corpus-v1.1"
CORPORA = ("seeds", "closure")
STORE = "library.sqlite3"  # the file `build-store` writes into its --out folder
FIELDS = (
    "definitions",
    "description",
    "synonyms",
    "related_terms",
    "subtypes",
    "supertypes",
    "examples",
)
TABLES = ("papers", "mentions", "occurrences", "occurrence_sources")
SCHEMA = """
CREATE TABLE papers(paper_id TEXT PRIMARY KEY, title TEXT, doi TEXT, corpus TEXT);
CREATE TABLE mentions(mention_id TEXT PRIMARY KEY, paper_id TEXT, term_loose TEXT,
  display_term TEXT, base_key TEXT, acronym TEXT, first_seen TEXT);
CREATE TABLE occurrences(occ_id TEXT PRIMARY KEY, mention_id TEXT, paper_id TEXT, field TEXT,
  text TEXT, text_loose TEXT, citations TEXT);
CREATE TABLE occurrence_sources(occ_id TEXT, source_id TEXT, record_index INT, item_index INT,
  raw_text TEXT, raw_citations TEXT, PRIMARY KEY(occ_id, source_id, record_index, item_index));
CREATE INDEX occ_mention ON occurrences(mention_id);
CREATE INDEX mention_paper ON mentions(paper_id);
"""

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")
_PAREN = re.compile(r"^(?P<a>.*?)\s*[(\[]\s*(?P<b>[^()\[\]]{1,60})\s*[)\]]\s*$")
_ARTICLES = ("the ", "a ", "an ")


def default_release() -> Path:
    return files.data_root() / RELEASE


def exact(text: str | None) -> str:
    """Unicode NFC, whitespace collapsed: the identity of a text."""
    return _WS.sub(" ", unicodedata.normalize("NFC", text or "")).strip()


def loose(text: str | None) -> str:
    """NFKC, case folded, punctuation to space: for matching only."""
    return " ".join(
        _PUNCT.sub(" ", unicodedata.normalize("NFKC", text or "").casefold()).split()
    )


def digest(*parts: str) -> str:
    """A row's id: the first 16 hex digits of the SHA-1 of its parts joined by the unit separator."""
    return hashlib.sha1("\x1f".join(parts).encode()).hexdigest()[:16]


def _is_acronym(short: str, long_form: str) -> bool:
    s = re.sub(r"[^A-Za-z0-9]", "", short)
    if not 2 <= len(s) <= 12 or " " in short.strip():
        return False
    if sum(c.isupper() for c in s) < max(2, len(s) // 2):
        return False
    words = [w for w in re.split(r"[\s\-/]+", long_form) if w]
    initials = "".join(w[0] for w in words).casefold()
    letters = [c for c in s.casefold() if c.isalpha()]
    # The acronym's letters must appear in order among the long form's initials or its letters.
    it = iter(initials + " " + long_form.casefold())
    return bool(letters) and all(c in it for c in letters)


def split_term(raw: str | None) -> tuple[str, str | None, str | None]:
    """(main, acronym, qualifier) of 'long form (ACR)', 'ACR (long form)' or 'term (qualifier)'."""
    text = exact(raw)
    m = _PAREN.match(text)
    if not m or not m["a"].strip():
        return text, None, None
    a, b = m["a"].strip(" ,;:-"), m["b"].strip()
    if _is_acronym(b, a):
        return a, b, None
    if _is_acronym(a, b):
        return b, a, None
    return a, None, b


def base_key(raw: str | None) -> tuple[str, str | None]:
    """A term's key inside its paper, and its acronym: 'Hostile Media Effect (HME)' -> ('hostile media effect', 'HME')."""
    main, acronym, qualifier = split_term(raw)
    key = loose(main)
    for article in _ARTICLES:
        if key.startswith(article) and len(key) > len(article) + 2:
            key = key[len(article) :]
            break
    if qualifier:
        key = f"{key} {loose(qualifier)}".strip()
    return key or loose(raw), acronym


def variants(derived: Path) -> list[tuple[str, Path, dict]]:
    """(source id, predictions.json, its DONE.json entry) of each variant of a corpus's `derived/` folder, in the order of its DONE.json. The paths inside DONE.json are where the run wrote them; the files are read from the variant's folder beside DONE.json."""
    done = json.loads((derived / "DONE.json").read_text(encoding="utf-8"))
    return [
        (tag(name), derived / tag(name) / "predictions.json", pick)
        for name, pick in done["picks"].items()
        if pick.get("pick")
    ]


def add_variant(
    db: sqlite3.Connection, source_id: str, corpus: str, path: Path, listed: dict
) -> dict:
    """Add one variant's predictions.json, after checking it against its DONE.json entry (sha256, records and definitions); returns its counts."""
    found = files.digest(path)
    if found != listed["predictions_sha256"]:
        raise SystemExit(
            f"{path}: its sha256 is {found}, not {listed['predictions_sha256']} as DONE.json lists"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    papers, mentions, occurrences, sources = [], {}, {}, []
    records = definitions = 0
    for paper in data["papers"]:
        pid, meta = paper["paper_id"], paper.get("metadata") or {}
        papers.append((pid, meta.get("title"), meta.get("doi"), corpus))
        for ri, record in enumerate(paper.get("definition_records") or []):
            records += 1
            key, acronym = base_key(record["term"])
            mid = "m" + digest(pid, key)
            mentions.setdefault(
                mid, (mid, pid, key, exact(record["term"]), key, acronym, source_id)
            )
            for field in FIELDS:
                for ii, item in enumerate(record.get(field) or []):
                    text = exact(item.get("text"))
                    if not text:
                        continue
                    definitions += field == "definitions"
                    oid = "o" + digest(mid, field, text)
                    cites = json.dumps(item.get("citation") or [], ensure_ascii=False)
                    occurrences.setdefault(
                        oid, (oid, mid, pid, field, text, loose(text), cites)
                    )
                    sources.append((oid, source_id, ri, ii, item.get("text"), cites))
    for name, count in (("records", records), ("definitions", definitions)):
        if listed.get(name) is not None and listed[name] != count:
            raise SystemExit(
                f"{path}: {count} {name}, but DONE.json lists {listed[name]}"
            )
    db.executemany("INSERT OR IGNORE INTO papers VALUES(?,?,?,?)", papers)
    db.executemany(
        "INSERT OR IGNORE INTO mentions VALUES(?,?,?,?,?,?,?)", mentions.values()
    )
    db.executemany(
        "INSERT OR IGNORE INTO occurrences VALUES(?,?,?,?,?,?,?)", occurrences.values()
    )
    db.executemany(
        "INSERT OR IGNORE INTO occurrence_sources VALUES(?,?,?,?,?,?)", sources
    )
    return dict(
        source=source_id,
        path=path,
        sha256=found,
        papers=len(papers),
        records=records,
        definitions=definitions,
    )


def build(release: Path, store: Path) -> list[dict]:
    """Write the corpus store of the release folder `release` to the new file `store`; the counts of each variant."""
    added = []
    db = sqlite3.connect(store)
    try:
        db.executescript(SCHEMA)
        for corpus in CORPORA:
            for variant, path, listed in variants(release / corpus / "derived"):
                added.append(
                    add_variant(db, f"{corpus}:{variant}", corpus, path, listed)
                )
        db.commit()
    finally:
        db.close()
    return added


def table_counts(store: Path) -> dict[str, int]:
    db = sqlite3.connect(f"file:{store}?immutable=1", uri=True)
    try:
        return {
            t: db.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in TABLES
        }
    finally:
        db.close()


def compare(built: Path, stored: Path) -> dict[str, str | None]:
    """Each table of `built` against the same table of `stored`, row for row in rowid order (the order the rows were written in): None when they are the same, else the first difference."""
    a = sqlite3.connect(f"file:{built}?immutable=1", uri=True)
    b = sqlite3.connect(f"file:{stored}?immutable=1", uri=True)
    found = {}
    try:
        for t in TABLES:
            columns = [
                [c[1] for c in db.execute(f"PRAGMA table_info({t})")] for db in (a, b)
            ]
            if columns[0] != columns[1]:
                found[t] = f"columns {columns[0]} against {columns[1]}"
                continue
            query = f"SELECT rowid, * FROM {t} ORDER BY rowid"
            pairs = itertools.zip_longest(a.execute(query), b.execute(query))
            first = next(((x, y) for x, y in pairs if x != y), None)
            found[t] = (
                None
                if first is None
                else f"first different row: {first[0]} against {first[1]}"
            )
    finally:
        a.close()
        b.close()
    return found


def build_command(
    release: Path | None, out: Path, overwrite: bool, stored: Path | None
) -> int:
    """`postprocess build-store`: build the store into `out`, write its receipt and, with `stored`, compare the tables with that store's."""
    release = release or default_release()
    for corpus in CORPORA:
        if not (release / corpus / "derived" / "DONE.json").is_file():
            raise SystemExit(
                f"input not found: {release / corpus / 'derived' / 'DONE.json'}"
            )
    if stored is not None and not stored.is_file():
        raise SystemExit(f"input not found: {stored}")
    store = out / STORE
    if store.exists() and not overwrite:
        raise SystemExit(f"{store} exists; pass --overwrite to replace it")
    out.mkdir(parents=True, exist_ok=True)
    part = store.with_name(store.name + ".part")
    part.unlink(missing_ok=True)
    try:
        added = build(release, part)
    except BaseException:  # a refused variant (or any failure) leaves no file behind
        part.unlink(missing_ok=True)
        raise
    part.replace(store)
    for a in added:
        print(
            f"{a['source']}: {a['papers']:,} papers, {a['records']:,} records, {a['definitions']:,} definitions "
            f"(sha256 {a['sha256'][:12]}… as DONE.json lists)"
        )
    counts = table_counts(store)
    print("; ".join(f"{t} {n:,} rows" for t, n in counts.items()))
    inputs = {
        f"{corpus}/derived/DONE.json": release / corpus / "derived" / "DONE.json"
        for corpus in CORPORA
    }
    inputs |= {a["source"]: a["path"] for a in added}
    receipt = files.receipt(
        "postprocess build-store",
        settings={"release": str(release), "corpora": list(CORPORA), "tables": counts},
        inputs=inputs,
        outputs={STORE: store},
    )
    status = 0
    if stored is not None:
        found = compare(store, stored)
        for t, difference in found.items():
            print(
                f"{t}: "
                + (
                    f"the same as {stored} row for row"
                    if difference is None
                    else f"DIFFERS, {difference}"
                )
            )
        status = int(any(found.values()))
        receipt["compared"] = {"store": files.describe(stored), "tables": found}
        print(
            f"{sum(d is not None for d in found.values())} of {len(TABLES)} tables differ from {stored}."
            if status
            else f"All {len(TABLES)} tables equal {stored} row for row."
        )
    files.write_json(out / files.RECEIPT, receipt)
    print(f"wrote {store}")
    return status
