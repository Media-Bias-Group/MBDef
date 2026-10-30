"""`mbdef dataset export`: the built tables as JSON files, simplest first, with SCHEMA.md, a copy of the dataset card and SHA256SUMS, in `<out>/taxonomy-dataset-json/` beside the build's `<out>/taxonomy-dataset/`.

  terms.json                 one entry per concept with everything the papers say about it, merged (the `terms` table)
  papers.json                the papers that `terms` sources name, by paper_id
  tables/*.json              the rows `terms` is built from: concepts, instances, definitions

terms.json leaves out what is empty in a source or item (no `def_id` outside definitions, no `as_written` when the paper wrote the item's text, no `links_to` when the item names no concept, no `variants` when nothing was merged), so that an entry reads like a page about its term; the other files keep every parquet column. Every file is pretty-printed UTF-8, and the export reads each back and stops unless it equals its parquet table.
"""

import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq

from mbdef import files

from . import DATASET, JSON_COPY
from .build import differing, not_written
from .gold import hub_url
from .schema import TERM_FIELDS

GOLD_URL = hub_url(files.MBDEF_GOLD_REPO)
FILES = {  # JSON file -> its parquet table in the dataset folder
    "terms.json": "data/terms.parquet",
    "papers.json": "data/papers.parquet",
    "tables/concepts.json": "data/concepts.parquet",
    "tables/instances.json": "data/instances.parquet",
    "tables/definitions.json": "data/definitions.parquet",
}


def prune_terms(rows: list[dict]) -> list[dict]:
    """terms rows without the empty parts of items and sources."""
    for r in rows:
        for f in TERM_FIELDS:
            for it in r[f]:
                if not it["links_to"]:
                    del it["links_to"]
                if not it["variants"]:
                    del it["variants"]
                for s in it["sources"]:
                    for k in ("def_id", "as_written"):
                        if s[k] is None:
                            del s[k]
    return rows


def dump(obj, path: Path) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    b = files.json_text(obj).encode("utf-8")
    path.write_bytes(b)
    return b


def shorten(entry: dict) -> dict:
    """One terms entry cut for SCHEMA.md: two items per field, two sources per item; "…" marks a cut."""

    def cut(xs, n):
        return xs[:n] + (["…"] if len(xs) > n else [])

    def item(it):
        it = {**it, "sources": cut(it["sources"], 2)}
        if "variants" in it:
            it["variants"] = cut(it["variants"], 3)
        return it

    return {
        k: cut([item(it) for it in v], 2)
        if k in TERM_FIELDS
        else cut(v, 3)
        if k == "spellings"
        else v
        for k, v in entry.items()
    }


def example(terms: list[dict]) -> dict | None:
    """The entry SCHEMA.md shows: "fake news" when it has a supertype, else the first entry that has one."""
    named = [t for t in terms if t["supertypes"]]
    return next((t for t in named if t["term"].casefold() == "fake news"), None) or (
        named[0] if named else None
    )


def schema_md(
    written: dict[str, tuple[int, bytes]], terms: list[dict], version: str
) -> str:
    ex = example(terms)
    if ex is None:
        raise SystemExit(
            "No entry of terms.json has a supertype, so SCHEMA.md has no example to show."
        )
    sup = ex["supertypes"][0]
    lines = [
        "# MBDef as JSON",
        "",
        f"The same data as MBDef's parquet files (`data/*.parquet`, version {version}), "
        "made by `mbdef dataset export`. The dataset "
        "card (`DATASET-CARD.md`, a copy of MBDef's `README.md`) is the overview; the pages it links explain what the data is, "
        "how it was made, its limits and its licence. This file describes the JSON files. The hand-labelled ground truth is a dataset of its "
        f"own, MBDef-Gold ({GOLD_URL}).",
        "",
        "## Start with `terms.json`",
        "",
        "One entry per concept, with everything its papers say about it: its definitions, description, synonyms, related "
        "terms, subtypes, supertypes and examples. Items that say the same thing are merged, and each item says how many "
        f"papers state it. For example, “{sup['text']}” is a supertype of “{ex['term']}” in {sup['papers']} papers.",
        "",
        "| Key | What it holds |",
        "|---|---|",
        "| `concept_id` | the concept's id (`g` plus 12 hex digits) |",
        "| `term` | its name: the most frequent spelling |",
        "| `spellings` | every spelling, with how many papers write it |",
        "| `papers` | how many papers define the term |",
        "| `strict` | true when at least one paper's use of it passes the stricter relevance cut |",
        "| `definitions`, `description`, `synonyms`, `related_terms`, `subtypes`, `supertypes`, `examples` | the merged "
        "items, most stated first; `[]` when no paper gives any |",
        "",
        "Each item:",
        "",
        "| Key | What it holds |",
        "|---|---|",
        "| `text` | the item, in its most frequent form |",
        "| `papers` | how many distinct papers state it |",
        "| `links_to` | names only (synonyms, related terms, subtypes, supertypes): `{concept_id, term}` of the concept the "
        "item names, when it is one of the dataset's concepts; missing otherwise |",
        "| `variants` | when forms were merged: each form with how many papers write it; missing when there was one form |",
        "| `sources` | every paper that states the item: `paper_id` (look it up in `papers.json`), `mention_id` (the row in "
        "`tables/instances.json`), `def_id` (definitions only), `as_written` (only when the paper's wording differs from "
        "`text`) and `citations` (the references the paper gives for it) |",
        "",
        "**How items are merged.** Names merge by the Label Match rule, the rule that merges the terms themselves (case, "
        "plurals, spelling variants, word order, hyphens, brackets, stop words). Sentences (definitions, descriptions, "
        "examples) merge only when they read the same once case, punctuation and spacing are set aside. Fields are kept "
        "as each paper puts them, so one name can appear under several fields of a term.",
        "",
        "## One entry, shortened",
        "",
        f"“{ex['term']}”, with two items per field and two sources per item (`…` marks a cut):",
        "",
        "```json",
        json.dumps(shorten(ex), ensure_ascii=False, indent=2),
        "```",
        "",
        "## All files",
        "",
        "| File | Entries | Size | SHA-256 |",
        "|---|---:|---:|---|",
    ]
    for name, (n, b) in written.items():
        lines.append(
            f"| `{name}` | {n:,} | {len(b) / 1e6:.1f} MB | `{files.sha256(b)[:16]}…` |"
        )
    lines += [
        "",
        "`papers.json` and `tables/*.json` are lists of records with exactly the columns of their "
        'parquet tables; the dataset card lists every column under "Columns". A missing value is `null`, and `[]` means '
        'the paper gives nothing for that field. In Python: `json.load(open(path, encoding="utf-8"))`; on the command '
        "line: `jq '.[] | select(.term == \"fake news\")' terms.json`.",
    ]
    return "\n".join(lines) + "\n"


def export(out: Path, same_data_as: Path | None = None) -> int:
    """Write `out/taxonomy-dataset-json/` from `out/taxonomy-dataset/`; 1 when a file differs from `same_data_as`."""
    src, dst = out / DATASET, out / JSON_COPY
    if not (src / "provenance.json").exists():
        raise SystemExit(
            f"{src} holds no built dataset; run `mbdef dataset build --out {out}` first."
        )
    present = [name for name, rel in FILES.items() if (src / rel).exists()]
    if dst.exists():
        shutil.rmtree(dst)
    written, terms = {}, []
    for name in present:
        rows = pq.read_table(src / FILES[name]).to_pylist()
        if name == "terms.json":
            rows = terms = prune_terms(rows)
        b = dump(rows, dst / name)
        if json.loads(b) != rows:
            raise SystemExit(f"{name} does not read back as its parquet table")
        written[name] = (len(rows), b)
        print(
            f"{name}: {len(rows):,} entries, {len(b) / 1e6:.1f} MB, equals the parquet table"
        )
    # terms.json against the tables it is built from: every item's sources name rows of its own concept
    inst = {r["mention_id"]: r for r in json.loads(written["tables/instances.json"][1])}
    defs = {r["def_id"] for r in json.loads(written["tables/definitions.json"][1])}
    seen = set()
    for t in terms:
        for f in TERM_FIELDS:
            for it in t[f]:
                for s in it["sources"]:
                    if inst[s["mention_id"]]["concept_id"] != t["concept_id"]:
                        raise SystemExit(
                            f"terms.json: a source of {t['concept_id']} is a row of another concept"
                        )
                    if f == "definitions":
                        seen.add(s["def_id"])
    if seen != defs:
        raise SystemExit(
            "terms.json does not hold every definition of tables/definitions.json once"
        )
    version = json.loads((src / "provenance.json").read_text())["version"]
    (dst / "SCHEMA.md").write_text(schema_md(written, terms, version))
    (dst / "DATASET-CARD.md").write_text((src / "README.md").read_text())
    (dst / "SHA256SUMS").write_text(
        "".join(f"{files.sha256(b)}  {n}\n" for n, (_, b) in written.items())
    )
    print(f"wrote {dst}")
    if same_data_as:
        mine = {n: files.sha256(b) for n, (_, b) in written.items()}
        differ = differing(same_data_as, JSON_COPY, mine)
        if differ:
            print(
                f"Differ from {same_data_as}: "
                + ", ".join(f"{JSON_COPY}/{p}" for p in differ)
            )
            return 1
        print(
            f"All {len(mine)} JSON files are byte-identical to the hashes in {same_data_as}."
        )
        others = not_written(same_data_as, JSON_COPY, mine)
        if others:
            print(
                "Not written by this export, though the listing records them: "
                + ", ".join(f"{JSON_COPY}/{p}" for p in others)
            )
    return 0
