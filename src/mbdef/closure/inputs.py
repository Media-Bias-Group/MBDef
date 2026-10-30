"""`mbdef closure inputs`: the four lists `closure bundle` reads, rebuilt from the closure run's files.

`closure bundle` (docs/step-4-closure/closure.md) reads four small lists besides the run's ledgers and folders:

- `all-candidates.json`: every collected closure paper, `{"paper_id", "tei"}`, the papers the duplicate check compares; one per conversion receipt (`*.grobid.tei.closure.json`) in the collected TEI folder, in the order of their paths in that folder, each TEI path written below `tei_prefix` (by default the folder as given, or `$DATA_ROOT/...`, so the list names no folder of this machine). The released lists have the order in which the file system of their machine listed the folder, and that machine's folder; docs/step-4-closure/closure.md shows how to compare.
- `papers-info.json`: `{paper_id: {"doi", "extracted", "n_citing_seed_papers"}}` for the same papers in the same order, as things stood when the duplicate check ran: the DOI the paper was collected under (its receipt's `requested_doi`), whether it was in the extraction batches that had run by then (`extracted_batches`), and how many seed papers cited it in the links of that time (`definition_citations` on the ledgers of that time, over those batches' papers, every cited work that became the paper counted, as the release's paper table counts).
  The duplicate rule's tie-breaks read it (`exclusions.rank`).
- `uncited.json`: the collected papers in the closure set that no link row reaches (`definition_citations` on the run's ledgers, with the existing-corpus works added and the copies of the duplicate check left out), apart from those copies, sorted: papers collected through a DOI that GROBID had put on the wrong reference entry.
- `existing-exclude.json`: the cited works found only in the existing corpus (`existing.add_existing`, nothing left out) whose TEI has the same text as another such work's (`exclusions.copies_among`): of each pair the one fewer seed papers cite is not added (on a tie the one with the smaller paper id), in the order the pairs are found.
"""

import json
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from mbdef import files
from mbdef.closure import bundle, definition_citations, exclusions, existing
from mbdef.collect.identity import CONVERSION_RECEIPTS, converted_tei

# The closure run's earlier ledgers (those of release v1: batches 1-6), the same files as `bundle.LEDGERS`.
EARLIER_LEDGERS = bundle.V11 / "links-inputs-v1"
# The released closure's duplicate check ran after batch 6 and before batch 7 was extracted.
EXTRACTED_BATCHES = 6


def write_listed(path: Path, value) -> None:
    """The byte format of the released lists: `json.dumps` with its defaults, one line, no final newline."""
    Path(path).write_text(
        files.json_text(value, indent=None, ensure_ascii=True, end=""), encoding="utf-8"
    )


def collected(closure_tei: Path) -> list[tuple[dict, Path]]:
    """(receipt, TEI path) of every collected paper in the folder, in the (bytewise) order of the TEI paths."""
    found = []
    for receipt in Path(closure_tei).rglob(CONVERSION_RECEIPTS):
        tei = converted_tei(receipt)
        found.append((json.loads(receipt.read_text(encoding="utf-8")), tei))
    return sorted(
        found, key=lambda pair: pair[1].relative_to(closure_tei).as_posix().encode()
    )


def candidates(
    papers: list[tuple[dict, Path]], closure_tei: Path, prefix: str
) -> list[dict]:
    """`all-candidates.json`: each paper's id and its TEI path, written below PREFIX instead of the folder."""
    return [
        {
            "paper_id": receipt["paperid"],
            "tei": f"{prefix.rstrip('/')}/{tei.relative_to(closure_tei).as_posix()}",
        }
        for receipt, tei in papers
    ]


def batches_rows(closure_set: Path, batches: int) -> list[dict]:
    """The extraction rows of the closure set's first BATCHES batches (rows-all.json lists them in batch order)."""
    rows = bundle.read(Path(closure_set) / bundle.ROWS_ALL)
    sizes = [b["papers"] for b in bundle.read(Path(closure_set) / "batches.json")]
    if sum(sizes) != len(rows):
        raise SystemExit(
            f"batches.json counts {sum(sizes)} papers, rows-all.json holds {len(rows)}."
        )
    return rows[: sum(sizes[:batches])]


def links(ledgers: Path, rows: Path, seeds: Path, **merges) -> dict:
    """`definition_citations.build` on one set of ledgers, as `closure bundle` calls it."""
    return definition_citations.build(
        inputs=ledgers,
        provider=ledgers / bundle.PROVIDER,
        downloads=ledgers / bundle.DOWNLOADS,
        grobid_ledger=ledgers / bundle.GROBID_LEDGER,
        union_dataset=ledgers / bundle.UNION_DATASET,
        rows_all=rows,
        seeds_derived=seeds / bundle.DERIVED,
        **merges,
    )


def citing_papers(result: dict) -> dict[str, int]:
    """{closure paper: how many seed papers cite one of the cited works it became}."""
    citing = defaultdict(set)
    for work in result["works"]:
        for paper_id in work["closure_paper_ids"]:
            citing[paper_id].update(work["citing_seed_papers"])
    return {paper_id: len(seeds) for paper_id, seeds in citing.items()}


def papers_info(
    papers: list[tuple[dict, Path]], extracted: set[str], citing: dict[str, int]
) -> dict:
    """`papers-info.json`."""
    return {
        receipt["paperid"]: {
            "doi": receipt.get("requested_doi") or "",
            "extracted": receipt["paperid"] in extracted,
            "n_citing_seed_papers": citing.get(receipt["paperid"], 0),
        }
        for receipt, _ in papers
    }


def existing_works(before: dict, seed_inputs: Path, scratch: Path) -> list[dict]:
    """Every cited work found only in the existing corpus, its TEI written into SCRATCH (none left out)."""
    return existing.add_existing(
        before,
        seed_inputs / bundle.EXISTING_ZIP,
        seed_inputs / bundle.EXISTING_SHA256,
        scratch / "grobid",
        set(),
    )


def existing_copies(added: list[dict]) -> list[str]:
    """`existing-exclude.json`: of two existing-corpus works with one text, the one fewer seed papers cite."""
    texts = {a["paper_id"]: exclusions.shingles(Path(a["tei"])) for a in added}
    cited = {a["paper_id"]: a["citing_seed_papers"] for a in added}
    drop = []
    for pair in exclusions.copies_among(texts):
        a, b = pair["paper_id"], pair["duplicate_of"]
        keep = max((a, b), key=lambda paper_id: (cited[paper_id], paper_id))
        drop.append(b if keep == a else a)
    return drop


def uncited(
    ledgers: Path,
    rows_all: Path,
    seeds: Path,
    copies: list[dict],
    added: list[dict],
    scratch: Path,
) -> list[str]:
    """`uncited.json`: closure papers no link row reaches, apart from the copies left out, sorted."""
    write_listed(scratch / bundle.EXCLUSIONS, copies)
    write_listed(scratch / bundle.EXISTING_ADDED, added)
    result = links(
        ledgers,
        rows_all,
        seeds,
        exclusions=scratch / bundle.EXCLUSIONS,
        existing_added=scratch / bundle.EXISTING_ADDED,
    )
    linked = {x["closure_paper_id"] for x in result["links"] if x["closure_paper_id"]}
    copied = {e["paper_id"] for e in copies}
    return [p for p in sorted(result["extracted"] - linked) if p not in copied]


def rebuild(
    *,
    ledgers: Path,
    earlier_ledgers: Path,
    closure_set: Path,
    closure_tei: Path,
    seeds: Path,
    seed_inputs: Path,
    extracted_batches: int,
    tei_prefix: str,
) -> tuple[dict[str, Any], dict]:
    """The four lists ({file name: value}) and the counts of each step."""
    papers = collected(closure_tei)
    with tempfile.TemporaryDirectory() as folder:
        scratch = Path(folder)
        rows = batches_rows(closure_set, extracted_batches)
        write_listed(scratch / "rows-extracted.json", rows)
        citing = citing_papers(
            links(earlier_ledgers, scratch / "rows-extracted.json", seeds)
        )
        info = papers_info(papers, {r["paper_id"] for r in rows}, citing)
        listed = candidates(papers, closure_tei, tei_prefix)
        added = existing_works(
            bundle.read(ledgers / "before.json"), seed_inputs, scratch
        )
        not_added = existing_copies(added)
        record = exclusions.find_duplicates(
            seed_inputs / bundle.SEED_TEI, listed, [closure_tei]
        )
        copies = exclusions.exclusions(record["duplicates"], info, [])
        not_cited = uncited(
            ledgers,
            closure_set / bundle.ROWS_ALL,
            seeds,
            copies,
            [a for a in added if a["paper_id"] not in set(not_added)],
            scratch,
        )
    lists = {
        bundle.ALL_CANDIDATES: listed,
        bundle.PAPERS_INFO: info,
        bundle.UNCITED: not_cited,
        bundle.EXISTING_EXCLUDE: not_added,
    }
    counts = {
        "candidates": len(listed),
        "extracted": len(rows),
        "copies": len(copies),
        "uncited": len(not_cited),
        "existing_only_works": len(added),
        "existing_not_added": len(not_added),
    }
    return lists, counts


def command(
    out: str,
    ledgers: str | None,
    earlier_ledgers: str | None,
    closure_set: str | None,
    closure_tei: str | None,
    seeds: str | None,
    seed_inputs: str | None,
    extracted_batches: int,
    tei_prefix: str | None,
) -> int:
    """`closure inputs`: write the four lists and a receipt into OUT."""
    folder = Path(out)
    if folder.exists() and any(folder.iterdir()):
        raise SystemExit(f"{folder} is not empty; give a new or empty folder.")
    found = {
        "ledgers": files.located(ledgers, str(bundle.LEDGERS)),
        "earlier_ledgers": files.located(earlier_ledgers, str(EARLIER_LEDGERS)),
        "closure_set": files.located(closure_set, str(bundle.CLOSURE_SET)),
        "closure_tei": files.located(closure_tei, str(bundle.CLOSURE_TEI)),
        "seeds": files.located(seeds, str(bundle.SEEDS)),
        "seed_inputs": files.located(seed_inputs, str(bundle.SEED_INPUTS)),
    }
    given = {name: path for name, (path, _) in found.items()}
    for name, path in given.items():
        if not path.exists():
            raise SystemExit(bundle.missing_input(name, path))
    prefix = tei_prefix or found["closure_tei"][1]
    started = time.time()
    lists, counts = rebuild(
        **given, extracted_batches=extracted_batches, tei_prefix=prefix
    )
    counts["seconds"] = round(time.time() - started)
    folder.mkdir(parents=True, exist_ok=True)
    for name, value in lists.items():
        write_listed(folder / name, value)
    inputs = {
        "ledgers": given["ledgers"],
        "earlier_ledgers": given["earlier_ledgers"],
        "rows_all": given["closure_set"] / bundle.ROWS_ALL,
        "batches": given["closure_set"] / "batches.json",
        "closure_tei": given["closure_tei"],
        "seeds_derived": given["seeds"] / bundle.DERIVED,
        "seed_tei": given["seed_inputs"] / bundle.SEED_TEI,
        "existing_zip": given["seed_inputs"] / bundle.EXISTING_ZIP,
        "existing_sha256": given["seed_inputs"] / bundle.EXISTING_SHA256,
    }
    receipt = files.receipt(
        "mbdef closure inputs",
        {
            "extracted_batches": extracted_batches,
            "tei_prefix": prefix,
            "duplicate_share": exclusions.SHARE,
            "shingle_words": exclusions.SHINGLE_WORDS,
            "shingle_step": exclusions.SHINGLE_STEP,
        },
        inputs,
        {name: folder / name for name in lists},
    )
    receipt["summary"] = counts
    files.write_json(folder / files.RECEIPT, receipt)
    print(json.dumps(counts, indent=2))
    print(f"lists: {folder} ({', '.join(lists)})")
    return 0
