"""`mbdef closure bundle`: the closure's release folder, rebuilt from the run's saved files.

The offline end of the closure step (Step 4), run in order, one folder of results (docs/step-4-closure/closure.md):

1. `exclusions.find_duplicates`, then `exclusions.exclusions`: closure papers whose text is a seed paper's or another closure paper's, and those no seed reference cites, are left out (`audit/`);
2. `existing.add_existing`: cited works found only in the project's existing corpus become closure papers (`existing/`, `audit/`); their TEI files and receipts must equal the collected ones;
3. `definition_citations.build` and `definition_citations.write`: which seed definition cites which cited work, and which closure paper that work became (`links/`);
4. the PDF identity check's decisions, round by round, and their merge (`identity/`): the `decide` step only, from the saved verdicts of the model (`pdf_identity`);
5. `release_folder.build` and `release_folder.finish`: the release folder with its README and MANIFEST.sha256 (`release/`), then `release_folder.verify`: every manifest line checks out.

What it does not do: collect the papers (`collect`), extract their definitions (`extract`) or run the model (`closure identity judge`); those results are inputs here. The default inputs are the files of the released closure (v1.1) in the data root (docs/setup/data-root.md); the swap points are the folders and the version name. With `--audit-reference` and `--identity-reference` the command also checks that its audit results and identity decisions equal the released ones (`compare_with`).
"""

import json
import re
import time
from pathlib import Path

from mbdef import files
from mbdef.closure import (
    definition_citations,
    existing,
    pdf_identity,
    release_folder,
)
from mbdef.closure import exclusions as left_out
from mbdef.collect import grobid as collect_grobid
from mbdef.collect.identity import receipt_path

VERSION = "v1.1"
# The released closure's (v1.1) inputs under DATA_ROOT (docs/setup/data-root.md): its own folders under `frozen/closure/`, as they were kept when it was built, and the seed run's under `frozen/runs/`.
CLOSURE_ROOT = Path("frozen/closure")
V11 = CLOSURE_ROOT / "closure-v11"
LEDGERS = V11 / "links-inputs"
CLOSURE_SET = V11 / "extraction" / "unpacked"
AUDIT_INPUTS = V11
RUN = CLOSURE_ROOT / "taxonomy-closure-20260928/prod/closure/run"
CLOSURE_TEI = RUN / "grobid"
LEFT_OUT_TEI = RUN / "grobid-left-out-v11"
IDENTITY = CLOSURE_ROOT / "taxonomy-closure-20260928/identity-llm"
SEED_RUN = Path("frozen/runs/production-20260925")
SEEDS = SEED_RUN / "corpus/seeds"
SEED_INPUTS = SEED_RUN / "inputs"
# In git (frozen inputs, `data/frozen/README.md`), relative to the repository root.
LOSS_AUDIT = Path(
    "data/frozen/docs/closure-loss-audit.md"
)  # the audit as the release recorded it

# Names inside the input folders.
SEED_ROWS = "runner-rows.json"
SEED_TEI = "seed-tei"
EXISTING_ZIP = "existing-corpus-tei.zip"
EXISTING_SHA256 = "existing-corpus-tei.sha256"
ALL_CANDIDATES = "all-candidates.json"
PAPERS_INFO = "papers-info.json"
UNCITED = "uncited.json"
EXISTING_EXCLUDE = "existing-exclude.json"
PROVIDER = "provider.csv"
DOWNLOADS = "download.csv"
GROBID_LEDGER = "grobid-laptop-snapshot.csv"
UNION_DATASET = "union-dataset.json"
ROWS_ALL = "rows-all.json"
DERIVED = "derived"
COMPARE = "compare"
IDENTITY_FILES = (pdf_identity.ACCEPTED, pdf_identity.DECISIONS)

# Names of the audit files (the released ones, for `--audit-reference`, are in data/frozen/results/closure-audit-20261001/).
DUPLICATES = "duplicate-texts.json"
DUPLICATES_RELEASED = "duplicate-texts-v11.json"  # its name in the frozen results
EXCLUSIONS = "exclusions.json"
EXISTING_ADDED = "existing-corpus-added.json"


def read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def step(text: str) -> None:
    print(text, flush=True)


def rounds_of(identity: Path) -> list[tuple[str, int]]:
    """The identity check's rounds in a folder: round 1's files at its top, round N's in `rN/`."""
    later = sorted(
        (int(p.name[1:]), p.name) for p in identity.glob("r[0-9]*") if p.is_dir()
    )
    return [("", 1), *[(name, number) for number, name in later]]


def run_identity(identity: Path, out: Path) -> dict:
    """`decide` on every round's saved evidence and verdicts, then the merges; the counts of each round."""
    merged = None
    summary = {}
    for folder, number in rounds_of(identity):
        where = identity / folder
        accepted, decisions = pdf_identity.decide(
            where / pdf_identity.EVIDENCE, where / pdf_identity.VERDICTS
        )
        pdf_identity.write(out / folder / pdf_identity.ACCEPTED, accepted)
        pdf_identity.write(out / folder / pdf_identity.DECISIONS, decisions)
        summary[f"round_{number}"] = len(accepted["accepted"])
        if number == 1:
            merged = accepted
            continue
        assert merged is not None
        merged = pdf_identity.merge(merged, accepted, number)
        name = "accepted-all.json" if number == 2 else f"accepted-all-r{number}.json"
        pdf_identity.write(out / name, merged)
        summary[f"after_round_{number}"] = len(merged["accepted"])
    return summary


def tei_path_in(folder: Path, added: dict) -> Path:
    """Where a work taken from the existing corpus lies below a GROBID folder (as `existing` wrote it)."""
    tei = Path(added["tei"])
    return Path(folder) / tei.parent.name / tei.name


def check_existing_collected(added: list[dict], collected: Path) -> int:
    """The existing-corpus TEI files and receipts `existing` wrote equal those in the collected TEI folder."""
    differ = []
    for entry in added:
        path = Path(entry["tei"])
        other = tei_path_in(collected, entry)
        for mine, theirs in ((path, other), (receipt_path(path), receipt_path(other))):
            if not theirs.is_file() or mine.read_bytes() != theirs.read_bytes():
                differ.append(theirs.name)
    if differ:
        raise SystemExit(
            f"{len(differ)} existing-corpus files differ from the collected TEI folder {collected} "
            f"(first: {differ[0]})."
        )
    return len(added)


def compare_with(file_set: dict[str, tuple[Path, Path]], what: str) -> dict[str, str]:
    """{name: "same"} for byte-identical pairs (mine, reference); a difference stops with a refusal."""
    found = {}
    for name, (mine, reference) in file_set.items():
        same = mine.read_bytes() == reference.read_bytes()
        found[name] = "same" if same else "DIFFERS"
    bad = [n for n, v in found.items() if v != "same"]
    if bad:
        raise SystemExit(f"{what} differs from the reference in: {', '.join(bad)}.")
    return found


def added_without_folder(path: Path) -> list[dict]:
    """The added list with each TEI path cut to its place below the GROBID folder (a path is a run detail)."""
    rows = read(path)
    for row in rows:
        row["tei"] = re.sub(r"^.*/grobid/", "", row["tei"])
    return rows


def bundle(
    *,
    out: Path,
    version: str,
    ledgers: Path,
    closure_set: Path,
    closure_tei: Path,
    left_out_tei: Path,
    audit_inputs: Path,
    seeds: Path,
    seed_inputs: Path,
    identity: Path,
    loss_audit: Path,
    audit_reference: Path | None,
    identity_reference: Path | None,
) -> dict:
    """Run the steps above into OUT and return their counts."""
    out = Path(out)
    audit = out / "audit"
    counts: dict = {}
    started = time.time()

    step("1. papers whose text is another paper's")
    candidates = read(audit_inputs / ALL_CANDIDATES)
    record = left_out.find_duplicates(
        seed_inputs / SEED_TEI, candidates, [closure_tei, left_out_tei]
    )
    duplicates = audit / DUPLICATES
    audit.mkdir(parents=True)
    left_out.write_duplicates(duplicates, record)
    rows = left_out.exclusions(
        record["duplicates"],
        read(audit_inputs / PAPERS_INFO),
        read(audit_inputs / UNCITED),
    )
    left_out.write_exclusions(audit / EXCLUSIONS, rows)
    counts["duplicates"] = {
        "seeds": record["seeds"],
        "candidates": record["candidates"],
        "of_a_seed": sum(d["kind"] == "seed" for d in record["duplicates"]),
        "of_another_candidate": sum(
            d["kind"] == "closure" for d in record["duplicates"]
        ),
    }
    counts["exclusions"] = left_out.counts(rows)

    step("2. works found only in the existing corpus")
    before = read(ledgers / "before.json")
    added = existing.add_existing(
        before,
        seed_inputs / EXISTING_ZIP,
        seed_inputs / EXISTING_SHA256,
        out / "existing" / collect_grobid.TEI_FOLDER,
        set(read(audit_inputs / EXISTING_EXCLUDE)),
    )
    existing.write_added(audit / EXISTING_ADDED, added)
    counts["existing"] = {
        "existing_only_works": len(existing.existing_only(before)),
        "added": len(added),
        "equal_to_collected": check_existing_collected(added, closure_tei),
    }

    step("3. which seed definition cites which cited work")
    result = definition_citations.build(
        inputs=ledgers,
        provider=ledgers / PROVIDER,
        downloads=ledgers / DOWNLOADS,
        grobid_ledger=ledgers / GROBID_LEDGER,
        union_dataset=ledgers / UNION_DATASET,
        rows_all=closure_set / ROWS_ALL,
        seeds_derived=seeds / DERIVED,
        exclusions=audit / EXCLUSIONS,
        existing_added=audit / EXISTING_ADDED,
    )
    summary = definition_citations.write(result, ledgers, out / "links")
    counts["links"] = {
        "works": summary["works"],
        "link_rows": summary["link_rows"],
        "closure_papers_linked_to_a_work": summary["closure_papers_linked_to_a_work"],
        "grouping_identical": summary["grouping_check_against_audit"]["identical"],
    }

    step("4. the PDF identity check's decisions")
    counts["identity"] = run_identity(identity, out / "identity")

    step("5. the release folder")
    facts = release_folder.build(
        out=out / "release",
        closure_set=closure_set,
        seeds_derived=seeds / DERIVED,
        seeds_rows=seed_inputs / SEED_ROWS,
        seed_tei=seed_inputs / SEED_TEI,
        closure_tei=closure_tei,
        links=out / "links",
        version=version,
        exclusions=audit / EXCLUSIONS,
        seeds_compare=seeds / COMPARE,
        extra=(loss_audit,),
    )
    listed = release_folder.finish(out / "release", facts)
    counts["release"] = {
        "manifest_files": listed,
        "verified": release_folder.verify(out / "release"),
        "seed_papers": facts["seeds"]["papers"],
        "closure_papers": facts["closure"]["papers"],
    }

    if audit_reference:
        reference = Path(audit_reference)
        names = {DUPLICATES: DUPLICATES_RELEASED, EXCLUSIONS: EXCLUSIONS}
        same = compare_with(
            {n: (audit / n, reference / r) for n, r in names.items()},
            "an audit result",
        )
        mine = added_without_folder(audit / EXISTING_ADDED)
        theirs = added_without_folder(reference / EXISTING_ADDED)
        if mine != theirs:
            raise SystemExit(
                f"{EXISTING_ADDED} differs from the reference, apart from its folder paths."
            )
        same[EXISTING_ADDED] = "same after cutting the folder from each TEI path"
        counts["audit_reference"] = same
    if identity_reference:
        reference = Path(identity_reference)
        names = [
            f"{folder}/{n}".lstrip("/")
            for folder, _ in rounds_of(identity)
            for n in IDENTITY_FILES
        ]
        names += ["accepted-all.json", "accepted-all-r3.json"]
        pairs = {
            n: (out / "identity" / n, reference / n)
            for n in names
            if (reference / n).exists()
        }
        counts["identity_reference"] = compare_with(pairs, "an identity decision")
    counts["seconds"] = round(time.time() - started)
    return counts


# The inputs of `closure bundle` and `closure inputs`, by their name in the commands: what each holds, and, for the ones data/data-root.tsv does not list, why they are not public.
INPUTS = {
    "ledgers": "the closure run's ledgers (links-inputs/)",
    "earlier_ledgers": "the closure run's earlier ledgers, as they were when the extracted batches ran (links-inputs-v1/)",
    "closure_set": "the closure papers' extraction (rows-all.json, batches.json, derived/)",
    "closure_tei": "the closure papers' TEI files and their conversion receipts",
    "left_out_tei": "the TEI files of the papers moved aside after the duplicate check",
    "audit_inputs": "the four lists of the released closure (closure inputs writes them)",
    "seeds": "the seed run's variants (derived/)",
    "seed_inputs": "the seed run's inputs (rows, manifest, the seed papers' TEIs, the earlier corpus)",
    "identity": "the identity checks' evidence and verdicts",
    "loss_audit": "the loss audit copied into the release",
}
NOT_PUBLIC = {
    "closure_set": "its records are not in the public deposit",
    "closure_tei": "they hold the papers' text",
    "left_out_tei": "they hold the papers' text",
}


def missing_input(name: str, path: Path) -> str:
    """The message for an input of `closure bundle` or `closure inputs` that is not there: what it is, whether it is public, and where the docs list every input and its flag."""
    what = f"{path} does not exist ({name}): {INPUTS.get(name, name)}."
    if name in NOT_PUBLIC:
        return (
            f"{what} It is not public: {NOT_PUBLIC[name]}; give your own with "
            f"--{name.replace('_', '-')} (docs/step-4-closure/closure.md lists every input and its flag)."
        )
    return (
        f"{what} `mbdef data fetch frozen/` gets the closure inputs that data/data-root.tsv "
        "lists (docs/step-4-closure/closure.md lists every input and its flag)."
    )


def command(
    out: str,
    version: str,
    ledgers: str | None,
    closure_set: str | None,
    closure_tei: str | None,
    left_out_tei: str | None,
    audit_inputs: str | None,
    seeds: str | None,
    seed_inputs: str | None,
    identity: str | None,
    loss_audit: str | None,
    audit_reference: str | None,
    identity_reference: str | None,
) -> int:
    """`closure bundle`: rebuild the release into OUT from the closure run's saved files."""
    folder = Path(out)
    if folder.exists() and any(folder.iterdir()):
        raise SystemExit(f"{folder} is not empty; give a new or empty folder.")
    root = files.data_root()
    given = {
        "ledgers": Path(ledgers) if ledgers else root / LEDGERS,
        "closure_set": Path(closure_set) if closure_set else root / CLOSURE_SET,
        "closure_tei": Path(closure_tei) if closure_tei else root / CLOSURE_TEI,
        "left_out_tei": Path(left_out_tei) if left_out_tei else root / LEFT_OUT_TEI,
        "audit_inputs": Path(audit_inputs) if audit_inputs else root / AUDIT_INPUTS,
        "seeds": Path(seeds) if seeds else root / SEEDS,
        "seed_inputs": Path(seed_inputs) if seed_inputs else root / SEED_INPUTS,
        "identity": Path(identity) if identity else root / IDENTITY,
        "loss_audit": Path(loss_audit) if loss_audit else LOSS_AUDIT,
    }
    for name, path in given.items():
        if not path.exists():
            raise SystemExit(missing_input(name, path))
    counts = bundle(
        out=folder,
        version=version,
        audit_reference=Path(audit_reference) if audit_reference else None,
        identity_reference=Path(identity_reference) if identity_reference else None,
        **given,
    )
    inputs = {
        name: path
        for name, path in given.items()
        if name not in ("seeds", "seed_inputs")
    } | {
        "seeds_derived": given["seeds"] / DERIVED,
        "seeds_compare": given["seeds"] / COMPARE,
        "seed_rows": given["seed_inputs"] / SEED_ROWS,
        "seed_tei": given["seed_inputs"] / SEED_TEI,
        "existing_zip": given["seed_inputs"] / EXISTING_ZIP,
    }
    receipt = files.receipt(
        "mbdef closure bundle",
        {
            "version": version,
            "duplicate_share": left_out.SHARE,
            "shingle_words": left_out.SHINGLE_WORDS,
            "shingle_step": left_out.SHINGLE_STEP,
            "audit_reference": audit_reference,
            "identity_reference": identity_reference,
        },
        inputs,
        {name: folder / name for name in ("audit", "links", "identity", "release")},
    )
    receipt["summary"] = counts
    files.write_json(folder / files.RECEIPT, receipt)
    print(json.dumps(counts, indent=2, sort_keys=True))
    print(
        f"release: {folder / 'release'} ({counts['release']['manifest_files']} files verified)"
    )
    return 0
