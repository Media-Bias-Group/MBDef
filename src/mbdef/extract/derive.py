"""`mbdef extract derive`: a run's variants from its saved generations, and their comparison.

A **variant** (i.e., selected extractor's configuration) is one stage of one or more parts: the records a part kept after its verifier and filter (docs/step-3-extraction/extract.md), for example P≥28. The run's files store the variants under `picks`. The step reads every needed stage, without calling a model, from the run's per-paper outputs (a run planned by `extract plan`, whose plan names each stage's folder) or from its part tarballs (the recorded runs), and then:

1. `derive`: per variant and paper, the parts' records in the listed order, merged with `records.merge` when there is more than one part; one file per paper, `predictions.json` per variant, and the union of all variants per paper (the closure's input).
2. `compare`: the variants against each other, without gold: per paper, a definition is the pair (term, definition text) after NFKC, case folding and white space runs collapsed; pairwise shared and Jaccard, and which part each variant's definitions come from.

The run's folder (`--set`) holds `picks.json` (the variants), `plans/grid.json` (the shards, each naming its manifest of papers) and `shards/`, and then either the per-paper outputs, one folder per part, shard and stage as the plan's `jobs` name them (`runs/…/papers/<sha256 of the paper id>/` with the file of PAPER_FILES), or `parts/<model>/<arm>/<format>/<shard>.tar.zst`, plus `<shard>.qv.tar.zst` for a Gemma part, which holds the Qwen verifier's stages. Both give the same files for the same records. The default is the seed run under DATA_ROOT (docs/setup/data-root.md). The swap points are the folder and the variants file (`--picks`). The files have the released run's byte format (`JSON_FORMAT`).
"""

import copy
import json
import os
import re
import subprocess
import tarfile
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import quote

from mbdef import files
from mbdef.extract import records as record_rules
from mbdef.extract.run import paper_key

# The saved seed run under DATA_ROOT: the cluster run folder's corpus/seeds/ (its seed manifest is at inputs/seed-manifest.jsonl in the same run folder, see `metadata_files`).
SEEDS = Path("frozen/runs/production-20260925/corpus/seeds")
# The byte format of the released run's files.
JSON_FORMAT: dict[str, Any] = {"indent": 1, "sort_keys": True}
# The four output variants, the one spelling of their names that the extraction and the closure use: folder name -> name in a run's DONE.json (the folder name is `tag` of the name).
VARIANTS = {
    "P20": "P>=20",
    "P24": "P>=24",
    "P28": "P>=28",
    "P28-single-model": "P>=28-single-model",
}
# Stages of a Gemma part that come from its .qv tarball (the Qwen verifier's pass).
QV_STAGES = ("verified-qwen", "filtered-title-qv", "filtered-ordinary-qv")
# The schema of a plan that `extract plan` wrote: its stages are read from per-paper folders.
PLAN_SCHEMA = (
    "taxonomy_pipeline/extract-plan-1"  # a recorded id: it keeps its name (files.py)
)
# A stage folder of a new run, by the stage's first word: the per-paper file that `extract run` (`combined`), `extract verify` (`verified-…`) or `extract filter` (`filtered-…`) writes, and the key of its records.
PAPER_FILES = {
    "combined": ("union.json", "records"),
    "verified": ("verified.json", "definition_records"),
    "filtered": ("filtered.json", "definition_records"),
}
# The union's note, as the released union/predictions.json words it (it is part of that file).
UNION_NOTE = (
    "per paper: the three picks' records concatenated (P>=20, P>=24, P>=28) and merged with "
    "src.extractor.merge; closure --dataset"
)
WORKERS = min(16, os.process_cpu_count() or 1)  # tarballs read at the same time
# The keys of a variant that its predictions.json repeats.
VARIANT_KEYS = (
    "pick",
    "config",
    "model",
    "stage",
    "stage_recipe",
    "pool",
    "parts",
    "test",
    "valid",
)


def read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    files.write_json(path, value, **JSON_FORMAT)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def part_id(part: dict) -> str:
    return f"{part['model']}/{part['arm']}/{part['format']}"


def tag(name: str) -> str:
    """A variant's folder: "P>=28-single-model" -> "P28-single-model"."""
    return re.sub(r"[^A-Za-z0-9._=-]+", "_", name.replace(">=", "")).strip("_")


def archive(set_dir: Path, part: dict, shard: str, qv: bool) -> Path:
    suffix = ".qv" if qv else ""
    return set_dir / "parts" / part_id(part) / f"{shard}{suffix}.tar.zst"


def read_archive(path: Path, shard: str, stages: list[str]) -> dict[str, dict]:
    """{stage: {paper id: records}} from one tarball: its files `cohorts/<shard>/<model>/<arm>/<format>/stages/<stage>/<paper>.json` (not status.json)."""
    found: dict[str, dict] = {stage: {} for stage in stages}
    pattern = re.compile(
        rf"cohorts/{re.escape(shard)}/[^/]+/[^/]+/[^/]+/stages/([^/]+)/([^/]+\.json)"
    )
    with subprocess.Popen(
        ["zstd", "-d", "-q", "-c", str(path)], stdout=subprocess.PIPE
    ) as zstd:
        assert zstd.stdout is not None
        with tarfile.open(fileobj=zstd.stdout, mode="r|") as tar:
            for member in tar:
                hit = pattern.fullmatch(member.name)
                if not member.isfile() or not hit or hit[2] == "status.json":
                    continue
                if hit[1] not in found:
                    continue
                handle = tar.extractfile(member)
                assert handle is not None
                value = json.loads(handle.read())
                if value["paper_id"] in found[hit[1]]:
                    raise SystemExit(
                        f"{path} has two {hit[1]} files of {value['paper_id']}."
                    )
                found[hit[1]][value["paper_id"]] = value["definition_records"]
        zstd.stdout.read()
    if zstd.returncode != 0:
        raise SystemExit(f"Cannot unpack {path} (zstd exit code {zstd.returncode}).")
    return found


def read_folder(folder: Path, stage: str, paper_ids: list[str]) -> dict[str, list]:
    """{paper id: records} of one stage folder: `papers/<sha256 of the paper id>/<file>`, the file of PAPER_FILES. A paper without its file has no output."""
    name, key = PAPER_FILES[stage.split("-")[0]]
    found = {}
    for pid in paper_ids:
        path = folder / "papers" / paper_key(pid) / name
        if path.is_file():
            value = read(path)
            if value["paper_id"] != pid:
                raise SystemExit(
                    f"{path} is the file of {value['paper_id']}, not {pid}."
                )
            found[pid] = value[key]
    return found


def read_folders(
    set_dir: Path, plan: dict, needed: dict[str, tuple[dict, set[str]]]
) -> tuple[dict[str, dict[str, dict]], list[str]]:
    """As `read_stages`, from the per-paper outputs of a run: the folder of each part's stage of each shard is the plan's `jobs` entry for that part and stage (`{shard}` filled in)."""
    folders = {(job["part"], job["stage"]): job["folder"] for job in plan["jobs"]}
    manifests = shard_manifests(set_dir, plan)
    loaded = {pid: {s: {} for s in stages} for pid, (_, stages) in needed.items()}
    problems = []
    for pid, (_part, stages) in needed.items():
        for stage in sorted(stages):
            if (pid, stage) not in folders:
                raise SystemExit(
                    f"The plan names no folder for stage {stage} of {pid}."
                )
            for shard, rows in manifests.items():
                folder = set_dir / folders[pid, stage].format(shard=shard)
                if not folder.is_dir():
                    problems.append(f"missing folder {folder.relative_to(set_dir)}")
                    continue
                papers = [row["paper_id"] for row in rows]
                loaded[pid][stage].update(read_folder(folder, stage, papers))
    return loaded, problems


def read_stages(
    set_dir: Path, plan: dict, needed: dict[str, tuple[dict, set[str]]]
) -> tuple[dict[str, dict[str, dict]], list[str]]:
    """{part id: {stage: {paper id: records}}} of every needed part and stage, and the problems (missing tarballs or folders). A run planned by `extract plan` (its plan's schema is PLAN_SCHEMA) is read from its per-paper folders (`read_folders`); a recorded run's stages come from the part tarballs, a Gemma part's Qwen-verifier stages from its .qv tarballs."""
    if plan.get("schema") == PLAN_SCHEMA:
        return read_folders(set_dir, plan, needed)
    shards = sorted(plan["inputs"])
    jobs, problems = [], []
    for pid, (part, stages) in needed.items():
        gemma = part["model"] == "gemma12b"
        groups = {
            False: sorted(s for s in stages if not (gemma and s in QV_STAGES)),
            True: sorted(s for s in stages if gemma and s in QV_STAGES),
        }
        for shard in shards:
            for qv, wanted in groups.items():
                if not wanted:
                    continue
                path = archive(set_dir, part, shard, qv)
                if path.is_file():
                    jobs.append((pid, path, shard, wanted))
                else:
                    problems.append(f"missing archive {path.relative_to(set_dir)}")
    loaded = {pid: {s: {} for s in stages} for pid, (_, stages) in needed.items()}
    with ProcessPoolExecutor(WORKERS) as pool:
        results = pool.map(
            read_archive,
            [j[1] for j in jobs],
            [j[2] for j in jobs],
            [j[3] for j in jobs],
        )
        for (pid, *_), found in zip(jobs, results, strict=True):
            for stage, papers in found.items():
                loaded[pid][stage].update(papers)
    return loaded, problems


def shard_manifests(set_dir: Path, plan: dict) -> dict[str, list[dict]]:
    """{shard: rows} of every shard manifest. The plan names each manifest by a path (the recorded run's by its path on the machine that ran it); it is read from `shards/` with that file name, and its sha256 checked."""
    manifests = {}
    for shard in sorted(plan["inputs"]):
        entry = plan["inputs"][shard]
        path = set_dir / "shards" / Path(entry["manifest"]).name
        data = path.read_bytes()
        expected = entry.get("manifest_sha256")
        if expected and files.sha256(data) != expected:
            raise SystemExit(f"{path} is not the manifest the plan names (sha256).")
        manifests[shard] = json.loads(data)
    return manifests


def shard_rows(set_dir: Path, plan: dict) -> dict[str, dict]:
    """{paper id: row} of every shard manifest (`shard_manifests`)."""
    return {
        row["paper_id"]: row
        for rows in shard_manifests(set_dir, plan).values()
        for row in rows
    }


def metadata_files(set_dir: Path) -> dict[str, Path]:
    """The files that give the papers' DOIs, those that are there: the run's seed manifest (`<run>/inputs/seed-manifest.jsonl`, the run folder being two levels above the set folder), and the set's `paper-metadata.json` (a closure run has one)."""
    file_set = {
        "seed-manifest": set_dir.parent.parent / "inputs" / "seed-manifest.jsonl",
        "paper-metadata": set_dir / "paper-metadata.json",
    }
    return {name: path for name, path in file_set.items() if path.is_file()}


def paper_metadata(set_dir: Path) -> dict[str, dict]:
    """{paper id: metadata} from `metadata_files`; the set's own file wins."""
    meta = {}
    found = metadata_files(set_dir)
    if "seed-manifest" in found:
        for line in found["seed-manifest"].read_text(encoding="utf-8").splitlines():
            if line.strip():
                m = json.loads(line)
                meta[m["paper_id"]] = m
    if "paper-metadata" in found:
        meta.update(read(found["paper-metadata"]))
    return meta


def doi_of(metadata: dict) -> str | None:
    return metadata.get("tei_header_doi") or metadata.get("doi") or None


def derive(set_dir: Path, out: Path, picks_path: Path) -> dict:
    """Write the variants of the run in SET_DIR into OUT (the run's `derived/`); returns the summary that is also written as OUT/DONE.json."""
    picks = read(picks_path)
    plan = read(set_dir / "plans" / "grid.json")
    rows = shard_rows(set_dir, plan)
    meta = paper_metadata(set_dir)
    needed: dict[str, tuple[dict, set[str]]] = {}
    for variant in picks["picks"].values():
        if not variant.get("pick"):
            continue
        for part in variant["parts"]:
            needed.setdefault(part_id(part), (part, set()))[1].add(variant["stage"])
    loaded, problems = read_stages(set_dir, plan, needed)

    summary = {
        "schema": "production-20260925-derived/1",
        "set": set_dir.name,
        "papers": len(rows),
        "picks": {},
        "problems": problems,
        "picks_sha256": files.digest(picks_path),
        "created_utc": now(),
    }
    union_records: dict[str, list] = {pid: [] for pid in rows}
    for name, variant in picks["picks"].items():
        if not variant.get("pick"):
            summary["picks"][name] = {"pick": None}
            continue
        folder = out / tag(name)
        papers, missing = [], []
        for pid in sorted(rows):
            flat, counts = [], {}
            for part in variant["parts"]:
                found = loaded[part_id(part)][variant["stage"]].get(pid)
                if found is None:
                    missing.append(f"{pid} {part_id(part)}")
                    found = []
                counts[part_id(part)] = len(found)
                flat += copy.deepcopy(found)  # merge changes the records it is given
            kept = flat if len(variant["parts"]) == 1 else record_rules.merge(flat)
            title = rows[pid]["frozen_title"]
            doi = doi_of(meta.get(pid, {}))
            write(
                folder / "papers" / f"{quote(pid, safe='')}.json",
                {
                    "paper_id": pid,
                    "pick": name,
                    "config": variant["pick"],
                    "stage": variant["stage"],
                    "parts": [part_id(p) for p in variant["parts"]],
                    "part_record_counts": counts,
                    "pool": variant["pool"]["merge"],
                    "title": title,
                    "definition_records": kept,
                },
            )
            papers.append(
                {
                    "paper_id": pid,
                    "metadata": {"title": title, **({"doi": doi} if doi else {})},
                    "definition_records": kept,
                }
            )
            union_records[pid].extend(copy.deepcopy(kept))
        predictions = folder / "predictions.json"
        write(
            predictions,
            {
                "paper_count": len(papers),
                "papers": papers,
                "pick": {k: variant[k] for k in VARIANT_KEYS},
                "missing_part_outputs": missing,
            },
        )
        summary["picks"][name] = {
            "pick": variant["pick"],
            "papers": len(papers),
            "records": sum(len(p["definition_records"]) for p in papers),
            "definitions": sum(
                len(r["definitions"]) for p in papers for r in p["definition_records"]
            ),
            "missing_part_outputs": len(missing),
            "predictions": str(predictions.resolve()),
            "predictions_sha256": files.digest(predictions),
        }
    union = []
    for pid, found in sorted(union_records.items()):
        doi = doi_of(meta.get(pid, {}))
        metadata = {"title": rows[pid]["frozen_title"], **({"doi": doi} if doi else {})}
        union.append(
            {
                "paper_id": pid,
                "metadata": metadata,
                "definition_records": record_rules.merge(found),
            }
        )
    union_path = out / "union" / "predictions.json"
    write(union_path, {"paper_count": len(union), "papers": union, "note": UNION_NOTE})
    summary["union"] = {
        "papers": len(union),
        "records": sum(len(p["definition_records"]) for p in union),
        "predictions": str(union_path.resolve()),
        "predictions_sha256": files.digest(union_path),
    }
    write(out / "DONE.json", summary)
    return summary


def norm_text(text: str) -> str:
    return re.sub(
        r"\s+", " ", unicodedata.normalize("NFKC", text or "").casefold()
    ).strip()


def definition_keys(found: list[dict]) -> set[tuple[str, str]]:
    """A paper's definitions as compare counts them: (term, text), normalised (`norm_text`)."""
    return {
        (norm_text(r.get("term", "")), norm_text(d.get("text", "")))
        for r in found
        for d in r.get("definitions") or []
    }


def compare(set_dir: Path, derived: Path, out: Path, picks_path: Path) -> dict:
    """Compare the variants in DERIVED (written by `derive`) without gold; write compare.json, per-paper.json and compare.md into OUT; returns compare.json's content."""
    picks = read(picks_path)
    preds, keys = {}, {}
    for name, variant in picks["picks"].items():
        if not variant.get("pick"):
            continue
        saved = read(derived / tag(name) / "predictions.json")
        preds[name] = {p["paper_id"]: p["definition_records"] for p in saved["papers"]}
        keys[name] = {pid: definition_keys(r) for pid, r in preds[name].items()}
    names = sorted(preds)
    papers = sorted(set().union(*[set(v) for v in preds.values()])) if preds else []
    per_paper = []
    for pid in papers:
        row: dict = {"paper_id": pid}
        for n in names:
            found = preds[n].get(pid, [])
            row[n] = {
                "records": len(found),
                "definitions": sum(len(r.get("definitions") or []) for r in found),
                "keys": len(keys[n].get(pid, ())),
            }
        per_paper.append(row)
    totals = {
        n: {
            "records": sum(r[n]["records"] for r in per_paper),
            "definitions": sum(r[n]["definitions"] for r in per_paper),
            "unique_definition_keys": sum(r[n]["keys"] for r in per_paper),
            "papers_with_zero_definitions": sum(
                1 for r in per_paper if r[n]["definitions"] == 0
            ),
        }
        for n in names
    }
    pairwise = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = sum(
                len(keys[a].get(p, set()) & keys[b].get(p, set())) for p in papers
            )
            only_a = sum(
                len(keys[a].get(p, set()) - keys[b].get(p, set())) for p in papers
            )
            only_b = sum(
                len(keys[b].get(p, set()) - keys[a].get(p, set())) for p in papers
            )
            total = shared + only_a + only_b
            pairwise[f"{a} vs {b}"] = {
                "shared": shared,
                f"only_{a}": only_a,
                f"only_{b}": only_b,
                "jaccard": round(shared / total, 4) if total else None,
            }
    # Counted only for exactly three variants, as the released comparison did (null with four).
    all_three = (
        sum(
            len(set.intersection(*[keys[n].get(p, set()) for n in names]))
            for p in papers
        )
        if len(names) == 3
        else None
    )
    union_keys = (
        sum(len(set.union(*[keys[n].get(p, set()) for n in names])) for p in papers)
        if names
        else 0
    )
    # Which parts' stage records hold each of a variant's definitions.
    plan = read(set_dir / "plans" / "grid.json")
    needed: dict[str, tuple[dict, set[str]]] = {}
    for n in names:
        for part in picks["picks"][n]["parts"]:
            needed.setdefault(part_id(part), (part, set()))[1].add(
                picks["picks"][n]["stage"]
            )
    loaded, _problems = read_stages(set_dir, plan, needed)
    contribution = {}
    for n in names:
        variant = picks["picks"][n]
        part_keys = {
            part_id(p): {
                pid: definition_keys(r)
                for pid, r in loaded[part_id(p)][variant["stage"]].items()
            }
            for p in variant["parts"]
        }
        counts = {
            p: {"definitions_in_part": 0, "only_from_this_part": 0} for p in part_keys
        }
        for pid in papers:
            for key in keys[n].get(pid, ()):
                owners = [p for p, kk in part_keys.items() if key in kk.get(pid, set())]
                for p in owners:
                    counts[p]["definitions_in_part"] += 1
                if len(owners) == 1:
                    counts[owners[0]]["only_from_this_part"] += 1
        contribution[n] = counts
    report = {
        "schema": "production-20260925-compare/1",
        "set": set_dir.name,
        "papers": len(papers),
        "totals": totals,
        "pairwise_definition_keys": pairwise,
        "shared_by_all_three": all_three,
        "union_definition_keys": union_keys,
        "part_contribution": contribution,
        "key": "per paper (NFKC-casefolded, whitespace-collapsed term, definition text); "
        "no gold is used",
        "picks": {n: picks["picks"][n]["pick"] for n in names},
        "created_utc": now(),
    }
    write(out / "compare.json", report)
    write(out / "per-paper.json", per_paper)
    lines = [
        f"# Pick comparison on the {set_dir.name} set (no gold)",
        "",
        f"Papers: {len(papers)}",
        "",
        "| Pick | Config | Records | Definitions | Unique definition keys "
        "| Papers with 0 definitions |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for n in names:
        t = totals[n]
        lines.append(
            f"| {n} | {picks['picks'][n]['pick']} | {t['records']} | {t['definitions']} "
            f"| {t['unique_definition_keys']} | {t['papers_with_zero_definitions']} |"
        )
    lines += [
        "",
        "| Pair | Shared | Only first | Only second | Jaccard |",
        "|---|---:|---:|---:|---:|",
    ]
    for pair, v in pairwise.items():
        a, b = pair.split(" vs ")
        lines.append(
            f"| {pair} | {v['shared']} | {v[f'only_{a}']} | {v[f'only_{b}']} "
            f"| {v['jaccard']} |"
        )
    lines += [
        "",
        f"Definition keys shared by all three: {all_three}; union of all three: "
        f"{union_keys}.",
        "",
        "## Part contribution",
        "",
    ]
    for n, counts in contribution.items():
        lines.append(f"**{n}**")
        for p, v in counts.items():
            lines.append(
                f"- {p}: {v['definitions_in_part']} of the pick's definitions appear in "
                f"this part's stage output; {v['only_from_this_part']} only there"
            )
        lines.append("")
    (out / "compare.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run(out: Path, set_dir: Path | None = None, picks: Path | None = None) -> int:
    """The command: derive, then compare, into OUT (new or empty), with a receipt."""
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; give a new or empty folder.")
    default = not set_dir
    set_dir = Path(set_dir) if set_dir else files.data_root() / SEEDS
    plan = set_dir / "plans" / "grid.json"
    if not plan.is_file():
        if default:
            raise SystemExit(
                f"{plan} is not there: the seed run's plan and saved parts ($DATA_ROOT/{SEEDS}/plans/ "
                "and parts/) are not public: the parts hold paper text. To derive variants of your "
                "own run, run `extract plan` and the jobs it lists on your TEIs, then "
                "`extract derive --set` on that run (docs/step-3-extraction/extract.md)."
            )
        raise SystemExit(
            f"{plan} does not exist: give --set a run folder that `extract plan` wrote "
            "(docs/step-3-extraction/extract.md)."
        )
    picks = Path(picks) if picks else set_dir / "picks.json"
    summary = derive(set_dir, out / "derived", picks)
    report = compare(set_dir, out / "derived", out / "compare", picks)
    planned = read(set_dir / "plans" / "grid.json").get("schema") == PLAN_SCHEMA
    stages = "runs" if planned else "parts"  # a planned run's folders, or the tarballs
    inputs = {
        "picks": picks,
        "plans": set_dir / "plans",
        "shards": set_dir / "shards",
        stages: set_dir / stages,
    }
    for name, path in metadata_files(set_dir).items():
        inputs[name] = path
    counts = {
        name: {"records": v["records"], "definitions": v["definitions"]}
        for name, v in summary["picks"].items()
        if v["pick"]
    }
    receipt = files.receipt(
        "mbdef extract derive",
        {"set": str(set_dir), "json_format": JSON_FORMAT, "workers": WORKERS},
        inputs,
        {"derived": out / "derived", "compare": out / "compare"},
    )
    receipt["summary"] = {
        "papers": summary["papers"],
        **counts,
        "union": {"records": summary["union"]["records"]},
        "problems": summary["problems"],
    }
    files.write_json(out / files.RECEIPT, receipt)
    for name, counts in summary["picks"].items():
        if counts["pick"]:
            print(
                f"{name}: {counts['papers']} papers, {counts['records']} records, "
                f"{counts['definitions']} definitions"
            )
    print(
        f"union: {summary['union']['papers']} papers, {summary['union']['records']} records"
    )
    for pair, v in report["pairwise_definition_keys"].items():
        print(f"{pair}: {v['shared']} definitions shared, Jaccard {v['jaccard']}")
    for problem in summary["problems"]:
        print(f"PROBLEM: {problem}")
    print("wrote", out)
    return 0
