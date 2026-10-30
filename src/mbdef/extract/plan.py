"""`mbdef extract plan`: a corpus run's shards and the jobs that make its variants.

A corpus (the 3,869 seed papers, or the closure's papers) is extracted in **shards**: groups of papers that one job reads at a time. The plan cuts the papers into shards of about the same work and lists, for every shard, the commands that make each stage the variants need, so the run can be done one shard at a time on any machine and then pooled by `extract derive --set OUT`:

1. **Rows.** One row per paper: its id, its title (`frozen_title`) and its TEI's sha256. From `--rows` (a JSON list; every key of a row is kept as given, and a row's `tei_sha256` must be its TEI's), else from every TEI in `--teis` with the TEI's own title. `--paper` keeps only the given papers.
2. **Windows.** Each paper's text is cut into windows at the three sizes of the released programs (`WINDOWS`), as `extract run` cuts it, and counted.
3. **Shards.** ceil(papers / `--shard-papers`) shards. The papers, longest first (16k plus 32k windows, then paper id), go one by one to the shard with the least work so far (then the fewest papers, then the first); a paper's work is its 16k and 32k windows plus half its 8k windows, rounded down. Each shard lists its papers by id. This is the recorded seed run's rule: with its rows it gives its 97 shards byte for byte.
4. **Jobs.** For each part of each variant in the variants file (`--picks`) and each stage it needs, with the stages these read first (`STAGES`): the folder the stage's per-paper files go to and the command that writes them. `combined` is `extract run` (the folder is the part's folder in its output), `verified-self` and `verified-qwen` are `extract verify` with the part's own model or Qwen, `filtered-<filter>` filters `verified-self`, `-noverify` the union, `-qv` `verified-qwen`.

Writes into `--out` (new or empty), the run's folder:

    shards/<prefix>-NNN.json     one shard's rows, sorted by paper id (indent 1, sorted keys)
    plans/grid.json              each shard's manifest, sha256, papers and windows (`inputs`), the
                                 jobs (`jobs`, the folder and command with `{shard}`) and settings
    plans/jobs.tsv               every job of every shard: shard, part, stage, the model server it
                                 needs (`-` for none) and the command, a shard's jobs in order
    picks.json                   the variants (a copy of --picks)
    paper-metadata.json          each paper's DOI for `extract derive`: a copy of --metadata, else
                                 the first DOI its TEI names (`parse_tei`; the header's, if any)
    windows-per-paper.json       each paper's windows at the three sizes
    receipt.json                 settings, input hashes, commit

The commands name the folders as given to this command, so run them from the same folder.
"""

import shlex
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from mbdef import files, models
from mbdef.extract import derive, train
from mbdef.extract import run as extract
from mbdef.extract.papers import parse_tei, read_teis, windows

SHARD_PAPERS = 40  # papers per shard, on average (the seed run's)
PREFIX = "shard"
PICKS = Path("data/extractors/picks.json")  # the released variants
# The windows of the released programs (train.WINDOWS): "characters/overlap".
WINDOWS = tuple(f"{size}/{overlap}" for size, overlap in train.WINDOWS.values())
# The stages a part can have, in an order in which each comes after the one it reads.
STAGES = (
    "combined",
    "verified-self",
    "filtered-title",
    "filtered-ordinary",
    "filtered-title-noverify",
    "filtered-ordinary-noverify",
    "verified-qwen",
    "filtered-title-qv",
    "filtered-ordinary-qv",
)
# The released parts' names (the released TEST run's arms and formats) -> `extract run` arguments.
RUN_PARTS = {"TR-bootstrap-8k": "w8k", "TR-bootstrap-16k32k": "w16k+w32k"}
RUN_FORMATS = {"chat-default-think": "chat", "json-default-think": "json"}
# A part's model -> the name and revision the server runs.
MODELS = {"gemma12b": models.PINNED["gemma"], "qwen": models.PINNED["qwen"]}
COMMAND = "mbdef"
WORKERS = 16  # TEIs parsed at the same time


def count_windows(tei: bytes) -> tuple[dict[str, int], str]:
    """A TEI's windows at each size of WINDOWS, and the first DOI it names (`parse_tei`)."""
    parsed = parse_tei(tei)
    counts = {
        key: len(windows(parsed["text"], *map(int, key.split("/")))) for key in WINDOWS
    }
    return counts, parsed["doi"]


def load_rows(
    teis: Path, rows_path: Path | None, paper_ids: list[str] | None
) -> tuple[list[dict], dict[str, bytes]]:
    """The rows (module docstring, 1) and every paper's TEI."""
    if rows_path:
        rows = derive.read(rows_path)
        if any(not {"paper_id", "frozen_title"} <= set(row) for row in rows):
            raise SystemExit(
                f"Every row of {rows_path} needs paper_id and frozen_title."
            )
        if paper_ids:
            missing = sorted(set(paper_ids) - {r["paper_id"] for r in rows})
            if missing:
                raise SystemExit(f"{rows_path} has no row for: {', '.join(missing)}")
            rows = [r for r in rows if r["paper_id"] in set(paper_ids)]
        found = read_teis(teis, [r["paper_id"] for r in rows])
        for row in rows:
            digest = files.sha256(found[row["paper_id"]])
            if row.get("tei_sha256", digest) != digest:
                raise SystemExit(
                    f"{row['paper_id']}: its TEI's sha256 is not the row's."
                )
    else:
        found = read_teis(teis, paper_ids)
        rows = []
        for pid, tei in found.items():
            rows.append(
                {
                    "frozen_title": parse_tei(tei)["title"],
                    "paper_id": pid,
                    "tei_sha256": files.sha256(tei),
                }
            )
    ids = [r["paper_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("Two rows have the same paper id.")
    empty = [r["paper_id"] for r in rows if not r["frozen_title"].strip()]
    if empty:
        raise SystemExit(
            f"No title for: {', '.join(empty)}; give them one with --rows (frozen_title)."
        )
    return rows, found


def cut_shards(rows: list[dict], counts: dict[str, dict], shard_papers: int):
    """The shards (module docstring, 3): lists of rows, each sorted by paper id."""
    n = max(1, -(-len(rows) // shard_papers))

    def long_windows(row):
        return (
            counts[row["paper_id"]]["16000/2000"]
            + counts[row["paper_id"]]["32000/4000"]
        )

    shards, load = [[] for _ in range(n)], [0] * n
    for row in sorted(rows, key=lambda r: (-long_windows(r), r["paper_id"])):
        i = min(range(n), key=lambda k: (load[k], len(shards[k]), k))
        shards[i].append(row)
        load[i] += long_windows(row) + counts[row["paper_id"]]["8000/1000"] // 2
    return [sorted(shard, key=lambda r: r["paper_id"]) for shard in shards]


def stage_input(stage: str) -> str | None:
    """The stage a stage reads (module docstring, 4)."""
    if stage == "combined":
        return None
    if stage.startswith("verified-") or stage.endswith("-noverify"):
        return "combined"
    return "verified-qwen" if stage.endswith("-qv") else "verified-self"


def make_jobs(picks: dict, set_dir: str, teis: str) -> list[dict]:
    """Every part's jobs (module docstring, 4), with `{shard}` in folders and commands."""
    needed: dict[str, tuple[dict, set[str]]] = {}
    for variant in picks["picks"].values():
        if not variant.get("pick"):
            continue
        for part in variant["parts"]:
            stages = needed.setdefault(derive.part_id(part), (part, set()))[1]
            stage = variant["stage"]
            while stage:
                if stage not in STAGES:
                    raise SystemExit(
                        f"Unknown stage {stage}; known: {', '.join(STAGES)}."
                    )
                stages.add(stage)
                stage = stage_input(stage)
    jobs = []
    for pid, (part, stages) in needed.items():
        if part["arm"] not in RUN_PARTS or part["format"] not in RUN_FORMATS:
            raise SystemExit(
                f"No `extract run` part for {pid}; add its arm to RUN_PARTS and its format "
                "to RUN_FORMATS in extract/plan.py."
            )
        model, revision = MODELS[part["model"]]
        own_model = (
            ["--model", model, "--revision", revision] if model != extract.MODEL else []
        )
        base = f"runs/{pid}/{{shard}}"
        folders = {s: f"{base}/{s}" for s in stages}
        folders["combined"] = f"{base}/extract/{RUN_PARTS[part['arm']]}"
        for stage in (s for s in STAGES if s in stages):
            out = f"{set_dir}/{folders[stage]}"
            given = stage_input(stage)
            source = f"{set_dir}/{folders[given]}" if given else ""
            if stage == "combined":
                server = part["model"]
                command = [
                    *("extract", "run", "--teis", teis),
                    *("--shard", f"{set_dir}/shards/{{shard}}.json"),
                    *("--part", RUN_PARTS[part["arm"]]),
                    *("--format", RUN_FORMATS[part["format"]]),
                    *("--out", f"{set_dir}/{base}/extract", *own_model),
                ]
            elif stage.startswith("verified-"):
                server = part["model"] if stage == "verified-self" else "qwen"
                verifier = (
                    ["--model", model, "--revision", revision]
                    if server != "qwen"
                    else []
                )
                command = [
                    *("extract", "verify", "--extraction", source, "--out", out),
                    *verifier,
                ]
            else:
                server = None
                name = stage.removeprefix("filtered-").removesuffix("-noverify")
                name = name.removesuffix("-qv")
                records = ["--records", "union"] if stage.endswith("-noverify") else []
                command = [
                    *("extract", "filter", "--run", source, "--out", out),
                    *("--filter", name, *records),
                ]
            jobs.append(
                {
                    "part": pid,
                    "stage": stage,
                    "input": given,
                    "model": server,
                    "folder": folders[stage],
                    "command": [COMMAND, *command],
                }
            )
    return jobs


def plan(
    out: Path,
    teis: Path,
    rows_path: Path | None = None,
    paper_ids: list[str] | None = None,
    picks_path: Path = PICKS,
    shard_papers: int = SHARD_PAPERS,
    prefix: str = PREFIX,
    metadata: Path | None = None,
) -> int:
    """The command (module docstring). Returns 0."""
    out, teis, picks_path = Path(out), Path(teis), Path(picks_path)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; give a new or empty folder.")
    picks = derive.read(picks_path)
    jobs = make_jobs(picks, str(out), str(teis))
    rows, found = load_rows(teis, rows_path, paper_ids)
    with ProcessPoolExecutor(WORKERS) as pool:
        counted = dict(zip(found, pool.map(count_windows, found.values()), strict=True))
    counts = {pid: counted[pid][0] for pid in found}
    plan_file = {"schema": derive.PLAN_SCHEMA, "inputs": {}}
    for i, shard in enumerate(cut_shards(rows, counts, shard_papers)):
        key = f"{prefix}-{i:03d}"
        path = out / "shards" / f"{key}.json"
        files.write_json(path, shard, **derive.JSON_FORMAT)
        plan_file["inputs"][key] = {
            "manifest": f"shards/{key}.json",
            "manifest_sha256": files.file_sha256(path),
            "papers": len(shard),
            "windows": {
                k: sum(counts[r["paper_id"]][k] for r in shard) for k in WINDOWS
            },
        }
    plan_file["shard_papers"] = shard_papers
    plan_file["jobs"] = jobs
    files.write_json(out / "plans" / "grid.json", plan_file, **derive.JSON_FORMAT)
    lines = ["shard\tpart\tstage\tmodel\tcommand"]
    for shard in plan_file["inputs"]:
        for job in jobs:
            command = [arg.format(shard=shard) for arg in job["command"]]
            lines.append(
                f"{shard}\t{job['part']}\t{job['stage']}\t{job['model'] or '-'}\t"
                f"{shlex.join(command)}"
            )
    (out / "plans" / "jobs.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "picks.json").write_bytes(picks_path.read_bytes())
    if metadata:
        (out / "paper-metadata.json").write_bytes(Path(metadata).read_bytes())
    else:
        dois = {
            pid: {"tei_header_doi": doi}
            for pid, (_, doi) in sorted(counted.items())
            if doi
        }
        files.write_json(out / "paper-metadata.json", dois, **derive.JSON_FORMAT)
    files.write_json(out / "windows-per-paper.json", counts, **derive.JSON_FORMAT)
    sizes = [entry["papers"] for entry in plan_file["inputs"].values()]
    summary = {
        "papers": len(rows),
        "shards": len(sizes),
        "papers_per_shard": {"min": min(sizes), "max": max(sizes)},
        "windows": {k: sum(c[k] for c in counts.values()) for k in WINDOWS},
        "jobs_per_shard": len(jobs),
    }
    inputs = {"teis": teis, "picks": picks_path}
    if rows_path:
        inputs["rows"] = Path(rows_path)
    if metadata:
        inputs["metadata"] = Path(metadata)
    receipt = files.receipt(
        "mbdef extract plan",
        {
            "shard_papers": shard_papers,
            "prefix": prefix,
            "windows": list(WINDOWS),
            "papers": "given" if paper_ids else "all",
        },
        inputs,
        {"shards": out / "shards", "plans": out / "plans"},
    )
    receipt["summary"] = summary
    files.write_json(out / files.RECEIPT, receipt)
    print(
        f"{len(rows)} papers in {len(sizes)} shards of {min(sizes)} to {max(sizes)} papers; "
        f"{len(jobs)} jobs per shard in {out / 'plans' / 'jobs.tsv'}"
    )
    return 0
