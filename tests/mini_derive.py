"""A miniature seed run for `extract derive`: two shards, three papers, three parts, four variants.

It has the layout of the saved seed run (the cluster run's `corpus/seeds/` folder, under `frozen/runs/production-20260925/` in the data root): `picks.json`, `plans/grid.json` (its shard manifests named by their cluster paths), `shards/`, and `parts/<model>/<arm>/<format>/` with one zstd tarball per shard, plus one `.qv` tarball per shard for the Gemma parts (the Qwen verifier's stages). Each tarball holds the stage files `cohorts/<shard>/<model>/<arm>/<format>/stages/<stage>/<paper>.json`, as the run wrote them.

The records are the worked example of docs/step-3-extraction/extract.md (paper `pdf-aaa`):

| Part | filtered-ordinary-noverify | verified-self | filtered-ordinary-qv |
|---|---|---|---|
| G8 (Gemma, w8k) | Media bias: 1 definition | Media bias: 1 | Media bias: 1 |
| G16 (Gemma, w16k+w32k) | News values: 1; MEDIA BIAS: the same definition in capitals | (none) | News values: 1 |
| Q (Qwen) | media bias: the same definition and a second one; Framing without a definition | | |
"""

import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path
from urllib.parse import quote

SEEDS = "frozen/runs/production-20260925/corpus/seeds"
MANIFEST = "frozen/runs/production-20260925/inputs/seed-manifest.jsonl"
RUN_ROOT = "/cluster/runs/production-20260925"  # a made-up run folder; the plan names its shards below it
FIELDS = (
    "definitions",
    "synonyms",
    "description",
    "related_terms",
    "subtypes",
    "supertypes",
    "examples",
)
G8 = {"arm": "TR-bootstrap-8k", "format": "chat-default-think", "model": "gemma12b"}
G16 = {
    "arm": "TR-bootstrap-16k32k",
    "format": "chat-default-think",
    "model": "gemma12b",
}
Q = {"arm": "A-minimal-8k", "format": "chat-default", "model": "qwen"}
PAPERS = {  # shard -> paper id -> title
    "shard-000": {"pdf-aaa": "Media bias in the news", "doi:10.1/b": "Framing effects"},
    "shard-001": {"pdf-ccc": "Gatekeeping"},
}
METADATA = [  # the seed manifest: a DOI from the TEI header, or one found elsewhere
    {
        "paper_id": "pdf-aaa",
        "doi": "10.9/aaa",
        "frozen_title": "Media bias in the news",
    },
    {"paper_id": "doi:10.1/b", "tei_header_doi": "10.1/b", "doi": "10.1/other"},
    {"paper_id": "pdf-ccc"},
]

SLANTED = "Media bias is slanted news coverage."
SELECTION = "Bias in the selection of news."
NEWS_VALUES = "News values are criteria for selecting news."
FRAMING = "A framing effect is a change in judgement caused by wording."
GATEKEEPING = "Gatekeeping is the selection of news."


def record(term: str, *definitions: str, citation: tuple[str, ...] = ()) -> dict:
    """A definition record with every field, as the stage files hold them."""
    return {
        "term": term,
        **{field: [] for field in FIELDS},
        "definitions": [{"text": d, "citation": list(citation)} for d in definitions],
    }


# part -> stage -> paper -> records (a paper left out has no stage file)
STAGES = {
    "G8": {
        "filtered-ordinary-noverify": {
            "pdf-aaa": [record("Media bias", SLANTED)],
            "doi:10.1/b": [],
            "pdf-ccc": [record("Gatekeeping", GATEKEEPING)],
        },
        "verified-self": {
            "pdf-aaa": [record("Media bias", SLANTED)],
            "doi:10.1/b": [],
            "pdf-ccc": [record("Gatekeeping", GATEKEEPING)],
        },
        "filtered-ordinary-qv": {
            "pdf-aaa": [record("Media bias", SLANTED)],
            "doi:10.1/b": [],
            "pdf-ccc": [record("Gatekeeping", GATEKEEPING)],
        },
    },
    "G16": {
        "filtered-ordinary-noverify": {
            "pdf-aaa": [
                record("News values", NEWS_VALUES),
                record("MEDIA BIAS", SLANTED.upper()),
            ],
            "doi:10.1/b": [],
            "pdf-ccc": [],
        },
        "verified-self": {"pdf-aaa": [], "doi:10.1/b": [], "pdf-ccc": []},
        "filtered-ordinary-qv": {
            "pdf-aaa": [record("News values", NEWS_VALUES)],
            "doi:10.1/b": [],
            "pdf-ccc": [],
        },
    },
    "Q": {
        "filtered-ordinary-noverify": {
            "pdf-aaa": [
                record("media bias", SLANTED.lower(), SELECTION, citation=("[1]",)),
                record("Framing"),
            ],
            "doi:10.1/b": [record("Framing effect", FRAMING)],
            "pdf-ccc": [],
        },
        "verified-self": {"pdf-aaa": [], "doi:10.1/b": [], "pdf-ccc": []},
    },
}
PARTS = {"G8": G8, "G16": G16, "Q": Q}
QV_STAGES = ("verified-qwen", "filtered-title-qv", "filtered-ordinary-qv")
POOL = "per paper: concatenate the parts' stage records in the listed order, then src.extractor.merge"


def part_id(part: dict) -> str:
    return f"{part['model']}/{part['arm']}/{part['format']}"


def variant(stage: str, *names: str, **extra) -> dict:
    """One variant of picks.json, with the keys the saved run's variants have."""
    parts = [{**PARTS[n], "family": "trained", "profile": "x"} for n in names]
    config = f"mini/{'+'.join(names)}/{stage}"
    return {
        "config": config,
        "dev": {"recall": 0.8},
        "model": "gemma12b",
        "parts": parts,
        "pick": f"test/{config}",
        "pool": {"kind": "cross-arm pool", "merge": POOL, "n_parts": len(parts)},
        "provisional": False,
        "stage": stage,
        "stage_recipe": {"stage": stage},
        "test": {"recall": 0.81},
        "valid": True,
        **extra,
    }


PICKS = {
    "schema": "production-20260925-picks/1",
    "picks": {
        "P>=20": variant("filtered-ordinary-noverify", "G8", "Q"),
        "P>=24": variant("verified-self", "G8"),
        "P>=28": variant("filtered-ordinary-qv", "G8", "G16"),
        "P>=28-single-model": variant(
            "filtered-ordinary-noverify",
            "G8",
            "G16",
            additional_output=True,
            label="ADDITIONAL OUTPUT: single-model variant of P>=28",
        ),
    },
}


def json_bytes(value) -> bytes:
    """JSON as the run wrote it: indent 1, sorted keys, UTF-8, a final newline."""
    return (
        json.dumps(value, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()


def write_json(path: Path, value) -> bytes:
    data = json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def pack(members: dict[str, bytes], archive: Path) -> None:
    """A zstd tarball of `members` ({path: bytes}), as the run packed its parts."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.GNU_FORMAT) as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(data), 1758844800, 0o644
            tar.addfile(info, io.BytesIO(data))
    archive.parent.mkdir(parents=True, exist_ok=True)
    done = subprocess.run(
        ["zstd", "-q", "-c"], input=buffer.getvalue(), capture_output=True, check=True
    )
    archive.write_bytes(done.stdout)


def paper_key(paper_id: str) -> str:
    return hashlib.sha256(paper_id.encode()).hexdigest()


# A miniature corpus for `extract plan`: five synthetic TEIs whose text is one paragraph of N letters "a" (no line break, so a window of size s and overlap o ends at s characters, and N characters give 1 + ceil((N - s) / (s - o)) windows when N > s):
#
# | Paper | N | 8000/1000 | 16000/2000 | 32000/4000 | 16k + 32k | load |
# |---|---:|---:|---:|---:|---:|---:|
# | p1 | 40,000 | 1 + ceil(32000/7000) = 6 | 1 + ceil(24000/14000) = 3 | 1 + ceil(8000/28000) = 2 | 5 | 3 + 2 + 6 // 2 = 8 |
# | doi:10.1/p2 | 20,000 | 1 + ceil(12000/7000) = 3 | 1 + ceil(4000/14000) = 2 | 1 | 3 | 2 + 1 + 1 = 4 |
# | p3 | 20,000 | 3 | 2 | 1 | 3 | 4 |
# | p4 | 5,000 | 1 | 1 | 1 | 2 | 2 |
# | p5 | 5,000 | 1 | 1 | 1 | 2 | 2 |
#
# With 2 papers per shard, 5 papers make ceil(5 / 2) = 3 shards. Longest first (16k + 32k windows, then paper id): p1 -> shard-000 (all empty, the first wins); doi:10.1/p2 -> shard-001 (empty, before shard-002); p3 -> shard-002; p4 -> shard-001 (load 4, 1 paper, before shard-002's 4 and 1); p5 -> shard-002 (load 4 against shard-001's 6).
CORPUS = {  # paper id -> (title, letters, DOI in the TEI header)
    "p1": ("Long paper", 40000, "10.5/P1"),
    "doi:10.1/p2": ("Middle paper", 20000, None),
    "p3": ("Another middle paper", 20000, None),
    "p4": ("Short paper", 5000, "10.5/p4"),
    "p5": ("Another short paper", 5000, None),
}


def tei(title: str, letters: int, doi: str | None = None) -> bytes:
    """A GROBID-like TEI with a title, an optional DOI and one paragraph of synthetic text."""
    idno = f'<idno type="DOI">{doi}</idno>' if doi else ""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc><titleStmt>'
        f'<title level="a" type="main">{title}</title></titleStmt>'
        f"<sourceDesc><biblStruct>{idno}</biblStruct></sourceDesc></fileDesc></teiHeader>"
        f"<text><body><div><p>{'a' * letters}</p></div></body></text></TEI>\n"
    ).encode()


def make_corpus(folder: Path) -> Path:
    """The five TEIs of CORPUS as `<paper id, URL-quoted>.tei.xml` files in FOLDER."""
    folder.mkdir(parents=True, exist_ok=True)
    for pid, (title, letters, doi) in CORPUS.items():
        (folder / f"{quote(pid, safe='')}.tei.xml").write_bytes(
            tei(title, letters, doi)
        )
    return folder


# What `extract run`, `extract verify` and `extract filter` write per paper, by the stage's first word: the file and the key of its records.
PAPER_FILES = {
    "combined": ("union.json", "records"),
    "verified": ("verified.json", "definition_records"),
    "filtered": ("filtered.json", "definition_records"),
}


def make_per_paper_run(folder: Path) -> Path:
    """The same run as `make_seed_run`, as a new run leaves it: per-paper outputs instead of tarballs. Each part's stage of each shard is one folder, `runs/<part>/<shard>/<stage>/`, with `papers/<sha256 of the paper id>/` and the file the command of that stage writes; the plan's `jobs` name the folders. The DOIs are in the run's `paper-metadata.json`. Returns the set folder (FOLDER/seeds)."""
    seeds = folder / "seeds"
    plan = {"inputs": {}, "jobs": [], "schema": "taxonomy_pipeline/extract-plan-1"}
    for shard, papers in PAPERS.items():
        rows = [
            {"frozen_title": title, "paper_id": pid, "tei_sha256": "0" * 64}
            for pid, title in papers.items()
        ]
        data = write_json(seeds / "shards" / f"{shard}.json", rows)
        plan["inputs"][shard] = {
            "manifest": f"shards/{shard}.json",
            "manifest_sha256": hashlib.sha256(data).hexdigest(),
            "papers": len(rows),
        }
    write_json(seeds / "picks.json", PICKS)
    write_json(seeds / "paper-metadata.json", {m["paper_id"]: m for m in METADATA})
    for name, stages in STAGES.items():
        part = PARTS[name]
        for stage, records in stages.items():
            folder_name = f"runs/{part_id(part)}/{{shard}}/{stage}"
            plan["jobs"].append(
                {"part": part_id(part), "stage": stage, "folder": folder_name}
            )
            file, key = PAPER_FILES[stage.split("-")[0]]
            for shard, papers in PAPERS.items():
                stage_dir = seeds / folder_name.format(shard=shard)
                stage_dir.mkdir(parents=True)
                for pid in papers:
                    if pid in records:
                        write_json(
                            stage_dir / "papers" / paper_key(pid) / file,
                            {"paper_id": pid, key: records[pid]},
                        )
    write_json(seeds / "plans" / "grid.json", plan)
    return seeds


def make_seed_run(root: Path) -> Path:
    """Write the miniature run under ROOT (a data root); returns its set folder."""
    seeds = root / SEEDS
    plan = {"inputs": {}, "schema": "mini-plan"}
    for shard, papers in PAPERS.items():
        rows = [
            {"frozen_title": title, "paper_id": pid, "tei_sha256": "0" * 64}
            for pid, title in papers.items()
        ]
        data = write_json(seeds / "shards" / f"{shard}.json", rows)
        plan["inputs"][shard] = {
            "manifest": f"{RUN_ROOT}/corpus/seeds/shards/{shard}.json",
            "manifest_sha256": hashlib.sha256(data).hexdigest(),
            "papers": len(rows),
        }
    write_json(seeds / "plans" / "grid.json", plan)
    write_json(seeds / "picks.json", PICKS)
    manifest = root / MANIFEST
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("".join(json.dumps(m) + "\n" for m in METADATA))
    for name, stages in STAGES.items():
        part = PARTS[name]
        folder = seeds / "parts" / part_id(part)
        for shard, papers in PAPERS.items():
            cohort = f"cohorts/{shard}/{part_id(part)}"
            archives: dict[bool, dict[str, bytes]] = {False: {}, True: {}}
            archives[False][f"{cohort}/manifest.json"] = b"{}\n"
            for stage, records in stages.items():
                qv = part["model"] == "gemma12b" and stage in QV_STAGES
                for pid in papers:
                    if pid not in records:
                        continue
                    value = {
                        "cohort": f"{shard}/{part_id(part)}",
                        "definition_records": records[pid],
                        "paper_id": pid,
                        "stage": stage,
                    }
                    member = f"{cohort}/stages/{stage}/{quote(pid, safe='')}.json"
                    archives[qv][member] = json_bytes(value)
                status = f"{cohort}/stages/{stage}/status.json"
                archives[qv][status] = b'{"paper_id": "not a paper"}\n'
            pack(archives[False], folder / f"{shard}.tar.zst")
            if archives[True]:
                pack(archives[True], folder / f"{shard}.qv.tar.zst")
    return seeds
