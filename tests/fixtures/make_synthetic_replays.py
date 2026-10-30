"""Build the synthetic replay fixtures: recorded runs on the made-up papers of tests/synthetic_papers.py.

Run from the repository root:

    PYTHONHASHSEED=0 uv run python tests/fixtures/make_synthetic_replays.py tests/fixtures

This repository holds no paper text, so the commands' tests replay runs on synthetic papers: these three zips hold recordings of the kinds a real run makes (request receipts, usage logs, saved outputs), made the same way. No model, network or data root is used: a stand-in model server on 127.0.0.1 answers every call with `synthetic_papers`' stand-in answers, and the training metric's pair scores come from its stand-in scorers. Each zip has a `SOURCES.tsv` with every member's sha256 and the step that wrote it.

1. `synthetic-training-replay.zip`: `extract train --window w8k` (in this process, so that the pair scores come from the stand-in scorers) on the TRAIN papers s1 to s4 and the DEV papers s5, s6:
   the training dataset and split of these papers (`dataset.json`, `split.json`), the run's request receipts and usage log (`w8k/requests-train/`, `w8k/usage-train.jsonl`), its pair scores (`w8k/pair-scores.json`, in the format of `--record-scores`; the file names the pinned scoring models only because `--replay-scores` requires them: the values are the stand-ins') and the trained program (`w8k/program.json`).
2. `synthetic-extract-replay.zip`: three programs (`programs/a`, `b`, `c`: the trained program, read on windows of 400/80, 650/120 and 900/150 characters, so that every message stays shorter than a real window), `extract run` of the parts `a` and `b+c` on the TEST papers t1 and t2 (`run/`), and `extract verify` of both parts with the one-word-per-token tokenizer (`verify/`).
3. `synthetic-identity-replay.zip`: five PDF identity checks of synthetic references against synthetic first pages (`evidence-5.json`): a candidate the model and the program checks accept, one the model accepts and the checks refuse, one the model calls a different work, a positive and a hard negative control; `closure identity judge` (its request receipts in `requests/`, its verdicts in `expected-verdicts.json`) and `decide` (`expected-decisions.json`).
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]
REPO = TESTS.parent
sys.path.insert(0, str(TESTS))
import synthetic_papers as world  # noqa: E402
from http_replay import NO_NETWORK, Answer, Replay  # noqa: E402

out_dir = Path(sys.argv[1])
FIXED = (2026, 10, 23, 0, 0, 0)
TOKENIZER = TESTS / "fixtures/word-tokenizer"
PROGRAMS = {"a": (400, 80), "b": (650, 120), "c": (900, 150)}
PARTS = ("a", "b+c")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def cli(work: Path, *args: str) -> None:
    """Run `mbdef ARGS` in WORK (its `.env` names the stand-in server); stop if it fails."""
    done = subprocess.run(
        [sys.executable, "-m", "mbdef", *args],
        cwd=work,
        capture_output=True,
        text=True,
        env={**os.environ, **NO_NETWORK},
    )
    if done.returncode != 0:
        raise SystemExit(f"{' '.join(args[:2])} failed:\n{done.stderr}")


def write_zip(path: Path, members: dict[str, bytes], step: dict[str, str]) -> None:
    """The fixture zip, with SOURCES.tsv: each member's sha256 and the step that wrote it."""
    rows = "".join(
        f"{name}\t{sha(data)}\t{step[name]}\n" for name, data in sorted(members.items())
    )
    members = {**members, "SOURCES.tsv": f"member\tsha256\tmade by\n{rows}".encode()}
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, FIXED)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            bundle.writestr(info, members[name], compresslevel=9)
    print(path, path.stat().st_size, "bytes,", len(members), "members")


def folder_members(folder: Path, prefix: str) -> dict[str, bytes]:
    """{prefix + relative path: bytes} for every file below FOLDER but its receipt (which holds the commit and the times of this run)."""
    return {
        prefix + path.relative_to(folder).as_posix(): path.read_bytes()
        for path in sorted(folder.rglob("*"))
        if path.is_file() and path.name != "receipt.json"
    }


# The stand-in model server ---------------------------------------------------------------------

IDENTITY: dict[
    str, dict
] = {}  # an identity item's PDF text: the stand-in's answer to it


def answer(request) -> Answer:
    body = json.loads(request.body)
    schema = (body.get("response_format") or {}).get("json_schema") or {}
    if schema.get("name") == "identity":
        prompt = body["messages"][-1]["content"]
        content = next(
            json.dumps(a, indent=2) for t, a in IDENTITY.items() if t in prompt
        )
    elif "[[ ## candidates ## ]]" in body["messages"][-1]["content"]:
        content = world.verifier_answer(body)
    else:
        content = world.extractor_answer(body)
    return Answer(body=world.completion(body, content))


# 1. Training -----------------------------------------------------------------------------------


def training(work: Path, base: str) -> bytes:
    """Train the w8k program on the synthetic papers; write the zip; return the program's bytes."""
    from mbdef.evaluate import metric
    from mbdef.extract import train

    teis = world.write_teis(work / "teis", (*world.TRAIN, *world.DEV))
    rows = [
        {
            "paper_id": pid,
            "split": "train" if pid in world.TRAIN else "dev",
            "title": world.PAPERS[pid][0],
            "tei_sha256": sha(world.tei(pid)),
            "text_sha256": sha(world.text_of(pid).encode()),
            "gold_records": world.gold(pid),
        }
        for pid in (*world.TRAIN, *world.DEV)
    ]
    dataset = {
        "schema": "synthetic-train-dataset/1",
        "contents": "the synthetic TRAIN and DEV papers of tests/synthetic_papers.py",
        "papers": rows,
    }
    split = {
        "schema": "synthetic-split/1",
        "trainset": list(world.TRAIN),
        "valset": list(world.DEV),
        "forbidden_test_ids": list(world.TEST),
    }
    members = {
        "dataset.json": (json.dumps(dataset, indent=1) + "\n").encode(),
        "split.json": (json.dumps(split, indent=1) + "\n").encode(),
    }
    for name, data in members.items():
        (work / name).write_bytes(data)
    os.environ |= {"DSPY_API_BASE": base, "DSPY_API_KEY": "EMPTY", **NO_NETWORK}
    os.environ["SCIDEF_NLI_DEVICE"] = "cpu"
    # The stand-in scorers in place of the real scoring models.
    metric.models = lambda device, batch_size=32: (world.similarity, world.entailment)  # ty: ignore[invalid-assignment]
    out = work / "train/w8k"
    scores = work / "pair-scores.json"
    train.train(
        out,
        "w8k",
        dataset=work / "dataset.json",
        split=work / "split.json",
        teis=teis,
        record_scores=str(scores),
        workers=1,
    )
    members |= folder_members(out / "requests-train", "w8k/requests-train/")
    members["w8k/usage-train.jsonl"] = (out / "usage-train.jsonl").read_bytes()
    members["w8k/pair-scores.json"] = scores.read_bytes()
    members["w8k/program.json"] = (out / "program.json").read_bytes()
    step = {
        name: "extract train --window w8k (stand-in model and scorers)"
        for name in members
    }
    step |= {
        "dataset.json": "the synthetic papers' training dataset",
        "split.json": "the synthetic papers' split",
    }
    write_zip(out_dir / "synthetic-training-replay.zip", members, step)
    return members["w8k/program.json"]


# 2. Extraction and the verifier --------------------------------------------------------------------


def extraction(work: Path, program: bytes) -> None:
    teis = world.write_teis(work / "teis", world.TEST)
    members: dict[str, bytes] = {}
    for name, (size, overlap) in PROGRAMS.items():
        manifest = {
            "schema": "taxonomy_pipeline/program-1",
            "program": "program.json",
            "program_sha256": sha(program),
            "window": name,
            "window_chars": size,
            "overlap_chars": overlap,
            "note": "the synthetic w8k program, read on smaller windows",
        }
        members[f"programs/{name}/program.json"] = program
        members[f"programs/{name}/manifest.json"] = (
            json.dumps(manifest, indent=2) + "\n"
        ).encode()
    for name, data in members.items():
        (work / name).parent.mkdir(parents=True, exist_ok=True)
        (work / name).write_bytes(data)
    step = dict.fromkeys(members, "the synthetic w8k program and its manifest")
    papers = [arg for pid in world.TEST for arg in ("--paper", pid)]
    parts = [arg for part in PARTS for arg in ("--part", part)]
    cli(
        work,
        *("extract", "run", "--teis", str(teis), "--programs", "programs"),
        *papers,
        *parts,
        *("--out", "run", "--workers", "1"),
    )
    found = folder_members(work / "run", "run/")
    members |= found
    step |= dict.fromkeys(found, "extract run (stand-in model)")
    for part in PARTS:
        cli(
            work,
            *("extract", "verify", "--extraction", f"run/{part}"),
            *("--out", f"verify/{part}", "--tokenizer", str(TOKENIZER)),
            "--workers",
            "1",
        )
        found = folder_members(work / "verify" / part, f"verify/{part}/")
        members |= found
        step |= dict.fromkeys(found, "extract verify (stand-in model, word tokenizer)")
    write_zip(out_dir / "synthetic-extract-replay.zip", members, step)


# 3. The PDF identity check ---------------------------------------------------------------------


def identity_item(kind: str, paper: str, pdf: str, ref: dict, text: str) -> dict:
    prefix = {"candidate": "cand", "control_same": "pos"}.get(kind, "neg")
    item_id = f"{prefix}:{paper}" + (f":{pdf}" if prefix == "neg" else "")
    return {
        "id": item_id,
        "kind": kind,
        "paperid": paper,
        "pdf_paperid": pdf,
        "ref": ref,
        "text": text,
        "pdf_sha256": sha(text.encode()),
    }


def verdict(
    same: bool, title: str, authors: list[str], year, evidence: str, why: str
) -> dict:
    return {
        "verdict": "same_work" if same else "different_work",
        "pdf_title": title,
        "pdf_authors": authors,
        "pdf_year": year,
        "evidence": evidence,
        "reasoning": why,
    }


def identity(work: Path, base: str) -> None:
    """Five checks that reach each outcome of the program checks (`pdf_identity.checks`)."""
    drift = {
        "title": "Headline drift in online news revisions",
        "authors": "Arden, M.; Kestrel, J.",
        "year": "2021",
        "doi": "10.5555/synthetic.1001",
    }
    tilt = {
        "title": "Source tilt and the regional expert list",
        "authors": "Brightwater, L.",
        "year": "2019",
        "doi": "10.5555/synthetic.1002",
    }
    gloom = {
        "title": "Gloom priming in economic reporting",
        "authors": "Corvane, P.; Dunmore, S.",
        "year": "2022",
        "doi": "10.5555/synthetic.1003",
    }
    control = {
        "title": "Byline masking and sponsored pages",
        "authors": "Ellery, R.; Fairholm, T.",
        "year": "2020",
        "doi": "10.5555/synthetic.1004",
    }
    pages = {
        # The title as printed, and an author: the checks accept the model's yes.
        "drift": "Headline Drift in Online News Revisions Mara Arden and Jon Kestrel "
        "Department of Journalism Abstract Editors change headlines after publication.",
        # A preprint whose printed title differs: the model says yes, but neither the cited title nor the title the model read is in the text, so the checks refuse.
        "tilt": "PREPRINT S OURCE T ILT AND THE R EGIONAL E XPERT L IST Lena Brightwater "
        "Newsroom Studies Group Abstract We study expert lists.",
        # Another paper entirely.
        "gloom": "Weather metaphors in sports commentary Ivo Greaves Abstract Commentators "
        "speak of storms and droughts when a team loses.",
        "control": "Byline Masking and Sponsored Pages Rhea Ellery and Tom Fairholm Abstract "
        "Sponsored content often carries a reporter's name.",
        "other": "Comment sections below the line Uma Hollis Abstract Reader comments are "
        "ranked by the newsroom.",
    }
    items = [
        identity_item(
            "candidate",
            "doi_10.5555_synthetic.1001",
            "doi_10.5555_synthetic.1001",
            drift,
            pages["drift"],
        ),
        identity_item(
            "candidate",
            "doi_10.5555_synthetic.1002",
            "doi_10.5555_synthetic.1002",
            tilt,
            pages["tilt"],
        ),
        identity_item(
            "candidate",
            "doi_10.5555_synthetic.1003",
            "doi_10.5555_synthetic.1003",
            gloom,
            pages["gloom"],
        ),
        identity_item(
            "control_same",
            "doi_10.5555_synthetic.1004",
            "doi_10.5555_synthetic.1004",
            control,
            pages["control"],
        ),
        identity_item(
            "control_different_hard",
            "doi_10.5555_synthetic.1004",
            "doi_10.5555_synthetic.1005",
            control,
            pages["other"],
        ),
    ]
    IDENTITY.update(
        {
            pages["drift"]: verdict(
                True,
                "Headline Drift in Online News Revisions",
                ["Mara Arden", "Jon Kestrel"],
                None,
                "Headline Drift in Online News Revisions Mara Arden and Jon Kestrel",
                "The printed title and the authors match the cited reference.",
            ),
            pages["tilt"]: verdict(
                True,
                "Source Tilt and the Regional Expert List",
                ["Lena Brightwater"],
                2019,
                "S OURCE T ILT AND THE R EGIONAL E XPERT L IST Lena Brightwater",
                "A preprint of the cited work: same title and author.",
            ),
            pages["gloom"]: verdict(
                False,
                "Weather metaphors in sports commentary",
                ["Ivo Greaves"],
                None,
                "Weather metaphors in sports commentary Ivo Greaves",
                "The PDF is about sports commentary, not the cited work.",
            ),
            pages["control"]: verdict(
                True,
                "Byline Masking and Sponsored Pages",
                ["Rhea Ellery", "Tom Fairholm"],
                None,
                "Byline Masking and Sponsored Pages Rhea Ellery and Tom Fairholm",
                "Title and authors match.",
            ),
            pages["other"]: verdict(
                False,
                "Comment sections below the line",
                ["Uma Hollis"],
                None,
                "Comment sections below the line Uma Hollis",
                "A different paper on reader comments.",
            ),
        }
    )
    evidence = {
        "schema": "identity-evidence/1",
        "prompt_version": "identity-llm/1",
        "items": items,
    }
    data = json.dumps(evidence, ensure_ascii=False, indent=1).encode()
    (work / "evidence-5.json").write_bytes(data)
    cli(
        work,
        *("closure", "identity", "judge", "--evidence", "evidence-5.json"),
        *("--out", "judged", "--base", base, "--workers", "1"),
    )
    cli(
        work,
        *("closure", "identity", "decide", "--evidence", "evidence-5.json"),
        *("--verdicts", "judged/verdicts.json", "--out", "decided"),
    )
    verdicts = json.loads((work / "judged/verdicts.json").read_text())["verdicts"]
    decisions = json.loads((work / "decided/identity-decisions.json").read_text())
    members = {
        "evidence-5.json": data,
        "expected-verdicts.json": json.dumps(
            verdicts, ensure_ascii=False, indent=1
        ).encode(),
        "expected-decisions.json": json.dumps(
            decisions, ensure_ascii=False, indent=1
        ).encode(),
        **folder_members(work / "judged/requests", "requests/"),
    }
    step = dict.fromkeys(members, "closure identity judge (stand-in model)")
    step["evidence-5.json"] = "five synthetic references and first pages"
    step["expected-decisions.json"] = "closure identity decide"
    write_zip(out_dir / "synthetic-identity-replay.zip", members, step)


if os.environ.get("PYTHONHASHSEED") != "0":
    raise SystemExit("Run with PYTHONHASHSEED=0, as the tests replay the training run.")
with tempfile.TemporaryDirectory() as scratch, Replay() as server:
    server.answer("POST /v1/chat/completions", answer)
    base = f"{server.url}/v1"
    work = Path(scratch)
    (work / ".env").write_text(f"DSPY_API_BASE={base}\nDSPY_API_KEY=EMPTY\n")
    program = training(work, base)
    extraction(work, program)
    identity(work, base)
    shutil.rmtree(work / "teis")
    assert server.misses == [], server.misses
