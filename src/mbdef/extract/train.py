"""`mbdef extract train`: a prompt program trained the way the released ones were.

The three released programs in `data/extractors/gemma12b-bootstrap/` (one per window size) were trained by DSPy's **BootstrapFewShot** on Gemma 4 12B with thinking on. MIPRO and GEPA were trained too and not selected. Bootstrap keeps the student's instruction and adds one worked example (a *demo*):

1. **Papers.** The training dataset (`DATASET`, frozen) holds the 80 TRAIN and 21 DEV papers of the ground truth with their titles, TEI hashes and reference records; `SPLIT` adds the TEST papers' ids, which are refused. `build_dataset` (`extract train-dataset`, at the end of this module) writes both from the ground truth and the TEIs, byte for byte as released. Each paper's TEI must have the recorded sha256, and its text too. The papers are put in one fixed order, `random.Random(9)` over the dataset's order (as the released runs did), then split: TRAIN is Bootstrap's training set; DEV is held out and never read (MIPRO and GEPA used it as their validation set).
2. **Bootstrap.** For each TRAIN paper in that order, the student `PaperExtractor(relevant_only=True)` reads every window of the paper (one model call each, no demo yet) and its merged raw records are scored against the paper's reference records with the full-record score (stored as `M3-full-v1`; match cutoff 0.5; 0 if an answer could not be parsed). The first paper that scores above 0 gives the demo, and Bootstrap stops (`MAX_BOOTSTRAPPED_DEMOS` = 1, no labelled demos).
3. **The demo** is one of that paper's window calls: the window, the title, and the model's reasoning and records as it answered them, before any string check. DSPy picks the window with a random draw seeded by a hash of the pickled calls, which depends on Python's per-process string hashing: set `PYTHONHASHSEED` to repeat a run's choice (the receipt records it).
4. **Checks.** The saved program must load, keep the student's signature, have at most one demo, and every demo must be, word for word, a window of a TRAIN paper with that paper's title (`inspect_program`, also `mbdef extract check-programs`).

Worked example (the released w8k run): the first TRAIN paper of the seed-9 order is "Racism and Health: Evidence and Needed Research" (f5c28232…, 9 windows of 8,000 characters). Its 9 calls scored above 0, so Bootstrap stopped after 9 calls; the released program's demo is its window 3. Replaying its 9 recorded answers made the same 9 requests and a program equal to the released one except for which of the 9 windows the draw picks (the recording holds paper text and is not public); tests/test_extract_train.py replays a run on synthetic papers the same way.

Everything goes into `--out` (new or empty), which is then a program folder for `extract run`:

    program.json, manifest.json        the program, and what `extract run --programs` reads
    requests-train/<request_sha256>.json, usage-train.jsonl     every model call (docs/setup/llm.md)
    receipt.json                       settings, the papers, input hashes, the papers tried, commit

The windows of one paper are sent at the same time (`--workers`); the outputs and the trace order are those of the one-by-one loop, so the program is the same.
"""

import contextlib
import json
import random
import sys
from pathlib import Path
from urllib.parse import quote

from mbdef import files
from mbdef.extract.papers import TEI_SUFFIX, parse_tei, read_teis, windows
from mbdef.extract.records import FIELDS, exact
from mbdef.extract.run import (
    MAX_TOKENS,
    MODEL,
    REVISION,
    SEED,
    TEIS,
    THINKING_SWITCH,
    TIMEOUT,
)

FROZEN = Path("data/frozen/results/training-20260925")
# The 80 TRAIN and 21 DEV papers with their titles, references and TEI hashes, and the split with the TEST ids that are refused (frozen copies of the released runner's inputs).
DATASET = FROZEN / "runner/train-dataset.json"
SPLIT = FROZEN / "runner/split.json"
WINDOWS = {"w8k": (8000, 1000), "w16k": (16000, 2000), "w32k": (32000, 4000)}
SHUFFLE_SEED = 9
OPTIMIZER = "dspy.BootstrapFewShot"
MAX_BOOTSTRAPPED_DEMOS = 1
MAX_LABELED_DEMOS = 0
MAX_ROUNDS = 1  # DSPy's default: one try per paper
# evaluate.metric.full_record, per paper, on the merged raw records
METRIC = "M3-full-v1"
THRESHOLD = 0.5  # the full-record score's match cutoff
WORKERS = 16  # windows of one paper sent at the same time
STAGE = "train"


# Papers


def load_dataset(dataset: Path, split: Path, paper_ids: list[str] | None):
    """(all dataset rows, the chosen rows in dataset order): every paper, or `paper_ids`.

    Refuses a TEST paper, a paper outside the dataset, and a dataset that disagrees with the split.
    """
    rows = json.loads(Path(dataset).read_text(encoding="utf-8"))["papers"]
    plan = json.loads(Path(split).read_text(encoding="utf-8"))
    test = set(plan["forbidden_test_ids"])
    train, dev = set(plan["trainset"]), set(plan["valset"])
    for row in rows:
        expected = (
            "train"
            if row["paper_id"] in train
            else "dev"
            if row["paper_id"] in dev
            else None
        )
        if row["paper_id"] in test or row["split"] != expected:
            raise SystemExit(
                f"{dataset}: paper {row['paper_id']} is {row['split']} there, but {split} does "
                "not have it in that set. Refusing to train."
            )
    if not paper_ids:
        return rows, rows
    known = {row["paper_id"] for row in rows}
    for pid in paper_ids:
        if pid in test:
            raise SystemExit(
                f"Paper {pid} is a TEST paper; training never reads TEST papers."
            )
        if pid not in known:
            raise SystemExit(f"Paper {pid} is not in the training dataset {dataset}.")
    chosen = set(paper_ids)
    return rows, [row for row in rows if row["paper_id"] in chosen]


def load_papers(rows: list[dict], teis: Path) -> list[dict]:
    """Each row's paper: id, title (the dataset's), split, text and references, after checking that its TEI and its text have the sha256 the dataset records."""
    found = read_teis(teis, [row["paper_id"] for row in rows])
    papers = []
    for row in rows:
        tei = found[row["paper_id"]]
        text = parse_tei(tei)["text"]
        for what, value, recorded in (
            ("TEI", files.sha256(tei), row["tei_sha256"]),
            ("text", files.sha256(text.encode()), row["text_sha256"]),
        ):
            if value != recorded:
                raise SystemExit(
                    f"Paper {row['paper_id']}: its {what} has sha256 {value}, not the "
                    f"{recorded} the training dataset records."
                )
        papers.append(
            {
                "paper_id": row["paper_id"],
                "title": row["title"],
                "split": row["split"],
                "text": text,
                "gold": row["gold_records"],
            }
        )
    return papers


def ordered(all_rows: list[dict], papers: list[dict]) -> list[dict]:
    """`papers` in the seed-9 order of the whole dataset (a subset keeps its papers' order)."""
    order = list(range(len(all_rows)))
    random.Random(SHUFFLE_SEED).shuffle(order)
    position = {all_rows[i]["paper_id"]: rank for rank, i in enumerate(order)}
    return sorted(papers, key=lambda p: position[p["paper_id"]])


# The student, with the windows of a paper sent at the same time


def student(workers: int):
    """`PaperExtractor(relevant_only=True)`, the released student, with its windows read `workers` at a time (it saves and loads as the released program: `workers` is no state)."""
    from mbdef.extract.extractor import PaperExtractor

    return PaperExtractor(relevant_only=True, workers=workers)


# Checks


def window_index(papers: list[dict], size: int, overlap: int) -> dict[str, dict]:
    """{window text: the paper it comes from and its place} for every window of `papers`."""
    index: dict[str, dict] = {}
    for paper in papers:
        parts = windows(paper["text"], size, overlap)
        for number, text in enumerate(parts):
            index.setdefault(
                text,
                {
                    "paper_id": paper["paper_id"],
                    "title": paper["title"],
                    "split": paper["split"],
                    "window": number,
                    "windows": len(parts),
                },
            )
    return index


def quotes(demo: dict) -> tuple[int, list[str], list[str]]:
    """(record strings, those found only in another case or spacing, those not in the window)."""
    window = demo["paper_text"]
    total, other_case, absent = 0, [], []
    for record in demo.get("definition_records") or []:
        values = [("term", record.get("term", ""))] + [
            (field, item["text"] if isinstance(item, dict) else item)
            for field in FIELDS
            for item in record.get(field) or []
        ]
        for field, value in values:
            total += 1
            if value in window:
                continue
            short = value if len(value) <= 60 else value[:60] + "…"
            (other_case if exact(window, value) else absent).append(
                f"{field} {short!r}"
            )
    return total, other_case, absent


def inspect_program(
    folder: Path, papers: list[dict]
) -> tuple[list[str], list[str], list[dict]]:
    """(report lines, problems, where each demo comes from) for the program in `folder`."""
    from mbdef.extract.extractor import PaperExtractor, load_program

    problems: list[str] = []
    try:
        program, manifest = load_program(folder)
    except SystemExit as refusal:
        return [f"{folder}: {refusal}"], [str(refusal)], []
    predictor = program.chunk_extractor.extract_v3.predict
    saved = predictor.dump_state()["signature"]
    expected = (
        PaperExtractor(
            relevant_only=True
        ).chunk_extractor.extract_v3.predict.dump_state()
    )["signature"]
    instruction = files.sha256(saved["instructions"].encode())
    if saved != expected:
        problems.append(f"{folder}: its signature is not the student's")
    demos = json.loads((folder / manifest["program"]).read_text(encoding="utf-8"))
    demos = demos["chunk_extractor.extract_v3.predict"]["demos"]
    if len(demos) > MAX_BOOTSTRAPPED_DEMOS:
        problems.append(
            f"{folder}: {len(demos)} demos, more than {MAX_BOOTSTRAPPED_DEMOS}"
        )
    size, overlap = manifest["window_chars"], manifest["overlap_chars"]
    lines = [
        f"{folder}: loads, sha256 {manifest['program_sha256'][:12]}… as its manifest records, "
        f"{'the' if saved == expected else 'NOT the'} student's signature (instruction "
        f"{instruction[:12]}…, {len(saved['instructions']):,} characters), {len(demos)} demo(s)"
    ]
    index = window_index(papers, size, overlap)
    places = []
    for number, demo in enumerate(demos, 1):
        place = index.get(demo["paper_text"])
        if place is None or place["split"] != "train":
            where = (
                "not a window of any TRAIN paper"
                if place is None
                else (f"a window of {place['split'].upper()} paper {place['paper_id']}")
            )
            problems.append(f"{folder}: demo {number} is {where}")
            lines.append(f"  demo {number}: {where}")
            continue
        if demo.get("title") != place["title"]:
            problems.append(f"{folder}: demo {number} has another title than its paper")
        total, other_case, absent = quotes(demo)
        found = total - len(other_case) - len(absent)
        detail = f"{found} in the window word for word"
        if other_case:
            detail += f", {len(other_case)} in another case or spacing ({'; '.join(other_case)})"
        if absent:
            detail += f", {len(absent)} not in the window ({'; '.join(absent)})"
        lines.append(
            f"  demo {number}: window {place['window']} of {place['windows']} (numbered from 0; {size:,}/"
            f'{overlap:,} characters) of TRAIN paper {place["paper_id"]} "{place["title"]}", '
            f"word for word; its {total} record strings: {detail}"
        )
        places.append({k: place[k] for k in ("paper_id", "window", "windows")})
    return lines, problems, places


def program_folders(folder: Path) -> list[Path]:
    """`folder` if it is a program folder (it has a manifest.json), else its program subfolders."""
    if (folder / "manifest.json").is_file():
        return [folder]
    found = sorted(p.parent for p in folder.glob("*/manifest.json"))
    if not found:
        raise SystemExit(f"{folder} has no program folder (no manifest.json).")
    return found


def check_programs(programs: list[Path], dataset: Path, split: Path, teis: Path) -> int:
    """`mbdef extract check-programs`: check saved programs; 1 if any check fails."""
    _all_rows, rows = load_dataset(dataset, split, None)
    papers = load_papers(rows, teis)
    folders = [f for folder in programs for f in program_folders(Path(folder))]
    problems = []
    for folder in folders:
        lines, found, _places = inspect_program(folder, papers)
        print("\n".join(lines))
        problems += found
    if problems:
        print(f"{len(problems)} problem(s):\n" + "\n".join(f"- {p}" for p in problems))
        return 1
    print(
        f"{len(folders)} program(s): all load, keep the student's signature, have at most "
        f"{MAX_BOOTSTRAPPED_DEMOS} demo, and every demo is a window of a TRAIN paper, word for word."
    )
    return 0


# The command


def train(
    out: Path,
    window: str,
    paper_ids: list[str] | None = None,
    dataset: Path = DATASET,
    split: Path = SPLIT,
    teis: Path = TEIS,
    model: str = MODEL,
    revision: str = REVISION,
    max_tokens: int = MAX_TOKENS,
    seed: int = SEED,
    threshold: float = THRESHOLD,
    replay: list[Path] | None = None,
    workers: int = WORKERS,
    replay_scores: str | None = None,
    record_scores: str | None = None,
    nli_batch_size: int | None = None,
    thinking: str = "on",
) -> int:
    """The step (module docstring). Returns 0; a failed or unrecorded call stops it (exit code 1)."""
    from mbdef import llm
    from mbdef.evaluate import metric
    from mbdef.evaluate import score as scoring

    # isort: split
    import dspy

    out = Path(out)
    if window not in WINDOWS:
        raise SystemExit(f"Unknown window {window}; use one of {', '.join(WINDOWS)}.")
    if thinking not in THINKING_SWITCH:
        raise SystemExit(f"--thinking is one of {', '.join(THINKING_SWITCH)}.")
    extra_body = THINKING_SWITCH[thinking]
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; give a new or empty folder.")
    size, overlap = WINDOWS[window]
    all_rows, rows = load_dataset(dataset, split, paper_ids)
    papers = ordered(all_rows, load_papers(rows, teis))
    trainset = [p for p in papers if p["split"] == "train"]
    valset = [p for p in papers if p["split"] == "dev"]
    if not trainset:
        raise SystemExit(
            "No TRAIN paper chosen: Bootstrap trains on TRAIN papers only."
        )
    scorer, write_scores, score_settings, score_inputs = scoring.pair_scorer(
        replay_scores, record_scores, nli_batch_size or metric.NLI_BATCH_SIZE
    )
    if replay:
        responses, api_base, api_key = llm.recorded(replay), None, None
    else:
        responses, api_base, api_key = (
            None,
            files.env("DSPY_API_BASE"),
            files.env("DSPY_API_KEY"),
        )
    requests = out / f"requests-{STAGE}"
    lm = llm.make_lm(
        model,
        api_base=api_base,
        api_key=api_key,
        max_tokens=max_tokens,
        seed=seed,
        extra_body=extra_body,
        timeout=TIMEOUT,
        usage_log=out / f"usage-{STAGE}.jsonl",
        requests=requests,
        replay=responses,
        stage=f"{window}:{STAGE}",
    )
    tried, unscored = [], []

    def objective(example, prediction, trace=None):
        """The full-record score of the paper's merged raw records; 0 for an answer that could not be parsed."""
        try:
            value = (
                0.0
                if getattr(prediction, "output_error", None)
                else metric.full_record(
                    example.gold_records,
                    prediction.raw_definition_records,
                    scorer,
                    threshold,
                    allow_partial=False,
                )["score"]
            )
        except metric.NotRecorded as error:
            # Bootstrap would count it as a failed paper and go on: remembered, to stop the run.
            unscored.append(str(error))
            raise
        tried.append({"paper_id": example.paper_id, "score": value})
        return value

    from mbdef.extract.extractor import paper_examples

    examples = paper_examples(trainset, size, overlap)
    optimizer = dspy.BootstrapFewShot(
        metric=objective,
        max_bootstrapped_demos=MAX_BOOTSTRAPPED_DEMOS,
        max_labeled_demos=MAX_LABELED_DEMOS,
        max_rounds=MAX_ROUNDS,
    )
    try:
        # DSPy prints its progress; the command's own output is the summary.
        with contextlib.redirect_stdout(sys.stderr), dspy.context(lm=lm):
            program = optimizer.compile(student(workers), trainset=examples)
    except llm.LMError as error:
        raise SystemExit(
            f"Stopped: a model call failed: {error}. No program was written."
        ) from error
    # Bootstrap counts a failed paper and goes on, so a call or a pair score that was not recorded is caught here, before anything is written.
    receipts = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(requests.glob("*.json"))
    ]
    missing = [
        r for r in receipts if (r.get("error") or {}).get("type") == "NotRecorded"
    ]
    if missing:
        raise SystemExit(
            f"Stopped: request_sha256 {missing[0]['request_sha256']} was not recorded (and "
            f"{len(missing) - 1} more); its request is in {requests}. No program was written."
        )
    if unscored:
        raise SystemExit(f"Stopped: {unscored[0]}. No program was written.")
    program_path = out / "program.json"
    program.save(str(program_path))
    manifest = {
        "schema": "taxonomy_pipeline/program-1",  # a recorded id: it keeps its name (files.py)
        "program": program_path.name,
        "program_sha256": files.sha256(program_path.read_bytes()),
        "window": window,
        "window_chars": size,
        "overlap_chars": overlap,
        "model": model,
        "revision": revision,
        "optimizer": OPTIMIZER,
        "receipt": files.RECEIPT,
    }
    files.write_json(out / "manifest.json", manifest)
    lines, problems, places = inspect_program(out, trainset)
    print("\n".join(lines), file=sys.stderr)
    if problems:
        (out / "manifest.json").unlink()
        raise SystemExit("The trained program fails its checks: " + "; ".join(problems))
    if write_scores and record_scores:
        write_scores(Path(record_scores))
    summary = {
        "window": window,
        "papers": {"train": len(trainset), "dev": len(valset)},
        "papers_tried": tried,
        "demo": places[0] if places else None,
        "model_calls": llm.usage_totals(out / f"usage-{STAGE}.jsonl").get("calls", 0),
        "failed_calls": sum(r.get("error") is not None for r in receipts),
    }
    settings = {
        "model": model,
        "revision": revision,
        "endpoint": "replay of recorded responses" if replay else api_base,
        "max_tokens": max_tokens,
        "seed": seed,
        "temperature": None,  # none is sent: the model's own generation config samples
        "extra_body": extra_body,
        "timeout_seconds": TIMEOUT,
        "workers": workers,
        "window": window,
        "window_chars": size,
        "overlap_chars": overlap,
        "shuffle_seed": SHUFFLE_SEED,
        "student": "PaperExtractor(relevant_only=True)",
        "optimizer": {
            "name": OPTIMIZER,
            "max_bootstrapped_demos": MAX_BOOTSTRAPPED_DEMOS,
            "max_labeled_demos": MAX_LABELED_DEMOS,
            "max_rounds": MAX_ROUNDS,
        },
        "metric": {
            "name": METRIC,
            "threshold": threshold,
            "allow_partial": False,
            "unparsed_answer_score": 0.0,
            **score_settings,
        },
    }
    inputs = {
        "dataset": Path(dataset),
        "split": Path(split),
        "teis": Path(teis),
        **score_inputs,
    }
    for number, folder in enumerate(replay or [], 1):
        inputs[f"replay:{number}"] = Path(folder)
    outputs = {
        "program": program_path,
        "manifest": out / "manifest.json",
        "requests": requests,
        "usage": out / f"usage-{STAGE}.jsonl",
    }
    if record_scores:
        outputs["pair_scores"] = Path(record_scores)
    receipt = files.receipt("mbdef extract train", settings, inputs, outputs)
    receipt["trainset"] = [p["paper_id"] for p in trainset]
    receipt["valset"] = [p["paper_id"] for p in valset]
    receipt["summary"] = summary
    files.write_json(out / files.RECEIPT, receipt)
    print(files.json_text(summary), end="")
    return 0


# The training dataset: `mbdef extract train-dataset`

# The TRAIN and DEV papers with their references; the other two files give titles and TEST ids.
# The first two are MBDef-Gold's record files, read from the data root (files.mbdef_gold) unless a flag names other files.
GROUND_TRUTH = "validated_gt.json"
NON_RELEVANT = "validated_gt_non_relevant.json"
# The names the released dataset and split give these two; a build from the data root writes the same names, so that it is byte-identical.
RELEASED_NAMES = {
    GROUND_TRUTH: "data/validated_gt.json",
    NON_RELEVANT: "data/validated_gt_non_relevant.json",
}
METADATA = Path("data/ground_truth/v3.json")
PLAN = FROZEN / "plan.json"
AMENDMENT = FROZEN / "plan-amendment-1.json"
SPLITS = ("train", "dev")
TEST_SPLITS = ("new_test", "original_test", "test")
# The released files' fixed values, word for word, so that a rebuild is byte-identical.
LOCAL_TEI = "papers"  # the folder the released dataset names in `local_tei`
DATASET_SCHEMA = "training-20260925-train-dataset/2"
SPLIT_SCHEMA = "training-20260925-split/2"
CONTENTS = "{train} TRAIN + {dev} DEV papers (no TEST); gold from {ground_truth} (biasdef view)"
TITLE_SOURCE = (
    "v3 GT metadata title or TEI title (src.data.load_papers, as the historical runs)"
)
ORDER = (
    "{ground_truth} order (src/experiment.py `bound`); the runner builds paper examples in this "
    "order and shuffles them once with random.Random(9), as src/experiment.py"
)
SPLIT_NOTE = "trainset = all TRAIN, valset = DEV (historical). TEST IDs abort the runner if they enter."


def gold_files(
    ground_truth: Path | None, non_relevant: Path | None
) -> list[tuple[Path, str]]:
    """The two ground-truth files as given, or else MBDef-Gold's in the data root.
    Each comes with the name the outputs give it: the path as given, or else its RELEASED_NAMES name."""
    return [
        (Path(given), str(given))
        if given
        else (files.mbdef_gold(name), RELEASED_NAMES[name])
        for given, name in ((ground_truth, GROUND_TRUTH), (non_relevant, NON_RELEVANT))
    ]


def build_dataset(
    ground_truth: Path | None = None,
    non_relevant: Path | None = None,
    metadata: Path = METADATA,
    teis: Path = TEIS,
    plan: Path = PLAN,
    amendment: Path = AMENDMENT,
) -> dict[str, str]:
    """{"train-dataset.json": text, "split.json": text}: the dataset and split that `train` reads (docs/step-3-extraction/extract.md, "The training dataset"). Reads TEST papers for their ids only."""
    gold = gold_files(ground_truth, non_relevant)
    sources = [path for path, _ in gold] + [Path(metadata)]
    names = [name for _, name in gold] + [str(metadata)]
    gold_name = names[0]  # the name the outputs and the messages give the ground truth
    loaded = [json.loads(p.read_text(encoding="utf-8")) for p in sources]
    plan_sha256 = files.digest(Path(plan))
    amendment_sha256 = files.digest(Path(amendment))
    amends = json.loads(Path(amendment).read_text(encoding="utf-8"))
    if amends.get("amends_plan_sha256") != plan_sha256:
        raise SystemExit(f"{amendment} does not amend {plan} (sha256 {plan_sha256}).")
    rows = [p for p in loaded[0]["papers"] if p["split"] in SPLITS]
    ids = [p["paper_id"] for p in rows]
    if len(set(ids)) != len(ids) or not any(p["split"] == SPLITS[0] for p in rows):
        raise SystemExit(f"{gold_name}: repeated paper ids, or no TRAIN paper.")
    forbidden = sorted(
        {
            p["paper_id"]
            for gt in loaded
            for p in gt["papers"]
            if p["split"] in TEST_SPLITS
        }
    )
    split_of = {p["paper_id"]: p["split"] for p in rows}
    for path, gt in zip(names, loaded, strict=True):
        for paper in gt["papers"]:
            if (
                paper["paper_id"] in split_of
                and paper["split"] != split_of[paper["paper_id"]]
            ):
                raise SystemExit(
                    f"Paper {paper['paper_id']} is {paper['split']} in {path} but "
                    f"{split_of[paper['paper_id']]} in {gold_name}."
                )
    if set(ids) & set(forbidden):
        raise SystemExit("A TRAIN or DEV paper is a TEST paper in another file.")
    titles = {
        p["paper_id"]: p.get("metadata", {}).get("title") for p in loaded[2]["papers"]
    }
    found = read_teis(Path(teis), ids)
    papers = []
    for position, row in enumerate(rows):
        tei = found[row["paper_id"]]
        parsed = parse_tei(tei)
        text = parsed["text"]
        if not text.strip():
            raise SystemExit(f"Paper {row['paper_id']}: its TEI has no text.")
        name = quote(row["paper_id"], safe="") + TEI_SUFFIX
        papers.append(
            {
                "paper_id": row["paper_id"],
                "position": position,
                "split": row["split"],
                "title": titles.get(row["paper_id"]) or parsed["title"],
                "local_tei": f"{LOCAL_TEI}/{name}",
                "tei_file": name,
                "tei_sha256": files.sha256(tei),
                "tei_bytes": len(tei),
                "text_sha256": files.sha256(text.encode()),
                "text_chars": len(text),
                "windows": {
                    key: len(windows(text, size, overlap))
                    for key, (size, overlap) in WINDOWS.items()
                },
                "gold_records": row["definition_records"],
            }
        )
    counts = {split: sum(p["split"] == split for p in rows) for split in SPLITS}
    dataset = {
        "schema": DATASET_SCHEMA,
        "plan_sha256": plan_sha256,
        "amendment_sha256": amendment_sha256,
        "contents": CONTENTS.format(**counts, ground_truth=gold_name),
        "title_source": TITLE_SOURCE,
        "order": ORDER.format(ground_truth=gold_name),
        "papers": papers,
    }
    split = {
        "schema": SPLIT_SCHEMA,
        "plan_sha256": plan_sha256,
        "amendment_sha256": amendment_sha256,
        "trainset": sorted(p["paper_id"] for p in rows if p["split"] == SPLITS[0]),
        "valset": sorted(p["paper_id"] for p in rows if p["split"] == SPLITS[1]),
        "forbidden_test_ids": forbidden,
        "test_splits": sorted(TEST_SPLITS),
        "forbidden_sources": names,
        "note": SPLIT_NOTE,
    }
    return {
        DATASET.name: files.json_text(dataset, sort_keys=True),
        SPLIT.name: files.json_text(split, sort_keys=True),
    }


def train_dataset(
    out: Path,
    ground_truth: Path | None = None,
    non_relevant: Path | None = None,
    metadata: Path = METADATA,
    teis: Path = TEIS,
    plan: Path = PLAN,
    amendment: Path = AMENDMENT,
) -> int:
    """`mbdef extract train-dataset`: write the dataset and split into `out`."""
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; give a new or empty folder.")
    texts = build_dataset(ground_truth, non_relevant, metadata, teis, plan, amendment)
    out.mkdir(parents=True, exist_ok=True)
    for name, text in texts.items():
        (out / name).write_text(text, encoding="utf-8")
    split = json.loads(texts[SPLIT.name])
    summary = {
        "papers": {"train": len(split["trainset"]), "dev": len(split["valset"])},
        "forbidden_test_ids": len(split["forbidden_test_ids"]),
        "sha256": {name: files.sha256(text.encode()) for name, text in texts.items()},
    }
    (relevant_path, _), (non_relevant_path, _) = gold_files(ground_truth, non_relevant)
    inputs = {
        "ground_truth": relevant_path,
        "non_relevant": non_relevant_path,
        "metadata": Path(metadata),
        "teis": Path(teis),
        "plan": Path(plan),
        "amendment": Path(amendment),
    }
    settings = {
        "splits": list(SPLITS),
        "test_splits": list(TEST_SPLITS),
        "windows": {k: list(v) for k, v in WINDOWS.items()},
    }
    receipt = files.receipt(
        "mbdef extract train-dataset",
        settings,
        inputs,
        {name: out / name for name in texts},
    )
    receipt["summary"] = summary
    files.write_json(out / files.RECEIPT, receipt)
    print(files.json_text(summary), end="")
    return 0
