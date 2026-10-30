"""`mbdef extract run`: definition records from papers, with the saved extractor programs.

A **part** is one or more programs, each read on its own window size: `w16k+w32k` runs the `w16k` program on 16,000-character windows and the `w32k` program on 32,000-character windows of the same paper. Every window is one model call; its records are sanitized, and each paper's windows are then merged into one union per part. The released extractor has two parts in the chat format (`PARTS`, the released TEST run's arms TR-bootstrap-8k and TR-bootstrap-16k32k); its later stages (the verifier, the relevance filter and the pool of the parts) are separate commands.

The papers are every TEI of `--teis`, or those of `--paper`, with the TEIs' titles or those of `--titles`; or the papers of one shard that `extract plan` wrote (`--shard`), with the shard's titles, and each TEI must have the shard's sha256. The defaults below are the released run's. Everything goes into `--out` (new or empty):

    <part>/papers/<sha256 of the paper id>/extraction/<branch>-<window>.json   one window's result
    <part>/papers/<...>/extraction-failures/<branch>-<window>.json   a window that gave no records
    <part>/papers/<...>/union.json                                    the paper's merged records
    <part>/requests-extract/<request_sha256>.json                     every model call, in full
    <part>/usage-extract.jsonl                                        one usage line per call
    predictions.jsonl                                                 every paper's records, the parts pooled
    receipt.json                                                      settings, input hashes, commit

`predictions.jsonl` is the parts' unions pooled per paper with `records.merge` (a predictions file, `records.py`): the extractor's records before the verifier and the filter, in the format `evaluate score --predictions` reads. With `--pdfs FOLDER` instead of `--teis`, every `<paper id>.pdf` of the folder is first converted by a running GROBID (`collect.grobid.convert_folder`) into `tei/<paper id>.tei.xml` in `--out`, and the run reads those.

`branch` numbers the part's programs from 0, `window` the windows from 0 (five digits). A window gives no records when the model's answer did not finish (`finish_reason` other than `stop`) or could not be parsed; the run goes on without it, as the released runs did. A failed model call (the server is unreachable, or a replayed call was not recorded) stops the run instead, before any union is written.
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from mbdef import files, models
from mbdef.extract.papers import (
    GROUND_TRUTH_TEIS,
    parse_tei,
    read_teis,
    windows,
)
from mbdef.extract.records import (
    PREDICTIONS,
    merge,
    sha_text,
    union,
    write_predictions,
)

MODEL, REVISION = models.PINNED[
    "gemma"
]  # the revision is recorded only: the server picks the weights
TEIS = GROUND_TRUTH_TEIS  # the 150 ground-truth papers' TEIs (paper text, not part of this repository)
PROGRAMS = Path("data/extractors/gemma12b-bootstrap")
PARTS = ("w8k", "w16k+w32k")
FORMAT = "chat"
MAX_TOKENS = 32768  # thinking included
SEED = 9
THINKING = {
    "chat_template_kwargs": {"enable_thinking": True}
}  # sent as the request's extra body
# `--thinking`: the extra body of each choice. `on` is the released runs'; `unset` sends none, for an API
# that refuses vLLM's `chat_template_kwargs`, so the model thinks or not as its provider decides.
THINKING_SWITCH = {
    "on": THINKING,
    "off": {"chat_template_kwargs": {"enable_thinking": False}},
    "unset": None,
}
TIMEOUT = (
    21600  # seconds per call; a transport bound only, long thinking under load needs it
)
WORKERS = 16  # windows sent at the same time; the server batches them
STAGE = "extract"


def paper_key(paper_id: str) -> str:
    """The folder name of a paper's outputs (paper ids can hold `/` and `:`)."""
    return files.sha256(paper_id.encode())


def load_parts(programs: Path, parts: list[str]):
    """[(part, [(branch, name, program, manifest)])] for parts like `w16k+w32k`."""
    from mbdef.extract.extractor import load_program

    loaded = {}
    plan = []
    for part in parts:
        branches = []
        for branch, name in enumerate(part.split("+")):
            if name not in loaded:
                loaded[name] = load_program(programs / name)
            branches.append((branch, name, *loaded[name]))
        plan.append((part, branches))
    return plan


def load_papers(
    teis: Path, paper_ids: list[str], titles: Path | None, shard: Path | None = None
) -> list[dict]:
    """Each paper's id, title, text and TEI hash. A title from `titles` wins over the TEI's.
    A `shard` of `extract plan` gives the papers, their titles and their TEIs' sha256."""
    chosen = json.loads(titles.read_text(encoding="utf-8")) if titles else {}
    hashes, source = {}, "titles"
    if shard:
        rows = json.loads(shard.read_text(encoding="utf-8"))
        paper_ids = [row["paper_id"] for row in rows]
        chosen = {row["paper_id"]: row["frozen_title"] for row in rows}
        hashes = {row["paper_id"]: row.get("tei_sha256") for row in rows}
        source = "shard"
    papers = []
    for paper_id, tei in read_teis(teis, paper_ids).items():
        digest = files.sha256(tei)
        if hashes.get(paper_id, digest) not in (None, digest):
            raise SystemExit(
                f"{paper_id}: its TEI's sha256 is not the one {shard} names."
            )
        parsed = parse_tei(tei)
        title = chosen.get(paper_id, parsed["title"])
        if not title.strip() or not parsed["text"].strip():
            raise SystemExit(
                f"Paper {paper_id} has an empty title or no text in its TEI."
            )
        papers.append(
            {
                "paper_id": paper_id,
                "title": title,
                "title_source": source if paper_id in chosen else "tei",
                "text": parsed["text"],
                "tei_sha256": digest,
            }
        )
    return papers


def unfinished(response: dict | None) -> str | None:
    choices = (response or {}).get("choices", [])
    if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
        return "Model response missing or unfinished"
    return None


def extract_window(lm, fmt: str, folder: Path, job: tuple) -> bool:
    """One window: one model call (two if the chat answer falls back to JSON). True if it gave records."""
    from mbdef import llm
    from mbdef.extract.extractor import extract

    paper, branch, index, source, name, program = job
    paper_dir = folder / "papers" / paper_key(paper["paper_id"])
    file = f"{branch}-{index:05d}.json"
    llm.LAST_RESPONSE.value = None
    llm.CALLS.value = []
    problem, prediction = None, None
    try:
        prediction = extract(program, lm, fmt, source, paper["title"])
        problem = unfinished(llm.LAST_RESPONSE.value)
        if not problem and getattr(prediction, "output_error", None):
            problem = "Malformed extractor output"
    except Exception as error:
        if llm.stops_the_run(error):
            raise  # a failed or unrecorded call stops the run
        # Anything else is an answer DSPy could not handle: the window gives no records.
        error = llm.unwrapped(error)
        problem = f"{type(error).__name__}: {error}"
    response = llm.LAST_RESPONSE.value
    if problem:
        choices = (response or {}).get("choices") or [{}]
        files.write_json(
            paper_dir / "extraction-failures" / file,
            {
                "paper_id": paper["paper_id"],
                "program": name,
                "branch": branch,
                "window": index,
                "text_sha256": sha_text(source),
                "problem": problem,
                "finish_reason": choices[0].get("finish_reason"),
                "output_error": getattr(prediction, "output_error", None),
                "requests": llm.CALLS.value,
                "last_response": response,
            },
        )
        return False
    assert prediction is not None
    files.write_json(
        paper_dir / "extraction" / file,
        {
            "paper_id": paper["paper_id"],
            "program": name,
            "branch": branch,
            "window": index,
            "source_text": source,
            "text_sha256": sha_text(source),
            "requests": llm.CALLS.value,
            "response": response,
            "definition_records": list(prediction.definition_records),
            "raw_definition_records": list(prediction.raw_definition_records),
        },
    )
    return True


def run(
    out: Path,
    teis: Path = TEIS,
    paper_ids: list[str] | None = None,
    titles: Path | None = None,
    programs: Path = PROGRAMS,
    parts: list[str] | tuple[str, ...] = PARTS,
    fmt: str = FORMAT,
    model: str = MODEL,
    revision: str = REVISION,
    max_tokens: int = MAX_TOKENS,
    seed: int = SEED,
    replay: list[Path] | None = None,
    workers: int = WORKERS,
    shard: Path | None = None,
    pdfs: Path | None = None,
    grobid_url: str | None = None,
    thinking: str = "on",
) -> int:
    """The step (module docstring). Returns 0; a failed model call stops it with exit code 1."""
    # DSPy takes a second to import, so it is imported here and not when the command line starts.
    from mbdef import llm
    from mbdef.extract.extractor import ADAPTERS

    out, teis, programs = Path(out), Path(teis), Path(programs)
    if fmt not in ADAPTERS:
        raise SystemExit(f"Unknown format {fmt}; use one of {', '.join(ADAPTERS)}.")
    if thinking not in THINKING_SWITCH:
        raise SystemExit(f"--thinking is one of {', '.join(THINKING_SWITCH)}.")
    extra_body = THINKING_SWITCH[thinking]
    if shard and (paper_ids or titles):
        raise SystemExit(
            "--shard names the papers and their titles; drop --paper and --titles."
        )
    if pdfs and shard:
        raise SystemExit("--pdfs gives the papers; drop --shard.")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; give a new or empty folder.")
    plan = load_parts(programs, list(parts))
    conversion = None
    if pdfs:
        from mbdef.collect import grobid

        teis = out / "tei"
        conversion = grobid.convert_folder(
            Path(pdfs), teis, grobid_url or grobid.GROBID_URL, list(paper_ids or [])
        )
    papers = load_papers(teis, list(paper_ids or []), titles, shard)
    if replay:
        responses, api_base, api_key = llm.recorded(replay), None, None
    else:
        responses, api_base, api_key = (
            None,
            files.env("DSPY_API_BASE"),
            files.env("DSPY_API_KEY"),
        )
    summary: dict = {}
    pooled: dict[
        str, list[dict]
    ] = {}  # each paper's records of every part, in part order
    for part, branches in plan:
        folder = out / part
        lm = llm.make_lm(
            model,
            api_base=api_base,
            api_key=api_key,
            max_tokens=max_tokens,
            seed=seed,
            extra_body=extra_body,
            timeout=TIMEOUT,
            usage_log=folder / f"usage-{STAGE}.jsonl",
            requests=folder / f"requests-{STAGE}",
            replay=responses,
            stage=f"{part}/{fmt}:{STAGE}",
        )
        jobs = [
            (paper, branch, index, source, name, program)
            for paper in papers
            for branch, name, program, manifest in branches
            for index, source in enumerate(
                windows(
                    paper["text"], manifest["window_chars"], manifest["overlap_chars"]
                )
            )
        ]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(extract_window, lm, fmt, folder, job) for job in jobs
            ]
            try:
                succeeded = sum(future.result() for future in as_completed(futures))
            except llm.LMError as error:
                pool.shutdown(cancel_futures=True)
                raise SystemExit(
                    f"Stopped: a model call failed in part {part}: {error}. No union was "
                    f"written; the windows done so far are in {folder}."
                ) from error
        counts = {"records": 0, "definitions": 0, "candidates": 0}
        for paper in papers:
            paper_dir = folder / "papers" / paper_key(paper["paper_id"])
            done = [
                json.loads(path.read_text(encoding="utf-8"))
                for branch, *_ in branches
                for path in sorted((paper_dir / "extraction").glob(f"{branch}-*.json"))
            ]
            merged, _contexts = union(paper, done)
            files.write_json(paper_dir / "union.json", merged)
            pooled.setdefault(paper["paper_id"], []).extend(merged["records"])
            counts["records"] += len(merged["records"])
            counts["definitions"] += sum(
                len(r["definitions"]) for r in merged["records"]
            )
            counts["candidates"] += len(merged["candidates"])
        summary[part] = {
            "papers": len(papers),
            "windows": len(jobs),
            "failed_windows": len(jobs) - succeeded,
            **counts,
            "model_calls": llm.usage_totals(folder / f"usage-{STAGE}.jsonl").get(
                "calls", 0
            ),
        }
    summary[PREDICTIONS] = write_predictions(
        out / PREDICTIONS,
        ((paper_id, merge(records)) for paper_id, records in pooled.items()),
    )
    inputs = {"teis": teis, **({"titles": titles} if titles else {})}
    if pdfs:
        inputs = {"pdfs": Path(pdfs), **inputs}
    if shard:
        inputs["shard"] = Path(shard)
    for _part, branches in plan:
        for _branch, name, _program, manifest in branches:
            inputs[f"program:{name}"] = programs / name / manifest["program"]
            inputs[f"manifest:{name}"] = programs / name / "manifest.json"
    for number, folder in enumerate(replay or [], 1):
        inputs[f"replay:{number}"] = Path(folder)
    settings = {
        "model": model,
        "revision": revision,
        "endpoint": "replay of recorded responses" if replay else api_base,
        "format": fmt,
        "max_tokens": max_tokens,
        "seed": seed,
        "temperature": None,  # none is sent: the model's own generation config samples
        "extra_body": extra_body,
        "timeout_seconds": TIMEOUT,
        "workers": workers,
        "parts": {
            part: [
                {
                    "branch": branch,
                    "program": name,
                    "program_sha256": manifest["program_sha256"],
                    "window_chars": manifest["window_chars"],
                    "overlap_chars": manifest["overlap_chars"],
                }
                for branch, name, _program, manifest in branches
            ]
            for part, branches in plan
        },
    }
    if conversion:
        settings["grobid"] = conversion
    receipt = files.receipt(
        "mbdef extract run",
        settings,
        inputs,
        {
            **{part: out / part for part, _ in plan},
            "predictions": out / PREDICTIONS,
            **({"tei": teis} if pdfs else {}),
        },
    )
    receipt["papers"] = [
        {k: paper[k] for k in ("paper_id", "title", "title_source", "tei_sha256")}
        for paper in papers
    ]
    receipt["summary"] = summary
    files.write_json(out / files.RECEIPT, receipt)
    print(files.json_text(summary), end="")
    return 0
