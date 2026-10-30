"""`mbdef extract verify`: the candidate verifier on one extraction run.

The input is one extraction run: a folder with `papers/<sha256 of the paper id>/union.json` and the paper's windows in `extraction/`, as `extract run` writes it for each part (and as the released runs wrote their cohorts' `run/`). Every paper's candidates are verified (`verifier.py` says how), several papers at a time and each paper's batches one after another. A failed model call (the server is unreachable, or a replayed call was not recorded) stops the run, as in `extract run`.

The defaults are the released verifier's: Qwen 3.6 35B-A3B, thinking off, temperature 0, seed 9, at most 4,096 output tokens, and the JSON format with the batch's JSON schema sent for constrained decoding. The released self-verification of the Gemma extractor is the same with `--model google/gemma-4-12B-it-qat-w4a16-ct`. Everything goes into `--out` (new or empty):

    papers/<sha256 of the paper id>/union.json            the union verified (copied)
    papers/<...>/verification/<batch>.json               one batch: ids, raw response, verdicts
    papers/<...>/decisions.json                          every verdict per candidate; the rejected
    papers/<...>/verified.json                           the kept records
    papers/<...>/paper-summary.json                      the paper's counts
    requests-verifier/<request_sha256>.json              every model call, in full
    usage-verifier.jsonl                                 one usage line per call
    predictions.jsonl                                    the kept records of every paper (records.py)
    receipt.json                                         settings, input hashes, commit

The per-paper files have the released runs' JSON format (sorted keys, indent 2).
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from mbdef import files, models
from mbdef.extract.records import PREDICTIONS, write_predictions


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


MODEL, REVISION = models.PINNED[
    "qwen"
]  # the revision is recorded; also the tokenizer's revision
MAX_TOKENS = 4096
SEED = 9
TEMPERATURE = 0.0
THINKING = {
    "chat_template_kwargs": {"enable_thinking": False}
}  # sent as the extra body
TIMEOUT = 3600  # seconds per call, a transport bound only
WORKERS = 16  # papers verified at the same time
STAGE = "verifier"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")


def load_tokenizer(model: str, revision: str, folder: Path | None):
    """The verifier model's tokenizer (from `folder`, else the Hugging Face cache) and its folder."""
    from transformers import AutoTokenizer

    if folder is None:
        from mbdef.cluster import cached_snapshot

        folder = cached_snapshot(model, revision, TOKENIZER_FILES[:2])
    return AutoTokenizer.from_pretrained(folder, local_files_only=True), folder


def verify(
    extraction: Path,
    out: Path,
    model: str = MODEL,
    revision: str = REVISION,
    tokenizer: Path | None = None,
    max_tokens: int = MAX_TOKENS,
    seed: int = SEED,
    replay: list[Path] | None = None,
    workers: int = WORKERS,
) -> int:
    """The step (module docstring). Returns 0; a failed model call stops it with exit code 1."""
    # DSPy takes a second to import, so it is imported here and not when the command line starts.
    from mbdef import llm
    from mbdef.extract import verifier
    from mbdef.extract.extractor import ControlledJSONAdapter

    extraction, out = Path(extraction), Path(out)
    paper_dirs = sorted(p.parent for p in (extraction / "papers").glob("*/union.json"))
    if not paper_dirs:
        raise SystemExit(f"{extraction} has no papers/*/union.json to verify.")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; give a new or empty folder.")
    loaded, tokenizer_folder = load_tokenizer(model, revision, tokenizer)

    def count_tokens(messages: list[dict]) -> int:
        encoded = loaded.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            **THINKING["chat_template_kwargs"],
        )
        return len(encoded["input_ids"] if hasattr(encoded, "keys") else encoded)

    if replay:
        responses, api_base, api_key = llm.recorded(replay), None, None
    else:
        responses, api_base, api_key = (
            None,
            files.env("DSPY_API_BASE"),
            files.env("DSPY_API_KEY"),
        )
    lm = llm.make_lm(
        model,
        api_base=api_base,
        api_key=api_key,
        max_tokens=max_tokens,
        seed=seed,
        temperature=TEMPERATURE,
        extra_body=THINKING,
        timeout=TIMEOUT,
        usage_log=out / f"usage-{STAGE}.jsonl",
        requests=out / f"requests-{STAGE}",
        replay=responses,
        stage=f"{extraction.name}:{STAGE}",
    )
    adapter = ControlledJSONAdapter(constrained=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                verifier.verify_paper,
                paper_dir,
                out / "papers" / paper_dir.name,
                lm,
                adapter,
                count_tokens,
                max_tokens,
            )
            for paper_dir in paper_dirs
        ]
        try:
            summaries = [future.result() for future in as_completed(futures)]
        except llm.LMError as error:
            pool.shutdown(cancel_futures=True)
            raise SystemExit(
                f"Stopped: a verifier call failed: {error}. No receipt was written; the papers "
                f"done so far are in {out / 'papers'}."
            ) from error
    keys = (
        "candidates",
        "batches",
        "unbatchable_contexts",
        "failed_batches",
        "missing_verdicts",
        "candidates_rejected_for_missing_verdict",
        "kept_records",
        "kept_definitions",
    )
    summary = {
        "papers": len(summaries),
        **{k: sum(s[k] for s in summaries) for k in keys},
        "model_calls": llm.usage_totals(out / f"usage-{STAGE}.jsonl").get("calls", 0),
    }
    summary[PREDICTIONS] = write_predictions(
        out / PREDICTIONS,
        (
            (
                read_json(paper_dir / "union.json")["paper_id"],
                read_json(out / "papers" / paper_dir.name / "verified.json")[
                    "definition_records"
                ],
            )
            for paper_dir in paper_dirs
        ),
    )
    inputs = {"extraction": extraction}
    for name in TOKENIZER_FILES:
        if (tokenizer_folder / name).is_file():
            inputs[f"tokenizer:{name}"] = tokenizer_folder / name
    for number, folder in enumerate(replay or [], 1):
        inputs[f"replay:{number}"] = Path(folder)
    settings = {
        "model": model,
        "revision": revision,
        "endpoint": "replay of recorded responses" if replay else api_base,
        "tokenizer": str(tokenizer) if tokenizer else f"{model} at {revision}",
        "max_tokens": max_tokens,
        "seed": seed,
        "temperature": TEMPERATURE,
        "extra_body": THINKING,
        "response_format": "each batch's JSON schema (constrained decoding)",
        "timeout_seconds": TIMEOUT,
        "max_candidates": verifier.MAX_CANDIDATES,
        "batch_tokens": verifier.BATCH_TOKENS,
        "policy": verifier.POLICY,
        "workers": workers,
    }
    receipt = files.receipt(
        "mbdef extract verify",
        settings,
        inputs,
        {"papers": out / "papers", "predictions": out / PREDICTIONS},
    )
    receipt["summary"] = summary
    files.write_json(out / files.RECEIPT, receipt)
    print(files.json_text(summary), end="")
    return 0
