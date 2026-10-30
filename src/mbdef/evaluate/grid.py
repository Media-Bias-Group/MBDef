"""`mbdef evaluate grid`: score every setup of the extractor grid.

The grid (a frozen plan and its amendments) ran two models, Qwen 3.6 35B and Gemma 4 12B, on the 21 DEV and 29 TEST papers in many ways; each way is a **setup**. This command scores every setup on both splits, 2,268 scorings in all. Choosing among them is the next command, `evaluate select` (selection.py). Read it in five parts:

1. **The shape of the grid.** A *cohort* is one model run with one prompt program (*arm*) and one output format on one split: `<split>/<model>/<arm>/<format>/`, with one folder per *stage* of its post-processing (`stages/<stage>/<paper>.json`): `combined` (all extracted records), `verified-self` (kept by the model's own verifier), `verified-qwen` (by the common Qwen verifier), and relevance-filtered versions of each (`filtered-title`, `filtered-ordinary`, their `-qv` and `-noverify` forms). A *setup* is a model, one arm or a pool of arms, one format or a pool of formats, and one stage: per paper, the pooled cohorts' records of that stage are concatenated in the listed order and merged (`extract.records.merge`). Every single cohort is a setup too. The trained-prompt arms (`TR-<optimizer>-8k`, `TR-<optimizer>-16k32k`) live in a second folder, the training run's cohorts.
2. **Cohorts.** `load_cohort` reads a cohort's stage files and its status and request records:
   whether it completed, how many extraction windows and verifier batches failed, and how many model calls it made (`cohort_accounting`).
3. **Scoring.** Each distinct set of records is scored once with the general scorer (`score.score`: DefScore one-to-one matches at 0.5 and the full-record score). A setup that cannot be scored stops the command, and nothing is written.
4. **Eligibility** (the plan's failure policy): every pooled cohort complete, every paper with an output, at most 5% failed extraction windows and verifier batches.
5. **Running it.** The work is split by split and model: all setups of one model on one split run in one process, so a definition pair that many setups share is scored once. Each process runs on one GPU; `--gpus` spreads them. The output, `scores.json`, holds every setup's scores and eligibility before any audit or pick, and a receipt.
"""

import copy
import hashlib
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

from mbdef import files
from mbdef.evaluate import metric, score
from mbdef.extract.records import merge

# 1. The shape of the grid (the frozen plan and its amendments 1-4)

SPLITS: dict[str, dict[str, Any]] = {
    "dev": {"papers": 21, "definitions": 115, "splits": ("dev",)},
    "test": {"papers": 29, "definitions": 365, "splits": ("original_test", "new_test")},
}
MODELS = ("qwen", "gemma12b")
ARMS = ("A-minimal-8k", "B-minimal-16k32k", "C-gepa-16k32k")
ORIGINAL_FORMATS = ("chat8192", "schema8192", "schema16384")
DEFAULT_FORMATS = ("chat-default", "json-default")  # amendment 1
THINK_FORMATS = ("chat-default-think", "json-default-think")  # amendment 4
FORMATS = ORIGINAL_FORMATS + DEFAULT_FORMATS + THINK_FORMATS
# Format pools of one arm. Thinking and non-thinking formats are never pooled together.
PASSES = {
    "chat": ("chat8192",),
    "chat+schema8192": ("chat8192", "schema8192"),
    "all3": ORIGINAL_FORMATS,
    "chat-default+json-default": DEFAULT_FORMATS,
    "chat-default-think+json-default-think": THINK_FORMATS,
}
SELF_STAGES = ("combined", "verified-self", "filtered-title", "filtered-ordinary")
NOVERIFY_STAGES = (
    "filtered-title-noverify",
    "filtered-ordinary-noverify",
)  # amendment 1
QV_STAGES = ("verified-qwen", "filtered-title-qv", "filtered-ordinary-qv")
STAGES = {m: SELF_STAGES + NOVERIFY_STAGES + QV_STAGES for m in MODELS}
# The verifier a stage's records went through: none, the model itself, or the common Qwen one.
VERIFIER = {
    "combined": None,
    "verified-self": "self",
    "filtered-title": "self",
    "filtered-ordinary": "self",
    "filtered-title-noverify": None,
    "filtered-ordinary-noverify": None,
    "verified-qwen": "qwen",
    "filtered-title-qv": "qwen",
    "filtered-ordinary-qv": "qwen",
}
# Pools across arms (amendment 2): arm-major concatenation, then merge per paper.
ARM_POOLS = {
    "AB-minimal-8k16k32k": ("A-minimal-8k", "B-minimal-16k32k"),
    "AC-minimal8k-gepa16k32k": ("A-minimal-8k", "C-gepa-16k32k"),
    "ABC-all": ("A-minimal-8k", "B-minimal-16k32k", "C-gepa-16k32k"),
}
# The format pools of the cross-arm pools: every single format, and three pools (amendments 2-4).
CROSS_ARM_PASSES = {
    **{f: (f,) for f in FORMATS},
    "chat+schema8192": PASSES["chat+schema8192"],
    "all3": PASSES["all3"],
    "chat-default+json-default": PASSES["chat-default+json-default"],
    "chat-default-think+json-default-think": PASSES[
        "chat-default-think+json-default-think"
    ],
}
GROUPS = ARMS + tuple(ARM_POOLS)
# Trained-prompt arms (the training run): TR-<optimizer>-8k and TR-<optimizer>-16k32k, thinking formats only, and their pool TR-<optimizer>-8k+16k32k.
TRAINED_OPTIMIZERS = ("bootstrap", "gepa", "mipro")
TRAINED_PREFIX = "TR-"
TRAINED_KINDS = ("8k", "16k32k")
TRAINED_FORMATS = THINK_FORMATS
FAILURE_CEILING = (5, 100)  # at most 5% failed extraction windows and verifier batches

# Inputs: the references in the repository's data folder (--data), the saved cohorts and the TEST evaluation's results under DATA_ROOT (data/data-root.tsv; copies of the cluster runs).
REFERENCES = {
    "dev": ("frozen/results/qwen-replication-sweep-20260923/dev-reference.json", None),
    "test": (
        "frozen/results/test-eval-20260924/inputs/test-reference.json",
        "frozen/results/test-eval-20260924/inputs/READY.json",
    ),
}
GRID = "frozen/runs/grid-20260925/cohorts"
TRAINED = "frozen/runs/training-20260925/eval/cohorts"
TEST_RESULTS = "frozen/artifacts/experiments/test-eval-20260924/collected/results.json"
SCORES = "scores.json"
SETS = "sets"  # --write-scorings: one folder of paper files per distinct record set
SCORES_SCHEMA = (
    "taxonomy_pipeline/grid-scores-1"  # a recorded id: it keeps its name (files.py)
)
JSON_FORMAT: dict[str, Any] = {"indent": 2, "ensure_ascii": False}
# The environment variables that the stored results record.
ENVIRONMENT = ("SCIDEF_NLI_DEVICE", "SCIDEF_NLI_BATCH_SIZE", "SLURM_JOB_ID")
# The four processes, largest first (the most definition pairs to score: TEST, then DEV).
FAMILIES = ("test/qwen", "test/gemma12b", "dev/qwen", "dev/gemma12b")
OPUS_ROW = "Rdb54680"  # its TEST calls come from the TEST evaluation's results
REPLICATE_ROW = "QWEN-POOL3"  # the plan's replicate note: the same recipe on TEST twice
REPLICATE_CONFIG = "test/qwen/C-gepa-16k32k/all3/filtered-title"


def now() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_reference(path: Path, ready: Path | None, split: str, name: str):
    """{paper id: reference records} of one split, and what the results say about it. The papers must be of the split, their counts as declared and as the plan expects, and the file's sha256 the one its READY record declares (TEST)."""
    raw = read_json(path)
    meta = (
        {k: v for k, v in raw.items() if k != "papers"} if isinstance(raw, dict) else {}
    )
    papers = raw.get("papers") if isinstance(raw, dict) else raw
    if not isinstance(papers, list):
        raise ValueError("Reference has no paper list")
    gold = {}
    for row in papers:
        pid = row.get("paper_id")
        if not isinstance(pid, str) or pid in gold:
            raise ValueError("Invalid or duplicate reference paper ID")
        if row.get("split") not in SPLITS[split]["splits"]:
            raise ValueError(f"Reference paper outside the cohort splits: {pid}")
        score.definition_count(row.get("definition_records"))
        gold[pid] = row["definition_records"]
    total = sum(score.definition_count(r) for r in gold.values())
    declared_papers = meta.get("paper_count", len(gold))
    declared_definitions = meta.get("definition_count", total)
    if declared_papers != len(gold) or declared_definitions != total:
        raise ValueError(
            "Reference paper/definition counts disagree with its declaration"
        )
    expected = (SPLITS[split]["papers"], SPLITS[split]["definitions"])
    if (len(gold), total) != expected:
        raise ValueError(
            f"{split} reference has {len(gold)}/{total}, expected {expected[0]}/{expected[1]}"
        )
    declared = ("schema", "paper_count", "definition_count", "source", "source_sha256")
    evidence: dict[str, Any] = {
        "path": name,
        "sha256": files.digest(path),
        "papers": len(gold),
        "definitions": total,
        "declared": {k: meta.get(k) for k in declared},
    }
    if ready is not None:  # the TEST reference has a READY record, the DEV one none
        evidence["ready"] = score.checked_ready(path, ready)
    return gold, evidence


# 2. Cohorts


def lookup(obj, *paths):
    """The first value that is not None along any dotted key path (dicts only)."""
    for path in paths:
        value = obj
        for key in path.split("."):
            value = value.get(key) if isinstance(value, dict) else None
        if value is not None:
            return value
    return None


def count_files(directory: Path, *names: str):
    for name in names:
        d = directory / name
        if d.is_dir():
            return len(list(d.glob("*.json")))
    return None


def cohort_accounting(directory: Path, status: dict | None) -> dict:
    """Failed windows and batches, and model calls, from the runner's status.json (whose schema the plan did not freeze, so several spellings are read) and its request records.
    Every model call, DSPy's fallback calls included, left one request record, so the records count the extraction calls where they exist."""
    s = status or {}
    reported = score.as_int(
        lookup(s, "requests.extract", "extraction_requests", "extraction.requests")
    )
    receipts = count_files(directory / "run", "requests-extract")
    fallback = score.as_int(
        lookup(
            s,
            "requests.fallback",
            "requests.extract_fallback",
            "fallback_requests",
            "extraction.fallback_requests",
        )
    )
    if receipts is not None:
        calls = max(receipts, reported or 0)
        source = "request receipts (includes fallback calls)"
    elif reported is not None:
        extra = (
            fallback
            if fallback and not lookup(s, "requests.extract_includes_fallback")
            else 0
        )
        calls = reported + extra
        source = "status requests.extract" + (
            " + reported fallback requests" if extra else ""
        )
    else:
        calls, source = None, "not reported"
    extraction = {
        "failed_windows": score.as_int(
            lookup(
                s,
                "failed_windows",
                "extraction.failed_windows",
                "extract.failed_windows",
            )
        ),
        "total_windows": score.as_int(
            lookup(s, "windows", "total_windows", "extraction.total_windows")
        ),
        "requests": calls,
        "requests_source": source,
        "reported_requests": reported,
        "receipt_files": receipts,
        "reported_fallback_requests": fallback,
    }
    self_requests = lookup(
        s, "requests.verifier", "verifier_requests", "verification.requests"
    )
    self_verifier = {
        "failed_batches": score.as_int(
            lookup(
                s,
                "failed_batches",
                "verification.failed_batches",
                "verifier.failed_batches",
                "verified-self.failed_batches",
            )
        ),
        "total_batches": score.as_int(
            lookup(
                s,
                "batches",
                "total_batches",
                "verification.total_batches",
                "verifier.total_batches",
            )
        ),
        "missing_verdicts": score.as_int(
            lookup(
                s,
                "missing_verdicts",
                "verification.missing_verdicts",
                "verifier.missing_verdicts",
            )
        ),
        "requests": score.as_int(self_requests)
        if self_requests is not None
        else count_files(directory / "run", "requests-verifier"),
    }
    qv = None
    for key in ("qwen_verifier", "verified-qwen", "qv", "qwen_verification"):
        if isinstance(s.get(key), dict):
            qv = s[key]
            break
    if qv is None:
        for name in (
            "status-verify-qwen.json",
            "status-qv.json",
            "status-verified-qwen.json",
        ):
            if (directory / name).is_file():
                qv = read_json(directory / name)
                break
    qv = qv or {}
    qv_requests = (
        lookup(qv, "requests.verifier", "verifier_requests")
        if qv
        else lookup(s, "qv_requests", "qwen_verifier_requests")
    )
    qwen_verifier = {
        "failed_batches": score.as_int(
            lookup(qv, "failed_batches")
            if qv
            else lookup(s, "qv_failed_batches", "qwen_verifier_failed_batches")
        ),
        "total_batches": score.as_int(
            lookup(qv, "batches", "total_batches")
            if qv
            else lookup(s, "qv_total_batches", "qwen_verifier_total_batches")
        ),
        "missing_verdicts": score.as_int(
            lookup(qv, "missing_verdicts") if qv else lookup(s, "qv_missing_verdicts")
        ),
        "requests": score.as_int(qv_requests)
        if qv_requests is not None
        else count_files(directory / "run-verify-qwen", "requests-verifier"),
        "complete": lookup(qv, "complete") if qv else None,
    }
    total_requests = (
        score.as_int(lookup(s, "requests.total"))
        if isinstance(s.get("requests"), dict)
        else score.as_int(s.get("requests"))
    )
    if (
        extraction["requests"] is None
        and self_verifier["requests"] is None
        and total_requests is not None
    ):
        # Only a cohort total is reported: it counts extraction and the self verifier.
        self_verifier["combined_total_requests"] = total_requests
    return {
        "extraction": extraction,
        "self": self_verifier,
        "qwen": qwen_verifier,
        "complete": s.get("complete"),
        "papers": score.as_int(s.get("papers")),
        "failed_paper_ids": sorted(s.get("failed_paper_ids") or []),
    }


def extraction_compute(directory: Path, status: dict | None) -> dict:
    """Extraction compute from the usage records (run/usage-extract.jsonl): output tokens (answer and reasoning), reasoning tokens, and generation wall time (first request start to last response). Reported only; never used to choose."""
    path = directory / "run" / "usage-extract.jsonl"
    result: dict[str, Any] = {
        "source": None,
        "responses": None,
        "output_tokens": None,
        "reasoning_tokens": None,
        "prompt_tokens": None,
        "wall_seconds": None,
        "request_seconds": None,
        "length_finishes": None,
    }
    if path.is_file():
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if rows:
            starts = [r["started"] for r in rows if r.get("started") is not None]
            ends = [
                r["started"] + r.get("seconds", 0)
                for r in rows
                if r.get("started") is not None
            ]
            result.update(
                source=str(path.relative_to(directory)),
                responses=len(rows),
                output_tokens=sum(r.get("completion_tokens") or 0 for r in rows),
                reasoning_tokens=sum(r.get("reasoning_tokens") or 0 for r in rows),
                prompt_tokens=sum(r.get("prompt_tokens") or 0 for r in rows),
                request_seconds=sum(r.get("seconds") or 0 for r in rows),
                wall_seconds=(max(ends) - min(starts)) if starts else None,
                length_finishes=sum(
                    "length" in (r.get("finish_reasons") or []) for r in rows
                ),
            )
            return result
    s = status or {}
    out = lookup(
        s,
        "usage.extract.completion_tokens",
        "extraction_usage.completion_tokens",
        "usage.completion_tokens",
    )
    if out is not None:
        result.update(
            source="status.json",
            output_tokens=score.as_int(out),
            reasoning_tokens=score.as_int(
                lookup(
                    s,
                    "usage.extract.reasoning_tokens",
                    "extraction_usage.reasoning_tokens",
                    "usage.reasoning_tokens",
                )
            ),
            wall_seconds=lookup(
                s,
                "usage.extract.wall_seconds",
                "extraction_usage.wall_seconds",
                "generation_seconds",
            ),
        )
    return result


def reasoning_text_counts(directory: Path) -> dict:
    """For a thinking cohort: how many responses carry separate reasoning text. A receipt that cannot be read stops the run (none of the saved cohorts' 32,797 receipts is unreadable)."""
    counts = {"receipts": 0, "with_reasoning_text": 0, "reasoning_chars": 0}
    for path in (directory / "run" / "requests-extract").glob("*.json"):
        try:
            response = read_json(path).get("response") or {}
        except (ValueError, AttributeError) as error:
            raise ValueError(
                f"Cannot read the request receipt {path}: {error}"
            ) from error
        counts["receipts"] += 1
        texts = [
            (
                (c.get("message") or {}).get("reasoning_content")
                or (c.get("message") or {}).get("reasoning")
                or ""
            )
            for c in response.get("choices") or []
        ]
        if any(texts):
            counts["with_reasoning_text"] += 1
            counts["reasoning_chars"] += sum(len(t) for t in texts)
    return counts


def load_cohort(directory: Path, cohort_ids: set[str], name: str) -> dict:
    """One cohort: its status, its records per stage and paper, the papers whose extraction windows all failed, its accounting and compute. A stage file that cannot be read, names a paper outside the split or repeats one is an error of its stage. `name` is how the outputs name the folder."""
    status = None
    errors = []
    status_path = directory / "status.json"
    if status_path.is_file():
        try:
            status = read_json(status_path)
        except Exception as e:
            errors.append(f"status.json unreadable: {type(e).__name__}")
    else:
        errors.append("status.json missing")
    stages = {}
    stages_dir = directory / "stages"
    for stage_dir in sorted(stages_dir.iterdir()) if stages_dir.is_dir() else []:
        if not stage_dir.is_dir():
            continue
        papers, stage_errors = {}, []
        for path in sorted(stage_dir.glob("*.json")):
            if path.name == "status.json":
                continue
            try:
                value = read_json(path)
                pid = value.get("paper_id") or unquote(path.stem)
                records = value["definition_records"]
                score.definition_count(records)
            except Exception as e:
                stage_errors.append(f"{path.name}: {type(e).__name__}: {e}")
                continue
            if pid not in cohort_ids:
                stage_errors.append(f"paper outside split: {pid}")
                continue
            if pid in papers:
                stage_errors.append(f"duplicate paper {pid}")
                continue
            papers[pid] = records
        stages[stage_dir.name] = {"papers": papers, "errors": stage_errors}
    # A paper whose extraction windows all failed has no output, whatever status.json says.
    zero_window = []
    papers_root = directory / "run" / "papers"
    if papers_root.is_dir():
        for pid in sorted(cohort_ids):
            paper_dir = papers_root / hashlib.sha256(pid.encode()).hexdigest()
            if not paper_dir.is_dir() or not any(
                (paper_dir / "extraction").glob("*.json")
            ):
                zero_window.append(pid)
    compute = extraction_compute(directory, status)
    if directory.name.endswith("-think"):
        compute["reasoning_text"] = reasoning_text_counts(directory)
    return {
        "directory": name,
        "status": status,
        "errors": errors,
        "stages": stages,
        "zero_window_papers": zero_window,
        "accounting": cohort_accounting(directory, status),
        "compute": compute,
        "status_sha256": files.digest(status_path) if status_path.is_file() else None,
    }


def verifier_kind(model: str, stage: str):
    """None, "self" or "qwen"; a Qwen cohort's own verifier is the common Qwen one."""
    kind = VERIFIER.get(stage)
    return "self" if model == "qwen" and kind == "qwen" else kind


def stage_calls(cohort: dict, stage: str):
    """(LLM calls of a cohort's stage or None, and why None): extraction requests plus the requests of the verifier the stage used."""
    acc = cohort["accounting"]
    kind = verifier_kind(cohort["model"], stage)
    extraction = acc["extraction"]["requests"]
    if kind is None:
        why = None if extraction is not None else "extraction requests not reported"
        return extraction, why
    if kind == "self":
        if acc["self"].get("combined_total_requests") is not None:
            return acc["self"]["combined_total_requests"], None
        verifier = acc["self"]["requests"]
    else:
        verifier = acc["qwen"]["requests"]
    if extraction is None or verifier is None:
        return None, "request counts not reported"
    return extraction + verifier, None


# 3. Scoring and 4. eligibility


def config_records(cohorts: list[dict], stage: str) -> dict[str, list]:
    """A setup's records per paper: a single cohort's as saved; for a pool, the cohorts' records concatenated in order and merged (merge changes its input, so on copies)."""
    if len(cohorts) == 1:
        return dict(cohorts[0]["stages"][stage]["papers"])
    merged, pids = {}, set()
    for c in cohorts:
        pids |= set(c["stages"][stage]["papers"])
    for pid in sorted(pids):
        parts = [
            copy.deepcopy(r)
            for c in cohorts
            for r in c["stages"][stage]["papers"].get(pid, [])
        ]
        merged[pid] = merge(parts)
    return merged


def config_eligibility(
    cohorts: list[dict], stage: str, cohort_ids: set[str]
) -> list[str]:
    """Why a setup is not eligible under the plan's failure policy (empty: eligible)."""
    reasons = []
    fw = tw = fb = tb = 0
    counts_known_w = counts_known_b = True
    kind = verifier_kind(cohorts[0]["model"], stage)
    for c in cohorts:
        name = c.get("label") or Path(c["directory"]).name
        acc = c["accounting"]
        reasons.extend(f"{name}: {e}" for e in c["errors"])
        reasons.extend(f"{name}/{stage}: {e}" for e in c["stages"][stage]["errors"])
        if acc["complete"] is not True:
            reasons.append(f"{name}: status.complete is not true")
        if acc["failed_paper_ids"]:
            reasons.append(f"{name}: failed papers {acc['failed_paper_ids']}")
        missing = (cohort_ids - set(c["stages"][stage]["papers"])) | set(
            c.get("zero_window_papers") or []
        )
        if missing:
            reasons.append(
                f"{name}/{stage}: {len(missing)}/{len(cohort_ids)} papers without output "
                "(incl. all windows failed)"
            )
        e = acc["extraction"]
        if e["failed_windows"] is None or not e["total_windows"]:
            counts_known_w = False
        else:
            fw += e["failed_windows"]
            tw += e["total_windows"]
        if kind is not None:
            v = acc[kind]
            if kind == "qwen" and v.get("complete") is False:
                reasons.append(f"{name}: Qwen verify-only pass incomplete")
            if v["failed_batches"] is None or not v["total_batches"]:
                counts_known_b = False
            else:
                fb += v["failed_batches"]
                tb += v["total_batches"]
    if not counts_known_w:
        reasons.append(
            "extraction window failure counts not reported (cannot verify <=5%)"
        )
    elif fw * FAILURE_CEILING[1] > FAILURE_CEILING[0] * tw:
        reasons.append(f"{fw}/{tw} extraction windows failed (>5%)")
    if kind is not None:
        if not counts_known_b:
            reasons.append(
                f"{kind} verifier batch failure counts not reported (cannot verify <=5%)"
            )
        elif fb * FAILURE_CEILING[1] > FAILURE_CEILING[0] * tb:
            reasons.append(f"{fb}/{tb} {kind} verifier batches failed (>5%)")
    return reasons


def summarize(scored: dict) -> dict:
    """A scoring as the results keep it: the counts and, per paper, its counts and score."""
    b = scored["binary_one_to_one"]
    return {
        "matches": b["tp"],
        "predicted": b["predicted"],
        "gold": b["gold"],
        "precision": b["precision"],
        "recall": b["recall"],
        "f1": b["f1"],
        "full_record": scored["full_record_macro_score"],
        "per_paper": [
            {
                "paper_id": r["paper_id"],
                **{k: r["binary_one_to_one"][k] for k in ("tp", "predicted", "gold")},
                "full_record": r["full_record"]["score"],
            }
            for r in scored["per_paper"]
        ],
    }


def pooled_compute(members: list[dict]) -> dict:
    """The parts' extraction compute added up (wall time: one after the other)."""
    out = {}
    for key in (
        "output_tokens",
        "reasoning_tokens",
        "prompt_tokens",
        "wall_seconds",
        "responses",
        "length_finishes",
    ):
        values = [m["compute"].get(key) for m in members]
        out[key] = sum(values) if all(v is not None for v in values) else None
    return out


def cohort_label(split: str, model: str, arm: str, fmt: str) -> str:
    return f"{split}/{model}/{arm}/{fmt}"


def score_part(
    family: dict, root: Path, root_name: str, group: str, arms, kind: str
) -> dict:
    """Every single cohort x stage and every setup of one group of one split and model.

    kind: "arm" (a grid arm), "cross" (an arm pool), "trained" (a trained-prompt arm) or "trained-pool" (TR-<optimizer>-8k+16k32k)."""
    split, model = family["split"], family["model"]
    ids = set(family["gold"])
    single_arm = kind in ("arm", "trained")
    trained = kind in ("trained", "trained-pool")
    part: dict[str, Any] = {
        "split": split,
        "model": model,
        "group": group,
        "reference": family["evidence"],
        "cohorts": {},
        "missing_cohorts": [],
        "missing_stages": [],
        "cohort_scores": [],
        "configs": [],
    }
    cohorts = {}
    for arm in arms:
        for fmt in FORMATS:
            d = root / split / model / arm / fmt
            cid = cohort_label(split, model, arm, fmt)
            if not d.is_dir():
                if kind == "arm":
                    part["missing_cohorts"].append(cid)
                continue
            if str(d) not in family["cohorts"]:
                name = f"{root_name}/{split}/{model}/{arm}/{fmt}"
                family["cohorts"][str(d)] = load_cohort(d, ids, name)
            c = family["cohorts"][str(d)]
            c["model"], c["label"] = model, cid
            cohorts[(arm, fmt)] = c
            if single_arm:  # each cohort is reported once, by its own arm
                part["cohorts"][cid] = {
                    "directory": c["directory"],
                    "errors": c["errors"],
                    "status": c["status"],
                    "status_sha256": c["status_sha256"],
                    "accounting": c["accounting"],
                    "compute": c["compute"],
                    "stages_present": {
                        st: len(v["papers"]) for st, v in c["stages"].items()
                    },
                }
                part["missing_stages"].extend(
                    f"{cid}/{st}" for st in STAGES[model] if st not in c["stages"]
                )
    if not cohorts:
        return part

    def scored(members, stage):
        predicted = config_records(members, stage)
        # Identical records give identical scores: each distinct set is scored once.
        key = hashlib.sha256(
            json.dumps(predicted, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        if "sets" in family:  # --write-scorings: keep the records, score nothing
            family["sets"].setdefault(key, predicted)
            return {"records_sha256": key}
        cache = family["scored"]
        if key not in cache:
            started = time.time()
            result = score.score(family["gold"], predicted, family["scorer"])
            cache[key] = summarize(result)
            cache[key]["seconds"] = time.time() - started
        return {**copy.deepcopy(cache[key]), "records_sha256": key}

    def entry_for(parts, stage, entry):
        """parts: [(arm, format)] in concatenation order (arm-major)."""
        absent = [
            f"{a}/{f}"
            for a, f in parts
            if (a, f) not in cohorts or stage not in cohorts[(a, f)]["stages"]
        ]
        entry["cohort_labels"] = [cohort_label(split, model, a, f) for a, f in parts]
        if absent:
            entry.update(state="missing", missing_parts=absent, eligible=False)
            return entry
        members = [cohorts[p] for p in parts]
        try:
            entry.update(state="scored", **scored(members, stage))
        except Exception as error:
            raise RuntimeError(
                f"{entry['id']}: {type(error).__name__}: {error}"
            ) from error
        calls, notes = [], []
        for m in members:
            n, why = stage_calls(m, stage)
            calls.append(n)
            if why:
                notes.append(why)
        entry["model_calls"] = sum(calls) if all(n is not None for n in calls) else None
        entry["model_calls_by_part"] = {
            cohort_label(split, model, a, f): n
            for (a, f), n in zip(parts, calls, strict=True)
        }
        entry["model_calls_note"] = "; ".join(sorted(set(notes))) or None
        entry["ineligible_reasons"] = config_eligibility(members, stage, ids)
        entry["eligible"] = not entry["ineligible_reasons"]
        entry["part_status_reasons"] = {
            m["label"]: config_eligibility([m], stage, ids) for m in members
        }
        entry["compute"] = pooled_compute(members)
        thinking = all(m["label"].endswith("-think") for m in members)
        entry["reasoning"] = "default-on" if thinking else "off"
        return entry

    for stage in STAGES[model]:
        if single_arm:
            arm = group
            for fmt in FORMATS:  # every single cohort x stage
                if (arm, fmt) in cohorts and stage in cohorts[(arm, fmt)]["stages"]:
                    entry = {
                        "id": f"{split}/{model}/{arm}/{fmt}/{stage}",
                        "split": split,
                        "model": model,
                        "arm": arm,
                        "arms": [arm],
                        "format": fmt,
                        "passes": fmt,
                        "formats": [fmt],
                        "stage": stage,
                        "candidate": True,  # amendment 3
                    }
                    part["cohort_scores"].append(entry_for([(arm, fmt)], stage, entry))
            for passes, fmts in PASSES.items():
                if (
                    trained
                    and len(fmts) > 1
                    and not all((arm, f) in cohorts for f in fmts)
                ):
                    continue  # trained programs: a pool only when all its parts exist
                if trained and len(fmts) == 1:
                    continue  # a trained arm's single formats are the cohort entries
                entry = {
                    "id": f"{split}/{model}/{arm}/{passes}/{stage}",
                    "split": split,
                    "model": model,
                    "arm": arm,
                    "arms": [arm],
                    "passes": passes,
                    "formats": list(fmts),
                    "stage": stage,
                    "candidate": True,
                    "trained": trained,
                }
                part["configs"].append(
                    entry_for([(arm, f) for f in fmts], stage, entry)
                )
        else:
            for passes, fmts in CROSS_ARM_PASSES.items():
                if trained and not all((a, f) in cohorts for a in arms for f in fmts):
                    continue
                entry = {
                    "id": f"{split}/{model}/{group}/{passes}/{stage}",
                    "split": split,
                    "model": model,
                    "arm": group,
                    "arms": list(arms),
                    "passes": passes,
                    "formats": list(fmts),
                    "stage": stage,
                    "candidate": True,
                    "cross_arm": True,
                    "trained": trained,
                }
                part["configs"].append(
                    entry_for([(a, f) for a in arms for f in fmts], stage, entry)
                )
    return part


# 5. Running it


def grid_parts(roots: list[tuple[Path, str]]) -> tuple[list[tuple], dict]:
    """Every part of the grid, in the stored order: the grid's arms and arm pools per split and model, then the trained-prompt arms and their pools found under the roots. A part is (split, model, root, root name, group, arms, kind). Also what was found and missing."""
    grid_root, grid_name = roots[0]
    parts: list[tuple] = [
        (
            split,
            model,
            grid_root,
            grid_name,
            group,
            ARM_POOLS.get(group, (group,)),
            "arm" if group in ARMS else "cross",
        )
        for split in SPLITS
        for model in MODELS
        for group in GROUPS
    ]
    found_arms: dict[str, list[str]] = {}
    missing: list[str] = []
    conflicts: list[str] = []
    for split in SPLITS:
        for model in MODELS:
            seen: set[str] = set()
            for base_root, base_name in roots:
                folder = base_root / split / model
                found = (
                    sorted(
                        d.name
                        for d in folder.iterdir()
                        if d.is_dir() and d.name.startswith(TRAINED_PREFIX)
                    )
                    if folder.is_dir()
                    else []
                )
                conflicts.extend(
                    f"{base_name}/{split}/{model}/{a}" for a in found if a in seen
                )
                found = [a for a in found if a not in seen]
                seen.update(found)
                if found:
                    found_arms[f"{split}/{model}@{base_name}"] = found
                for arm in found:
                    parts.append(
                        (split, model, base_root, base_name, arm, (arm,), "trained")
                    )
                singles = {a for a in found if "+" not in a}
                for optimizer in sorted({a.rsplit("-", 1)[0] for a in singles}):
                    pool = tuple(f"{optimizer}-{k}" for k in TRAINED_KINDS)
                    name = "+".join([pool[0], *TRAINED_KINDS[1:]])
                    if name not in found and all(p in singles for p in pool):
                        parts.append(
                            (
                                split,
                                model,
                                base_root,
                                base_name,
                                name,
                                pool,
                                "trained-pool",
                            )
                        )
            for optimizer in TRAINED_OPTIMIZERS:
                for k in TRAINED_KINDS:
                    for fmt in TRAINED_FORMATS:
                        arm = f"{TRAINED_PREFIX}{optimizer}-{k}"
                        if not any(
                            (r / split / model / arm / fmt).is_dir() for r, _ in roots
                        ):
                            missing.append(cohort_label(split, model, arm, fmt))
    found = {
        "trained_roots": [name for _, name in roots],
        "trained_root_conflicts": conflicts,
        "trained_arms_found": found_arms,
        "missing_trained_cohorts": missing,
    }
    return parts, found


def write_family_scorings(task: dict) -> int:
    """--write-scorings: every distinct record set of one split and model as a folder of paper files, and their list for `evaluate score` (`<split>-<model>.json`), into the task's folder. Scores nothing. Returns the number of record sets."""
    split, model = task["split"], task["model"]
    reference, ready, name = task["reference"]
    gold, evidence = load_reference(reference, ready, split, name)
    family = {
        "split": split,
        "model": model,
        "gold": gold,
        "evidence": evidence,
        "scorer": None,
        "cohorts": {},
        "scored": {},
        "sets": {},
    }
    for _, (_, _, root, root_name, group, arms, kind) in task["parts"]:
        score_part(family, root, root_name, group, arms, kind)
    folder = Path(task["write"])
    entries = []
    for key, predicted in family["sets"].items():
        papers = folder / SETS / key
        papers.mkdir(parents=True, exist_ok=True)
        for pid, records in predicted.items():
            value = {"paper_id": pid, "definition_records": records}
            (papers / f"{quote(pid, safe='')}.json").write_text(
                json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        entries.append(
            {
                "id": key,
                "reference": f"references/{split}.json",
                "papers": f"{SETS}/{key}",
            }
        )
    files.write_json(folder / f"{split}-{model}.json", entries, indent=1)
    print(f"{split}/{model}: {len(entries)} distinct record sets written", flush=True)
    return len(entries)


def write_scorings_command(
    target: Path, data_dir: Path, tasks: list[dict], workers: int, roots: list
) -> int:
    """Write every family's record sets and scorings list into TARGET, with copies of the two references (the lists name them relative to TARGET), and a receipt."""
    for split, (reference, _) in REFERENCES.items():
        copy_to = target / "references" / f"{split}.json"
        copy_to.parent.mkdir(parents=True, exist_ok=True)
        copy_to.write_bytes((data_dir / reference).read_bytes())
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)),
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            counts = list(pool.map(write_family_scorings, tasks))
    else:
        counts = [write_family_scorings(task) for task in tasks]
    lists = {
        f"{t['split']}-{t['model']}.json": target / f"{t['split']}-{t['model']}.json"
        for t in tasks
    }
    receipt = files.receipt(
        "evaluate grid --write-scorings",
        settings={"families": [f"{t['split']}/{t['model']}" for t in tasks]},
        inputs={name: root for root, name in roots}
        | {f"{s}_reference": data_dir / ref for s, (ref, _) in REFERENCES.items()},
        outputs={**lists, SETS: target / SETS},
    )
    files.write_json(target / files.RECEIPT, receipt)
    print(
        f"{sum(counts)} distinct record sets in {len(tasks)} scorings lists; "
        f"wrote {target}"
    )
    return 0


def score_family(task: dict) -> dict[int, dict]:
    """All parts of one split and model, in one process with one pair-score cache, on the GPU the task names. Returns {part number: part}."""
    if task["gpu"] is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = task["gpu"]
    split, model = task["split"], task["model"]
    reference, ready, name = task["reference"]
    gold, evidence = load_reference(reference, ready, split, name)
    scorer, write, _, _ = score.pair_scorer(
        task["replay"], task["record"], task["nli_batch_size"]
    )
    family = {
        "split": split,
        "model": model,
        "gold": gold,
        "evidence": evidence,
        "scorer": scorer,
        "cohorts": {},
        "scored": {},
    }
    started = time.time()
    done = {}
    for number, (_, _, root, root_name, group, arms, kind) in task["parts"]:
        part_started = time.time()
        done[number] = score_part(family, root, root_name, group, arms, kind)
        n = len(done[number]["configs"]) + len(done[number]["cohort_scores"])
        print(
            f"{split}/{model}/{group}: {n} setups scored in "
            f"{time.time() - part_started:.0f} s",
            flush=True,
        )
    if write and task["record"]:
        write(Path(task["record"]))
    print(
        f"{split}/{model}: {len(family['scored'])} distinct record sets in "
        f"{time.time() - started:.0f} s",
        flush=True,
    )
    return done


def opus_test_calls(results: dict) -> int | None:
    """Opus's TEST calls, from its row of the TEST evaluation."""
    rows = {r["row"]: r for r in results.get("rows", [])}
    return score.as_int((rows.get(OPUS_ROW, {}).get("status") or {}).get("requests"))


def replicate(scores: dict, results: dict, path: Path, name: str) -> dict | None:
    """The plan's replicate note: the TEST result of the recipe that the TEST evaluation ran first (QWEN-POOL3), next to this grid's fresh generation of it."""
    rows = {r["row"]: r for r in results.get("rows", [])}
    old = rows.get(REPLICATE_ROW, {}).get("score", {}).get("binary_one_to_one")
    if not old:
        return None
    new = next((c for c in scores["configs"] if c["id"] == REPLICATE_CONFIG), None)
    text = (
        "First TEST generation of the same recipe (QWEN-POOL3): "
        f"{old['tp']}/{old['predicted']}."
    )
    if new and new.get("state") == "scored":
        text += (
            f" This grid's fresh generation: {new['matches']}/{new['predicted']} "
            f"(difference {new['matches'] - old['tp']:+d} matches)."
        )
    else:
        text += " This grid's fresh generation is not available."
    return {
        "previous_results": name,
        "previous_sha256": files.digest(path),
        "qwen_pool3": old,
        "text": text,
    }


def grid_command(
    out: str,
    grid_arg: str | None,
    trained_arg: str | None,
    data: str,
    test_results_arg: str | None,
    families_arg: list[str] | None,
    workers: int,
    gpus: int,
    replay: str | None,
    record: str | None,
    nli_batch_size: int,
    write_scorings: bool = False,
) -> int:
    """Score the grid; write scores.json and a receipt into OUT. With `write_scorings`, write the distinct record sets and their scorings lists for `evaluate score` instead."""
    target, data_dir = Path(out), Path(data)
    if target.exists() and any(target.iterdir()):
        raise SystemExit(f"{out} is not empty; choose a new folder for --out.")
    if write_scorings and (replay or record):
        raise SystemExit(
            "--write-scorings scores nothing: no --replay- or --record-scores."
        )
    if replay and record:
        raise SystemExit("Give --replay-scores or --record-scores, not both.")
    if record and Path(record).exists():
        raise SystemExit(f"{record} exists; --record-scores writes a new folder.")
    if replay and not Path(replay).is_dir():
        raise SystemExit(
            f"{replay} is not a folder of recorded pair scores. The selection's are the "
            "output of its cluster job (docs/step-3-extraction/selection.md, the cluster job)."
        )
    grid_root, grid_name = files.located(grid_arg, GRID)
    trained_root, trained_name = files.located(trained_arg, TRAINED)
    results_path, results_name = files.located(test_results_arg, TEST_RESULTS)
    for folder in (grid_root, trained_root):
        if not folder.is_dir():
            raise SystemExit(
                f"{folder} is not a folder. The saved cohorts hold the papers' windows, so they are "
                "not public; give your own runs with --grid and --trained (docs/step-3-extraction/selection.md)."
            )
    previous = read_json(results_path)
    roots = [(grid_root, grid_name)]
    if trained_root.resolve() != grid_root.resolve():
        roots.append((trained_root, trained_name))
    parts, found = grid_parts(roots)
    chosen = [f for f in FAMILIES if not families_arg or f in families_arg]
    os.environ["SCIDEF_NLI_BATCH_SIZE"] = str(nli_batch_size)
    device = None if replay or write_scorings else files.env("SCIDEF_NLI_DEVICE")
    if record:
        Path(record).mkdir(parents=True)
    visible = [
        g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip()
    ] or [str(i) for i in range(gpus)]
    spread = device is not None and device.startswith("cuda") and workers > 1
    tasks = []
    for position, family in enumerate(chosen):
        split, model = family.split("/")
        reference, ready = REFERENCES[split]
        tasks.append(
            {
                "split": split,
                "model": model,
                "reference": (
                    data_dir / reference,
                    data_dir / ready if ready else None,
                    str(data_dir / reference),
                ),
                "parts": [
                    (n, p)
                    for n, p in enumerate(parts)
                    if (p[0], p[1]) == (split, model)
                ],
                "replay": str(Path(replay) / f"{split}-{model}.json")
                if replay
                else None,
                "record": str(Path(record) / f"{split}-{model}.json")
                if record
                else None,
                "nli_batch_size": nli_batch_size,
                "gpu": visible[position % max(1, gpus)] if spread else None,
                "write": str(target) if write_scorings else None,
            }
        )
    if write_scorings:
        return write_scorings_command(target, data_dir, tasks, workers, roots)
    started = time.time()
    try:
        if workers > 1 and len(tasks) > 1:
            with ProcessPoolExecutor(
                max_workers=min(workers, len(tasks)),
                mp_context=multiprocessing.get_context("spawn"),
                max_tasks_per_child=1,
            ) as pool:
                outcomes = list(pool.map(score_family, tasks))
        else:
            outcomes = [score_family(task) for task in tasks]
    except (RuntimeError, ValueError, metric.NotRecorded) as error:
        rerun = (
            " The scoring asks for a pair score the recording lacks, so it no longer "
            "scores what the recorded run scored: rerun that run with the models "
            "(docs/step-3-extraction/selection.md, the cluster job)."
            if replay and "no recorded" in str(error)
            else ""
        )
        raise SystemExit(
            f"Scoring failed: {error}. No results were written.{rerun}"
        ) from error
    done = {n: part for outcome in outcomes for n, part in outcome.items()}
    scores: dict[str, Any] = {
        "schema": SCORES_SCHEMA,
        "created_utc": now(),
        "grid_root": grid_name,
        "references": {},
        "cohorts": {},
        "missing_cohorts": [],
        "missing_trained_cohorts": found["missing_trained_cohorts"],
        "missing_stages": [],
        "cohort_scores": [],
        "configs": [],
        "workers": workers,
        "environment": {k: os.environ.get(k) for k in ENVIRONMENT},
        "trained_roots": found["trained_roots"],
        "trained_root_conflicts": found["trained_root_conflicts"],
        "trained_arms_found": found["trained_arms_found"],
    }
    for number in sorted(done):  # the stored order: split, model, group
        part = done[number]
        scores["references"].setdefault(part["split"], part["reference"])
        for key in ("missing_cohorts", "missing_stages", "cohort_scores", "configs"):
            scores[key].extend(part[key])
        scores["cohorts"].update(part["cohorts"])
    note = replicate(scores, previous, results_path, results_name)
    if note:
        scores["replicate"] = note
    scores["opus_test_calls"] = opus_test_calls(previous)
    target.mkdir(parents=True, exist_ok=True)
    files.write_json(target / SCORES, scores, **JSON_FORMAT)
    outputs = {SCORES: target / SCORES}
    if record:
        outputs["pair_scores"] = Path(record)
    test_reference, test_ready = REFERENCES["test"]
    inputs = {
        "grid": grid_root,
        "trained": trained_root,
        "test_results": results_path,
        "dev_reference": data_dir / REFERENCES["dev"][0],
        "test_reference": data_dir / test_reference,
        "test_ready": data_dir / str(test_ready),
    }
    if replay:
        inputs["pair_scores"] = Path(replay)
    receipt = files.receipt(
        "evaluate grid",
        settings={
            "threshold": metric.MATCH_THRESHOLD,
            "nli_batch_size": nli_batch_size,
            "device": device,
            "pair_scores": f"replayed from {replay}"
            if replay
            else f"computed on {device}",
            **metric.model_names(),
            "families": chosen,
            "workers": workers,
            "gpus": gpus,
            "seconds": round(time.time() - started),
        },
        inputs=inputs,
        outputs=outputs,
    )
    files.write_json(target / files.RECEIPT, receipt)
    print(
        f"{len(scores['configs'])} setups and {len(scores['cohort_scores'])} single cohorts "
        f"scored in {time.time() - started:.0f} s; wrote {out}"
    )
    return 0
