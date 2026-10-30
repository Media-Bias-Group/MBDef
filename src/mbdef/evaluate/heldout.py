"""`mbdef evaluate heldout`: the held-out TEST rescoring.

The TEST evaluation ran 12 frozen rows (older extractor candidates and the Opus reference) on the 29 held-out TEST papers, whose reference has 365 definitions. This command rescores the saved rows from scratch and applies the plan's frozen rules:

1. **Inputs.** The frozen plan (its sha256 is checked), the reference and its readiness record (whose declared sha256 must match), and the cohort: the 29 papers of test-sources.json.
2. **Rows.** Each row folder holds `status.json` and `papers/<paper>.json`. A file that cannot be read, or names a paper outside the cohort, is listed as an error of the row; a cohort paper without a file counts as no predictions.
3. **Scoring.** Every row with a folder is scored with the general scorer (`score.score`). A row that cannot be scored stops the command, and nothing is written.
4. **Eligibility** (frozen): every paper has an output file, status.complete, and at most 5% of the extraction windows and of the verifier batches failed.
5. **Pick** (frozen, verbatim in the plan): among eligible rows with recall >= 80% (292 of 365), the highest precision, but prefer a self-hosted Qwen row within 2 matches of that leader; if no row reaches 80%, the highest recall among rows with precision >= 15%. In the released run no row reached 80%, so the fallback picked QWEN-POOL3 (291 of 365, precision 24.8%).
6. **Outputs.** results.json and results.md in the format of the stored results, and a receipt.

The outputs match what the original scorer wrote for the same rows; the test fixtures hold its outputs for small made-up rows.
"""

import hashlib
import json
import os
import time
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

from mbdef import files
from mbdef.evaluate import metric, score

# Inputs in the repository's data folder (--data): the frozen plan, reference and readiness.
TEST_EVAL = "frozen/results/test-eval-20260924"
PLAN = f"{TEST_EVAL}/plan.json"
# The plan's neutral copy: the original plan (sha256 7cb89ade…, which the readiness record and the grid audit list) with only its `authorization` field reworded so that it names no person; every rule the rescoring reads is unchanged (docs/step-3-extraction/evaluate.md).
PLAN_SHA256 = "bd80acc8da32d6a97815358327076a2fd715f1656ebc6100783f34ca70767e9d"
REFERENCE = f"{TEST_EVAL}/inputs/test-reference.json"
READY = f"{TEST_EVAL}/inputs/READY.json"
# Inputs under DATA_ROOT (data/data-root.tsv): the cohort and the saved rows of the run.
RUN = "frozen/artifacts/experiments/test-eval-20260924"
SOURCES = f"{RUN}/final/inputs/test-sources.json"
ROWS = f"{RUN}/collected/rows"
# The frozen rules of the plan.
TEST_SPLITS = ("original_test", "new_test")
RECALL_FLOOR = (80, 100)  # recall >= 80%
PRECISION_FLOOR = (15, 100)  # precision >= 15%
QWEN_MARGIN_MATCHES = 2  # "within 2 matches of the leader"
FAILURE_CEILING = (5, 100)  # at most 5% failed windows / batches
# Outputs, in --out, in the format of the stored results.
RESULTS = "results.json"
RESULTS_MD = "results.md"
RESULTS_SCHEMA = "test-eval-20260924-results/1"
JSON_FORMAT: dict[str, Any] = {"indent": 2, "ensure_ascii": False}
MATCHER = (
    "mbdef.evaluate.score.score: per paper the M3 one-to-one matches of the "
    "definitions at 0.50 (calibrated_base_equal_mean), pooled over the cohort; full record = "
    "mean per-paper M3-full-v1 score at 0.50"
)
SCORING_CODE = ("metric.py", "score.py", "heldout.py")  # in this folder


def now() -> str:
    return datetime.now(UTC).isoformat()


# 1. Inputs


def check_plan(path: Path) -> dict:
    if files.digest(path) != PLAN_SHA256:
        raise SystemExit(
            f"{path} is not the frozen TEST plan (sha256 {PLAN_SHA256[:8]}…)."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_reference(path: Path, ready: Path, cohort: set[str] | None):
    """Reference records by paper id; checks the declared counts, the cohort and READY."""
    raw = json.loads(path.read_text(encoding="utf-8"))
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
        if row.get("split") not in TEST_SPLITS:
            raise ValueError(f"Reference paper outside the cohort splits: {pid}")
        score.definition_count(row.get("definition_records"))
        gold[pid] = row
    total = sum(score.definition_count(r["definition_records"]) for r in gold.values())
    if (
        meta.get("paper_count", len(gold)) != len(gold)
        or meta.get("definition_count", total) != total
    ):
        raise ValueError(
            "Reference paper/definition counts disagree with its declaration"
        )
    if cohort is not None and set(gold) != cohort:
        raise ValueError("Reference papers differ from the frozen cohort")
    evidence: dict[str, Any] = {
        "path": str(path),
        "sha256": files.digest(path),
        "papers": len(gold),
        "definitions": total,
        "declared": {
            k: meta.get(k)
            for k in (
                "schema",
                "paper_count",
                "definition_count",
                "source",
                "source_sha256",
            )
        },
    }
    if not ready.is_file():
        raise FileNotFoundError(f"Reference READY marker missing: {ready}")
    evidence["ready"] = score.checked_ready(path, ready)
    return gold, evidence


# 2. Rows


def discover_rows(roots: list[Path]) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if (
                child.is_dir()
                and not child.name.startswith(".")
                and ((child / "status.json").exists() or (child / "papers").is_dir())
            ):
                found.setdefault(child.name, []).append(child)
    return found


def load_row(directory: Path, cohort: set[str]) -> dict:
    result = {"directory": str(directory), "errors": [], "status": None, "papers": {}}
    status_path = directory / "status.json"
    if status_path.is_file():
        try:
            result["status"] = json.loads(status_path.read_text())
            result["status_sha256"] = files.digest(status_path)
        except Exception as e:
            result["errors"].append(f"status.json unreadable: {type(e).__name__}: {e}")
    else:
        result["errors"].append("status.json missing")
    papers_dir = directory / "papers"
    file_set = sorted(papers_dir.rglob("*.json")) if papers_dir.is_dir() else []
    digests = {}
    for path in file_set:
        try:
            value = json.loads(path.read_text())
            pid = value["paper_id"]
            records = value["definition_records"]
            score.definition_count(records)
        except Exception as e:
            result["errors"].append(
                f"invalid paper file {path.name}: {type(e).__name__}: {e}"
            )
            continue
        if pid not in cohort:
            result["errors"].append(f"paper outside cohort: {pid}")
            continue
        if pid in result["papers"]:
            result["errors"].append(f"duplicate paper file for {pid}")
            continue
        if value.get("row") not in (None, directory.name):
            result["errors"].append(f"paper file names another row: {path.name}")
        result["papers"][pid] = records
        digests[pid] = hashlib.sha256(path.read_bytes()).hexdigest()
    result["paper_file_sha256"] = digests
    status = result["status"] or {}
    if status.get("row") not in (None, directory.name):
        result["errors"].append(
            f"status.json row {status.get('row')!r} differs from directory {directory.name!r}"
        )
    return result


# 4. Eligibility


def eligibility(row: dict, cohort_size: int) -> tuple[bool, list[str]]:
    """Eligible only if every paper produced output and at most 5% of the row's extraction windows and verifier batches failed."""
    reasons = []
    status = row.get("status") or {}
    if row.get("errors"):
        reasons.extend(row["errors"])
    if row.get("conflict"):
        reasons.append("row directory appears under more than one rows root")
    if status.get("complete") is not True:
        reasons.append("status.complete is not true")
    present = len(row.get("papers", {}))
    if present != cohort_size:
        reasons.append(f"only {present}/{cohort_size} papers have output files")
    declared = score.as_int(status.get("papers"))
    if declared is not None and declared != cohort_size:
        reasons.append(f"status reports {declared}/{cohort_size} papers with output")
    failed_papers = status.get("failed_paper_ids") or []
    if failed_papers:
        reasons.append(f"status lists failed papers: {sorted(failed_papers)}")
    fw, tw = (
        score.as_int(status.get("failed_windows")),
        score.as_int(status.get("total_windows")),
    )
    if fw is None or tw is None or tw <= 0:
        reasons.append(
            "extraction window failure counts not reported (cannot verify <=5%)"
        )
    elif fw * FAILURE_CEILING[1] > FAILURE_CEILING[0] * tw:
        reasons.append(f"{fw}/{tw} extraction windows failed (>5%)")
    fb, tb = (
        score.as_int(status.get("failed_batches")),
        score.as_int(status.get("total_batches")),
    )
    if tb is not None and tb > 0:
        if fb is None:
            reasons.append("verifier batch failure count not reported")
        elif fb * FAILURE_CEILING[1] > FAILURE_CEILING[0] * tb:
            reasons.append(f"{fb}/{tb} verifier batches failed (>5%)")
    elif fb:
        reasons.append(f"{fb} failed verifier batches reported without a batch total")
    return not reasons, reasons


# 5. Pick


def matches_of(row) -> int:
    return row["score"]["binary_one_to_one"]["tp"]


def precision_of(row) -> Fraction:
    b = row["score"]["binary_one_to_one"]
    return Fraction(b["tp"], b["predicted"]) if b["predicted"] else Fraction(0)


def best_by_precision(rows):
    """Highest precision; ties -> more matches -> row id (exact fractions)."""
    ranked = sorted(rows, key=lambda r: (-precision_of(r), -matches_of(r), r["row"]))
    return ranked[0] if rows else None


def best_by_recall(rows):
    """Highest recall (= most matches, one shared denominator); ties -> precision -> id."""
    ranked = sorted(rows, key=lambda r: (-matches_of(r), -precision_of(r), r["row"]))
    return ranked[0] if rows else None


def apply_pick_rule(rows: list[dict], reference_definitions: int) -> dict:
    """The plan's rule, verbatim: "Among eligible rows with TEST recall >= 80%, pick the highest precision, but prefer a self-hosted Qwen row if its recall is within 2 matches of the leader. If no row reaches 80%, pick the highest recall among rows with precision >= 15%. Report every row either way."

    Its reading, fixed before any TEST score: the leader is the highest-precision eligible row with recall >= 80%; a self-hosted Qwen row with recall >= 80% and at least the leader's matches - 2 is preferred (the highest-precision such row). The reading that also admits Qwen rows below 80% is evaluated and reported, never used.
    """
    eligible = [r for r in rows if r["eligible"]]
    at_floor = [
        r
        for r in eligible
        if score.ge(
            r["score"]["binary_one_to_one"]["tp"], reference_definitions, RECALL_FLOOR
        )
    ]
    decision = {
        "rule": "frozen plan pick_rule",
        "eligible_rows": [r["row"] for r in eligible],
        "rows_with_recall_at_least_80": [r["row"] for r in at_floor],
    }
    if at_floor:
        leader = best_by_precision(at_floor)
        lt = leader["score"]["binary_one_to_one"]["tp"]
        decision["leader"] = leader["row"]
        qwen = [
            r
            for r in at_floor
            if r["self_hosted_qwen"] and matches_of(r) >= lt - QWEN_MARGIN_MATCHES
        ]
        preferred = best_by_precision(qwen)
        alt_pool = [
            r
            for r in eligible
            if r["self_hosted_qwen"] and matches_of(r) >= lt - QWEN_MARGIN_MATCHES
        ]
        alternative = (
            best_by_precision(alt_pool) if not leader["self_hosted_qwen"] else leader
        )
        if leader["self_hosted_qwen"]:
            pick = leader
            why = (
                f"{leader['row']} has the highest precision among eligible rows with recall "
                ">= 80% and is itself a self-hosted Qwen row."
            )
        elif preferred is not None:
            pick = preferred
            why = (
                f"Leader by precision among eligible rows with recall >= 80% is "
                f"{leader['row']} ({lt} matches); self-hosted Qwen row {preferred['row']} also "
                f"has recall >= 80% and {preferred['score']['binary_one_to_one']['tp']} matches, "
                f"within {QWEN_MARGIN_MATCHES} of the leader, so it is preferred."
            )
        else:
            pick = leader
            why = (
                f"{leader['row']} has the highest precision among eligible rows with recall "
                f">= 80%; no eligible self-hosted Qwen row with recall >= 80% is within "
                f"{QWEN_MARGIN_MATCHES} matches ({lt - QWEN_MARGIN_MATCHES}+) of it."
            )
        decision["branch"] = "recall >= 80% available"
        decision["alternative_reading"] = {
            "description": "Also admit eligible self-hosted Qwen rows below 80% recall "
            "within 2 matches of the leader",
            "pick": (alternative or leader)["row"],
            "differs": (alternative or leader)["row"] != pick["row"],
        }
    else:
        pool = [
            r
            for r in eligible
            if score.ge(
                r["score"]["binary_one_to_one"]["tp"],
                r["score"]["binary_one_to_one"]["predicted"],
                PRECISION_FLOOR,
            )
        ]
        decision["branch"] = "no eligible row reaches 80% recall"
        decision["rows_with_precision_at_least_15"] = [r["row"] for r in pool]
        pick = best_by_recall(pool)
        why = (
            f"No eligible row reaches 80% recall; {pick['row']} has the highest recall "
            "among eligible rows with precision >= 15%."
            if pick
            else "No eligible row reaches 80% recall and none has precision >= 15%; no "
            "row can be picked."
        )
    decision["pick"] = pick["row"] if pick else None
    decision["reasoning"] = why
    return decision


# 6. Outputs


def pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def render_markdown(result: dict) -> str:
    """results.md, in the format of the stored results."""
    ref = result["reference"]
    lines = [
        "# TEST evaluation results (29 held-out papers)",
        "",
        f"Created {result['created_utc']}. Reference: {ref['papers']} papers with "
        f"{ref['definitions']} reference definitions. A predicted definition counts as a match "
        "when the frozen definition matcher pairs it one-to-one with a reference definition of "
        "the same paper at similarity 0.50 or more.",
        "",
        "Precision = matches / predicted definitions. Recall = matches / all reference "
        "definitions. Full-record score = average per-paper agreement over all record fields "
        "(terms, definitions, synonyms, and so on).",
        "",
        "| Row | Model family | Matches / predicted | Precision | Recall | F1 | Full-record | "
        "Papers with output | Failed windows | Failed verifier batches | Requests | Cost (USD) "
        "| Eligible |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in result["rows"]:
        if r["state"] == "missing":
            lines.append(
                f"| {r['row']} | {r.get('family') or ''} | missing | | | | | | | | | | no "
                "(not produced by the deadline) |"
            )
            continue
        b = r["score"]["binary_one_to_one"]
        s = r.get("status") or {}
        fw = f"{s.get('failed_windows')}/{s.get('total_windows')}"
        fb = (
            f"{s.get('failed_batches')}/{s.get('total_batches')}"
            if score.as_int(s.get("total_batches"))
            else "n/a"
        )
        cost = s.get("cost_usd")
        requests = s.get("requests")
        elig = (
            "yes"
            if r["eligible"]
            else "no (" + "; ".join(r["ineligible_reasons"]) + ")"
        )
        if not r.get("planned"):
            elig = "not a planned row (reported only)"
        lines.append(
            f"| {r['row']} | {r.get('family') or ''} | {b['tp']} / {b['predicted']} | "
            f"{pct(b['precision'])} | {pct(b['recall'])} | {pct(b['f1'])} | "
            f"{r['score']['full_record_macro_score']:.3f} | "
            f"{r['papers_with_output']}/{ref['papers']} | {fw} | {fb} | "
            f"{requests if requests is not None else 'n/a'} | "
            f"{cost if cost is not None else 'n/a'} | {elig} |"
        )
    lines += ["", "Row descriptions (from the frozen plan):", ""]
    for r in result["rows"]:
        if r.get("what"):
            lines.append(
                f"- {r['row']} ({r.get('family')}, {r.get('cost')}): {r['what']}"
            )
    d = result["decision"]
    lines += [
        "",
        "## Pick",
        "",
        f"Rule (frozen before any TEST result): {result['pick_rule_text']}",
        "",
        f"Eligibility rule: {result['eligibility_rule_text']}",
        "",
        f"**Pick: {d['pick'] or 'none'}.** {d['reasoning']}",
        "",
    ]
    alt = d.get("alternative_reading")
    if alt:
        lines.append(
            f"Reading check: admitting self-hosted Qwen rows below 80% recall would pick "
            f"{alt['pick']}"
            + (
                " (different; flagged for the owner)."
                if alt["differs"]
                else " (same pick)."
            )
        )
        lines.append("")
    lines += ["## Caveats", ""] + [f"- {c}" for c in result["caveats"]]
    if result.get("unplanned_rows"):
        lines += [
            "",
            "Rows found but not in the plan (reported, never picked): "
            f"{', '.join(result['unplanned_rows'])}.",
        ]
    return "\n".join(lines) + "\n"


def rescore(
    reference: Path,
    ready: Path,
    rows_roots: list[Path],
    cohort: set[str] | None,
    plan: dict,
    pair_scorer: metric.PairScorer,
    environment: dict,
) -> dict:
    """The results of every planned (and every found) row, and the pick."""
    gold, ref_evidence = load_reference(reference, ready, cohort)
    planned = {r["row"]: r for r in plan["rows"]}
    found = discover_rows(rows_roots)
    references = {pid: paper["definition_records"] for pid, paper in gold.items()}
    rows = []
    for row_id in list(planned) + sorted(set(found) - set(planned)):
        meta = planned.get(row_id, {})
        entry = {
            "row": row_id,
            "planned": row_id in planned,
            "family": meta.get("family"),
            "cost": meta.get("cost"),
            "what": meta.get("what"),
            "self_hosted_qwen": meta.get("family") == "Qwen"
            and meta.get("cost") == "self-hosted",
        }
        dirs = found.get(row_id, [])
        if not dirs:
            entry.update(
                state="missing",
                eligible=False,
                ineligible_reasons=["row not present at scoring time"],
            )
            rows.append(entry)
            continue
        loaded = load_row(dirs[0], set(gold))
        if len(dirs) > 1:
            loaded["conflict"] = [str(d) for d in dirs]
        entry.update(
            state="scored",
            directory=loaded["directory"],
            status=loaded["status"],
            errors=loaded["errors"],
            conflict=loaded.get("conflict"),
            papers_with_output=len(loaded["papers"]),
            paper_file_sha256=loaded["paper_file_sha256"],
            status_sha256=loaded.get("status_sha256"),
        )
        started = time.time()
        try:
            entry["score"] = score.score(references, loaded["papers"], pair_scorer)
        except Exception as error:
            raise SystemExit(
                f"Scoring {row_id} failed: {type(error).__name__}: {error}. "
                "No results were written."
            ) from error
        entry["scoring_seconds"] = time.time() - started
        ok, reasons = eligibility({**loaded, "row": row_id}, len(gold))
        if not entry["planned"]:
            ok, reasons = False, ["not a planned row"] + reasons
        entry.update(eligible=ok, ineligible_reasons=reasons)
        rows.append(entry)
        print(score.summary(row_id, entry["score"]), flush=True)
    decision = apply_pick_rule(
        [r for r in rows if r["state"] == "scored" and r["planned"]],
        ref_evidence["definitions"],
    )
    here = Path(__file__).parent
    return {
        "schema": RESULTS_SCHEMA,
        "created_utc": now(),
        "plan_sha256": PLAN_SHA256,
        "pick_rule_text": plan["pick_rule"],
        "eligibility_rule_text": plan["eligibility"],
        "matcher": MATCHER,
        "reference": ref_evidence,
        "cohort": sorted(gold),
        "rows_roots": [str(p) for p in rows_roots],
        "rows": rows,
        "missing_rows": [r["row"] for r in rows if r["state"] == "missing"],
        "unplanned_rows": [r["row"] for r in rows if not r["planned"]],
        "decision": decision,
        "caveats": plan.get("caveats", []),
        "environment": environment,
        "scoring_code_sha256": {
            f"src/mbdef/evaluate/{name}": files.digest(here / name)
            for name in SCORING_CODE
        },
    }


def heldout_command(
    out: str,
    data: str,
    plan_arg: str | None,
    reference_arg: str | None,
    ready_arg: str | None,
    sources_arg: str | None,
    rows_args: list[str] | None,
    replay: str | None,
    record: str | None,
    nli_batch_size: int,
) -> int:
    """Rescore the saved TEST rows and write results.json, results.md and a receipt into OUT."""
    target, data_dir = Path(out), Path(data)
    if target.exists() and any(target.iterdir()):
        raise SystemExit(f"{out} is not empty; choose a new folder for --out.")
    plan_path = Path(plan_arg) if plan_arg else data_dir / PLAN
    reference = Path(reference_arg) if reference_arg else data_dir / REFERENCE
    ready = Path(ready_arg) if ready_arg else data_dir / READY
    sources = Path(sources_arg) if sources_arg else files.data_root() / SOURCES
    roots = [Path(r) for r in rows_args] if rows_args else [files.data_root() / ROWS]
    plan = check_plan(plan_path)
    size = plan["cohort"]["papers"]
    cohort = {p["paper_id"] for p in json.loads(sources.read_text())["papers"]}
    if len(cohort) != size:
        raise SystemExit("Source cohort size differs from the plan")
    pair_scorer, write, settings, inputs = score.pair_scorer(
        replay, record, nli_batch_size
    )
    environment = {
        "SCIDEF_NLI_DEVICE": settings["device"],
        "SCIDEF_NLI_BATCH_SIZE": str(nli_batch_size),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    result = rescore(reference, ready, roots, cohort, plan, pair_scorer, environment)
    if len(result["cohort"]) != size:
        raise SystemExit("Reference cohort size differs from the plan")
    target.mkdir(parents=True, exist_ok=True)
    (target / RESULTS).write_text(
        files.json_text(result, **JSON_FORMAT), encoding="utf-8"
    )
    (target / RESULTS_MD).write_text(render_markdown(result), encoding="utf-8")
    outputs = {RESULTS: target / RESULTS, RESULTS_MD: target / RESULTS_MD}
    if write and record:
        write(Path(record))
        outputs["pair_scores"] = Path(record)
    receipt = files.receipt(
        "evaluate heldout",
        settings={
            "plan_sha256": PLAN_SHA256,
            "threshold": metric.MATCH_THRESHOLD,
            "recall_floor": RECALL_FLOOR,
            "precision_floor": PRECISION_FLOOR,
            "qwen_margin_matches": QWEN_MARGIN_MATCHES,
            "failure_ceiling": FAILURE_CEILING,
            "json_format": JSON_FORMAT,
            **settings,
        },
        inputs={
            "plan": plan_path,
            "reference": reference,
            "ready": ready,
            "sources": sources,
            **{
                ("rows" if i == 0 else f"rows-{i}"): root
                for i, root in enumerate(roots)
            },
            **inputs,
        },
        outputs=outputs,
    )
    files.write_json(target / files.RECEIPT, receipt)
    d = result["decision"]
    print(f"Pick: {d['pick'] or 'none'}. {d['reasoning']}")
    print("wrote", out)
    return 0
