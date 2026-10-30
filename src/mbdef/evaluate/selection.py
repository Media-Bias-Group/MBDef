"""`mbdef evaluate select`: pick the extractors from the scored grid.

`evaluate grid` (grid.py) scores every setup of the extractor grid on DEV and TEST. This command applies the frozen rules to those scores. Read it in five parts:

1. **The plan.** The rules are in the frozen plan and its amendments 1-9, whose sha256 are checked (`check_plan`); their texts are copied into the results.
2. **The audit.** An independent audit judged every cohort (one model run on one split): `ok`, or a defect with reasons, and whether its failure counts make it eligible. A setup is excluded if any of its cohorts has a defect (`audit_gate`), and the audit's eligibility replaces the one computed from the cohort's own status (`apply_audit_eligibility`).
3. **The pick rule** (amendment 8, `apply_floor_pick`). For each precision floor, 20%, 24% and 28%: among the eligible, audit-passing TEST setups with at least 292 of the 365 TEST definitions found (80% recall) and at least that precision, take the one that finds the most; setups within 2 matches of it are tied, and the tie goes to the higher precision, then to fewer LLM calls. Amendment 9 adds a fourth output that no rule picks: the single-model variant of the P>=28 pick, the same Gemma generations with the relevance filter but without the Qwen verifier (`single_model_variant`). The superseded rules (amendment 5 on TEST, amendment 1 on DEV) are still reported, for information.
4. **Outputs.** results.json and results.md in the format of the stored results (`render_markdown`), and a receipt.
5. **Correcting an audit** (`--replaces`). The first audit (sha256 806075de…) wrongly excluded all 48 trained-prompt cohorts, so the scores were selected again with the corrected audit (7cdcba40…). With `--replaces RESULTS --reason TEXT` the command first proves that RESULTS's own audit, applied to these scores, gives RESULTS's picks, writes that replay into `replaced/`, and records the correction in a `reselect` block.

A sixth part holds the command line of the cluster job that rescored the grid.
"""

import copy
import json
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

from mbdef import files, models
from mbdef.evaluate.grid import (
    ARMS,
    CROSS_ARM_PASSES,
    FAMILIES,
    GROUPS,
    MODELS,
    NOVERIFY_STAGES,
    PASSES,
    QV_STAGES,
    SELF_STAGES,
    SPLITS,
    STAGES,
    TRAINED_PREFIX,
    VERIFIER,
)
from mbdef.evaluate.score import ge

# 1. The plan (in the repository's data folder, --data)

PLAN_FOLDER = "frozen/results/grid-20260925"
PLAN_SHA256 = {
    "plan.json": "5604ef104203d3bb3e41a1b498c69399291e710df592d462e74d6d8045910322",
    "plan-amendment-1.json": "31fd6c999bd577f75699c62eff32a06cc0b23352433b94c53967a2670d812b6d",
    "plan-amendment-2.json": "8a297def8e45fce12900a2c3545e4b0967bd7a01470e37cb7f19dce391b947c9",
    "plan-amendment-3.json": "24182d648469bddc14dda1b20c47df58d3f6a476755ea00071e1cb9ca517ac0b",
    "plan-amendment-4.json": "a4eb5480444c0c238b0a18b18d4933bdcf55d0bd508ab2b00ff695995bd11004",
    "plan-amendment-5.json": "61ea57369b117a0ebec1eb33b55501b50e60e24d60bda399b285fccfa3ebb1af",
    "plan-amendment-6.json": "b87d791869ccf53b0d16f23e287ad943b59858db520623ef1fa797a76c0db6ad",
    "plan-amendment-7.json": "3836d574db6f78f88b234a9781480bf01a3cf68668ef65cd5a92aba8cd34b8ed",
    "plan-amendment-8.json": "08733f9da1cb34431d5ee96a2c5f421a644446f5962421a1d43dc3646d2f4680",
    "plan-amendment-9.json": "5dccd3f99cfe18e52ab8c1474d0f4f7c003f651c34de49d168e201c423eae628",
}
AUDIT = f"{PLAN_FOLDER}/audit/audit.json"  # the corrected audit (7cdcba40…)
# The rules' numbers. Fractions are compared exactly (no float edge cases). The three of the final rule (amendment 8) are the defaults of `evaluate select`'s flags --match-floor, --precision-floors and --tie-matches; the superseded rules' numbers below are fixed, since they only reproduce decisions the results report for information.
FINAL_MATCH_FLOOR = 292  # amendment 8: >= 292 of 365 TEST matches (80% recall)
FINAL_PRECISION_FLOORS = {"P>=20": 20, "P>=24": 24, "P>=28": 28}
FINAL_TIE_MATCHES = 2
SINGLE_MODEL_OUTPUT = "P>=28-single-model"  # amendment 9
TEST_MATCH_FLOOR = 290  # amendment 5 (superseded): Opus's 290 TEST matches
RECALL_FLOOR, PRECISION_FLOOR = (80, 100), (15, 100)  # amendment 1 (superseded), on DEV
CALL_TIE = (
    110,
    100,
)  # amendments 1 and 5: within 10% of the fewest calls counts as tied
RECALL_LEVELS = (50, 55, 60, 65, 70, 75, 80, 85, 90)  # the cheapest-setup table
# Opus's reference row Rdb54680 (amendment 1, acceptance_on_test). Its TEST calls come with the scores (`opus_test_calls`, from the TEST evaluation's results).
OPUS: dict[str, Any] = {
    "row": "Rdb54680",
    "test": {"matches": 290, "predicted": 1002},
    "dev": {"matches": 100, "predicted": 521, "model_calls": 270},
}
# How each stage was made: (verifier, relevance filter), and the filters' frozen settings.
STAGE_RECIPES = {
    "combined": (None, None),
    "verified-self": ("self", None),
    "filtered-title": ("self", "title"),
    "filtered-ordinary": ("self", "ordinary"),
    "filtered-title-noverify": (None, "title"),
    "filtered-ordinary-noverify": (None, "ordinary"),
    "verified-qwen": ("qwen", None),
    "filtered-title-qv": ("qwen", "title"),
    "filtered-ordinary-qv": ("qwen", "ordinary"),
}
FILTERS = {
    "title": {
        "filter_row": "R0bea6bc",
        "classifier": "semantic-title-relevance-v2",
        "cutoff": 0.2371840437933272,
    },
    "ordinary": {
        "filter_row": "R6e24b3d",
        "classifier": "semantic-relevance-v1",
        "cutoff": 0.3087965279332379,
    },
}
MODEL_IDS = {  # as the stored results name them: repository@revision
    "qwen": "@".join(models.PINNED["qwen"]),
    "gemma12b": "@".join(models.PINNED["gemma"]),
}
POOL_MERGE = (
    "per paper: concatenate the parts' stage records in the listed order, then "
    "mbdef.extract.records.merge"
)
# A Qwen cohort's own verifier is the common Qwen verifier (the same-setup table).
QWEN_EQUIVALENT = {
    "verified-qwen": "verified-self",
    "filtered-title-qv": "filtered-title",
    "filtered-ordinary-qv": "filtered-ordinary",
}
# The single-model variant of a setup verified by the common Qwen verifier (amendment 9).
WITHOUT_QWEN_VERIFIER = {
    "verified-qwen": "combined",
    "filtered-title-qv": "filtered-title-noverify",
    "filtered-ordinary-qv": "filtered-ordinary-noverify",
}
OK_VERDICTS = {"ok", "pass", "passed", "clean"}
# Outputs, in --out, in the format of the stored results.
RESULTS = "results.json"
RESULTS_MD = "results.md"
REPLACED = (
    "replaced"  # with --replaces: the replaced results' audit applied to these scores
)
RESULTS_SCHEMA = "grid-20260925-results/5"
JSON_FORMAT: dict[str, Any] = {"indent": 2, "ensure_ascii": False}
# The released results' top-level keys, in their order.
RESULTS_KEYS = (
    "schema",
    "created_utc",
    "plan_sha256",
    "amendment_sha256",
    "amendment2_sha256",
    "amendment3_sha256",
    "amendment4_sha256",
    "reporting_text",
    "pick_candidates_text",
    "pick_rule_text",
    "original_pick_rule_text",
    "acceptance_text",
    "quality_gate_text",
    "equivalence_note",
    "failure_policy_text",
    "pooling_text",
    "grid_root",
    "audit",
    "references",
    "cohorts",
    "missing_cohorts",
    "missing_trained_cohorts",
    "missing_stages",
    "cohort_scores",
    "configs",
    "workers",
    "final_pick_rule_text",
    "optimism_note",
    "amendment5_sha256",
    "amendment6_sha256_withdrawn",
    "amendment7_sha256",
    "amendment8_sha256",
    "final_picks_rules",
    "environment",
    "trained_roots",
    "trained_root_conflicts",
    "trained_arms_found",
    "dev_rule_decision",
    "decision",
    "final_picks",
    "pick_test",
    "pick_dev",
    "acceptance",
    "opus_reference",
    "cheapest",
    "comparison",
    "reasoning_comparison",
    "replicate",
    "reselect",
)


def now() -> str:
    return datetime.now(UTC).isoformat()


def check_plan(folder: Path) -> dict[str, dict]:
    """The plan and its amendments, by file name, after checking every sha256 and that each amendment amends exactly the files before it (amendment 7 withdraws amendment 6)."""
    plan = {}
    for name, expected in PLAN_SHA256.items():
        path = folder / name
        if not path.is_file() or files.digest(path) != expected:
            raise SystemExit(
                f"{path} is not the frozen grid plan file (sha256 {expected[:8]}…)."
            )
        plan[name] = json.loads(path.read_text(encoding="utf-8"))
    names = list(PLAN_SHA256)
    if plan["plan-amendment-1.json"]["amends_plan_sha256"] != PLAN_SHA256["plan.json"]:
        raise SystemExit("Amendment 1 does not amend this plan.")
    for number in (2, 3, 4, 5, 6, 8, 9):  # amendment 7 only withdraws amendment 6
        name = f"plan-amendment-{number}.json"
        if plan[name]["amends"] != {n: PLAN_SHA256[n] for n in names[:number]}:
            raise SystemExit(f"{name} does not amend this plan.")
    withdrawn = {"plan-amendment-6.json": PLAN_SHA256["plan-amendment-6.json"]}
    if plan["plan-amendment-7.json"].get("withdraws") != withdrawn:
        raise SystemExit("Amendment 7 does not withdraw amendment 6.")
    return plan


def plan_fields(plan: dict[str, dict]) -> dict:
    """The plan's hashes and rule texts, as the results carry them."""
    p, a1, a2, a3 = (
        plan["plan.json"],
        plan["plan-amendment-1.json"],
        plan["plan-amendment-2.json"],
        plan["plan-amendment-3.json"],
    )
    a4, a5 = plan["plan-amendment-4.json"], plan["plan-amendment-5.json"]
    return {
        "plan_sha256": PLAN_SHA256["plan.json"],
        "amendment_sha256": PLAN_SHA256["plan-amendment-1.json"],
        "amendment2_sha256": PLAN_SHA256["plan-amendment-2.json"],
        "amendment3_sha256": PLAN_SHA256["plan-amendment-3.json"],
        "amendment4_sha256": PLAN_SHA256["plan-amendment-4.json"],
        "reporting_text": a4["reporting"],
        "pick_candidates_text": a3["pick_candidates"],
        "pick_rule_text": a1["pick_rule_replaced"],
        "original_pick_rule_text": p["pick_rule"],
        "acceptance_text": a1["acceptance_on_test"],
        "quality_gate_text": a2["quality_gate"],
        "equivalence_note": a2["equivalence_note"],
        "failure_policy_text": p["failure_policy"],
        "pooling_text": p["pooling"]
        + "; amendment 1: "
        + a1["added_pools"]
        + "; amendment 2: "
        + a2["cross_arm_scope"],
        "final_pick_rule_text": a5["final_pick_rule"],
        "optimism_note": a5["reporting_note"],
        "amendment5_sha256": PLAN_SHA256["plan-amendment-5.json"],
        "amendment6_sha256_withdrawn": PLAN_SHA256["plan-amendment-6.json"],
        "amendment7_sha256": PLAN_SHA256["plan-amendment-7.json"],
        "amendment8_sha256": PLAN_SHA256["plan-amendment-8.json"],
        "final_picks_rules": plan["plan-amendment-8.json"]["final_picks"],
    }


# 2. The audit


def load_audit(path: Path | None, name: str | None = None) -> dict:
    """The audit's verdict per cohort (ok, or a defect with reasons) and its eligibility table.

    The reader is tolerant: the verdicts may sit under "cohorts" or "verdicts", as a text or as {"verdict", "reasons"}; eligibility is cohorts[label].metrics.eligibility. `name` is how the outputs name the file."""
    if not path or not path.is_file():
        return {"present": False, "path": name, "verdicts": {}}
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = (
        raw.get("cohorts", raw.get("verdicts", raw)) if isinstance(raw, dict) else raw
    )
    verdicts, eligibility = {}, {}
    items = (
        entries.items()
        if isinstance(entries, dict)
        else (
            (e.get("cohort") or e.get("label") or e.get("id"), e) for e in entries or []
        )
    )
    for label, value in items:
        if not isinstance(label, str) or label in (
            "schema",
            "created_utc",
            "status",
            "summary",
        ):
            continue
        if isinstance(value, str):
            verdict, reasons = value, []
        elif isinstance(value, dict):
            verdict = value.get("verdict") or value.get("status") or value.get("result")
            reasons = (
                value.get("reasons") or value.get("defects") or value.get("notes") or []
            )
            if isinstance(reasons, str):
                reasons = [reasons]
        else:
            continue
        verdicts[label.strip("/")] = {
            "verdict": str(verdict).lower() if verdict is not None else None,
            "reasons": list(reasons),
        }
        e = (
            (value.get("metrics") or {}).get("eligibility")
            if isinstance(value, dict)
            else None
        )
        if isinstance(e, dict):
            eligibility[label.strip("/")] = e
    meta = (
        {k: v for k, v in raw.items() if not isinstance(v, (dict, list))}
        if isinstance(raw, dict)
        else {}
    )
    return {
        "present": True,
        "path": name or str(path),
        "sha256": files.digest(path),
        "verdicts": verdicts,
        "eligibility": eligibility,
        "meta": meta,
    }


def merge_audits(main: dict, extra_paths: list[Path]) -> dict:
    """Add the verdicts and eligibility of further audit files (for example one that covers the training run's cohorts); the main audit's entries win. Uncovered cohorts fail."""
    merged: dict[str, Any] = {
        **main,
        "verdicts": dict(main.get("verdicts", {})),
        "eligibility": dict(main.get("eligibility", {})),
        "extra": [],
    }
    for path in extra_paths:
        extra = load_audit(path)
        if not extra["present"]:
            continue
        merged["extra"].append(
            {
                "path": extra["path"],
                "sha256": extra["sha256"],
                "cohorts": len(extra["verdicts"]),
            }
        )
        for key in ("verdicts", "eligibility"):
            for label, value in extra[key].items():
                merged[key].setdefault(label, value)
    return merged


def apply_audit_eligibility(entry: dict, audit: dict) -> None:
    """The audit's eligibility of a cohort (eligible_qv for a Gemma setup verified by Qwen, eligible_self otherwise) replaces the one from its status; a pool is eligible only if every part is. Parts without an audit entry keep the status check. Recomputed from the status every time, never sticky."""
    table = audit.get("eligibility") or {}
    labels = entry.get("cohort_labels", [])
    if "status_eligible" not in entry:
        entry["status_eligible"] = entry["eligible"]
        entry["status_ineligible_reasons"] = list(entry["ineligible_reasons"])
    entry["eligible"] = entry["status_eligible"]
    entry["ineligible_reasons"] = list(entry["status_ineligible_reasons"])
    key = (
        "eligible_qv"
        if VERIFIER.get(entry["stage"]) == "qwen" and entry["model"] == "gemma12b"
        else "eligible_self"
    )
    decided = {
        label: table[label].get(key)
        for label in labels
        if isinstance(table.get(label, {}).get(key), bool)
    }
    entry["audit_eligibility"] = {"field": key, "parts": decided}
    if not decided:
        return
    reasons = []
    for label in labels:
        if label in decided:
            if not decided[label]:
                e = table[label]
                reasons.append(
                    f"{label}: audit {key}=false (failed windows "
                    f"{e.get('failed_window_fraction')}, failed batches "
                    f"{e.get('failed_batch_fraction_self', e.get('failed_batch_fraction'))}, "
                    f"papers with output {e.get('papers_with_output')})"
                )
        else:
            reasons.extend(entry.get("part_status_reasons", {}).get(label, []))
    entry["ineligible_reasons"] = reasons
    entry["eligible"] = not reasons


def audit_gate(entry: dict, audit: dict) -> list[str]:
    """Why the audit excludes a setup (empty: it passes): a cohort the audit does not cover, or one whose verdict is not ok (for a Gemma setup verified by Qwen, also the verdict of its Qwen verification, `<label>+qv` or `<label>/verify-qwen`)."""
    if not audit["present"]:
        return ["audit.json missing"]
    reasons = []
    qv = VERIFIER.get(entry["stage"]) == "qwen" and entry["model"] == "gemma12b"
    for label in entry.get("cohort_labels", []):
        keys = [label] + ([f"{label}+qv", f"{label}/verify-qwen"] if qv else [])
        found = [k for k in keys if k in audit["verdicts"]]
        if label not in audit["verdicts"]:
            reasons.append(f"{label}: not covered by the audit")
        for k in found:
            v = audit["verdicts"][k]
            if v["verdict"] not in OK_VERDICTS:
                reasons.append(
                    f"{k}: {v['verdict'] or 'no verdict'}"
                    + (
                        f" ({'; '.join(map(str, v['reasons']))})"
                        if v["reasons"]
                        else ""
                    )
                )
    return reasons


# 3. The pick rules


def precision_of(c) -> Fraction:
    return Fraction(c["matches"], c["predicted"]) if c["predicted"] else Fraction(0)


def calls_of(c):
    return c["model_calls"] if c.get("model_calls") is not None else float("inf")


def apply_floor_pick(
    configs: list[dict],
    floor: int,
    min_matches: int = FINAL_MATCH_FLOOR,
    tie_matches: int = FINAL_TIE_MATCHES,
) -> dict:
    """Amendment 8: among eligible TEST setups with at least `min_matches` matches and precision >= floor%, the most matches; setups within `tie_matches` (2) matches of the most are tied, and the tie goes to higher precision, then fewer LLM calls (then more matches, then id)."""
    pool = [
        c
        for c in configs
        if c.get("eligible")
        and c["matches"] >= min_matches
        and ge(c["matches"], c["predicted"], (floor, 100))
    ]
    if not pool:
        return {
            "pick": None,
            "qualifying": 0,
            "reason": f"no eligible config with TEST matches >= {min_matches} and "
            f"precision >= {floor}%",
        }
    best = max(c["matches"] for c in pool)
    tied = [c for c in pool if c["matches"] >= best - tie_matches]
    pick = sorted(
        tied, key=lambda c: (-precision_of(c), calls_of(c), -c["matches"], c["id"])
    )[0]
    return {
        "pick": pick["id"],
        "qualifying": len(pool),
        "highest_matches": best,
        "tied": [c["id"] for c in tied],
        "reason": (
            f"{len(pool)} configs qualify; highest TEST matches {best}; {len(tied)} within "
            f"{tie_matches} matches; {pick['id']} has the highest precision among them "
            f"({pick['matches']}/{pick['predicted']}, {pick.get('model_calls')} calls)."
        ),
    }


def apply_pick_rule(configs: list[dict], gold: int) -> dict:
    """Amendment 1 (superseded, reported for information): among eligible setups with DEV recall >= 80% and DEV precision >= 15%, the fewest LLM calls; setups within 10% of the fewest are tied and the tie goes to higher DEV precision (then more matches, fewer calls, id). Otherwise the highest DEV recall with DEV precision >= 15%."""
    eligible = [c for c in configs if c.get("eligible")]
    floor = [
        c
        for c in eligible
        if ge(c["matches"], gold, RECALL_FLOOR)
        and ge(c["matches"], c["predicted"], PRECISION_FLOOR)
    ]
    decision: dict[str, Any] = {
        "eligible_dev_configs": len(eligible),
        "qualifying": [c["id"] for c in floor],
    }
    known = [c for c in floor if c.get("model_calls") is not None]
    decision["qualifying_without_call_counts"] = [
        c["id"] for c in floor if c.get("model_calls") is None
    ]
    if known:
        fewest = min(c["model_calls"] for c in known)
        tied = [
            c for c in known if c["model_calls"] * CALL_TIE[1] <= CALL_TIE[0] * fewest
        ]
        pick = sorted(
            tied,
            key=lambda c: (-precision_of(c), -c["matches"], c["model_calls"], c["id"]),
        )[0]
        limit = fewest * CALL_TIE[0] / CALL_TIE[1]
        decision.update(
            branch="DEV recall >= 80% and precision >= 15% available",
            fewest_calls=fewest,
            tie_limit_calls=limit,
            tied=[c["id"] for c in tied],
        )
        why = (
            f"{len(floor)} eligible configs reach DEV recall >= 80% with precision >= 15%. "
            f"The fewest LLM calls is {fewest}; {len(tied)} config(s) are within 10% of that "
            f"({limit:.1f} calls). Among them {pick['id']} has the highest DEV precision "
            f"({pick['matches']}/{pick['predicted']} = {100 * pick['precision']:.1f}%, recall "
            f"{100 * pick['recall']:.1f}%, {pick['model_calls']} calls)."
        )
    elif floor:
        pick, why = (
            None,
            "Configs qualify but none reports LLM call counts; the rule cannot be applied.",
        )
        decision["branch"] = "call counts missing"
    else:
        pool = [
            c for c in eligible if ge(c["matches"], c["predicted"], PRECISION_FLOOR)
        ]
        decision.update(
            branch="no eligible DEV config reaches 80% recall with precision >= 15%",
            dev_configs_with_precision_at_least_15=[c["id"] for c in pool],
        )
        ranked = sorted(
            pool, key=lambda c: (-c["matches"], -precision_of(c), calls_of(c), c["id"])
        )
        pick = ranked[0] if ranked else None
        why = (
            f"No eligible config reaches 80% DEV recall; {pick['id']} has the highest DEV "
            f"recall ({pick['matches']}/{gold}) among eligible configs with DEV precision "
            ">= 15%."
            if pick
            else "No eligible config reaches 80% DEV recall and none has DEV precision "
            ">= 15%; no pick."
        )
    decision["pick"] = pick["id"] if pick else None
    decision["reasoning"] = why
    return decision


def apply_test_pick_rule(configs: list[dict]) -> dict:
    """Amendment 5 (superseded by amendment 8, reported): among eligible setups with TEST matches >= 290, the fewest LLM calls; within 10% of the fewest are tied and the tie goes to higher TEST precision (then more matches, fewer calls, id). If none reaches 290, the highest TEST recall with TEST precision >= 15%."""
    eligible = [c for c in configs if c.get("eligible")]
    floor = [c for c in eligible if c["matches"] >= TEST_MATCH_FLOOR]
    known = [c for c in floor if c.get("model_calls") is not None]
    decision: dict[str, Any] = {
        "rule": "plan-amendment-5 final_pick_rule (TEST)",
        "eligible_test_configs": len(eligible),
        "qualifying": [c["id"] for c in floor],
        "qualifying_without_call_counts": [
            c["id"] for c in floor if c.get("model_calls") is None
        ],
    }
    if known:
        fewest = min(c["model_calls"] for c in known)
        tied = [
            c for c in known if c["model_calls"] * CALL_TIE[1] <= CALL_TIE[0] * fewest
        ]
        pick = sorted(
            tied,
            key=lambda c: (-precision_of(c), -c["matches"], c["model_calls"], c["id"]),
        )[0]
        decision.update(
            branch=f"TEST matches >= {TEST_MATCH_FLOOR} available",
            fewest_calls=fewest,
            tied=[c["id"] for c in tied],
        )
        why = (
            f"{len(floor)} eligible configs reach at least {TEST_MATCH_FLOOR}/365 TEST "
            f"matches (Opus level). The fewest LLM calls is {fewest}; {len(tied)} config(s) "
            f"are within 10% of that. Among them {pick['id']} has the highest TEST precision "
            f"({pick['matches']}/{pick['predicted']} = {100 * pick['precision']:.1f}%, recall "
            f"{100 * pick['recall']:.1f}%, {pick['model_calls']} calls)."
        )
    elif floor:
        pick, why = (
            None,
            "Configs reach 290 TEST matches but none reports LLM call counts; the rule "
            "cannot be applied.",
        )
        decision["branch"] = "call counts missing"
    else:
        pool = [
            c for c in eligible if ge(c["matches"], c["predicted"], PRECISION_FLOOR)
        ]
        decision.update(
            branch=f"no eligible config reaches {TEST_MATCH_FLOOR} TEST matches",
            test_configs_with_precision_at_least_15=[c["id"] for c in pool],
        )
        ranked = sorted(
            pool, key=lambda c: (-c["matches"], -precision_of(c), calls_of(c), c["id"])
        )
        pick = ranked[0] if ranked else None
        why = (
            f"No eligible config reaches {TEST_MATCH_FLOOR} TEST matches; {pick['id']} has "
            f"the highest TEST recall ({pick['matches']}/365) among eligible configs with "
            "TEST precision >= 15%."
            if pick
            else f"No eligible config reaches {TEST_MATCH_FLOOR} TEST matches and none has "
            "TEST precision >= 15%; no pick."
        )
    decision["pick"] = pick["id"] if pick else None
    decision["reasoning"] = why
    return decision


def pick_parts(entry: dict) -> dict:
    """How to rebuild a setup on new papers: its cohorts, its stage's recipe and its pool."""
    verifier, flt = STAGE_RECIPES[entry["stage"]]
    model = entry["model"]
    verifier_model = (
        None
        if verifier is None
        else (
            MODEL_IDS["qwen"]
            if verifier == "qwen" or model == "qwen"
            else MODEL_IDS[model]
        )
    )
    parts = []
    for label in entry.get("cohort_labels", []):
        _, m, arm, fmt = label.split("/")
        parts.append(
            {
                "model": m,
                "model_id": MODEL_IDS.get(m),
                "arm": arm,
                "trained_program": arm.startswith(TRAINED_PREFIX),
                "format": fmt,
                "cohort_path_template": f"<split>/{m}/{arm}/{fmt}",
                "stage_dir": f"stages/{entry['stage']}",
            }
        )
    arms = list(dict.fromkeys(p["arm"] for p in parts))
    formats = list(dict.fromkeys(p["format"] for p in parts))
    kind = (
        "single"
        if len(parts) == 1
        else "cross-arm pool"
        if len(arms) > 1
        else "format pool"
    )
    return {
        "parts": parts,
        "stage_recipe": {
            "stage": entry["stage"],
            "verifier": verifier,
            "verifier_model": verifier_model,
            "verifier_policy": None
            if verifier is None
            else "definitions-strict (common strict verifier, thinking off, temperature 0)",
            "filter": flt,
            "filter_spec": FILTERS.get(flt) if flt else None,
            "noverify": verifier is None and flt is not None,
            "order": "extraction -> verifier (if any) -> relevance filter (if any), per "
            "part, then pool",
        },
        "pool": {
            "kind": kind,
            "arms": arms,
            "formats": formats,
            "n_parts": len(parts),
            "merge": "as saved (single part)" if len(parts) == 1 else POOL_MERGE,
        },
    }


def final_pick_entry(pick_id: str, by_id: dict) -> dict:
    """A pick: its TEST and DEV numbers and its rebuild recipe."""
    keep = (
        "matches",
        "predicted",
        "precision",
        "recall",
        "f1",
        "full_record",
        "model_calls",
        "eligible",
        "passes_audit",
        "records_sha256",
    )
    t = by_id[pick_id]
    d = by_id.get("dev/" + pick_id.split("/", 1)[1])
    return {
        "pick": pick_id,
        "config": pick_id.split("/", 1)[1],
        "model": t["model"],
        "arms": t.get("arms"),
        "formats": t.get("formats"),
        "stage": t["stage"],
        "passes": t.get("passes"),
        "calls": t.get("model_calls"),
        "test": {k: t.get(k) for k in keep},
        "dev": ({k: d.get(k) for k in keep} | {"state": d.get("state")})
        if d
        else {"state": "missing"},
        **pick_parts(t),
    }


def single_model_variant(results: dict) -> dict | None:
    """Amendment 9's fourth output: the pick of the highest precision floor (P>=28) without the common Qwen verifier, that is the same setup with the stage made from the model's own output (no rule picks it). None if that pick uses no Qwen verifier."""
    picks = results["final_picks"]
    pick = picks[max(picks, key=lambda name: int(name.removeprefix("P>=")))].get("pick")
    if pick in (None, "none"):
        return None
    *setup, stage = pick.split("/")
    if stage not in WITHOUT_QWEN_VERIFIER:
        return None
    by_id = {c["id"]: c for c in results["configs"] + results["cohort_scores"]}
    variant = "/".join([*setup, WITHOUT_QWEN_VERIFIER[stage]])
    return final_pick_entry(variant, by_id) if variant in by_id else None


def acceptance(pick_test) -> dict:
    """Amendment 1: the pick 'matches or beats Opus' on TEST if it finds >= 290."""
    opus = OPUS["test"]
    definitions = SPLITS["test"]["definitions"]
    base = {
        "opus": f"{OPUS['row']} {opus['matches']}/{opus['predicted']} (precision "
        f"{100 * opus['matches'] / opus['predicted']:.1f}%, recall "
        f"{100 * opus['matches'] / definitions:.1f}%)"
    }
    if not pick_test or pick_test.get("state") != "scored":
        return {**base, "status": "not available", "matches_or_beats_opus": None}
    return {
        **base,
        "status": "checked",
        "matches_or_beats_opus": pick_test["matches"] >= opus["matches"],
        "test_matches": pick_test["matches"],
        "test_predicted": pick_test["predicted"],
        "test_precision": pick_test["precision"],
        "test_recall": pick_test["recall"],
        "opus_precision": opus["matches"] / opus["predicted"],
        "test_eligible": pick_test.get("eligible"),
    }


# The summary tables of the results


def cheapest_table(configs: list[dict], all_configs: list[dict]) -> list[dict]:
    """At each DEV recall level, the eligible DEV setup (precision >= 15%) with the fewest LLM calls, and the same setup on TEST (for information; TEST never chooses here)."""
    dev = [
        c
        for c in configs
        if c["split"] == "dev"
        and c.get("state") == "scored"
        and c.get("eligible")
        and c.get("model_calls") is not None
        and ge(c["matches"], c["predicted"], PRECISION_FLOOR)
    ]
    index = {c["id"]: c for c in all_configs}
    keep = (
        "id",
        "state",
        "matches",
        "predicted",
        "precision",
        "recall",
        "f1",
        "full_record",
        "eligible",
        "model_calls",
    )
    rows: list[dict[str, Any]] = []
    for level in RECALL_LEVELS:
        pool = [
            c
            for c in dev
            if ge(c["matches"], SPLITS["dev"]["definitions"], (level, 100))
        ]
        if not pool:
            rows.append({"dev_recall_at_least": level, "config": None})
            continue
        best = sorted(
            pool,
            key=lambda c: (c["model_calls"], -precision_of(c), -c["matches"], c["id"]),
        )[0]
        test = index.get("test/" + best["id"].split("/", 1)[1])
        rows.append(
            {
                "dev_recall_at_least": level,
                "config": best["id"],
                "dev": {k: best.get(k) for k in keep},
                "test": {k: test.get(k) for k in keep} if test else None,
            }
        )
    return rows


def comparison_table(configs: list[dict]) -> list[dict]:
    """The same setup with Qwen and with Gemma, on DEV and TEST."""
    keep = (
        "id",
        "state",
        "matches",
        "predicted",
        "precision",
        "recall",
        "f1",
        "full_record",
        "eligible",
        "model_calls",
    )
    index = {
        (c["split"], c["model"], c["arm"], c["passes"], c["stage"]): c for c in configs
    }
    rows = []
    for arm in GROUPS:
        for passes in PASSES if arm in ARMS else CROSS_ARM_PASSES:
            for stage in SELF_STAGES + NOVERIFY_STAGES + QV_STAGES:
                row: dict[str, Any] = {"arm": arm, "passes": passes, "stage": stage}
                for split in SPLITS:
                    q_stage = QWEN_EQUIVALENT.get(stage, stage)
                    row[f"qwen_{split}"] = index.get(
                        (split, "qwen", arm, passes, q_stage)
                    )
                    row[f"gemma12b_{split}"] = index.get(
                        (split, "gemma12b", arm, passes, stage)
                    )
                rows.append(
                    {
                        k: (
                            v
                            if not isinstance(v, dict)
                            else {x: v.get(x) for x in keep}
                        )
                        for k, v in row.items()
                    }
                )
    return rows


REASONING_PAIRS = (
    ("chat-default", "chat-default-think"),
    ("json-default", "json-default-think"),
    ("chat-default+json-default", "chat-default-think+json-default-think"),
)


def reasoning_table(entries: list[dict]) -> list[dict]:
    """Per arm and model: reasoning off against default-on (amendment 4), same format and stage."""
    index = {
        (c["split"], c["model"], c["arm"], c["passes"], c["stage"]): c for c in entries
    }
    keep = (
        "id",
        "state",
        "matches",
        "predicted",
        "precision",
        "recall",
        "eligible",
        "model_calls",
        "compute",
        "passes_audit",
    )
    rows = []
    for model in MODELS:
        verified = "filtered-title" if model == "qwen" else "filtered-title-qv"
        for arm in ARMS:
            for off, on in REASONING_PAIRS:
                for stage in ("combined", verified, "filtered-title-noverify"):
                    row: dict[str, Any] = {
                        "model": model,
                        "arm": arm,
                        "passes_off": off,
                        "passes_on": on,
                        "stage": stage,
                    }
                    for split in SPLITS:
                        for label, passes in (("off", off), ("on", on)):
                            c = index.get((split, model, arm, passes, stage))
                            row[f"{split}_{label}"] = (
                                {k: c.get(k) for k in keep} if c else None
                            )
                    rows.append(row)
    return rows


def finalize(
    out: dict,
    audit: dict,
    opus_test_calls=None,
    rule: dict | None = None,
) -> dict:
    """The audit, the pick rules and the summary tables, on scored setups (no scoring). RULE holds the final rule's numbers (`final_rule`; default: amendment 8's)."""
    rule = rule or final_rule(
        FINAL_MATCH_FLOOR, list(FINAL_PRECISION_FLOORS.values()), FINAL_TIE_MATCHES
    )
    out["audit"] = audit
    everything = (
        out["configs"] + out["cohort_scores"]
    )  # amendment 3: all are candidates
    for entry in everything:
        if entry.get("state") == "scored":
            apply_audit_eligibility(entry, audit)
    for entry in everything:
        if entry.get("state") == "scored":
            entry["audit_exclusions"] = audit_gate(entry, audit)
            entry["passes_audit"] = not entry["audit_exclusions"]
    gate = audit["present"]
    by_id = {c["id"]: c for c in everything}

    def candidates(split):
        return [
            c
            for c in everything
            if c["split"] == split
            and c.get("state") == "scored"
            and c["candidate"]
            and (c["passes_audit"] or not gate)
        ]

    def excluded(split):
        return [
            {"id": c["id"], "reasons": c["audit_exclusions"]}
            for c in everything
            if c["split"] == split
            and c.get("state") == "scored"
            and c["candidate"]
            and gate
            and not c["passes_audit"]
        ]

    dev_rule = apply_pick_rule(candidates("dev"), SPLITS["dev"]["definitions"])
    out["dev_rule_decision"] = {
        **dev_rule,
        "status": "superseded by plan-amendment-5 (final pick on TEST); for information only",
    }
    decision = apply_test_pick_rule(candidates("test"))
    decision["audit_excluded_test_configs"] = excluded("test")
    decision["audit_excluded_dev_configs"] = excluded("dev")
    decision["valid"] = gate
    if not gate:
        decision["provisional_pick"] = decision["pick"]
        decision["pick"] = None
        decision["reasoning"] = (
            "NO VALID PICK: the independent audit (audit.json) is missing. Provisional, "
            "audit-free result of the rule: "
            + (decision["provisional_pick"] or "none")
            + ". "
            + decision["reasoning"]
        )
    decision["status"] = (
        "superseded by plan-amendment-8 (three precision-floor picks); reported for "
        "information"
    )
    out["decision"] = decision
    out["final_picks"] = {}
    test_pool = candidates("test")
    for name, floor in rule["final_precision_floors"].items():
        chosen = apply_floor_pick(
            test_pool, floor, rule["final_match_floor"], rule["final_tie_matches"]
        )
        entry = (
            final_pick_entry(chosen["pick"], by_id)
            if chosen["pick"]
            else {"pick": "none"}
        )
        entry.update(
            rule=out["final_picks_rules"][name] if "final_picks_rules" in out else name,
            selection=chosen,
            valid=gate,
            provisional=not gate,
        )
        out["final_picks"][name] = entry
    shown = decision["pick"] or decision.get("provisional_pick")
    if shown:
        out["pick_test"] = {k: v for k, v in by_id[shown].items() if k != "per_paper"}
        out["pick_test"]["provisional"] = not gate
        dev = by_id.get("dev/" + shown.split("/", 1)[1])
        out["pick_dev"] = (
            {k: v for k, v in dev.items() if k != "per_paper"}
            if dev
            else {"id": "dev/" + shown.split("/", 1)[1], "state": "missing"}
        )
    else:
        out["pick_test"] = out["pick_dev"] = None
    out["acceptance"] = acceptance(out["pick_test"])
    if not gate:
        out["acceptance"]["valid"] = False
    out["opus_reference"] = {
        **OPUS,
        "test": {**OPUS["test"], "model_calls": opus_test_calls},
    }
    out["cheapest"] = cheapest_table(
        [
            c
            for c in everything
            if c["candidate"] and (c.get("passes_audit") or not gate)
        ],
        everything,
    )
    out["comparison"] = comparison_table(out["configs"])
    out["reasoning_comparison"] = reasoning_table(everything)
    return out


def in_released_order(results: dict) -> dict:
    """The results with their top-level keys in the released order."""
    unknown = [k for k in results if k not in RESULTS_KEYS]
    if unknown:
        raise ValueError(f"Unexpected keys in the results: {unknown}")
    return {k: results[k] for k in RESULTS_KEYS if k in results}


def pick_summary(results: dict) -> dict:
    return {
        "final_picks": {
            k: v.get("pick") for k, v in (results.get("final_picks") or {}).items()
        },
        "superseded_amendment5_pick": (results.get("decision") or {}).get("pick"),
        "superseded_dev_rule_pick": (results.get("dev_rule_decision") or {}).get(
            "pick"
        ),
    }


# 4. Outputs


def cell(c) -> str:
    if c is None:
        return "not run"
    if c.get("state") == "missing":
        return "missing"
    if c.get("state") == "error":
        return "error"
    mark = "" if c.get("eligible") else "*"
    return (
        f"{c['matches']}/{c['predicted']} ({100 * c['precision']:.1f}% / "
        f"{100 * c['recall']:.1f}%){mark}"
    )


def render_markdown(r: dict) -> str:
    """results.md, in the format of the stored results."""
    refs = r["references"]
    audit: dict[str, Any] = r.get("audit") or {"present": False}
    lines: list[str] = []
    if not audit.get("present"):
        lines += [
            "# AUDIT MISSING: NO PICK IS VALID",
            "",
            "## The independent audit file (audit.json) was not found. The scores below are "
            "published without the required quality check, and no configuration may be picked "
            "or claimed until the audit has been run and this report regenerated.",
            "",
        ]
    fp = r.get("final_picks") or {}
    if fp:
        lines += [
            "# Final picks (plan amendment 8): three TEST precision floors",
            "",
            "**These picks were chosen on TEST, so their TEST numbers are optimistic estimates, "
            "not held-out results.** DEV numbers of the same configs are shown beside them.",
            "",
            "| Floor | Pick | TEST matches/predicted | TEST precision | TEST recall | DEV "
            "matches/predicted | DEV precision | DEV recall | LLM calls (TEST) |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for name, e in fp.items():
            if e.get("pick") in (None, "none"):
                lines.append(f"| {name} | none | | | | | | | |")
                continue
            t, dv = e["test"], e["dev"]
            dtxt = (
                f"{dv['matches']}/{dv['predicted']} | {100 * dv['precision']:.1f}% | "
                f"{100 * dv['recall']:.1f}%"
                if dv.get("matches") is not None
                else "missing | | "
            )
            provisional = (
                " (provisional: audit missing)" if e.get("provisional") else ""
            )
            lines.append(
                f"| {name} | {e['config']}{provisional} | {t['matches']}/{t['predicted']} | "
                f"{100 * t['precision']:.1f}% | {100 * t['recall']:.1f}% | {dtxt} | "
                f"{e.get('calls')} |"
            )
        lines += [
            "",
            "Each pick's exact rebuild recipe (cohort parts, stage, pool) is in results.json "
            "under final_picks.",
            "",
        ]
    lines += [
        "# Controlled model grid: Qwen 3.6 35B vs Gemma 4 12B",
        "",
        f"Created {r['created_utc']}. Frozen plan SHA-256 {r['plan_sha256'][:12]}.",
        "",
        "A predicted definition counts as a match when the frozen definition matcher pairs it "
        "one-to-one with a reference definition of the same paper at similarity 0.50 or more. "
        "Precision = matches / predicted definitions; recall = matches / all reference "
        "definitions of the split.",
        "",
    ]
    for split, e in refs.items():
        lines.append(
            f"- {split.upper()} reference: "
            + (
                f"{e['papers']} papers, {e['definitions']} definitions"
                if "papers" in e
                else f"not usable ({e['error']})"
            )
        )
    d = r["decision"]
    lines += [
        "",
        "## Superseded: fewest-calls TEST pick (plan amendment 5, replaced by amendment 8)",
        "",
        f"Rule: {r.get('final_pick_rule_text', '')}",
        "",
        "LLM calls = extraction requests (including every automatic format-fallback request) "
        "+ requests of the verifier the stage uses.",
        "",
        f"**Pick: {d['pick'] or 'none'}.** {d['reasoning']}",
        "",
        "**Because TEST was used to choose, the pick's TEST result is an optimistic estimate, "
        "not a held-out result.** Its DEV result is shown beside it.",
        "",
    ]
    pdv = r.get("pick_dev")
    if pdv:
        calls = (
            f", {pdv.get('model_calls')} LLM calls"
            if pdv.get("state") == "scored"
            else ""
        )
        lines += [f"DEV result of the same config: {cell(pdv)}{calls}", ""]
    dr = r.get("dev_rule_decision") or {}
    if dr:
        lines += [
            "For information only (superseded DEV-only rule of plan amendment 1): "
            f"{dr.get('pick') or 'none'}.",
            "",
        ]
    lines += [f"Candidates (plan amendment 3): {r.get('pick_candidates_text', '')}", ""]
    if r.get("pick_test"):
        t = r["pick_test"]
        prefix = (
            "PROVISIONAL (audit missing, not a valid pick): "
            if t.get("provisional")
            else ""
        )
        extra = (
            f", full-record {t['full_record']:.3f}, {t.get('model_calls')} LLM calls"
            if t.get("state") == "scored"
            else ""
        )
        lines += [
            f"{prefix}TEST result of the pick (optimistic, used to choose): {cell(t)}{extra}",
            "",
        ]
    a = r.get("acceptance") or {}
    if a.get("status") == "checked":
        prefix = "PROVISIONAL (audit missing): " if a.get("valid") is False else ""
        verdict = "matches or beats" if a["matches_or_beats_opus"] else "does not match"
        note = (
            ""
            if a.get("test_eligible")
            else " Note: the pick's TEST run is not eligible under the failure policy."
        )
        lines += [
            f"{prefix}TEST check against Opus ({a['opus']}): the pick {verdict} Opus with "
            f"{a['test_matches']} TEST matches (needs >= {OPUS['test']['matches']}); its "
            f"TEST precision is {100 * a['test_precision']:.1f}% against Opus "
            f"{100 * a['opus_precision']:.1f}%.{note}",
            "",
        ]
    elif a:
        lines += [f"TEST check against Opus ({a['opus']}): not available.", ""]
    lines += ["## Quality audit", ""]
    if audit.get("present"):
        verdicts = audit.get("verdicts", {})
        bad = {k: v for k, v in verdicts.items() if v["verdict"] not in OK_VERDICTS}
        lines += [
            f"Audit file {audit['path']} (SHA-256 {audit['sha256'][:12]}): {len(verdicts)} "
            f"cohort verdicts, {len(bad)} defect(s).",
            "",
        ]
        for k, v in sorted(bad.items()):
            lines.append(
                f"- {k}: {v['verdict']}"
                + (f" - {'; '.join(map(str, v['reasons']))}" if v["reasons"] else "")
            )
        excluded = (d.get("audit_excluded_test_configs") or []) + (
            d.get("audit_excluded_dev_configs") or []
        )
        lines += [
            "",
            f"Configs excluded from the pick by the audit (TEST and DEV): {len(excluded)}.",
        ]
        for x in excluded:
            lines.append(f"- {x['id']}: {'; '.join(x['reasons'])}")
        lines.append("")
    else:
        lines += [
            "**Audit missing. No pick is valid.** The pick shown above is provisional and was "
            "computed without the quality gate.",
            "",
        ]
    lines += [
        "## Cheapest configs by DEV recall level",
        "",
        "For each DEV recall level: the eligible config (DEV precision >= 15%) with the fewest "
        "LLM calls on DEV, and the same config on TEST. TEST is shown for information and "
        "never used to choose.",
        "",
        "| DEV recall at least | Config | DEV matches/predicted | DEV precision | DEV recall | "
        "DEV LLM calls | TEST matches/predicted | TEST precision | TEST recall | TEST LLM "
        "calls |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    def pr(c, key):
        return (
            f"{100 * c[key]:.1f}%"
            if c and c.get("state") == "scored" and c.get(key) is not None
            else ""
        )

    def mp(c):
        return (
            f"{c['matches']}/{c['predicted']}"
            if c and c.get("state") == "scored"
            else ("missing" if c else "not run")
        )

    for row in r.get("cheapest", []):
        if not row["config"]:
            lines.append(f"| {row['dev_recall_at_least']}% | none | | | | | | | | |")
            continue
        dv, ts = row["dev"], row["test"]
        ts_calls = (
            (ts or {}).get("model_calls") if ts and ts.get("state") == "scored" else ""
        )
        lines.append(
            f"| {row['dev_recall_at_least']}% | {row['config'].split('/', 1)[1]} | {mp(dv)} | "
            f"{pr(dv, 'precision')} | {pr(dv, 'recall')} | {dv.get('model_calls')} | "
            f"{mp(ts)} | {pr(ts, 'precision')} | {pr(ts, 'recall')} | {ts_calls} |"
        )
    o = r.get("opus_reference") or OPUS
    od, ot = o["dev"], o["test"]
    dev_n, test_n = SPLITS["dev"]["definitions"], SPLITS["test"]["definitions"]
    lines.append(
        f"| reference | Opus {o['row']} (paid API) | {od['matches']}/{od['predicted']} | "
        f"{100 * od['matches'] / od['predicted']:.1f}% | "
        f"{100 * od['matches'] / dev_n:.1f}% | {od['model_calls']} | "
        f"{ot['matches']}/{ot['predicted']} | {100 * ot['matches'] / ot['predicted']:.1f}% | "
        f"{100 * ot['matches'] / test_n:.1f}% | {ot.get('model_calls') or ''} |"
    )
    lines.append("")
    lines += [
        "## Reasoning off vs default-on (same model, arm, format and stage)",
        "",
        "Off = thinking disabled, temperature 0. Default-on = the model's own thinking and "
        "shipped sampling settings. The Qwen-verified stage uses the common Qwen verifier "
        "(thinking off) for both models. Output tokens include reasoning tokens. Compute is "
        "reported only; it is not used to choose.",
        "",
        "| Model | Arm | Formats (off / on) | Stage | DEV off | DEV on | TEST off | TEST on | "
        "DEV LLM calls off / on | DEV output tokens off / on | DEV reasoning tokens on | DEV "
        "generation minutes off / on |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def comp(c, key, scale=1.0, fmt="{:.0f}"):
        v = ((c or {}).get("compute") or {}).get(key)
        return fmt.format(v / scale) if v is not None else ""

    def calls(c):
        return "" if not c or c.get("model_calls") is None else str(c["model_calls"])

    for row in r.get("reasoning_comparison", []):
        do, dn, to, tn = row["dev_off"], row["dev_on"], row["test_off"], row["test_on"]
        lines.append(
            f"| {row['model']} | {row['arm']} | {row['passes_off']} / {row['passes_on']} | "
            f"{row['stage']} | {cell(do)} | {cell(dn)} | {cell(to)} | {cell(tn)} | "
            f"{calls(do)} / {calls(dn)} | {comp(do, 'output_tokens')} / "
            f"{comp(dn, 'output_tokens')} | {comp(dn, 'reasoning_tokens')} | "
            f"{comp(do, 'wall_seconds', 60, '{:.1f}')} / "
            f"{comp(dn, 'wall_seconds', 60, '{:.1f}')} |"
        )
    lines += [
        "",
        "## Same setup, different model",
        "",
        "Cells: matches/predicted (precision / recall); * = not eligible under the failure "
        "policy. For Qwen, the rows 'checked by Qwen' show Qwen's own verifier, which is the "
        "common Qwen verifier.",
        "",
        "| Arm | Passes | Stage | Qwen DEV | Gemma DEV | Qwen TEST | Gemma TEST |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in r["comparison"]:
        lines.append(
            f"| {row['arm']} | {row['passes']} | {row['stage']} | {cell(row['qwen_dev'])} | "
            f"{cell(row['gemma12b_dev'])} | {cell(row['qwen_test'])} | "
            f"{cell(row['gemma12b_test'])} |"
        )
    lines += [
        "",
        "## Every config (DEV and TEST)",
        "",
        "| Model | Arm | Passes | Stage | DEV | DEV F1 | DEV full-record | TEST | TEST F1 | "
        "TEST full-record | Model calls (DEV / TEST) | Output tokens (DEV / TEST) | Reasoning "
        "tokens (DEV / TEST) | Generation minutes (DEV / TEST) | Eligible (DEV / TEST) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    index = {
        (c["split"], c["model"], c["arm"], c["passes"], c["stage"]): c
        for c in r["configs"]
    }
    trained_groups = sorted(
        {
            c["arm"]
            for c in r["configs"] + r["cohort_scores"]
            if c.get("trained") or str(c.get("arm", "")).startswith(TRAINED_PREFIX)
        }
    )
    index.update(
        {
            (c["split"], c["model"], c["arm"], c["passes"], c["stage"]): c
            for c in r["cohort_scores"]
            if str(c.get("arm", "")).startswith(TRAINED_PREFIX)
        }
    )

    def f(c, key, fmt):
        return (
            fmt.format(c[key])
            if c and c.get("state") == "scored" and c.get(key) is not None
            else ""
        )

    def e(c):
        return (
            ""
            if not c or c.get("state") != "scored"
            else ("yes" if c["eligible"] else "no")
        )

    def k(c, key, scale=1.0, fmt="{:.0f}"):
        v = (
            ((c or {}).get("compute") or {}).get(key)
            if c and c.get("state") == "scored"
            else None
        )
        return fmt.format(v / scale) if v is not None else ""

    for model in MODELS:
        for arm in GROUPS + tuple(trained_groups):
            single = arm in ARMS or (arm.startswith(TRAINED_PREFIX) and "+" not in arm)
            for passes in PASSES if single else CROSS_ARM_PASSES:
                for stage in STAGES[model]:
                    dv = index.get(("dev", model, arm, passes, stage))
                    ts = index.get(("test", model, arm, passes, stage))
                    if arm.startswith(TRAINED_PREFIX) and dv is None and ts is None:
                        continue
                    lines.append(
                        f"| {model} | {arm} | {passes} | {stage} | {cell(dv)} | "
                        f"{f(dv, 'f1', '{:.3f}')} | {f(dv, 'full_record', '{:.3f}')} | "
                        f"{cell(ts)} | {f(ts, 'f1', '{:.3f}')} | "
                        f"{f(ts, 'full_record', '{:.3f}')} | {f(dv, 'model_calls', '{}')} / "
                        f"{f(ts, 'model_calls', '{}')} | {k(dv, 'output_tokens')} / "
                        f"{k(ts, 'output_tokens')} | {k(dv, 'reasoning_tokens')} / "
                        f"{k(ts, 'reasoning_tokens')} | "
                        f"{k(dv, 'wall_seconds', 60, '{:.1f}')} / "
                        f"{k(ts, 'wall_seconds', 60, '{:.1f}')} | {e(dv)} / {e(ts)} |"
                    )
    lines += [
        "",
        "## Single-format cohorts",
        "",
        "| Split | Model | Arm | Format | Stage | Result | Model calls | Eligible |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in r["cohort_scores"]:
        state = "error" if c.get("error") else "scored"
        calls_text = c.get("model_calls") if c.get("model_calls") is not None else ""
        lines.append(
            f"| {c['split']} | {c['model']} | {c['arm']} | {c['format']} | {c['stage']} | "
            f"{cell({**c, 'state': state})} | {calls_text} | "
            f"{'yes' if c.get('eligible') else 'no'} |"
        )
    if r.get("replicate"):
        lines += ["", "## Run-to-run spread", "", r["replicate"]["text"]]
    ineligible = [
        c for c in r["configs"] if c.get("state") == "scored" and not c["eligible"]
    ]
    if ineligible:
        lines += ["", "## Why configs are not eligible", ""] + [
            f"- {c['id']}: {'; '.join(c['ineligible_reasons'])}" for c in ineligible
        ]
    missing_trained = r.get("missing_trained_cohorts", [])
    lines += [
        "",
        f"Trained-program cohorts not (yet) present ({len(missing_trained)}): "
        + (", ".join(missing_trained) or "none"),
    ]
    lines += [
        "",
        f"Missing cohorts ({len(r['missing_cohorts'])}): "
        + (", ".join(r["missing_cohorts"]) or "none"),
        "",
        f"Missing stages in present cohorts ({len(r['missing_stages'])}): "
        + (", ".join(r["missing_stages"]) or "none"),
    ]
    return "\n".join(lines) + "\n"


def reselect_note(results: dict) -> str:
    """The line a reselection puts above results.md."""
    s = results["reselect"]
    return (
        f"> Reselection {s['created_utc']}: eligibility and picks recomputed from the stored "
        f"scores of {s['source_results']} (no rescoring) with audit SHA-256 "
        f"{s['audit_sha256']} (previous {s['previous_audit_sha256']}). Reason: "
        f"{s['reason']}\n\n"
    )


def write_results(results: dict, folder: Path) -> dict[str, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / RESULTS).write_text(
        files.json_text(results, **JSON_FORMAT), encoding="utf-8"
    )
    note = reselect_note(results) if "reselect" in results else ""
    (folder / RESULTS_MD).write_text(note + render_markdown(results), encoding="utf-8")
    return {RESULTS: folder / RESULTS, RESULTS_MD: folder / RESULTS_MD}


def pick_line(name: str, entry: dict | None) -> str:
    if not entry or entry.get("pick") in (None, "none"):
        return f"{name}: none"
    t, d = entry["test"], entry["dev"]
    dev = (
        f"{d['matches']}/{d['predicted']}"
        if d.get("matches") is not None
        else "missing"
    )
    return (
        f"{name}: {entry['config']}: TEST {t['matches']}/{t['predicted']} "
        f"({100 * t['recall']:.1f}% recall, {100 * t['precision']:.1f}% precision), DEV {dev}"
    )


# 5. The command


def named(given: str | None, under_data: str, data: Path) -> tuple[Path, str]:
    """An input's path, and how the outputs name it: as given, or `<data>/<path>`."""
    if given:
        return Path(given), given
    return data / under_data, str(data / under_data)


def final_rule(match_floor: int, precision_floors: list[int], tie_matches: int) -> dict:
    """The final pick rule's numbers as the receipt names them: {"P>=20": 20, …} for the floors."""
    return {
        "final_match_floor": match_floor,
        "final_precision_floors": {f"P>={f}": f for f in precision_floors},
        "final_tie_matches": tie_matches,
    }


def select_command(
    scores_arg: str,
    out: str,
    data: str,
    audit_arg: str | None,
    extra_audits: list[str] | None,
    replaces_arg: str | None,
    reason: str | None,
    match_floor: int = FINAL_MATCH_FLOOR,
    precision_floors: list[int] | None = None,
    tie_matches: int = FINAL_TIE_MATCHES,
) -> int:
    """Apply the audit and the pick rules to the scores; write results.json, results.md and a receipt into OUT (and, with --replaces, the replay of the replaced results into OUT/replaced/). The final rule's numbers default to amendment 8's."""
    rule = final_rule(
        match_floor,
        precision_floors or list(FINAL_PRECISION_FLOORS.values()),
        tie_matches,
    )
    target, data_dir = Path(out), Path(data)
    if any((target / name).exists() for name in (RESULTS, RESULTS_MD, REPLACED)):
        raise SystemExit(f"{out} already holds results; choose a new folder for --out.")
    if bool(replaces_arg) != bool(reason):
        raise SystemExit("--replaces and --reason go together.")
    plan = check_plan(data_dir / PLAN_FOLDER)
    scores_path = Path(scores_arg)
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    audit_path, audit_name = named(audit_arg, AUDIT, data_dir)
    if not audit_path.is_file():
        raise SystemExit(f"The audit {audit_name} does not exist.")
    extras = [Path(p) for p in extra_audits or []]
    audit = merge_audits(load_audit(audit_path, audit_name), extras)
    opus_calls = scores.pop(
        "opus_test_calls",
        (scores.get("opus_reference") or {}).get("test", {}).get("model_calls"),
    )
    scores.pop("reselect", None)
    base = {**scores, "schema": RESULTS_SCHEMA, **plan_fields(plan)}
    inputs: dict[str, Path] = {"scores": scores_path, "audit": audit_path}
    inputs.update({f"audit-{i + 1}": p for i, p in enumerate(extras)})
    inputs.update({name: data_dir / PLAN_FOLDER / name for name in PLAN_SHA256})
    outputs: dict[str, Path] = {}
    reselect = None
    if replaces_arg:
        replaced_path = Path(replaces_arg)
        old = json.loads(replaced_path.read_text(encoding="utf-8"))
        replay = in_released_order(
            finalize(copy.deepcopy(base), old["audit"], opus_calls, rule)
        )
        check = {
            "stored_audit_sha256": old["audit"].get("sha256"),
            "stored_picks": pick_summary(old),
            "replayed_picks": pick_summary(replay),
            "reproduced_exactly": pick_summary(replay) == pick_summary(old),
        }
        if not check["reproduced_exactly"]:
            print(json.dumps(check, indent=2))
            raise SystemExit(
                f"Refused: the audit of {replaces_arg}, applied to these scores, does not "
                "give its picks. Nothing was written."
            )
        written = write_results(replay, target / REPLACED)
        outputs.update({f"{REPLACED}/{k}": v for k, v in written.items()})
        inputs["replaces"] = replaced_path
        reselect = {
            "created_utc": now(),
            "reason": reason,
            "source_results": scores_arg,
            "source_results_sha256": files.digest(scores_path),
            "source_scored_utc": scores.get("created_utc"),
            "previous_audit_sha256": old["audit"].get("sha256"),
            "audit_path": audit_name,
            "audit_sha256": audit.get("sha256"),
            "extra_audits": audit.get("extra"),
            "scores_recomputed": False,
            "reproduction_check": check,
            "method": "mbdef evaluate select (evaluate/selection.py finalize) on "
            "the stored per-config scores; the same rule code as without --replaces",
        }
    results = finalize(copy.deepcopy(base), audit, opus_calls, rule)
    if reselect:
        results["reselect"] = reselect
    results = in_released_order(results)
    outputs.update(write_results(results, target))
    receipt = files.receipt(
        "evaluate select",
        settings={
            **rule,
            "plan_sha256": PLAN_SHA256,
            "reason": reason,
            "json_format": JSON_FORMAT,
        },
        inputs=inputs,
        outputs=outputs,
    )
    files.write_json(target / files.RECEIPT, receipt)
    for name in rule["final_precision_floors"]:
        print(pick_line(name, results["final_picks"][name]))
    print(
        pick_line(f"{SINGLE_MODEL_OUTPUT} (amendment 9)", single_model_variant(results))
    )
    print("wrote", out)
    return 0


# 6. The cluster job that rescored the grid (docs/step-3-extraction/selection.md, the cluster job): `evaluate score` on the scorings that `evaluate grid --write-scorings` wrote. With its pair scores, `evaluate grid --replay-scores` and `evaluate select` rerun everything else without the models.

# The cluster job's output under DATA_ROOT (data/data-root.tsv): scores.json, its receipt, the job's record and logs, and the pair scores, one file per split and model.
JOB = "selection/rescore/2026-10-20"
JOB_PAIR_SCORES = f"{JOB}/pair-scores"
# What the job runs, as one command line for the cluster launcher (cluster/launcher.sbatch runs it in bash with DATA_ROOT and the node's own TMPDIR set): unpack the scorings that `evaluate grid --write-scorings` wrote (69,736 small files, one list per split and model; a tar at $DATA_ROOT/<JOB>/scorings.tar.zst, packed with zstd's long window) onto the node's disk, score each list with the general scorer in its own process, four at once on the node's one GPU, recording every pair score, then write the hashes of the pair-score files, to compare after they are copied back. A process that fails fails the job.
JOB_TIME = "02:00:00"
JOB_STEP = (
    f"O=$DATA_ROOT/{JOB}; S=$TMPDIR/scorings; mkdir -p $O/pair-scores && "
    "tar -I 'zstd --long=31' -xf $O/scorings.tar.zst -C $TMPDIR && "
    f"printf '%s\\n' {' '.join(f.replace('/', '-') for f in FAMILIES)} | "
    "xargs -P 4 -I{} mbdef evaluate score --scorings $S/{}.json "
    "--record-scores $O/pair-scores/{}.json --out $O/scores/{} && "
    "(cd $O/pair-scores && sha256sum *.json > ../pair-scores.sha256)"
)
