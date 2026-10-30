"""A miniature extractor grid on disk, laid out as the grid and training runs wrote theirs.

`make_grid(folder)` writes a few cohorts (one model run with one prompt arm and one output format on one split) for two grid arms and one trained-prompt arm, with stage files for three papers of each split, status records, request records and usage records, and some defects: a cohort without status.complete, a paper whose extraction windows all failed, a stage file of a paper outside the split. Predictions copy part of the real reference definitions and add made-up ones, so a made-up pair scorer finds matches. `made_up_scorers()` is that scorer: deterministic word overlap, so the original implementation and the package can be given the same one.
"""

import hashlib
import json
from pathlib import Path
from urllib.parse import quote

REPO = Path(__file__).resolve().parents[1]
REFERENCES = {
    "dev": "data/frozen/results/qwen-replication-sweep-20260923/dev-reference.json",
    "test": "data/frozen/results/test-eval-20260924/inputs/test-reference.json",
}
READY = "data/frozen/results/test-eval-20260924/inputs/READY.json"
STAGES = ("combined", "verified-self", "filtered-title", "filtered-title-noverify")
QV = ("verified-qwen", "filtered-title-qv")
# (root, split, model, arm, format, status changes, extra stages)
COHORTS = [
    ("grid", "test", "qwen", "A-minimal-8k", "chat8192", {}, ()),
    ("grid", "test", "qwen", "A-minimal-8k", "schema8192", {}, ()),
    ("grid", "test", "qwen", "B-minimal-16k32k", "chat8192", {"complete": False}, ()),
    ("grid", "test", "gemma12b", "A-minimal-8k", "chat8192", {}, QV),
    ("grid", "dev", "qwen", "A-minimal-8k", "chat8192", {}, ()),
    ("grid", "dev", "gemma12b", "A-minimal-8k", "chat8192", {}, QV),
    ("trained", "test", "gemma12b", "TR-bootstrap-8k", "chat-default-think", {}, QV),
    (
        "trained",
        "test",
        "gemma12b",
        "TR-bootstrap-16k32k",
        "chat-default-think",
        {},
        QV,
    ),
    ("trained", "dev", "gemma12b", "TR-bootstrap-8k", "chat-default-think", {}, QV),
]


def reference(split: str) -> dict:
    return json.loads((REPO / REFERENCES[split]).read_text())


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n")


def predictions(papers: list[dict], salt: int, stage: int) -> dict[str, list]:
    """Records of three papers: every (salt+stage)-th reference definition kept, and one made-up record per paper."""
    out = {}
    for n, paper in enumerate(papers[:3]):
        records = []
        for record in paper["definition_records"]:
            kept = [
                d
                for i, d in enumerate(record["definitions"])
                if (i + salt + stage + n) % (2 + stage % 2) == 0
            ]
            if kept:
                records.append({**record, "definitions": kept})
        records.append(
            {
                "term": f"Made-up term {salt}",
                "definitions": [
                    {
                        "text": f"a made-up definition number {salt} of paper {n}",
                        "citation": [],
                    }
                ],
                "synonyms": [],
                "description": [],
                "related_terms": [],
                "subtypes": [],
                "supertypes": [],
                "examples": [],
            }
        )
        out[paper["paper_id"]] = records
    return out


def make_grid(folder: Path) -> dict[str, Path]:
    roots = {"grid": folder / "grid", "trained": folder / "trained"}
    for salt, (root, split, model, arm, fmt, changes, extra) in enumerate(COHORTS):
        cohort = roots[root] / split / model / arm / fmt
        papers = reference(split)["papers"]
        for stage_number, stage in enumerate(STAGES + extra):
            for pid, records in predictions(papers, salt, stage_number).items():
                write(
                    cohort / "stages" / stage / f"{quote(pid, safe='')}.json",
                    {"paper_id": pid, "definition_records": records},
                )
        status = {
            "complete": True,
            "papers": len(papers),
            "windows": 40,
            "failed_windows": 1,
            "batches": 20,
            "failed_batches": 0,
            "requests": {"extract": 40, "verifier": 20},
            **changes,
        }
        write(cohort / "status.json", status)
        for i in range(3 + salt % 2):
            response = {"choices": [{"message": {"content": "{}"}}]}
            if fmt.endswith("-think") and i == 0:
                response["choices"][0]["message"]["reasoning_content"] = "thinking"
            write(
                cohort / "run" / "requests-extract" / f"{i:06d}.json",
                {"response": response},
            )
        usage = [
            {
                "started": 100.0 + i,
                "seconds": 2.5,
                "completion_tokens": 50,
                "prompt_tokens": 9,
            }
            for i in range(3)
        ]
        (cohort / "run" / "usage-extract.jsonl").write_text(
            "".join(json.dumps(u) + "\n" for u in usage)
        )
        for n, paper in enumerate(papers[:3]):
            sha = hashlib.sha256(paper["paper_id"].encode()).hexdigest()
            extraction = cohort / "run" / "papers" / sha / "extraction"
            extraction.mkdir(parents=True, exist_ok=True)
            if not (salt == 1 and n == 2):  # one paper whose windows all failed
                (extraction / "0-00000.json").write_text("{}")
        if extra:
            write(
                cohort / "status-verify-qwen.json",
                {
                    "complete": True,
                    "batches": 10,
                    "failed_batches": 0,
                    "requests": {"verifier": 10},
                },
            )
    # A stage file of a paper outside the split.
    write(
        roots["grid"] / "test/qwen/A-minimal-8k/chat8192/stages/combined/stray.json",
        {"paper_id": "not-a-test-paper", "definition_records": []},
    )
    return roots


def audit_labels() -> list[str]:
    return sorted(
        {f"{split}/{model}/{arm}/{fmt}" for _, split, model, arm, fmt, *_ in COHORTS}
    )


def write_audit(path: Path) -> Path:
    cohorts = {}
    for label in audit_labels():
        entry: dict[str, object] = {"verdict": "ok", "reasons": []}
        if label == "test/qwen/A-minimal-8k/schema8192":
            entry = {"verdict": "defect", "reasons": ["made-up defect"]}
        if label == "test/gemma12b/A-minimal-8k/chat8192":
            entry["metrics"] = {
                "eligibility": {"eligible_self": True, "eligible_qv": False}
            }
        cohorts[label] = entry
    write(path, {"schema": "mini-audit", "cohorts": cohorts})
    return path


def words(text: str) -> set[str]:
    return set(text.casefold().split())


def overlap(a: str, b: str) -> float:
    left, right = words(a), words(b)
    return len(left & right) / len(left | right) if left | right else 1.0


def made_up_scorers():
    """(similarity, entailment): word overlap, and word overlap shaded by the direction."""

    def similarity(pairs):
        return [overlap(a, b) for a, b in pairs]

    def entailment(pairs):
        return [overlap(a, b) * (0.9 if len(a) > len(b) else 1.0) for a, b in pairs]

    return similarity, entailment
