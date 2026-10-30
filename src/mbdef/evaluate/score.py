"""The general scorer: how many reference definitions a set of predicted records finds.

A **scoring** is the predicted definition records of one setup on the papers of one reference (for example one TEST row, or one selection config on DEV). `score` scores it as the released TEST and selection runs did: per paper of the reference, in paper-id order,

- the one-to-one DefScore matches of its definitions at 0.5 (`metric.one_to_one`): tp, fp, fn;
- the full-record score (`metric.full_record`);

then tp, fp and fn are pooled over the papers: precision = tp / predicted definitions, recall = tp / all reference definitions (a paper without predictions keeps its reference definitions in the denominator), and the full-record score is the mean over the papers.

`mbdef evaluate score` (`score_command`) runs `score` over a list of scorings (`--scorings`), or over predictions files (`--predictions`, `records.read_predictions`) against one split of MBDef-Gold (`--split`: train, validation or test; `GOLD_SPLITS`), and fails if any of them fails; then it writes nothing.

Worked example: a predictions file with the records of 2 TEST papers, scored with `--split test`, is scored against all 29 TEST papers (365 definitions), so the other 27 papers count as found nothing; `--paper ID` (repeatable) makes the reference only those papers. A paper of the file outside the split is counted (`papers_outside_reference`) and not scored.
"""

import json
import time
from pathlib import Path
from typing import Any

from mbdef import files
from mbdef.evaluate import metric
from mbdef.extract.records import read_predictions

SCORES = "scores.json"
SCORES_SCHEMA = (
    "taxonomy_pipeline/scores-1"  # a recorded id: it keeps its name (files.py)
)


def as_int(value):
    """A count from a status file: an int (a bool or a whole float as one), a list's length, or None."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, (list, tuple)):
        return len(value)
    return None


def ge(numerator: int, denominator: int, fraction: tuple[int, int]) -> bool:
    """Exact rational comparison numerator/denominator >= a/b (no float edge cases)."""
    a, b = fraction
    return denominator > 0 and numerator * b >= a * denominator


def definition_count(records) -> int:
    """The number of definitions in a list of records; a malformed record is an error (with the original scorer's messages)."""
    if not isinstance(records, list):
        raise ValueError("Definition records must be a list")
    n = 0
    for record in records:
        if not isinstance(record, dict) or not isinstance(
            record.get("definitions"), list
        ):
            raise ValueError("Invalid definition record")
        if not isinstance(record.get("term"), str):
            raise ValueError("Invalid record term")
        for item in record["definitions"]:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("text"), str)
                or not item["text"].strip()
            ):
                raise ValueError("Invalid definition text")
            n += 1
    return n


def counted(pid: str, records) -> int:
    try:
        return definition_count(records)
    except ValueError as error:
        raise ValueError(f"{pid}: {error}") from None


def score(
    gold: dict[str, list[dict]],
    predicted: dict[str, list[dict]],
    pair_scorer: metric.PairScorer,
    threshold: float = metric.MATCH_THRESHOLD,
) -> dict:
    """Score predicted records ({paper id: records}) against reference records ({paper id:
    records}). Papers of `predicted` that are not in `gold` are not scored."""
    per_paper: list[dict[str, Any]] = []
    for pid in sorted(gold):
        left = gold[pid]
        right = predicted.get(pid, [])
        expected_gold, expected_predicted = counted(pid, left), counted(pid, right)
        binary = metric.one_to_one(left, right, pair_scorer, threshold)
        if (
            binary["gold"] != expected_gold
            or binary["predicted"] != expected_predicted
            or binary["tp"] + binary["fn"] != expected_gold
            or binary["tp"] + binary["fp"] != expected_predicted
        ):
            raise ValueError(f"DefScore one-to-one counts disagree: {pid}")
        full = metric.full_record(left, right, pair_scorer, threshold)
        per_paper.append(
            {
                "paper_id": pid,
                "has_output": pid in predicted,
                "binary_one_to_one": {
                    k: binary[k] for k in ("tp", "fp", "fn", "gold", "predicted")
                },
                "full_record": {
                    "score": full["score"],
                    "binary_macro_f1": full["binary_macro_f1"],
                    "fields": full["fields"],
                },
            }
        )
    counts = {
        k: sum(r["binary_one_to_one"][k] for r in per_paper)
        for k in ("tp", "fp", "fn", "gold", "predicted")
    }
    reference_total = sum(definition_count(records) for records in gold.values())
    if counts["gold"] != reference_total:
        raise ValueError("Reference definition count changed during scoring")
    precision = counts["tp"] / counts["predicted"] if counts["predicted"] else 0.0
    recall = counts["tp"] / reference_total if reference_total else 0.0
    return {
        "binary_one_to_one": {
            **counts,
            "precision": precision,
            "recall": recall,
            "f1": metric.f1(precision, recall),
        },
        "full_record_macro_score": sum(r["full_record"]["score"] for r in per_paper)
        / len(per_paper),
        "full_record_binary_macro_f1": sum(
            r["full_record"]["binary_macro_f1"] for r in per_paper
        )
        / len(per_paper),
        "per_paper": per_paper,
    }


def summary(name: str, result: dict) -> str:
    """One line per scoring, as `evaluate score` prints it."""
    b = result["binary_one_to_one"]
    return (
        f"{name}: {b['tp']} of {b['gold']} reference definitions found, {b['predicted']} "
        f"predicted (precision {100 * b['precision']:.1f}%, recall {100 * b['recall']:.1f}%), "
        f"full-record score {result['full_record_macro_score']:.3f}"
    )


# The pair scorer of a command: the real models, or recorded scores


def pair_scorer(
    replay: str | None, record: str | None, nli_batch_size: int
) -> tuple[metric.PairScorer, Any, dict, dict[str, Path]]:
    """(pair scorer, write-the-recording or None, receipt settings, receipt inputs)."""
    if replay and record:
        raise SystemExit("Give --replay-scores or --record-scores, not both.")
    if record and Path(record).exists():
        raise SystemExit(f"{record} exists; --record-scores writes a new file.")
    if replay:
        similarity, entailment = metric.recorded(Path(replay))
        how, device, inputs = (
            f"replayed from {replay}",
            None,
            {"pair_scores": Path(replay)},
        )
    else:
        device = files.env("SCIDEF_NLI_DEVICE")
        similarity, entailment = metric.models(device, nli_batch_size)
        how, inputs = f"computed on {device}", {}
    write = None
    if record:
        similarity, entailment, write = metric.recording(similarity, entailment)
        how += f", recorded to {record}"

    def scorer(pairs):
        return metric.defscore_pair_scores(pairs, similarity, entailment)

    settings = {
        "m3": {
            "slope": metric.DEFSCORE_SLOPE,
            "intercept": metric.DEFSCORE_INTERCEPT,
            "pair_score": metric.DEFSCORE_FORMULA,
        },
        "similarity_model": {
            "name": metric.BGE_MODEL,
            "revision": metric.BGE_REVISION,
            "max_length": metric.BGE_MAX_LENGTH,
            "batch_size": metric.BGE_BATCH_SIZE,
        },
        "entailment_model": {
            "name": metric.NLI_MODEL,
            "revision": metric.NLI_REVISION,
            "max_length": metric.NLI_MAX_LENGTH,
            "batch_size": nli_batch_size,
        },
        "device": device,
        "pair_scores": how,
    }
    return scorer, write, settings, inputs


# The command


def load_reference(path: Path) -> tuple[dict[str, list[dict]], dict]:
    """{paper id: reference records} of a reference file ({"papers": [...]} or a list), and what the results say about it."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    papers = raw.get("papers") if isinstance(raw, dict) else raw
    if not isinstance(papers, list):
        raise ValueError(f"{path}: no paper list")
    gold: dict[str, list[dict]] = {}
    for paper in papers:
        pid = paper.get("paper_id")
        if not isinstance(pid, str) or pid in gold:
            raise ValueError(f"{path}: invalid or duplicate paper id {pid!r}")
        counted(pid, paper.get("definition_records"))
        gold[pid] = paper["definition_records"]
    return gold, {
        "path": str(path),
        "sha256": files.digest(path),
        "papers": len(gold),
        "definitions": sum(definition_count(r) for r in gold.values()),
    }


# Where a reference's readiness record (READY.json) names the reference's sha256: under one of these keys, or under its file name in one of these lists (the released records use both forms).
READY_KEYS = ("sha256", "reference_sha256", "test_reference_sha256")
READY_LISTS = ("files", "files_sha256")


def checked_ready(reference: Path, ready: Path) -> dict:
    """What the results say about READY, the readiness record of REFERENCE, after checking that it names the reference's sha256 and that the reference has it. A record that names none is refused: it would let any reference through. (The TEST rescoring and the grid read the TEST reference this way.)"""
    marker = json.loads(ready.read_text(encoding="utf-8"))
    declared = None
    if isinstance(marker, dict):
        for key in READY_KEYS:
            if isinstance(marker.get(key), str):
                declared = marker[key]
        listed = {}
        for key in READY_LISTS:
            if isinstance(marker.get(key), dict):
                listed.update(marker[key])
        for name, value in listed.items():
            if Path(name).name == reference.name:
                declared = (
                    value
                    if isinstance(value, str)
                    else (value or {}).get("sha256", declared)
                )
    if declared is None:
        raise ValueError(
            f"The READY record names no sha256 of {reference.name}, so the reference "
            "cannot be checked against it"
        )
    if declared != files.digest(reference):
        raise ValueError("Reference bytes differ from the READY marker")
    return {
        "path": str(ready),
        "sha256": files.digest(ready),
        "declared_reference_sha256": declared,
    }


def load_predictions(folder: Path, cohort: set[str]) -> dict[str, list[dict]]:
    """{paper id: predicted records} from the JSON files below `folder`, each {"paper_id": ..., "definition_records": [...]}. A paper without a file has no predictions."""
    if not folder.is_dir():
        raise ValueError(f"{folder}: no such folder")
    predicted: dict[str, list[dict]] = {}
    for path in sorted(folder.rglob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        pid = value.get("paper_id")
        if pid not in cohort:
            raise ValueError(f"{path.name}: paper {pid!r} is not in the reference")
        if pid in predicted:
            raise ValueError(f"{path.name}: a second file for paper {pid}")
        counted(pid, value.get("definition_records"))
        predicted[pid] = value["definition_records"]
    return predicted


# MBDef-Gold's splits, as its dataset card names them, and the `split` values of its record file.
GOLD_SPLITS = {
    "train": ("train",),
    "validation": ("dev",),
    "test": ("original_test", "new_test"),
}
GOLD_SPLIT = "test"


def load_gold_split(
    path: Path, split: str, paper_ids: list[str] | None = None
) -> tuple[dict[str, list[dict]], dict]:
    """{paper id: reference records} of one split of MBDef-Gold's `validated_gt.json` (or only PAPER_IDS of it), and what the results say about it."""
    if split not in GOLD_SPLITS:
        raise SystemExit(f"--split is one of {', '.join(GOLD_SPLITS)}, not {split}.")
    raw = json.loads(path.read_text(encoding="utf-8"))
    gold = {
        paper["paper_id"]: paper["definition_records"]
        for paper in raw["papers"]
        if paper["split"] in GOLD_SPLITS[split]
    }
    if paper_ids:
        outside = [p for p in paper_ids if p not in gold]
        if outside:
            raise SystemExit(
                f"Not papers of MBDef-Gold's {split} split: {', '.join(outside)}"
            )
        gold = {p: gold[p] for p in paper_ids}
    for pid, records in gold.items():
        counted(pid, records)
    return gold, {
        "path": str(path),
        "sha256": files.digest(path),
        "split": split,
        "papers": len(gold),
        "definitions": sum(definition_count(r) for r in gold.values()),
    }


def scorings_jobs(listing: Path) -> list[tuple[str, Any]]:
    """(id, load) of every scoring of a --scorings list; `load()` gives (reference, its evidence, predictions, the result's fields, the receipt's inputs)."""
    entries = json.loads(listing.read_text(encoding="utf-8"))
    ids = [e["id"] for e in entries]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{listing}: scoring ids must be unique.")

    def loader(entry: dict):
        reference = listing.parent / entry["reference"]
        papers = listing.parent / entry["papers"]

        def load():
            gold, evidence = load_reference(reference)
            predicted = load_predictions(papers, set(gold))
            inputs = {
                f"{entry['id']}/reference": reference,
                f"{entry['id']}/papers": papers,
            }
            return gold, evidence, predicted, {"papers": str(entry["papers"])}, inputs

        return load

    return [(entry["id"], loader(entry)) for entry in entries]


def predictions_jobs(
    predictions: list[str], gold: dict[str, list[dict]], evidence: dict
) -> list[tuple[str, Any]]:
    """(id, load) of every --predictions file, scored on the reference GOLD: its papers outside GOLD are counted and left out."""

    def loader(path: Path):
        def load():
            everything = read_predictions(path)
            predicted = {p: r for p, r in everything.items() if p in gold}
            fields = {
                "predictions": str(path),
                "papers_outside_reference": len(everything) - len(predicted),
            }
            return gold, evidence, predicted, fields, {str(path): path}

        return load

    if len(set(predictions)) != len(predictions):
        raise SystemExit("Each --predictions file once, please.")
    return [(name, loader(Path(name))) for name in predictions]


def score_command(
    scorings: str | None,
    out: str,
    replay: str | None,
    record: str | None,
    threshold: float,
    nli_batch_size: int,
    predictions: list[str] | None = None,
    gold: str | None = None,
    split: str = GOLD_SPLIT,
    paper_ids: list[str] | None = None,
) -> int:
    """Score every scoring of the SCORINGS file, or every PREDICTIONS file on one split of MBDef-Gold, and write scores.json and a receipt into OUT.

    SCORINGS is a JSON list of {"id", "reference", "papers"}: the reference file and the folder of predicted paper files, relative to the SCORINGS file's folder unless absolute. A PREDICTIONS file is a predictions file (`records.read_predictions`); its reference is the SPLIT papers of GOLD (default MBDef-Gold's `validated_gt.json` in the data root), or only PAPER_IDS of them.
    """
    target = Path(out)
    if bool(scorings) == bool(predictions):
        raise SystemExit("Give --predictions (one or more files) or --scorings.")
    if (gold or paper_ids) and scorings:
        raise SystemExit("--gold and --paper go with --predictions, not --scorings.")
    if target.exists() and any(target.iterdir()):
        raise SystemExit(f"{out} is not empty; choose a new folder for --out.")
    if scorings:
        listing = Path(scorings)
        jobs, inputs = scorings_jobs(listing), {"scorings": listing}
    else:
        gold_path = Path(gold) if gold else files.mbdef_gold("validated_gt.json")
        reference, evidence = load_gold_split(gold_path, split, paper_ids)
        jobs = predictions_jobs(predictions or [], reference, evidence)
        inputs = {"gold": gold_path}
        print(
            f"Reference: the {split} split of {gold_path.name}, {evidence['papers']} papers "
            f"with {evidence['definitions']} definitions",
            flush=True,
        )
    scorer, write, settings, scorer_inputs = pair_scorer(replay, record, nli_batch_size)
    inputs = {**inputs, **scorer_inputs}
    results = []
    for name, load in jobs:
        started = time.time()
        try:
            gold_records, evidence, predicted, fields, more_inputs = load()
            result = score(gold_records, predicted, scorer, threshold)
        except Exception as error:
            raise SystemExit(
                f"Scoring {name} failed: {type(error).__name__}: {error}. "
                "No results were written."
            ) from error
        inputs.update(more_inputs)
        results.append(
            {
                "id": name,
                "reference": evidence,
                **fields,
                "papers_with_output": len(predicted),
                "seconds": round(time.time() - started, 3),
                "score": result,
            }
        )
        print(summary(name, result), flush=True)
        if predictions:
            print(
                f"  {len(predicted)} of the {evidence['papers']} papers have predictions; "
                f"F1 {100 * result['binary_one_to_one']['f1']:.1f}%"
                + (
                    f"; {fields['papers_outside_reference']} other papers were not scored"
                    if fields["papers_outside_reference"]
                    else ""
                ),
                flush=True,
            )
    target.mkdir(parents=True, exist_ok=True)
    files.write_json(target / SCORES, {"schema": SCORES_SCHEMA, "scorings": results})
    outputs = {SCORES: target / SCORES}
    if write and record:
        write(Path(record))
        outputs["pair_scores"] = Path(record)
    receipt = files.receipt(
        "evaluate score",
        settings={"threshold": threshold, **settings},
        inputs=inputs,
        outputs=outputs,
    )
    files.write_json(target / files.RECEIPT, receipt)
    print("wrote", out)
    return 0
