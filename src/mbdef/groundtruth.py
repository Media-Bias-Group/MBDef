"""The ground-truth step (paper Step 3, its ground truth): load it, read the annotations, release it.

Read it in three parts, top to bottom:

1. **Loading**: `load_ground_truth` reads a ground-truth JSON (schema 3); `dataset_views` gives the three views of the same papers that extraction is trained and evaluated on, and `ascii_words` the text form they join titles by.
2. **Curation**: `AnnotationExports` reads the raw Label Studio exports against V3 and works out each record's votes and consensus; `validate_record`, `object_digest` and `final_review_complete` are the checks the curator's export must pass.
3. **Release**: `build` checks the curator's export against the raw annotations and V3, then writes the relevant and non-relevant datasets, their statistics, the release report (PDF, via Tectonic), a hash manifest, the release zip and a receipt into one new folder.

The settings below are pinned to MBDef-Gold's release, so `build` with no flags rebuilds it. docs/step-3-extraction/groundtruth.md says what to change for another export.
"""

import copy
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from mbdef import files
from mbdef.stats import alpha_from_counts, pair_counts

# Pinned to MBDef-Gold's release: the curator's export, and its pseudonymised copy for publication (`groundtruth pseudonymise`), which builds the same data. `build` accepts either.
EXPORT_SHA256 = "27ac4f2946deab4f3f8539b3a264b210f862b649cbd55353f6fe9f5bb29f9f46"
PSEUDONYMISED_EXPORT_SHA256 = (
    "db6bc11b909a273d0356cace850c23d99be613245d842542b99a53282b693aaf"
)
PINNED_EXPORTS = {
    EXPORT_SHA256: "the curator's approved export",
    PSEUDONYMISED_EXPORT_SHA256: "its pseudonymised copy",
}
REPORT_DATE_EPOCH = (
    1788926168  # the released PDF's creation date (seconds since 1970, UTC)
)
ZIP_MEMBER_TIME = (
    2026,
    9,
    9,
    12,
    56,
    8,
)  # the released zip's member time (local, Tokyo)
RELEASE_ID = "MBDef-Gold-1.0.0"
EDIT_EXAMPLE = (
    "3457607:2",
    "Behavioral Bias",
)  # the report's worked example of an edit
CITATION_EXAMPLE = "0209e602acaeab882fee84e244caf574cf345ef9:5"  # its repaired citation

# Inputs outside git, under DATA_ROOT (docs/setup/data-root.md; listed in data/data-root.tsv): the two exports, which may name the annotators (not public), and their pseudonymised copies (`pseudonymise`, public).
EXPORT = "groundtruth/validated-v3.zip"
ANNOTATIONS = "groundtruth/newAnnotations/Annotations.zip"
PSEUDONYMISED = "groundtruth/pseudonymised"  # the copies, at the originals' paths below groundtruth/
EXPORT_COPY = f"{PSEUDONYMISED}/validated-v3.zip"
ANNOTATIONS_COPY = f"{PSEUDONYMISED}/newAnnotations/Annotations.zip"
WIKI = "groundtruth/newAnnotations/raw/IE-wiki-snapshot.txt"


def exports(
    export: str | None, annotations: str | None
) -> tuple[tuple[Path, str], tuple[Path, str]]:
    """The curator's export and the raw exports, each (path, how outputs name it; `files.located`): as given, else the originals in the data root, or their pseudonymised copies where the originals are not there (the originals are not public, so a reader's data root has only the copies)."""
    if export and annotations:
        return files.located(export, EXPORT), files.located(annotations, ANNOTATIONS)
    copies = (
        not (files.data_root() / EXPORT).exists()
        and (files.data_root() / EXPORT_COPY).exists()
    )
    return (
        files.located(export, EXPORT_COPY if copies else EXPORT),
        files.located(annotations, ANNOTATIONS_COPY if copies else ANNOTATIONS),
    )


# Inputs in the repository's data folder (--data).
V3 = "ground_truth/v3.json"
V2 = "ground_truth/v2.json"
SCIDEF = "ground_truth/scidef.json"
LABEL_STUDIO_XML = "frozen/labelStudioXML.md"

# The release's fixed texts: files of MBDef-Gold, in the data root's MBDef-Gold folder, that `build` copies as they are (files.mbdef_gold).
DATA_README = "history/data-README.md"  # the notes on the earlier ground truths
REPORT_MD = "validated_gt_report.md"
FIGURES = "figures"  # the report's figures (PNG) and their README

# The release report's LaTeX template, next to this file.
TEMPLATE = Path(__file__).with_name("groundtruth_report.tex")

# Outputs, inside --out.
PRIMARY = "validated_gt.json"
SEPARATE = "validated_gt_non_relevant.json"
REPORT = "validated_gt_report.pdf"
TEX = "validated_gt_report.tex"
STATS = "validated_gt_stats.json"
GUIDE = "validated_gt_README.md"
MANIFEST = "validated_gt_manifest.json"
BUNDLE = "validated_gt_release.zip"
EVIDENCE = "newAnnotations/raw/final-export-20260908"  # the export, unpacked

FIELDS = (
    "definitions",
    "synonyms",
    "description",
    "related_terms",
    "subtypes",
    "supertypes",
    "examples",
)
QUESTIONS = ("definition_is_definition", "term_relevant")
SPLITS = ["train", "dev", "original_test", "new_test"]
LABELS = {
    "definitions": "Definitions",
    "synonyms": "Synonyms",
    "description": "Description",
    "related_terms": "Related terms",
    "subtypes": "Subtypes",
    "supertypes": "Supertypes",
    "examples": "Examples",
}
EXPORT_MEMBERS = {
    "validated.json",
    "irrelevant-candidates.json",
    "original-v3.json",
    "annotations-original.zip",
    "curation-provenance.json",
    "relevance-labels.json",
    "curation-suggestions.json",
    "final-touches.json",
    "review-coverage.json",
    "pdf-sources.json",
    "labelStudioXML.md",
    "README.txt",
}


# 1. Loading


def load_ground_truth(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("paper_count") != len(data.get("papers", [])):
        raise ValueError(f"invalid ground truth: {path}")
    return data


def records(dataset: dict) -> list[dict]:
    return [r for p in dataset["papers"] for r in p["definition_records"]]


def ascii_words(text) -> str:
    """Lower-case ASCII letters and digits, one space between words ("Media-Bias (2020)" -> "media bias 2020"). The dataset views join titles by it, and the scoring matcher (`evaluate.metric`) compares terms, citations and exact texts by it.

    It drops every non-ASCII letter (after case folding) because the scorer and the dataset views of the released runs did, and their scores depend on it; for titles in any script, collect uses `collect.identity.unicode_words`."""
    return " ".join(re.findall(r"[a-z0-9]+", str(text or "").casefold()))


DATASETS = ("scidef", "biasdef", "biasdef-combined")


def dataset_views(data: Path = Path("data")) -> dict:
    """Three views of the same publication splits, read from the data folder `data`:

    - `scidef`: the original SciDef definitions (`ground_truth/scidef.json`), grouped by V3 paper (V2 maps SciDef's ids to V3's; its two duplicate publications are grouped, not dropped); only definitions are evaluated.
    - `biasdef`: MBDef-Gold's relevant set (`validated_gt.json`), all fields.
    - `biasdef-combined`: the relevant records first, then the non-relevant ones (`validated_gt_non_relevant.json`) of the same papers.

    Every view keeps V3's split of each paper; a different split is an error.
    """
    canonical = load_ground_truth(data / V3)
    by_id = {p["paper_id"]: p for p in canonical["papers"]}
    by_title = {ascii_words(p["metadata"]["title"]): p for p in canonical["papers"]}
    old = json.loads((data / SCIDEF).read_text())
    v2 = json.loads((data / V2).read_text())
    original, mapping = {}, []
    for entry in v2["papers"]:
        paper = by_id.get(entry["paper_id"]) or by_title[ascii_words(entry["title"])]
        pid = paper["paper_id"]
        target = original.setdefault(
            pid,
            {
                **copy.deepcopy(paper),
                "definition_records": [],
                "original_ids": [],
            },
        )
        target["original_ids"].append(entry["paper_id"])
        for term, value in old[entry["paper_id"]].items():
            target["definition_records"].append(
                {
                    "term": term,
                    **{field: [] for field in FIELDS},
                    "definitions": [{"text": value["definition"], "citation": []}],
                }
            )
        mapping.append(
            {
                "original_id": entry["paper_id"],
                "paper_id": pid,
                "old_split": entry["split"],
                "split": paper["split"],
            }
        )
    relevant = load_ground_truth(data / PRIMARY)
    irrelevant = load_ground_truth(data / SEPARATE)
    combined = {p["paper_id"]: copy.deepcopy(p) for p in irrelevant["papers"]}
    for p in relevant["papers"]:
        if combined[p["paper_id"]]["split"] != p["split"]:
            raise ValueError(
                f"Relevant and non-relevant splits differ: {p['paper_id']}"
            )
        combined[p["paper_id"]]["definition_records"] = (
            copy.deepcopy(p["definition_records"])
            + combined[p["paper_id"]]["definition_records"]
        )
    views: dict[str, dict[str, Any]] = {
        "scidef": {
            "papers": list(original.values()),
            "evaluated_fields": ["definitions"],
            "mapping": mapping,
            "relevant_only": False,
        },
        "biasdef": {
            **relevant,
            "evaluated_fields": list(FIELDS),
            "relevant_only": True,
        },
        "biasdef-combined": {
            "papers": list(combined.values()),
            "evaluated_fields": list(FIELDS),
            "relevant_only": False,
            "supporting_fields": "Non-relevant supporting fields were not systematically human-verified.",
        },
    }
    for name, view in views.items():
        view["name"] = name
        view["paper_count"] = len(view["papers"])
        # All variants must use the same publication split, including the original aliases.
        for p in view["papers"]:
            if p["split"] != by_id[p["paper_id"]]["split"]:
                raise ValueError(f"Cross-dataset split leakage: {p['paper_id']}")
    return views


# 2. Curation: the raw annotations, read against V3


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def object_digest(value) -> str:
    return files.sha256(canonical(value).encode())


def control(name: str) -> tuple[int, str]:
    """Return task-local record index and record-local question name."""
    match = re.fullmatch(r"example_(\d+)_(\d+)_(fit|type|error|error_text)", name)
    if match:
        return int(match[1]), f"example_{match[2]}_{match[3]}"
    match = re.fullmatch(r"(.+)_(\d+)", name)
    if not match:
        raise ValueError(f"Unrecognized annotation control: {name}")
    return int(match[2]), match[1]


def display(items: list[dict]) -> str:
    return (
        "\n".join(
            f"{i + 1}. {x['text']}"
            + (
                "  [citation: " + ", ".join(x["citation"]) + "]"
                if x["citation"]
                else ""
            )
            for i, x in enumerate(items)
        )
        or "—"
    )


def majority(answers: list[str | None]) -> str | None:
    """The consensus rule: the answer of at least two raters and of more raters than the other.

    One answer is not two-rater agreement, and a missing answer (None) is never a vote.
    """
    counts = Counter(a for a in answers if a is not None)
    yes, no = counts["Yes"], counts["No"]
    return "Yes" if yes > no and yes >= 2 else "No" if no > yes and no >= 2 else None


def consensus(votes: list[dict], question: str) -> dict:
    answers = [v["answers"].get(question, [None])[0] for v in votes]
    counts = Counter(answers)
    return {
        "value": majority(answers),
        "yes": counts["Yes"],
        "no": counts["No"],
        "missing": counts[None],
    }


def control_fields(name: str) -> tuple[str, ...]:
    if name.startswith("term_definition_"):
        return ("term", "definitions")
    if name.startswith("example_"):
        return ("examples",)
    return tuple(field for field in FIELDS if name.startswith(field + "_"))


class AnnotationExports:
    """The raw Label Studio exports read against V3: one row per V3 record, with its votes."""

    def __init__(self, archive: Path, baseline: Path):
        self.archive, self.baseline = Path(archive), Path(baseline)
        self.fingerprint = {
            "archive_sha256": files.digest(self.archive),
            "v3_sha256": files.digest(self.baseline),
        }
        self.original = json.loads(self.baseline.read_text(encoding="utf-8"))
        self.papers = {p["paper_id"]: p for p in self.original["papers"]}
        self.rows: dict[str, dict[str, Any]] = {}
        self.inventory: list[dict[str, Any]] = []
        self.duplicates: list[dict[str, Any]] = []
        self.task_counts = Counter()
        self.annotators = defaultdict(Counter)
        term_index = {}
        for p in self.papers.values():
            for i, record in enumerate(p["definition_records"]):
                key = f"{p['paper_id']}:{i}"
                term_key = (p["paper_id"], record["term"])
                if term_key in term_index:
                    raise ValueError(f"Ambiguous V3 term: {term_key}")
                term_index[term_key] = key
                self.rows[key] = {
                    "key": key,
                    "paper_id": p["paper_id"],
                    "index": i,
                    "title": p["metadata"]["title"],
                    "split": p["split"],
                    "record": record,
                    "votes": [],
                    "targeted": [],
                }
        fragment_map = {}
        raw_tasks = []
        with zipfile.ZipFile(self.archive) as archive_file:
            for member in archive_file.infolist():
                if not member.filename.endswith(".json"):
                    continue
                tasks = json.loads(archive_file.read(member))
                stage = (
                    "third"
                    if member.filename.startswith("Annotation_Disagreements/")
                    else "rest"
                    if member.filename.startswith("Annotation_rest/")
                    else "weekly"
                )
                self.inventory.append(
                    {
                        "file": member.filename,
                        "bytes": member.file_size,
                        "tasks": len(tasks),
                        "annotations": sum(len(t["annotations"]) for t in tasks),
                        "stage": stage,
                    }
                )
                raw_tasks.extend((stage, member.filename, task) for task in tasks)
        # Establish the fragment-local indices before reading the third-round maps.
        for stage, _member, task in raw_tasks:
            d = task["data"]
            pid = d.get("original_paper_id", d["paper_id"].split("__")[0])
            mapping = {}
            for record in d["definition_records"]:
                key = term_index[(pid, record["term"])]
                base = self.rows[key]["record"]
                for field in FIELDS:
                    if field == "examples":
                        expected = [
                            x["text"]
                            + (
                                "  [citation: " + ", ".join(x["citation"]) + "]"
                                if x["citation"]
                                else ""
                            )
                            for x in base[field]
                        ]
                        # Export examples sometimes omit the citation display suffix.
                        if record[field] not in (
                            expected,
                            [x["text"] for x in base[field]],
                        ):
                            raise ValueError(f"Example payload differs from V3: {key}")
                    elif record[field + "_display"] != display(base[field]):
                        raise ValueError(f"Payload differs from V3: {key} / {field}")
                if record["idx"] in mapping:
                    raise ValueError("Duplicate task-local index")
                mapping[record["idx"]] = key
            if len(mapping) != d["num_records"]:
                raise ValueError("Incorrect task record count")
            if stage != "third":
                if (
                    d["paper_id"] in fragment_map
                    and mapping != fragment_map[d["paper_id"]]
                ):
                    raise ValueError("Fragment index changed between exports")
                fragment_map[d["paper_id"]] = mapping
        self.fragments = len(fragment_map)
        for stage, member, task in raw_tasks:
            d = task["data"]
            pid = d.get("original_paper_id", d["paper_id"].split("__")[0])
            mapping = {
                r["idx"]: term_index[(pid, r["term"])] for r in d["definition_records"]
            }
            if stage == "third":
                for record in d["definition_records"]:
                    key = mapping[record["idx"]]
                    if (
                        fragment_map[d["source_paper_id"]][
                            record["source_record_index"]
                        ]
                        != key
                    ):
                        raise ValueError(f"Third-round source index mismatch: {key}")
                    self.rows[key]["targeted"] = record["disagreement_questions"]
                expected_map = [
                    {
                        "queue_record_index": r["idx"],
                        "source_record_index": r["source_record_index"],
                        "disagreement_questions": r["disagreement_questions"],
                    }
                    for r in d["definition_records"]
                ]
                if d["disagreement_record_map"] != expected_map:
                    raise ValueError("Third-round map is inconsistent")
            seen = {}
            for ann in task["annotations"]:
                self.task_counts[(stage, "raw")] += 1
                if ann["was_cancelled"]:
                    self.task_counts[(stage, "cancelled")] += 1
                    continue
                answers = defaultdict(dict)
                note_meta = defaultdict(dict)
                for result in ann["result"]:
                    idx, name = control(result["from_name"])
                    if idx not in mapping or name in answers[idx]:
                        raise ValueError("Invalid or repeated annotation control")
                    value = result["value"].get("choices", result["value"].get("text"))
                    if not isinstance(value, list) or not all(
                        isinstance(v, str) for v in value
                    ):
                        raise ValueError("Invalid result value")
                    if name in QUESTIONS and value not in (["Yes"], ["No"]):
                        raise ValueError("Unexpected binary label")
                    answers[idx][name] = value
                    if name.endswith("error_text"):
                        note_meta[idx][name] = {
                            "raw_control": result["from_name"],
                            "object_name": result["to_name"],
                        }
                signature = (ann["completed_by"], canonical(answers))
                if signature in seen:
                    prior = seen[signature]
                    for vote in prior:
                        vote["duplicate_annotation_ids"].append(ann["id"])
                    self.duplicates.append(
                        {
                            "file": member,
                            "task": task["id"],
                            "annotation": ann["id"],
                            "same_as": prior[0]["annotation"],
                        }
                    )
                    continue
                self.task_counts[(stage, "unique")] += 1
                self.annotators[ann["completed_by"]][stage + "_tasks"] += 1
                self.annotators[ann["completed_by"]][stage + "_records"] += len(mapping)
                votes = []
                for idx, key in mapping.items():
                    vote: dict[str, Any] = {
                        "annotator": ann["completed_by"],
                        "stage": stage,
                        "project": task["project"],
                        "task": task["id"],
                        "annotation": ann["id"],
                        "file": member,
                        "fragment": d["paper_id"],
                        "local_index": idx,
                        "created_at": ann["created_at"],
                        "answers": answers[idx],
                        "note_meta": note_meta[idx],
                        "duplicate_annotation_ids": [],
                    }
                    self.rows[key]["votes"].append(vote)
                    votes.append(vote)
                seen[signature] = votes
        for row in self.rows.values():
            row["votes"].sort(key=lambda v: (v["stage"] == "third", v["annotator"]))
            initial = [v for v in row["votes"] if v["stage"] != "third"]
            if len(initial) != 2 or len({v["annotator"] for v in row["votes"]}) != len(
                row["votes"]
            ):
                raise ValueError(
                    f"Expected two independent initial raters and distinct third rater: {row['key']}"
                )
            row["initial_disagreements"] = [
                q
                for q in QUESTIONS
                if {v["answers"].get(q, [None])[0] for v in initial} == {"Yes", "No"}
            ]
            if set(row["initial_disagreements"]) != set(row["targeted"]):
                raise ValueError(f"Disagreement routing mismatch: {row['key']}")
            row["labels"] = {q: consensus(row["votes"], q) for q in QUESTIONS}
            row["notes"] = [
                {
                    "id": f"{v['annotation']}:{name}",
                    "annotator": v["annotator"],
                    "stage": v["stage"],
                    "annotation": v["annotation"],
                    "task": v["task"],
                    "file": v["file"],
                    "field": name,
                    "text": "\n".join(values),
                    **v["note_meta"][name],
                }
                for v in row["votes"]
                for name, values in v["answers"].items()
                if name.endswith("error_text") and any(values)
            ]
            names = {name for v in row["votes"] for name in v["answers"]}
            row["field_conflicts"] = [
                name
                for name in sorted(names)
                if name not in QUESTIONS
                and not name.endswith(("error", "error_text"))
                and len(
                    {
                        canonical(v["answers"][name])
                        for v in row["votes"]
                        if name in v["answers"]
                    }
                )
                > 1
            ]
            row["negative_fields"] = sorted(
                {
                    name
                    for v in row["votes"]
                    for name, values in v["answers"].items()
                    if "Does not fit" in values
                }
            )
            row["error_flags"] = sorted(
                {
                    name
                    for v in row["votes"]
                    for name in v["answers"]
                    if name.endswith("error")
                }
            )
            row["missing_initial"] = [
                q for q in QUESTIONS if any(q not in v["answers"] for v in initial)
            ]
            row["unmapped"] = sorted(
                {
                    name
                    for v in row["votes"]
                    for name in v["answers"]
                    if (
                        name.startswith("example_")
                        and int(name.split("_")[1]) >= len(row["record"]["examples"])
                    )
                    or (
                        name.endswith("_fit")
                        and name[:-4] in FIELDS
                        and not row["record"][name[:-4]]
                    )
                }
            )
            row["other_example_types"] = sorted(
                {
                    name
                    for v in row["votes"]
                    for name, values in v["answers"].items()
                    if name.startswith("example_")
                    and name.endswith("_type")
                    and any(x != "Example" for x in values)
                }
            )
            field_issues = defaultdict(set)
            for names_, reason in (
                (row["field_conflicts"], "Annotators disagree"),
                (row["negative_fields"], "Does not fit"),
                (row["error_flags"], "Error flagged"),
                (row["unmapped"], "Answer has no matching item"),
                (
                    row["other_example_types"],
                    "Labeled as a counterexample or other material",
                ),
                ([n["field"] for n in row["notes"]], "Correction note"),
            ):
                for name in names_:
                    for field in control_fields(name):
                        field_issues[field].add(reason)
            row["field_issues"] = {
                field: sorted(reasons) for field, reasons in field_issues.items()
            }
            missing = any(x["value"] is None for x in row["labels"].values())
            row["review_phase"] = "work" if field_issues or missing else "confirm"
            row["priority"] = (
                0
                if field_issues
                else 1
                if missing
                else 2
                if row["initial_disagreements"]
                else 3
            )

    def statistics(self) -> dict:
        rows = list(self.rows.values())
        stats: dict[str, Any] = {
            "papers": len(self.papers),
            "records": len(rows),
            "fragments": self.fragments,
            "definitions": sum(len(r["record"]["definitions"]) for r in rows),
            "raw_annotations": sum(x["annotations"] for x in self.inventory),
            "duplicate_annotations": len(self.duplicates),
            "third_records": sum(bool(r["targeted"]) for r in rows),
            "notes": sum(len(r["notes"]) for r in rows),
            "records_with_notes": sum(bool(r["notes"]) for r in rows),
            "records_with_field_conflicts": sum(
                bool(r["field_conflicts"]) for r in rows
            ),
            "records_with_negative_fields": sum(
                bool(r["negative_fields"]) for r in rows
            ),
            "records_with_unmapped": sum(bool(r["unmapped"]) for r in rows),
            "records_with_field_issues": sum(bool(r["field_issues"]) for r in rows),
            "review_phases": dict(Counter(r["review_phase"] for r in rows)),
            "questions": {},
            "outcomes": {},
            "splits": {},
        }
        for q in QUESTIONS:
            pairs = []
            for row in rows:
                values = [
                    v["answers"].get(q, [None])[0]
                    for v in row["votes"]
                    if v["stage"] != "third"
                ]
                if None not in values:
                    pairs.append(values)
            n = len(pairs)
            yes_yes, no_no, disagree = pair_counts(pairs)
            stats["questions"][q] = {
                "complete_pairs": n,
                "missing_pairs": len(rows) - n,
                "yes_yes": yes_yes,
                "no_no": no_no,
                "disagree": disagree,
                "agreement": 1 - disagree / n,
                "alpha_nominal": alpha_from_counts(yes_yes, no_no, disagree),
                "final": dict(
                    Counter(r["labels"][q]["value"] or "unresolved" for r in rows)
                ),
            }
        stats["outcomes"] = dict(
            Counter(
                " / ".join(r["labels"][q]["value"] or "unresolved" for q in QUESTIONS)
                for r in rows
            )
        )
        for split in sorted({r["split"] for r in rows}):
            subset = [r for r in rows if r["split"] == split]
            stats["splits"][split] = {
                "papers": len({r["paper_id"] for r in subset}),
                "records": len(subset),
                "both_yes": sum(
                    all(r["labels"][q]["value"] == "Yes" for q in QUESTIONS)
                    for r in subset
                ),
                "valid_definitions": sum(
                    r["labels"][QUESTIONS[0]]["value"] == "Yes" for r in subset
                ),
                "disagreements": sum(bool(r["targeted"]) for r in subset),
            }
        return stats


def validate_record(record: dict) -> None:
    if not isinstance(record, dict) or set(record) != {"term", *FIELDS}:
        raise ValueError("Record must have term and exactly the seven V3 list fields")
    if not isinstance(record["term"], str) or not record["term"].strip():
        raise ValueError("Term must be non-empty")
    for field in FIELDS:
        if not isinstance(record[field], list):
            raise ValueError(f"{field} must be a list")
        for item in record[field]:
            if not isinstance(item, dict) or set(item) != {"text", "citation"}:
                raise ValueError(f"{field} items must contain text and citation")
            if not isinstance(item["text"], str) or not item["text"].strip():
                raise ValueError(f"{field} contains empty text")
            if not isinstance(item["citation"], list) or not all(
                isinstance(c, str) and c.strip() for c in item["citation"]
            ):
                raise ValueError("Citations must be a list of non-empty strings")


def final_review_complete(decision: dict, touch: dict) -> bool:
    return not touch or (
        decision.get("final_review_sha256") == object_digest(touch)
        and decision.get("final_record_sha256")
        == object_digest({"record": decision["record"], "labels": decision["labels"]})
    )


# 3. Release


def profile(dataset: dict) -> dict:
    rs = records(dataset)
    return {
        "paper_entries": len(dataset["papers"]),
        "nonempty_papers": sum(
            bool(p["definition_records"]) for p in dataset["papers"]
        ),
        "records": len(rs),
        "unique_terms_casefold": len({r["term"].casefold().strip() for r in rs}),
        "fields": {f: sum(len(r[f]) for r in rs) for f in FIELDS},
        "populated_records": {f: sum(bool(r[f]) for r in rs) for f in FIELDS},
        "citation_markers": sum(
            len(x["citation"]) for r in rs for f in FIELDS for x in r[f]
        ),
        "definition_multiplicity": dict(
            sorted(Counter(len(r["definitions"]) for r in rs).items())
        ),
        "splits": {
            s: {
                "papers": sum(p["split"] == s for p in dataset["papers"]),
                "nonempty_papers": sum(
                    p["split"] == s and bool(p["definition_records"])
                    for p in dataset["papers"]
                ),
                "records": sum(
                    len(p["definition_records"])
                    for p in dataset["papers"]
                    if p["split"] == s
                ),
                "definitions": sum(
                    len(r["definitions"])
                    for p in dataset["papers"]
                    if p["split"] == s
                    for r in p["definition_records"]
                ),
            }
            for s in SPLITS
        },
    }


def split_origins(primary: dict, separate: dict, v2: dict) -> list[dict]:
    original_ids = {p["paper_id"] for p in v2["papers"]}
    rows = []
    for label, splits in [
        ("Train", {"train"}),
        ("Development", {"dev"}),
        ("Test", {"original_test", "new_test"}),
        ("Total", set(SPLITS)),
    ]:
        row: dict[str, Any] = {"split": label}
        for name, dataset in [("relevant", primary), ("non_relevant", separate)]:
            papers = [
                p
                for p in dataset["papers"]
                if p["split"] in splits and p["definition_records"]
            ]
            row[name] = {}
            for origin, predicate in [("original", True), ("added", False)]:
                selected = [
                    p for p in papers if (p["paper_id"] in original_ids) == predicate
                ]
                row[name][origin] = {
                    "papers": len(selected),
                    "records": sum(len(p["definition_records"]) for p in selected),
                }
        rows.append(row)
    return rows


def analyze(export: Path, annotations: Path, data: Path, export_sha256: str | None):
    """Check the curator's export against the raw annotations and V3; compute the statistics.

    The export's sha256 must be `export_sha256`, or, when that is None, one of PINNED_EXPORTS."""
    found = files.digest(export)
    if export_sha256 and found != export_sha256:
        raise SystemExit(
            f"The export's sha256 is {found}, not {export_sha256} (--export-sha256)."
        )
    if not export_sha256 and found not in PINNED_EXPORTS:
        pinned = " or ".join(f"{h} ({what})" for h, what in PINNED_EXPORTS.items())
        raise SystemExit(
            f"The export's sha256 is {found}, not a pinned one: {pinned}. "
            "For another export, pass its hash with --export-sha256 (docs/step-3-extraction/groundtruth.md)."
        )
    with zipfile.ZipFile(export) as archive:
        assert archive.testzip() is None, "Corrupt export archive"
        raw = {name: archive.read(name) for name in archive.namelist()}
    assert set(raw) == EXPORT_MEMBERS, (
        "Unexpected export members: inspect before packaging"
    )
    primary, separate, baseline, provenance, labels, touches, coverage, pdfs = (
        json.loads(raw[n])
        for n in [
            "validated.json",
            "irrelevant-candidates.json",
            "original-v3.json",
            "curation-provenance.json",
            "relevance-labels.json",
            "final-touches.json",
            "review-coverage.json",
            "pdf-sources.json",
        ]
    )
    corpus = AnnotationExports(annotations, data / V3)
    assert provenance["fingerprint"] == corpus.fingerprint
    assert raw["original-v3.json"] == corpus.baseline.read_bytes()
    assert raw["annotations-original.zip"] == corpus.archive.read_bytes()
    assert raw["labelStudioXML.md"] == (data / LABEL_STUDIO_XML).read_bytes()
    decisions = provenance["decisions"]
    assert (
        set(decisions) == set(corpus.rows) == {entry["source_key"] for entry in labels}
    )
    assert (
        len(labels)
        == len({entry["source_key"] for entry in labels})
        == len(corpus.rows)
    )
    assert set(provenance["history"]) == set(decisions)
    assert set(provenance["annotations"]) == set(decisions)
    assert all(
        d["status"] == "reviewed" and d["scope"] == "both" for d in decisions.values()
    )
    assert all(
        final_review_complete(decisions[k], t) for k, t in touches["records"].items()
    )
    assert provenance["scope"] == "both"
    assert provenance["final_audit_sha256"] == touches["sha256"]
    assert (
        object_digest({k: v for k, v in touches.items() if k != "sha256"})
        == touches["sha256"]
    )
    for key, d in decisions.items():
        validate_record(d["record"])
        if d["labels"][QUESTIONS[0]] == "Yes":
            assert d["record"]["definitions"]
        hist = provenance["history"][key]
        latest = max(hist, key=lambda h: h["revision"])
        assert latest == {k: v for k, v in d.items() if k != "key"}, key
    label_map = {entry["source_key"]: entry for entry in labels}
    for dataset, predicate in [
        (primary, lambda d: all(d["labels"][q] == "Yes" for q in QUESTIONS)),
        (
            separate,
            lambda d: d["labels"] == dict(zip(QUESTIONS, ("Yes", "No"), strict=True)),
        ),
    ]:
        assert dataset["schema_version"] == baseline["schema_version"] == "3"
        assert (
            dataset["paper_count"] == len(dataset["papers"]) == len(baseline["papers"])
        )
        assert [p["paper_id"] for p in dataset["papers"]] == [
            p["paper_id"] for p in baseline["papers"]
        ]
        for p, old in zip(dataset["papers"], baseline["papers"], strict=True):
            assert {k: v for k, v in p.items() if k != "definition_records"} == {
                k: v for k, v in old.items() if k != "definition_records"
            }
            selected = [
                f"{p['paper_id']}:{i}"
                for i in range(len(old["definition_records"]))
                if predicate(decisions[f"{p['paper_id']}:{i}"])
            ]
            assert p["definition_records"] == [decisions[k]["record"] for k in selected]
            if dataset is primary:
                for i, key in enumerate(selected):
                    assert (
                        label_map[key]["retained"]
                        and label_map[key]["output_record_index"] == i
                    )
    primary_keys = [entry["source_key"] for entry in labels if entry["retained"]]
    separate_keys = [
        entry["source_key"] for entry in labels if entry["in_irrelevant_candidates"]
    ]
    assert not set(primary_keys) & set(separate_keys)
    assert len(primary_keys) == len(records(primary))
    assert len(separate_keys) == len(records(separate))
    assert len({p["paper_id"] for p in baseline["papers"]}) == len(baseline["papers"])
    # The relevant set leaves out only the paper entries that kept no record.
    removed_papers = [p for p in primary["papers"] if not p["definition_records"]]
    kept_papers = [p for p in primary["papers"] if p["definition_records"]]
    primary = {**primary, "papers": kept_papers, "paper_count": len(kept_papers)}
    changed = [
        k for k in primary_keys if decisions[k]["record"] != corpus.rows[k]["record"]
    ]
    unchanged = [k for k in primary_keys if k not in changed]
    changed_fields = {
        f: sum(
            decisions[k]["record"][f] != corpus.rows[k]["record"][f]
            for k in primary_keys
        )
        for f in ["term", *FIELDS]
    }
    # Item deltas are exact multiset comparisons, not guesses about semantic additions/deletions.
    item_diff = {}
    exact_moves = []
    for f in FIELDS:
        old = Counter(
            (k, object_digest(x))
            for k in primary_keys
            for x in corpus.rows[k]["record"][f]
        )
        new = Counter(
            (k, object_digest(x))
            for k in primary_keys
            for x in decisions[k]["record"][f]
        )
        item_diff[f] = {
            "retained_v3_cohort": sum(old.values()),
            "final": sum(new.values()),
            "exact_unchanged": sum((old & new).values()),
            "before_only": sum((old - new).values()),
            "after_only": sum((new - old).values()),
        }
    for k in primary_keys:
        removed, added = [], []
        for f in FIELDS:
            old, new = (
                Counter(object_digest(x) for x in r[f])
                for r in [corpus.rows[k]["record"], decisions[k]["record"]]
            )
            removed.extend((f, x) for x, n in (old - new).items() for _ in range(n))
            added.extend((f, x) for x, n in (new - old).items() for _ in range(n))
        for f, x in removed:
            match = next(((g, y) for g, y in added if g != f and y == x), None)
            if match:
                added.remove(match)
                exact_moves.append(
                    {"source_key": k, "from": f, "to": match[0], "item_sha256": x}
                )
    duplicates, repeated_names = [], []
    for p in primary["papers"]:
        names = Counter(r["term"].casefold().strip() for r in p["definition_records"])
        repeated_names.extend(
            {"paper_id": p["paper_id"], "term": name, "records": n}
            for name, n in names.items()
            if n > 1
        )
        for i, r in enumerate(p["definition_records"]):
            for f in FIELDS:
                if len({object_digest(x) for x in r[f]}) != len(r[f]):
                    duplicates.append((p["paper_id"], i, f))
    assert not duplicates
    v1 = json.loads((data / SCIDEF).read_text(encoding="utf-8"))
    v2 = json.loads((data / V2).read_text(encoding="utf-8"))
    touch_outcomes = []
    for k, t in touches["records"].items():
        r = decisions[k]["record"]
        outcome = (
            "all_applied"
            if all(
                r[e["field"]] == e["after"]
                for issue in t["issues"]
                for e in issue["changes"]
            )
            else "kept_existing"
            if r == t["base_record"]
            else "manual_or_partial"
        )
        touch_outcomes.append({"source_key": k, "term": r["term"], "outcome": outcome})
    events = {
        (v["stage"], v["project"], v["task"], v["annotation"]): v
        for vs in provenance["annotations"].values()
        for v in vs
    }
    effort = {
        str(a): dict(
            Counter(v["stage"] for v in events.values() if v["annotator"] == a)
        )
        for a in sorted({v["annotator"] for v in events.values()})
    }
    original_stats = corpus.statistics()
    main_coverage = [u for u in coverage if u["key"] in set(primary_keys)]
    fit_counts = Counter(min(u["answer_count"], 2) for u in main_coverage)
    validity = Counter(
        d["labels"][QUESTIONS[0]] or "Unknown" for d in decisions.values()
    )
    primary_profile, separate_profile, v3_profile = map(
        profile, [primary, separate, baseline]
    )
    stats: dict[str, Any] = {
        "release_id": RELEASE_ID,
        "export_created_at": provenance["created_at"],
        "export_sha256": found,
        "primary": primary_profile,
        "non_relevant": separate_profile,
        "v3": v3_profile,
        "split_origin_counts": split_origins(primary, separate, v2),
        "packaging": {
            "relevant_empty_papers_removed": len(removed_papers),
            "removed_relevant_paper_ids": [p["paper_id"] for p in removed_papers],
            "retained_paper_objects_unchanged": True,
            "original_export_preserved": True,
            "non_relevant_export_unchanged": True,
        },
        "earlier_stages": {
            "defextra_scidef": {
                "paper_ids": len(v1),
                "records": sum(len(x) for x in v1.values()),
                "unique_publications_documented": 73,
            },
            "v2": {
                "paper_ids": len(v2["papers"]),
                "records": sum(len(p["definitions"]) for p in v2["papers"]),
                "unique_publications_documented": 73,
            },
        },
        "fates": {
            "primary": len(primary_keys),
            "non_relevant": len(separate_keys),
            "definition_no": validity["No"],
            "definition_unknown_relevance_no": validity["Unknown"],
        },
        "approval_basis_all": dict(
            Counter(entry["approval_basis"] for entry in labels)
        ),
        "approval_basis_primary": dict(
            Counter(entry["approval_basis"] for entry in labels if entry["retained"])
        ),
        "non_relevant_supporting_fields_individually_reviewed": sum(
            entry["supporting_fields_reviewed"]
            for entry in labels
            if entry["in_irrelevant_candidates"]
        ),
        "changed_primary_records": len(changed),
        "unchanged_primary_records": len(unchanged),
        "changed_record_keys": changed,
        "changed_fields": changed_fields,
        "exact_item_comparison": item_diff,
        "exact_cross_field_moves": exact_moves,
        "same_paper_repeated_terms": repeated_names,
        "final_touch_outcomes": touch_outcomes,
        "annotation_archive": original_stats,
        "canonical_task_annotations_by_stage": dict(
            Counter(v["stage"] for v in events.values())
        ),
        "annotator_task_annotation_counts": effort,
        "primary_original_supporting_fit_coverage": {
            "units": len(main_coverage),
            "zero_fit_answers": fit_counts[0],
            "one_fit_answer": fit_counts[1],
            "two_or_more_fit_answers": fit_counts[2],
        },
        "missing_paper_metadata": {
            f: sum(not p["metadata"].get(f) for p in primary["papers"])
            for f in ["title", "authors", "publication_year", "doi", "venue"]
        },
        "pdf_identity": {
            "papers": len(pdfs),
            "available": sum(x["available"] for x in pdfs.values()),
            "same_input": sum(x["same_input"] for x in pdfs.values()),
        },
        "integrity_checks": {
            "all_2488_decisions_reviewed": True,
            "all_final_touches_approved": True,
            "all_exported_fields_equal_saved_decisions": True,
            "all_paper_metadata_and_splits_preserved": True,
            "raw_originals_byte_identical": True,
            "no_duplicate_items_within_primary_fields": True,
        },
        "counting_notes": [
            "Definition record = paper-specific concept; definition statement = one item in definitions; example = one item in examples.",
            "Earlier unique-publication counts follow the supplied IE wiki and data README; identical normalized titles alone are not treated as publication identity.",
            "Before-only and after-only items include text/citation edits and moves. They are not counts of semantic deletions and additions.",
            "Supporting fit coverage refers to original V3 items, not the resulting edited items; absent votes are not agreement.",
            "annotation_policy records preserve accepted human outcomes but do not imply individual curator checking of every supporting field.",
        ],
    }
    return raw, primary, separate, baseline, provenance, labels, stats, corpus


def tex(value) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(c, c) for c in str(value))


def build_pdf(
    primary: dict,
    provenance: dict,
    s: dict,
    corpus: AnnotationExports,
    out: Path,
    report_date: int,
) -> None:
    """Fill the LaTeX template and compile it with Tectonic, at the pinned date."""
    # This report documents the fixed exported release, never the mutable live database.
    missing = [
        k
        for k, d in provenance["decisions"].items()
        if d["labels"][QUESTIONS[0]] is None
    ]
    for key in missing:
        votes = [a for a in provenance["annotations"][key] if a["stage"] != "third"]
        assert len(votes) == 2
        assert all(a["answers"].get("term_relevant") == ["No"] for a in votes)
        assert provenance["decisions"][key]["labels"]["term_relevant"] == "No"

    descriptions = {
        "definitions": "Statements of the term's meaning",
        "synonyms": "Alternative names for the same concept",
        "description": "Additional explanatory information",
        "related_terms": "Associated concepts",
        "subtypes": "More specific kinds of the concept",
        "supertypes": "Broader categories containing the concept",
        "examples": "Instances that illustrate the concept",
    }
    n_primary, n_separate = s["primary"]["records"], s["non_relevant"]["records"]
    field_rows = [
        rf"\file{{term}} & Concept name & {n_primary:,} & {n_primary:,} & {n_separate:,} \\"
    ]
    for field in FIELDS:
        n = s["exact_item_comparison"][field]
        field_rows.append(
            rf"\file{{{tex(field)}}} & {descriptions[field]} & "
            rf"{n['retained_v3_cohort']:,} & {n['final']:,} & "
            rf"{s['non_relevant']['fields'][field]:,} \\"
        )
    split_rows = []
    for row in s["split_origin_counts"]:
        cells = []
        for name in ["relevant", "non_relevant"]:
            for metric in ["papers", "records"]:
                a, b = (row[name][origin][metric] for origin in ["original", "added"])
                cells.append(f"{a + b:,} ({a:,} + {b:,})")
        split_rows.append(tex(row["split"]) + " & " + " & ".join(cells) + r" \\")
    bars = []
    for field, n in sorted(s["changed_fields"].items(), key=lambda x: -x[1]):
        name = LABELS.get(field, "Term")
        width = 28 * n / max(s["changed_fields"].values())
        bars.append(rf"\barrow{{{name}}}{{{width:.2f}}}{{{n}}}")

    def pair(key):
        return corpus.rows[key]["record"], provenance["decisions"][key]["record"]

    before, after = pair(EDIT_EXAMPLE[0])
    assert after["term"] == EDIT_EXAMPLE[1]
    assert after in records(primary)
    lines = ["{", '  "term": ' + json.dumps(after["term"]) + ","]
    for i, field in enumerate(FIELDS):
        suffix = "," if i < len(FIELDS) - 1 else ""
        items = after[field]
        if not items:
            lines.append(f'  "{field}": []{suffix}')
            continue
        lines.append(f'  "{field}": [')
        for j, item in enumerate(items):
            lines.append(
                "    {"
                + '"text": '
                + json.dumps(item["text"], ensure_ascii=False)
                + ","
            )
            lines.append(
                '     "citation": '
                + json.dumps(item["citation"], ensure_ascii=False)
                + "}"
                + ("," if j < len(items) - 1 else "")
            )
        lines.append("  ]" + suffix)
    lines.append("}")
    sample = "\n".join(lines)
    assert json.loads(sample) == after
    # Escape only JSON keys into LaTeX; stored strings remain literal in listings.
    sample = re.sub(
        r'"(\w+)"(?=:)',
        lambda m: r'(*@\textcolor{inkblue}{"' + tex(m[1]) + r'"}@*)',
        sample,
    )
    old_cite, new_cite = pair(CITATION_EXAMPLE)
    values = {
        "FIELD_ROWS": "\n".join(field_rows),
        "SPLIT_ROWS": "\n".join(split_rows),
        "BAR_ROWS": "\n".join(bars),
        "BEHAVIOR_BEFORE": tex(before["definitions"][0]["text"]),
        "BEHAVIOR_AFTER": tex(after["definitions"][0]["text"]),
        "CITATION_BEFORE": tex(json.dumps(old_cite["definitions"][0]["citation"])),
        "CITATION_AFTER": tex(json.dumps(new_cite["definitions"][0]["citation"])),
        "JSON_SAMPLE": sample,
    }
    source = TEMPLATE.read_text(encoding="utf-8")
    for key, value in values.items():
        assert source.count("<<" + key + ">>") == 1
        source = source.replace("<<" + key + ">>", value)
    assert not re.search(r"<<[A-Z_]+>>", source)
    (out / TEX).write_text(source, encoding="utf-8")
    with tempfile.TemporaryDirectory() as logs:
        command = ["tectonic", "--keep-logs", "--outdir", logs, str(out / TEX)]
        try:
            done = subprocess.run(
                command,
                capture_output=True,
                text=True,
                env={**os.environ, "SOURCE_DATE_EPOCH": str(report_date)},
            )
        except FileNotFoundError:
            raise SystemExit(
                "Tectonic is not installed; the release report needs it (docs/step-3-extraction/groundtruth.md)."
            ) from None
        if done.returncode != 0:
            raise SystemExit(f"Tectonic failed:\n{done.stdout}{done.stderr}")
        (out / REPORT).write_bytes((Path(logs) / REPORT).read_bytes())


def write_zip(path: Path, members: dict[str, Path]) -> None:
    """The release zip: every member, then SHA256SUMS, all at the release's member time."""

    def info(name: str) -> zipfile.ZipInfo:
        member = zipfile.ZipInfo(name, ZIP_MEMBER_TIME)
        member.compress_type = zipfile.ZIP_DEFLATED
        member.external_attr = 0o600 << 16  # what ZipFile.writestr gives a name
        return member

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, source in members.items():
            z.writestr(info(name), source.read_bytes())
        z.writestr(
            info("SHA256SUMS"),
            "".join(
                f"{files.digest(source)}  {name}\n"
                for name, source in sorted(members.items())
            ),
        )


def build(
    out: str,
    export: str | None,
    annotations: str | None,
    wiki: str | None,
    data: str,
    export_sha256: str | None = None,
) -> int:
    """Build the ground-truth release into OUT, a new or empty folder, from the export whose sha256 is `export_sha256` (when that is None: one of PINNED_EXPORTS)."""
    target, data_dir = Path(out), Path(data)
    (export_path, _), (annotations_path, _) = exports(export, annotations)
    inputs = {
        "export": export_path,
        "annotations": annotations_path,
        "wiki": Path(wiki) if wiki else files.data_root() / WIKI,
    }
    if target.exists() and any(target.iterdir()):
        raise SystemExit(f"{target} is not empty; choose a new folder for --out.")
    raw, primary, separate, _, provenance, _, stats, corpus = analyze(
        inputs["export"], inputs["annotations"], data_dir, export_sha256
    )
    target.mkdir(parents=True, exist_ok=True)
    # Unpack only the known flat members: the complete export, kept as evidence.
    for name, content in raw.items():
        assert Path(name).name == name
        (target / EVIDENCE).mkdir(parents=True, exist_ok=True)
        (target / EVIDENCE / name).write_bytes(content)
    files.write_json(target / PRIMARY, primary)
    (target / SEPARATE).write_bytes(raw["irrelevant-candidates.json"])
    files.write_json(target / STATS, stats)
    texts = {name: files.mbdef_gold(name) for name in (GUIDE, REPORT_MD, DATA_README)}
    shutil.copyfile(texts[GUIDE], target / GUIDE)
    build_pdf(primary, provenance, stats, corpus, target, REPORT_DATE_EPOCH)
    shutil.copyfile(texts[REPORT_MD], target / REPORT_MD)
    figures = sorted(files.mbdef_gold(FIGURES).glob("*.png")) + [
        files.mbdef_gold(f"{FIGURES}/README.md")
    ]
    (target / FIGURES).mkdir()
    for figure in figures:
        shutil.copyfile(figure, target / FIGURES / figure.name)

    members = {
        name: target / name
        for name in [PRIMARY, SEPARATE, REPORT, TEX, STATS, GUIDE, REPORT_MD]
    }
    members.update({f"provenance/{name}": target / EVIDENCE / name for name in raw})
    members["provenance/validated-v3.zip"] = inputs["export"]
    members["history/scidef.json"] = data_dir / SCIDEF
    members["history/v2.json"] = data_dir / V2
    members[DATA_README] = texts[DATA_README]
    members["code/groundtruth.py"] = Path(__file__)
    members["code/groundtruth_report.tex"] = TEMPLATE
    for figure in figures:
        members[f"figures/{figure.name}"] = target / FIGURES / figure.name
    if inputs["wiki"].exists():
        members["history/IE-wiki-snapshot.txt"] = inputs["wiki"]
    manifest = {
        "release_id": stats["release_id"],
        "working_name": "MBDef-Gold",  # the key stays for readers of the manifest; no code reads it
        "export_created_at": stats["export_created_at"],
        "source_export_sha256": stats["export_sha256"],
        "scope": "definition Yes and relevance Yes; valid non-relevant definitions separate",
        "schema_version": "3",
        "counts": {
            "relevant_papers": stats["primary"]["paper_entries"],
            "non_relevant_paper_entries": stats["non_relevant"]["paper_entries"],
            "non_relevant_papers_with_records": stats["non_relevant"][
                "nonempty_papers"
            ],
            "primary_records": stats["primary"]["records"],
            "non_relevant_records": stats["non_relevant"]["records"],
        },
        "files": {
            name: {"sha256": files.digest(path), "bytes": path.stat().st_size}
            for name, path in members.items()
        },
        "identity_notes": [
            "Relevant GT omits 20 empty paper entries and sets paper_count to 130; all retained paper objects are unchanged.",
            "Non-relevant JSON and the complete original export remain byte-identical to the source export.",
            "Source record identities and final output-index mapping are in provenance/relevance-labels.json.",
            "The manifest lists content members; SHA256SUMS additionally covers the manifest itself.",
        ],
    }
    files.write_json(target / MANIFEST, manifest)
    members[MANIFEST] = target / MANIFEST
    write_zip(target / BUNDLE, members)

    outputs = {
        p.relative_to(target).as_posix(): p
        for p in sorted(target.rglob("*"))
        if p.is_file()
    }
    receipt = files.receipt(
        "groundtruth build",
        settings={
            "export_sha256": stats["export_sha256"],
            "report_date_epoch": REPORT_DATE_EPOCH,
            "zip_member_time": list(ZIP_MEMBER_TIME),
            "release_id": RELEASE_ID,
        },
        inputs={
            **{name: path for name, path in inputs.items() if path.exists()},
            "v3": data_dir / V3,
            "v2": data_dir / V2,
            "scidef": data_dir / SCIDEF,
            "label_studio_xml": data_dir / LABEL_STUDIO_XML,
            "guide": texts[GUIDE],
            "data_readme": texts[DATA_README],
            "report_md": texts[REPORT_MD],
            "figures": files.mbdef_gold(FIGURES),
        },
        outputs=outputs,
    )
    files.write_json(target / files.RECEIPT, receipt)
    print(
        json.dumps(
            {
                "primary": stats["primary"]["records"],
                "non_relevant": stats["non_relevant"]["records"],
                "primary_definition_statements": stats["primary"]["fields"][
                    "definitions"
                ],
                "changed_primary_records": stats["changed_primary_records"],
                "report": str(target / REPORT),
                "package": str(target / BUNDLE),
                "package_sha256": files.digest(target / BUNDLE),
                "receipt": str(target / files.RECEIPT),
            },
            indent=2,
        )
    )
    return 0
