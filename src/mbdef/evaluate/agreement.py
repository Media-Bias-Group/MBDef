"""The evaluate step: the agreement and curation numbers of the ground truth (paper Table 3).

Read it in four parts, top to bottom:

1. **Votes**: `load_votes` reads every first-round and third-round answer from the raw Label Studio exports and joins it to its V3 record.
2. **Agreement**: `agreement` computes Table 3 (for the first two annotators of each record: N, Yes/Yes, No/No, disagree, % agree, Krippendorff's alpha, for Q1 "is it a definition?" and Q2 "is the concept relevant?") and the numbers a reader asks next: chance agreement, order-free measures, the third round, the curator's final labels, intervals, and cross-checks with the ground-truth code and the `krippendorff` package.
3. **Curation**: `curation` measures what the curator did, from the curator's export.
4. **Figures**: the two figures of the agreement numbers.

The statistics themselves are in `stats`. The settings below are pinned to the released numbers (agreement.json and curation.json), so `agreement_command` with no flags remakes them; docs/step-3-extraction/evaluate.md says what to change for another annotation round. numpy, matplotlib and krippendorff are imported where they are used, so the other commands of the command line run without them.
"""

import json
import random
import re
import zipfile
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

from mbdef import files, groundtruth, stats

# Inputs outside git, under DATA_ROOT (docs/setup/data-root.md; listed in data/data-root.tsv).
ANNOTATIONS = groundtruth.ANNOTATIONS  # the raw Label Studio exports
EXPORT = (
    groundtruth.EXPORT
)  # the curator's export: released labels and curation records

# Inputs in the repository's data folder (--data).
V3 = groundtruth.V3
LABEL_STUDIO_XML = groundtruth.LABEL_STUDIO_XML  # the labelling form
# The agreement table as two earlier outputs stored it, for the cross-check: the annotation audit in the data folder and MBDef-Gold's statistics in the data root.
# Each with the keys that lead to the table.
STORED = {
    "annotation-audit.json": ("statistics", "questions"),
    "validated_gt_stats.json": ("annotation_archive", "questions"),
}
AUDIT = "newAnnotations/annotation-audit.json"  # in --data


def stored_files(data: Path) -> dict[str, Path]:
    """Where the two stored tables are."""
    return {
        "annotation-audit.json": data / AUDIT,
        "validated_gt_stats.json": files.mbdef_gold(groundtruth.STATS),
    }


# Outputs, in --out.
AGREEMENT = "agreement.json"
CURATION = "curation.json"
FIGURES = ("fig1_agreement_tables", "fig2_observed_vs_chance")  # each as .png and .svg
JSON_FORMAT: dict[str, Any] = {
    "indent": 1,
    "ensure_ascii": False,
}  # both JSON files of the released numbers
GENERATED_BY = "mbdef evaluate agreement"

# Pinned settings: the released numbers were made with these.
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEEDS = {
    "by_paper": 1,
    "by_batch": 2,
}  # numpy default_rng; unit = paper, batch
KAPPA_ORDERINGS = 1000  # random rater orderings for the range of Cohen's kappa
KAPPA_SEED = 20261016  # Python random.Random
LARGE_PAIR = (
    100  # an annotator pair with at least this many records gets its alpha range
)
ORGANISERS = (2, 3)  # the two organisers' Label Studio accounts
SVG_HASH_SALT = "taxonomy_pipeline"  # fixed SVG element ids (matplotlib's svg.hashsalt); the salt is part of the recorded SVG bytes, so it keeps its name

STAGES = (
    ("Annotations_Weekly", "weekly"),
    ("Annotation_rest", "rest"),
    ("Annotation_Disagreements", "third"),
)
Q = {"Q1": "definition_is_definition", "Q2": "term_relevant"}
QTEXT = {
    "Q1": "Is a definition?",
    "Q2": "Is the concept relevant for the Media Bias Taxonomy task?",
}
CONTROL = re.compile(r"(definition_is_definition|term_relevant)_(\d+)")

# Table 3 as printed in the paper; the reproduction status compares the recomputed numbers with these.
PAPER = {
    "Q1": {
        "N": 2486,
        "yes_yes": 2095,
        "no_no": 59,
        "disagree": 332,
        "pct_agree": "86.7",
        "alpha": "0.19",
    },
    "Q2": {
        "N": 2488,
        "yes_yes": 737,
        "no_no": 1050,
        "disagree": 701,
        "pct_agree": "71.8",
        "alpha": "0.43",
    },
}

# How each part was computed, as the released numbers say it (stored in agreement.json).
HOW = {
    "design": "Votes read from every JSON under data/newAnnotations/raw/{Annotations_Weekly,"
    "Annotation_rest,Annotation_Disagreements}; joined to V3 records by exact (original_paper_id, "
    "term); exact repeated submissions (same annotator, same task, identical answers) dropped. "
    "'First round' = weekly + rest (completion) stages; 'third round' = Annotation_Disagreements.",
    "status": "[printed value, recomputed value, printed == recomputed rounded to the printed "
    "precision]",
    "n_difference": "A record enters a question's N only if both first-round votes contain that "
    "question's answer. Q1 is missing in exactly one first-round vote of two records; Q2 is never "
    "missing.",
    "third_round": "Third-round votes (Annotation_Disagreements) compared with the two first-round "
    "votes. 'Targeted' = the question listed in the task's disagreement_questions.",
    "final_labels": "Majority = answer given by at least 2 of the 2-3 votes (src/curate.py "
    "consensus rule). Released labels = data/newAnnotations/raw/final-export-20260908/"
    "relevance-labels.json (byte-identical to provenance/relevance-labels.json in "
    "data/validated_gt_release.zip).",
    "approval_basis": "relevance-labels.json approval_basis: 'curator' = individually reviewed by "
    "the curator; 'annotation_policy' = accepted in one batch under policy "
    "curator-accept-annotation-outcomes-20260908.",
    "rater_order": "Cohen's kappa needs a 'rater A' and a 'rater B'. Here the two people differ "
    "from batch to batch, so any A/B assignment is arbitrary; kappa changes with that choice, "
    "alpha does not.",
    "annotators": "Per first-round account: answers given and how many of them disagree with the "
    "other first-round answer on the same record. Per pair: the two first-round accounts of a "
    "record (unordered, lower id first). Alpha per pair is None when it is undefined (only one "
    "answer category used).",
    "alpha_identity": "Krippendorff's alpha with the 11 account identities as columns of a sparse "
    "account x record matrix equals the 'first/second' value exactly, because alpha pools the "
    "values within each record and never uses who gave them.",
    "uncertainty": "Percentile bootstrap, {replicates:,} replicates, resampling whole papers "
    "({papers}) or whole batches ({batches}) with replacement, because records of one batch share "
    "the same two annotators and records of one paper share context. Point estimates are "
    "unchanged.",
}
# The cross-check with the code that first produced the numbers, `groundtruth.AnnotationExports`; the check keeps its stored name.
CORPUS_CHECK = "src/curate.py Corpus.statistics()"
PACKAGE_CHECK = "krippendorff package (sparse account x record matrix)"


# 1. Votes


def load_v3(path: Path) -> tuple[dict, dict]:
    """V3's records by key ('<paper_id>:<index>'), and the key of each (paper_id, term)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    records, term_index = {}, {}
    for paper in data["papers"]:
        for i, rec in enumerate(paper["definition_records"]):
            key = f"{paper['paper_id']}:{i}"
            term_index[(paper["paper_id"], rec["term"])] = key
            records[key] = {
                "key": key,
                "paper_id": paper["paper_id"],
                "index": i,
                "term": rec["term"],
                "split": paper["split"],
                "title": paper["metadata"]["title"],
                "definitions": [d["text"] for d in rec["definitions"]],
            }
    return records, term_index


def load_votes(annotations: Path, term_index: dict):
    """One vote = one annotator's answers for one record. Exact repeated submissions are dropped.

    The exports are the JSON files in the zip's stage folders, read stage by stage and file by file in name order.
    """
    votes, inventory, duplicates, third_targets = [], [], [], {}
    with zipfile.ZipFile(annotations) as archive:
        names = archive.namelist()
        for folder, stage in STAGES:
            file_set = sorted(
                n
                for n in names
                if n.startswith(folder + "/")
                and n.endswith(".json")
                and n.count("/") == 1
            )
            for file in file_set:
                tasks = json.loads(archive.read(file))
                inventory.append(
                    {
                        "file": file,
                        "stage": stage,
                        "tasks": len(tasks),
                        "submissions": sum(len(t["annotations"]) for t in tasks),
                    }
                )
                for task in tasks:
                    read_task(task, stage, file, term_index, votes, duplicates)
                    if stage == "third":
                        mapping = task_mapping(task, term_index)
                        for r in task["data"]["definition_records"]:
                            third_targets[mapping[r["idx"]]] = sorted(
                                r["disagreement_questions"]
                            )
    return votes, inventory, duplicates, third_targets


def task_mapping(task: dict, term_index: dict) -> dict:
    """The task's local record index -> V3 key."""
    d = task["data"]
    pid = d.get("original_paper_id", d["paper_id"].split("__")[0])
    return {r["idx"]: term_index[(pid, r["term"])] for r in d["definition_records"]}


def read_task(task, stage, file, term_index, votes, duplicates) -> None:
    """Add one vote per record and submission of TASK; drop exact repeats and cancellations."""
    d = task["data"]
    mapping = task_mapping(task, term_index)
    batch = d.get("source_paper_id", d["paper_id"])  # third-round tasks point back
    seen = {}
    for ann in task["annotations"]:
        if ann["was_cancelled"]:
            continue
        sig = (
            ann["completed_by"],
            json.dumps(
                sorted(
                    (r["from_name"], json.dumps(r["value"], sort_keys=True))
                    for r in ann["result"]
                )
            ),
        )
        if sig in seen:
            duplicates.append(
                {
                    "file": file,
                    "task": task["id"],
                    "annotation": ann["id"],
                    "same_as": seen[sig],
                    "annotator": ann["completed_by"],
                }
            )
            continue
        seen[sig] = ann["id"]
        answers = defaultdict(dict)
        controls = defaultdict(list)
        for r in ann["result"]:
            m = re.search(r"_(\d+)$", r["from_name"])
            ex = re.fullmatch(r"example_(\d+)_\d+_.+", r["from_name"])
            if ex:
                idx = int(ex[1])
            elif m:
                idx = int(m[1])
            else:
                raise ValueError(f"Unrecognized annotation control: {r['from_name']}")
            controls[idx].append(r["from_name"])
            cm = CONTROL.fullmatch(r["from_name"])
            if cm:
                answers[idx][cm[1]] = r["value"]["choices"][0]
        for idx, key in mapping.items():
            votes.append(
                {
                    "key": key,
                    "annotator": ann["completed_by"],
                    "stage": stage,
                    "project": task["project"],
                    "task": task["id"],
                    "annotation": ann["id"],
                    "file": file,
                    "batch": batch,
                    "created_at": ann["created_at"],
                    "Q1": answers[idx].get(Q["Q1"]),
                    "Q2": answers[idx].get(Q["Q2"]),
                    "controls": sorted(controls[idx]),
                }
            )


# 2. Agreement


def agreement(
    annotations: Path,
    export: Path,
    data: Path,
) -> dict:
    """Table 3 and the numbers around it, from the raw exports, V3 and the released labels."""
    records, term_index = load_v3(data / V3)
    votes, inventory, duplicates, third_targets = load_votes(annotations, term_index)
    by_rec = defaultdict(list)
    for v in votes:
        by_rec[v["key"]].append(v)
    initial = {
        k: sorted(
            [v for v in vs if v["stage"] != "third"], key=lambda v: v["annotator"]
        )
        for k, vs in by_rec.items()
    }
    third = {k: [v for v in vs if v["stage"] == "third"] for k, vs in by_rec.items()}
    out: dict[str, Any] = {}

    # ---- design: who annotated what
    n_initial = Counter(len(v) for v in initial.values())
    n_third = Counter(len(v) for v in third.values())
    n_people = Counter(len({v["annotator"] for v in by_rec[k]}) for k in records)
    stage_combo = Counter(
        "+".join(sorted(v["stage"] for v in initial[k])) for k in records
    )
    first_round_accounts = sorted(
        {v["annotator"] for vs in initial.values() for v in vs}
    )
    third_accounts = sorted({v["annotator"] for vs in third.values() for v in vs})
    judgments = Counter(v["annotator"] for vs in initial.values() for v in vs)
    organisers = sum(judgments[a] for a in ORGANISERS)
    batches = {v["batch"] for vs in initial.values() for v in vs}
    out["design"] = {
        "how": HOW["design"],
        "v3_records": len(records),
        "records_with_votes": len(by_rec),
        "export_inventory": inventory,
        "duplicate_submissions_dropped": duplicates,
        "first_round_votes_per_record": dict(n_initial),
        "third_round_votes_per_record": dict(n_third),
        "distinct_annotators_per_record": dict(n_people),
        "first_round_stage_combination": dict(stage_combo),
        "first_round_judgments_total": sum(judgments.values()),
        "first_round_accounts": first_round_accounts,
        "third_round_accounts": third_accounts,
        "accounts_total": len(set(first_round_accounts) | set(third_accounts)),
        "first_round_judgments_by_account": {
            str(a): judgments[a] for a in first_round_accounts
        },
        "accounts_2_and_3_share": {
            "judgments": organisers,
            "of": sum(judgments.values()),
            "share": organisers / sum(judgments.values()),
            "note": "Paper: 'Two organisers supplied 47% of these judgments'.",
        },
        "batches_first_round": len(batches),
        "records_routed_to_third_round": len(third_targets),
    }

    # ---- the table
    table: dict[str, Any] = {}
    pairs_by_q: dict[str, list] = {}
    missing: dict[str, list] = {}
    for q in ("Q1", "Q2"):
        pairs, miss = [], []
        for k in records:
            a, b = (v[q] for v in initial[k])
            if a is None or b is None:
                miss.append(k)
            else:
                pairs.append((k, a, b))
        pairs_by_q[q] = pairs
        missing[q] = miss
        yy, nn, dis = stats.pair_counts([(a, b) for _, a, b in pairs])
        nums = stats.two_rater_numbers(yy, nn, dis)
        general = stats.krippendorff_alpha([[a, b] for _, a, b in pairs])
        sparse = stats.krippendorff_alpha(
            [[v[q] for v in initial[k]] for k in records]
        )  # incl. the 1-answer records
        table[q] = {
            "question": QTEXT[q],
            **nums,
            "alpha_general_coincidence_matrix": general["alpha"],
            "alpha_all_records_incomplete_units_dropped_by_alpha_itself": sparse[
                "alpha"
            ],
            "pct_agree_1dp": f"{100 * nums['observed_agreement']:.1f}",
            "pct_agree_2dp": f"{100 * nums['observed_agreement']:.2f}",
            "alpha_2dp": f"{nums['krippendorff']['alpha']:.2f}",
            "records_missing_an_answer": miss,
        }
    out["table"] = table

    # ---- reproduction status vs the paper
    status: dict[str, Any] = {}
    for q in ("Q1", "Q2"):
        t, p = table[q], PAPER[q]
        status[q] = {
            "N": [p["N"], t["N"], p["N"] == t["N"]],
            "Yes/Yes": [p["yes_yes"], t["yes_yes"], p["yes_yes"] == t["yes_yes"]],
            "No/No": [p["no_no"], t["no_no"], p["no_no"] == t["no_no"]],
            "Disagree": [p["disagree"], t["disagree"], p["disagree"] == t["disagree"]],
            "% agree": [
                p["pct_agree"],
                f"{100 * t['observed_agreement']:.4f}",
                p["pct_agree"] == t["pct_agree_1dp"],
            ],
            "alpha": [
                p["alpha"],
                f"{t['krippendorff']['alpha']:.4f}",
                p["alpha"] == t["alpha_2dp"],
            ],
            "row_sums_to_N": t["yes_yes"] + t["no_no"] + t["disagree"] == t["N"],
        }
    status["how"] = HOW["status"]
    out["reproduction_status"] = status

    # ---- the records behind the difference between the two N
    form = data / LABEL_STUDIO_XML
    xml_evidence = []
    for i, line in enumerate(form.read_text(encoding="utf-8").splitlines(), 1):
        if re.search(
            r'<Choices name="(definition_is_definition|term_relevant)_\{\{idx\}\}"',
            line,
        ):
            xml_evidence.append(
                {
                    "file": form.name,
                    "line": i,
                    "control": line.split('"')[1],
                    "required": 'required="true"' in line,
                    "text": line.strip(),
                }
            )
    detail: list[dict[str, Any]] = []
    for k in missing["Q1"]:
        r = records[k]
        detail.append(
            {
                "record": k,
                "term": r["term"],
                "paper_title": r["title"],
                "split": r["split"],
                "definition_shown": r["definitions"],
                "votes": [
                    {
                        x: v[x]
                        for x in (
                            "annotator",
                            "stage",
                            "project",
                            "task",
                            "annotation",
                            "file",
                            "batch",
                            "created_at",
                            "Q1",
                            "Q2",
                            "controls",
                        )
                    }
                    for v in initial[k]
                ],
                "routed_to_third_round": k in third_targets,
                "released_labels": None,
            }
        )
    with zipfile.ZipFile(export) as archive:
        released_labels = json.loads(archive.read("relevance-labels.json"))
    released = {x["source_key"]: x for x in released_labels}
    for d in detail:
        x = released[d["record"]]
        d["released_labels"] = {
            "definition_is_definition": x["definition_is_definition"],
            "term_relevant": x["term_relevant"],
            "retained": x["retained"],
        }
    q1_missing_all = Counter((v["stage"]) for v in votes if v["Q1"] is None)
    q2_missing_all = Counter((v["stage"]) for v in votes if v["Q2"] is None)
    out["n_difference"] = {
        "how": HOW["n_difference"],
        "q1_N": table["Q1"]["N"],
        "q2_N": table["Q2"]["N"],
        "records_without_a_complete_q1_pair": detail,
        "missing_q1_answers_by_stage": dict(q1_missing_all),
        "missing_q2_answers_by_stage": dict(q2_missing_all),
        "label_studio_form": xml_evidence,
    }

    # ---- third round
    th: dict[str, Any] = {"how": HOW["third_round"]}
    union = [
        k
        for k in records
        if any(len({v[q] for v in initial[k]} - {None}) == 2 for q in ("Q1", "Q2"))
    ]
    th["records_with_first_round_disagreement_on_Q1_or_Q2"] = len(union)
    th["routing_matches_disagreements_exactly"] = set(union) == set(
        third_targets
    ) and all(
        set(third_targets[k])
        == {Q[q] for q in ("Q1", "Q2") if len({v[q] for v in initial[k]} - {None}) == 2}
        for k in union
    )
    th["records_disputed_on_both"] = sum(
        len(third_targets[k]) == 2 for k in third_targets
    )
    for q in ("Q1", "Q2"):
        targeted = [k for k in third_targets if Q[q] in third_targets[k]]
        third_ans = Counter(third[k][0][q] for k in targeted)
        other = [k for k in third_targets if Q[q] not in third_targets[k]]
        cross = Counter()
        for k in other:
            agreed = initial[k][0][q]
            cross[(agreed, third[k][0][q])] += 1
        agree_other = sum(c for (a, b), c in cross.items() if a == b)
        th[q] = {
            "disputed_records": len(targeted),
            "third_annotator_answer_on_disputed": dict(third_ans),
            "missing_third_answer_on_disputed": third_ans.get(None, 0),
            "not_disputed_but_reanswered_records": len(other),
            "first_round_agreed_vs_third_answer": {
                f"agreed {a} / third {b}": c
                for (a, b), c in sorted(
                    cross.items(), key=lambda x: (str(x[0][0]), str(x[0][1]))
                )
            },
            "third_agrees_with_first_round_agreement": agree_other,
            "third_agrees_share": agree_other / len(other) if other else None,
        }
    # pooled alpha including the third round (shown only to explain why it is NOT reported)
    for q in ("Q1", "Q2"):
        all3 = stats.krippendorff_alpha([[v[q] for v in by_rec[k]] for k in records])
        sub3 = stats.krippendorff_alpha(
            [[v[q] for v in by_rec[k]] for k in third_targets]
        )
        th[q]["alpha_if_third_round_pooled_in_all_records"] = all3["alpha"]
        th[q]["alpha_three_raters_on_the_931_disputed_records_only"] = sub3["alpha"]
    out["third_round"] = th

    # ---- final labels: majority vs released curator labels
    fin: dict[str, Any] = {"how": HOW["final_labels"]}
    for q in ("Q1", "Q2"):
        trans = Counter()
        override = []
        for k in records:
            maj = groundtruth.majority([v[q] for v in by_rec[k]])
            rel = released[k][Q[q]]
            trans[(maj, rel)] += 1
            if maj is not None and rel != maj:
                override.append(
                    {
                        "record": k,
                        "term": records[k]["term"],
                        "majority": maj,
                        "released": rel,
                        "first_round": [v[q] for v in initial[k]],
                        "third": [v[q] for v in third[k]],
                    }
                )
        fin[q] = {
            "majority_vs_released": {
                f"majority {a} -> released {b}": c
                for (a, b), c in sorted(
                    trans.items(), key=lambda x: (str(x[0][0]), str(x[0][1]))
                )
            },
            "curator_overrides": override,
        }
    fates = Counter()
    for k in records:
        x = released[k]
        if x["retained"]:
            fates["BiasDef130 (definition and relevant)"] += 1
        elif x["definition_is_definition"] == "Yes":
            fates["non-relevant file (definition, not relevant)"] += 1
        elif x["definition_is_definition"] == "No":
            fates["not a definition"] += 1
        else:
            fates[f"definition label {x['definition_is_definition']!r}"] += 1
    fin["release_fates"] = dict(fates)
    fin["approval_basis"] = {
        "how": HOW["approval_basis"],
        "disputed_records": dict(
            Counter(released[k]["approval_basis"] for k in third_targets)
        ),
        "disputed_records_retained_in_BiasDef130": dict(
            Counter(
                released[k]["approval_basis"]
                for k in third_targets
                if released[k]["retained"]
            )
        ),
        "not_disputed_records": dict(
            Counter(
                released[k]["approval_basis"] for k in records if k not in third_targets
            )
        ),
    }
    out["final_labels"] = fin

    # ---- rater-order dependence (why Cohen's kappa is not used) and order-free measures
    order: dict[str, Any] = {}
    for q in ("Q1", "Q2"):
        recs = {k: (initial[k][0], initial[k][1]) for k, _, _ in pairs_by_q[q]}
        by_id = [
            (a[q], b[q]) for a, b in recs.values()
        ]  # lower account id first (the curation code's sort order)
        by_time = [
            tuple(v[q] for v in sorted(p, key=lambda v: v["created_at"]))
            for p in recs.values()
        ]
        by_stage = [
            tuple(
                v[q]
                for v in sorted(
                    p, key=lambda v: (v["stage"] != "weekly", v["created_at"])
                )
            )
            for p in recs.values()
        ]
        rnd = random.Random(KAPPA_SEED)
        kappas = []
        for _ in range(KAPPA_ORDERINGS):
            flipped = [(a, b) if rnd.random() < 0.5 else (b, a) for a, b in by_id]
            kappas.append(stats.cohen_kappa(flipped)[0])
        order[q] = {
            "cohen_kappa_lower_account_id_first": stats.cohen_kappa(by_id)[0],
            "cohen_kappa_earlier_submission_first": stats.cohen_kappa(by_time)[0],
            "cohen_kappa_weekly_before_completion": stats.cohen_kappa(by_stage)[0],
            "cohen_kappa_1000_random_orderings_min_max": [min(kappas), max(kappas)],
            "table_2x2_earlier_submission_first": {
                "first Yes / second Yes": sum(a == b == "Yes" for a, b in by_time),
                "first Yes / second No": sum(
                    a == "Yes" and b == "No" for a, b in by_time
                ),
                "first No / second Yes": sum(
                    a == "No" and b == "Yes" for a, b in by_time
                ),
                "first No / second No": sum(a == b == "No" for a, b in by_time),
            },
        }
    order["how"] = HOW["rater_order"]
    out["rater_order"] = order

    # ---- per annotator and per annotator pair
    per_ann: dict[str, Any] = {}
    for a in first_round_accounts:
        mine = [(k, v) for k in records for v in initial[k] if v["annotator"] == a]
        row: dict[str, Any] = {"record_judgments": len(mine)}
        for q in ("Q1", "Q2"):
            ans = Counter(v[q] for _, v in mine)
            dis = sum(
                1
                for k, v in mine
                if v[q] is not None
                and any(
                    o[q] is not None and o[q] != v[q] for o in initial[k] if o is not v
                )
            )
            partner = [next(o[q] for o in initial[k] if o is not v) for k, v in mine]
            no_matched = sum(
                v[q] == "No" and p == "No"
                for (_, v), p in zip(mine, partner, strict=True)
            )
            row[q] = {
                "yes": ans["Yes"],
                "no": ans["No"],
                "missing": ans[None],
                "share_yes": ans["Yes"] / (ans["Yes"] + ans["No"]),
                "in_disagreement": dis,
                "no_answers_matched_by_partner": no_matched,
                "share_of_all_first_round_no_answers": ans["No"]
                / table[q]["no_answers"],
            }
        per_ann[str(a)] = row
    pair_rows: defaultdict[str, dict[str, Any]] = defaultdict(
        lambda: {"records": 0, "batches": set(), "Q1": [], "Q2": []}
    )
    for k in records:
        a, b = initial[k]
        key = f"{a['annotator']}+{b['annotator']}"
        pr = pair_rows[key]
        pr["records"] += 1
        pr["batches"].add(a["batch"])
        for q in ("Q1", "Q2"):
            if a[q] is not None and b[q] is not None:
                pr[q].append((a[q], b[q]))
    pairs_out: dict[str, Any] = {}
    for key, pr in sorted(pair_rows.items(), key=lambda x: -x[1]["records"]):
        cells: dict[str, Any] = {
            "records": pr["records"],
            "batches": len(pr["batches"]),
        }
        for q in ("Q1", "Q2"):
            yy, nn, dis = stats.pair_counts(pr[q])
            cells[q] = {
                "N": len(pr[q]),
                "yes_yes": yy,
                "no_no": nn,
                "disagree": dis,
                "observed_agreement": (yy + nn) / len(pr[q]),
                "alpha": stats.alpha_from_counts(yy, nn, dis)
                if (yy + dis) and (nn + dis)
                else None,
            }
        pairs_out[key] = cells
    big = {k: v for k, v in pairs_out.items() if v["records"] >= LARGE_PAIR}
    out["annotators"] = {
        "how": HOW["annotators"],
        "per_account": per_ann,
        "distinct_pairs": len(pairs_out),
        "pairs_with_at_least_100_records": len(big),
        "records_in_pairs_with_at_least_100_records": sum(
            v["records"] for v in big.values()
        ),
        "pairs_with_undefined_alpha": {
            q: sum(v[q]["alpha"] is None for v in pairs_out.values())
            for q in ("Q1", "Q2")
        },
        "alpha_range_in_pairs_with_at_least_100_records": {
            q: [
                min(v[q]["alpha"] for v in big.values()),
                max(v[q]["alpha"] for v in big.values()),
            ]
            if big
            else None
            for q in ("Q1", "Q2")
        },
        "per_pair": pairs_out,
        "alpha_identity_matrix_note": HOW["alpha_identity"],
    }

    # ---- step-by-step arithmetic (as printed in explanation.md)
    arith: dict[str, Any] = {}
    for q in ("Q1", "Q2"):
        t = table[q]
        yy, nn, dis, n, m = (
            t["yes_yes"],
            t["no_no"],
            t["disagree"],
            t["N"],
            t["answers_total"],
        )
        yes, no = t["yes_answers"], t["no_answers"]
        kd = t["krippendorff"]
        arith[q] = [
            f"records N = {yy} + {nn} + {dis} = {n}",
            f"observed agreement = ({yy} + {nn}) / {n} = {yy + nn} / {n} = {t['observed_agreement']:.6f}",
            f"answers M = 2 x {n} = {m}; Yes answers = 2 x {yy} + {dis} = {yes}; No answers = 2 x {nn} + {dis} = {no}",
            f"share of Yes answers = {yes} / {m} = {yes / m:.6f}",
            f"observed disagreement Do = {dis} / {n} = {kd['observed_disagreement_Do']:.6f}",
            f"expected disagreement De = 2 x {yes} x {no} / ({m} x {m - 1}) = {2 * yes * no} / {m * (m - 1)} "
            f"= {kd['expected_disagreement_De']:.6f}",
            f"chance agreement = 1 - De = {kd['chance_agreement_1_minus_De']:.6f}",
            f"alpha = 1 - Do / De = 1 - {kd['observed_disagreement_Do']:.6f} / {kd['expected_disagreement_De']:.6f} "
            f"= {kd['alpha']:.6f}",
            f"same thing as (observed - chance) / (1 - chance) = ({t['observed_agreement']:.6f} - "
            f"{kd['chance_agreement_1_minus_De']:.6f}) / {kd['expected_disagreement_De']:.6f} = {kd['alpha']:.6f}",
        ]
    out["arithmetic"] = arith

    # ---- the paper's claims around the table, recomputed
    out["paper_claims"] = {
        "first_round_accounts (paper: 11 annotator accounts)": len(
            first_round_accounts
        ),
        "third_round_accounts (paper: 5 further accounts, 16 in total)": [
            len(third_accounts),
            out["design"]["accounts_total"],
        ],
        "share_accounts_2_3 (paper: 47%)": round(
            100 * organisers / sum(judgments.values()), 2
        ),
        "third_round_records (paper: 931 of 2,488, 37.4%)": [
            len(third_targets),
            round(100 * len(third_targets) / len(records), 2),
        ],
        "q1_pct_agree_exact (paper: 86.7)": round(
            100 * table["Q1"]["observed_agreement"], 4
        ),
        "q1_disagreement_share": round(
            100 * table["Q1"]["disagree"] / table["Q1"]["N"], 2
        ),
        "q1_no_answers_without_a_matching_no": [
            table["Q1"]["no_answers"] - 2 * table["Q1"]["no_no"],
            table["Q1"]["no_answers"],
        ],
        "q1_records_with_any_no_that_are_no_no": [
            table["Q1"]["no_no"],
            table["Q1"]["records_with_at_least_one_no"],
        ],
    }

    # ---- uncertainty
    unc: dict[str, Any] = {
        "how": HOW["uncertainty"].format(
            replicates=BOOTSTRAP_REPLICATES,
            papers=len({r["paper_id"] for r in records.values()}),
            batches=len(batches),
        )
    }
    for q in ("Q1", "Q2"):
        by_paper, by_batch = defaultdict(list), defaultdict(list)
        for k, a, b in pairs_by_q[q]:
            by_paper[records[k]["paper_id"]].append((a, b))
            by_batch[initial[k][0]["batch"]].append((a, b))
        unc[q] = {
            "by_paper": stats.cluster_bootstrap(
                by_paper, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEEDS["by_paper"]
            ),
            "by_batch": stats.cluster_bootstrap(
                by_batch, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEEDS["by_batch"]
            ),
        }
    out["uncertainty"] = unc

    # ---- cross-checks
    cross: dict[str, Any] = {}
    for name, path in stored_files(data).items():
        keys = STORED[name]
        stored = json.loads(path.read_text(encoding="utf-8"))
        for kk in keys:
            stored = stored[kk]
        cross[name] = {
            q: {
                "complete_pairs": stored[Q[q]]["complete_pairs"] == table[q]["N"],
                "yes_yes": stored[Q[q]]["yes_yes"] == table[q]["yes_yes"],
                "no_no": stored[Q[q]]["no_no"] == table[q]["no_no"],
                "disagree": stored[Q[q]]["disagree"] == table[q]["disagree"],
                "alpha_abs_diff": abs(
                    stored[Q[q]]["alpha_nominal"] - table[q]["krippendorff"]["alpha"]
                ),
            }
            for q in ("Q1", "Q2")
        }
    try:  # the code that first produced the numbers
        orig = groundtruth.AnnotationExports(annotations, data / V3).statistics()[
            "questions"
        ]
        cross[CORPUS_CHECK] = {
            q: {
                "complete_pairs": orig[Q[q]]["complete_pairs"],
                "yes_yes": orig[Q[q]]["yes_yes"],
                "no_no": orig[Q[q]]["no_no"],
                "disagree": orig[Q[q]]["disagree"],
                "alpha_nominal": orig[Q[q]]["alpha_nominal"],
                "alpha_abs_diff": abs(
                    orig[Q[q]]["alpha_nominal"] - table[q]["krippendorff"]["alpha"]
                ),
            }
            for q in ("Q1", "Q2")
        }
    except Exception as exc:  # an export the curation code rejects still gets its table
        print(f"Warning: the cross-check with {CORPUS_CHECK} did not run: {exc!r}")
        cross[CORPUS_CHECK] = f"not run: {exc!r}"
    cross[PACKAGE_CHECK] = package_alpha(records, votes)
    out["cross_checks"] = cross
    return out


def package_alpha(records: dict, votes: list[dict]) -> dict:
    """Alpha from the `krippendorff` package, on the account x record matrix of answers."""
    import krippendorff
    import numpy as np

    keys = sorted(records)
    col = {k: i for i, k in enumerate(keys)}
    ref: dict[str, Any] = {"package": f"krippendorff {version('krippendorff')}"}
    for stages, label in (
        (("weekly", "rest"), "first_round_accounts_x_records"),
        (("weekly", "rest", "third"), "all_rounds_pooled_accounts_x_records"),
    ):
        accts = sorted({v["annotator"] for v in votes if v["stage"] in stages})
        row = {a: i for i, a in enumerate(accts)}
        for q in ("Q1", "Q2"):
            mat = np.full((len(accts), len(keys)), np.nan)
            for v in votes:
                if v["stage"] in stages and v[q] is not None:
                    mat[row[v["annotator"]], col[v["key"]]] = (
                        1.0 if v[q] == "Yes" else 0.0
                    )
            ref[f"{label}_{q}"] = {
                "accounts": len(accts),
                "alpha": float(
                    krippendorff.alpha(
                        reliability_data=mat, level_of_measurement="nominal"
                    )
                ),
            }
    return ref


# 3. Curation

RECORD_FIELDS = ["term", *groundtruth.FIELDS]
CURATION_MEMBERS = (
    "curation-provenance.json",  # per-record decisions, annotator answers, edit history
    "relevance-labels.json",  # approval basis and final labels
    "curation-suggestions.json",  # the assistant's proposals
    "final-touches.json",  # the final audit
    "original-v3.json",  # before
    "validated.json",  # after
)
WORKED_EXAMPLES = 5  # records whose note led to an accepted change, one per field


def items(rec: dict, field: str) -> list:
    v = rec.get(field)
    if field == "term":
        return [v or ""]
    return [(i.get("text") or "", tuple(i.get("citation") or [])) for i in (v or [])]


def field_change(before: dict, after: dict, field: str) -> dict | None:
    """None, or how the field changed: items added, removed, text or citation edited."""
    b, a = items(before, field), items(after, field)
    if b == a:
        return None
    if field == "term":
        return {"edited": 1}
    bt, at = Counter(b), Counter(a)
    removed, added = sum((bt - at).values()), sum((at - bt).values())
    edited = min(removed, added)
    return {"edited": edited, "added": added - edited, "removed": removed - edited}


def action_kind(text: str) -> str:
    if text.startswith("Accepted assistant recommendation"):
        return "accepted the assistant's proposed fix"
    if text.startswith("Curator chose the original V3 value"):
        return "kept the original text after considering the note"
    return "other: " + text[:60]


def curation(export: Path, export_name: str) -> dict:
    """What the curator did, measured from the curator's export (named EXPORT_NAME)."""
    with zipfile.ZipFile(export) as archive:
        prov, labels_list, sugg, final, v3, val = (
            json.loads(archive.read(name)) for name in CURATION_MEMBERS
        )
    labels = {r["source_key"]: r for r in labels_list}
    dec, ann, hist = prov["decisions"], prov["annotations"], prov["history"]
    before = {
        f"{p['paper_id']}:{i}": r
        for p in v3["papers"]
        for i, r in enumerate(p["definition_records"])
    }
    after_by_pos = {
        (p["paper_id"], i): r
        for p in val["papers"]
        for i, r in enumerate(p["definition_records"])
    }
    keys = sorted(dec)
    if not set(keys) == set(before) == set(labels):
        raise SystemExit(
            "The curator's export does not cover V3: its decisions, its labels and "
            "original-v3.json name different records."
        )
    basis = Counter(labels[k]["approval_basis"] for k in keys)
    curated = [k for k in keys if labels[k]["approval_basis"] == "curator"]
    retained = [k for k in keys if labels[k]["retained"]]
    out: dict[str, Any] = {"export": export_name, "records": len(keys)}

    # 1. workload
    out["workload"] = {
        "individually_reviewed_by_curator": len(curated),
        "of_these_supporting_fields_reviewed": sum(
            dec[k]["supporting_fields_reviewed"] for k in curated
        ),
        "of_these_edited_at_least_once": sum(dec[k]["revision"] > 1 for k in curated),
        "of_these_retained_in_biasdef130": sum(labels[k]["retained"] for k in curated),
        "accepted_under_policy": basis["annotation_policy"],
        "policy_retained_unchanged_v3": sum(
            labels[k]["retained"]
            for k in keys
            if labels[k]["approval_basis"] == "annotation_policy"
        ),
        "policy_not_retained": sum(
            not labels[k]["retained"]
            for k in keys
            if labels[k]["approval_basis"] == "annotation_policy"
        ),
        "policy_text": final["policy"],
        "revisions_per_edited_record": dict(
            sorted(
                Counter(
                    dec[k]["revision"] for k in keys if dec[k]["revision"] > 1
                ).items()
            )
        ),
        "biasdef130_records": len(retained),
        "biasdef130_individually_curated": sum(
            labels[k]["approval_basis"] == "curator" for k in retained
        ),
    }

    # 2. labels the curator set directly
    direct = [k for k in keys if dec[k].get("label_choice") is None]
    out["labels"] = {
        "label_set_by_curator_not_by_vote": len(direct),
        "confirmed_yes_yes": sum(
            dec[k]["labels"]
            == {"definition_is_definition": "Yes", "term_relevant": "Yes"}
            for k in direct
        ),
        "overrides": [],
    }
    for k in direct:
        first = [x for x in ann[k] if x["stage"] != "third"]
        votes = [
            (x.get("answers", {}).get("definition_is_definition") or [None])[0]
            for x in ann[k]
        ]
        if (
            dec[k]["labels"]["definition_is_definition"] == "No"
            and votes.count("Yes") >= 2
        ):
            notes = [
                {"annotator": x["annotator"], "stage": x["stage"], "note": t}
                for x in ann[k]
                for q, v in x["answers"].items()
                if q.endswith("_text")
                for t in v
                if str(t).strip()
            ]
            out["labels"]["overrides"].append(
                {
                    "record": k,
                    "term": before[k]["term"],
                    "definition": [d["text"] for d in before[k]["definitions"]],
                    "votes_is_definition": votes,
                    "first_round": len(first),
                    "annotator_notes": notes,
                    "curator_note": dec[k].get("note") or None,
                    "history": [
                        {
                            "revision": h["revision"],
                            "status": h["status"],
                            "is_definition": h["labels"]["definition_is_definition"],
                            "updated_at": h["updated_at"],
                        }
                        for h in sorted(hist[k], key=lambda h: h["revision"])
                    ],
                }
            )

    # 3. annotator notes and what was done with them
    notes = [
        (k, x["annotator"], x["stage"], q[: -len("_text")], t)
        for k in keys
        for x in ann[k]
        for q, v in x["answers"].items()
        if q.endswith("_text")
        for t in v
        if str(t).strip()
    ]
    note_recs = sorted({n[0] for n in notes})
    acted = {k for k in keys if dec[k].get("note_actions")}
    actions = [(k, v) for k in keys for v in dec[k].get("note_actions", {}).values()]

    def outcome(k: str) -> str:
        return (
            "individually curated"
            if labels[k]["approval_basis"] == "curator"
            else "accepted under the policy"
        ) + (", in BiasDef130" if labels[k]["retained"] else ", not in BiasDef130")

    out["notes"] = {
        "free_text_notes": len(notes),
        "records_with_notes": len(note_recs),
        "by_field": dict(
            Counter(
                n[3]
                .replace("term_definition_error", "definition")
                .replace("_error", "")
                for n in notes
            ).most_common()
        ),
        "records_with_notes_by_outcome": dict(Counter(outcome(k) for k in note_recs)),
        "records_with_a_recorded_note_action": len(acted),
        "note_actions": len(actions),
        "note_actions_by_kind": dict(
            Counter(action_kind(v) for _, v in actions).most_common()
        ),
        "notes_in_records_without_action_by_outcome": dict(
            Counter(outcome(k) for k in note_recs if k not in acted)
        ),
    }

    # 4. field decisions and changes against V3
    choices = [
        (k, f, v) for k in keys for f, v in dec[k].get("field_choices", {}).items()
    ]
    out["field_choices"] = {
        "total": len(choices),
        "records": len({c[0] for c in choices}),
        "by_source": dict(Counter(v for _, _, v in choices).most_common()),
        "by_field": dict(Counter(f for _, f, _ in choices).most_common()),
        "source_meaning": {
            "assistant": "the curator accepted the assistant's proposed text",
            "manual": "the curator wrote the text",
            "v3": "the curator kept the original V3 text",
        },
    }
    changed, per_field = [], Counter()
    per_kind = Counter()
    for k in retained:
        r = labels[k]
        after = after_by_pos.get((r["paper_id"], r["output_record_index"]))
        if (
            after is None
            or after["term"].strip().casefold()
            != dec[k]["record"]["term"].strip().casefold()
        ):
            raise SystemExit(f"The kept record {k} is not where its label puts it.")
        diff = {f: c for f in RECORD_FIELDS if (c := field_change(before[k], after, f))}
        if diff:
            changed.append((k, diff))
            for f, c in diff.items():
                per_field[f] += 1
                for kind, n in c.items():
                    per_kind[(f, kind)] += n
    out["changes_vs_v3_in_biasdef130"] = {
        "records": len(retained),
        "changed": len(changed),
        "unchanged": len(retained) - len(changed),
        "changed_by_basis": dict(
            Counter(labels[k]["approval_basis"] for k, _ in changed)
        ),
        "records_changed_per_field": dict(per_field.most_common()),
        "items_per_field": {
            f: {
                kind: per_kind[(f, kind)]
                for kind in ("edited", "added", "removed")
                if per_kind[(f, kind)]
            }
            for f in RECORD_FIELDS
            if per_field[f]
        },
        "final_audit_says_changed": final["summary"]["changed_records"],
    }

    # The audit counted its changes before its final touches: the records changed only by a final touch explain the gap.
    def norm(r: dict) -> dict:
        return {f: items(r, f) for f in RECORD_FIELDS}

    only_touch = [
        k
        for k, x in final["records"].items()
        if norm(x["base_record"]) == norm(before[k])
    ]
    out["changes_vs_v3_in_biasdef130"]["changed_only_by_a_final_touch"] = len(
        only_touch
    )
    if final["summary"]["changed_records"] + len(only_touch) != len(changed):
        raise SystemExit(
            "The final audit's changed records plus those changed only by a final touch "
            f"({final['summary']['changed_records']} + {len(only_touch)}) are not the "
            f"{len(changed)} changed kept records."
        )

    # 5. the assistant's proposals
    recs = sugg["records"]
    batch2 = (
        set(sugg["review_batches"][0]["record_keys"])
        if sugg.get("review_batches")
        else set()
    )
    out["assistant"] = {
        "who": sugg["author"],
        "method": sugg["method"],
        "records_with_a_proposal": len(recs),
        "second_batch_records": len(batch2),
        "proposal_records_retained": sum(
            labels.get(k, {}).get("retained", False) for k in recs
        ),
        "fields_where_curator_accepted_assistant_text": out["field_choices"][
            "by_source"
        ].get("assistant", 0),
        "fields_where_curator_kept_v3_instead": out["field_choices"]["by_source"].get(
            "v3", 0
        ),
        "fields_curator_wrote_by_hand": out["field_choices"]["by_source"].get(
            "manual", 0
        ),
    }

    # 6. final audit
    s = final["summary"]
    out["final_audit"] = {
        k: s[k]
        for k in (
            "human_reviewed",
            "changed_records",
            "source_checks",
            "final_touch_records",
            "final_touch_categories",
            "pending_single_or_missing_fit_units",
            "pending_single_or_missing_fit_records",
            "policy_acceptances",
        )
    }
    out["final_audit"]["primary_field_items_checked_against_pdf_text"] = sum(
        s["primary_fields"].values()
    )
    out["final_audit"]["review_limits"] = final["review_limits"]

    # 7. worked examples: records whose note led to an accepted change, one per field kind
    examples, seen = [], set()
    for k, diff in changed:
        acts = dec[k].get("note_actions", {})
        if not acts:
            continue
        field = next(iter(diff))
        if field in seen:
            continue
        seen.add(field)
        r = labels[k]
        after = after_by_pos[(r["paper_id"], r["output_record_index"])]
        rec_notes = [
            {"annotator": n[1], "field": n[3], "note": n[4]} for n in notes if n[0] == k
        ]
        examples.append(
            {
                "record": k,
                "term": before[k]["term"],
                "changed_fields": diff,
                "annotator_notes": rec_notes[:3],
                "curator_actions": list(acts.values())[:2],
                "before": {f: [i[0] for i in items(before[k], f)] for f in diff},
                "after": {f: [i[0] for i in items(after, f)] for f in diff},
            }
        )
        if len(examples) == WORKED_EXAMPLES:
            break
    out["examples"] = examples
    return out


# 4. Figures

INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb"
BLUE_RAMP = [
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
]
GRAY = "#c9c8c3"
BLUE = "#2a78d6"
DPI = 200
TABLE_CELLS = (  # the 2x2 table of fig1, row by row
    "first Yes / second Yes",
    "first Yes / second No",
    "first No / second Yes",
    "first No / second No",
)


def figure_numbers(res: dict) -> dict[str, dict]:
    """The numbers each figure plots, taken from agreement.json; `figures` draws only these.

    A figure is checked by them: after a library upgrade its image bytes change, its numbers do not. Worked example, the released round: fig1 draws Q1's table 2,095 / 185 / 147 / 59 and "same answer on 86.6%" (observed agreement 0.86645), and says 2 records lack an answer; fig2 draws Q1's chance agreement 0.83534, the observed 0.86645 and alpha 0.18897, and "About 91% of all Q1 answers are 'Yes'" (share 0.90949).
    """
    table, order = res["table"], res["rater_order"]
    lacking = len(table["Q1"]["records_missing_an_answer"])
    return {
        FIGURES[0]: {
            **{
                q: {
                    "table_2x2": {
                        cell: order[q]["table_2x2_earlier_submission_first"][cell]
                        for cell in TABLE_CELLS
                    },
                    "observed_agreement": table[q]["observed_agreement"],
                }
                for q in ("Q1", "Q2")
            },
            "Q1_records_missing_an_answer": lacking,
        },
        FIGURES[1]: {
            **{
                q: {
                    "chance_agreement": table[q]["krippendorff"][
                        "chance_agreement_1_minus_De"
                    ],
                    "observed_agreement": table[q]["observed_agreement"],
                    "alpha": table[q]["krippendorff"]["alpha"],
                }
                for q in ("Q1", "Q2")
            },
            "Q1_share_yes_answers": table["Q1"]["share_yes_answers"],
        },
    }


def figures(res: dict, out_dir: Path) -> None:
    """fig1: the two 2x2 tables of the first two answers; fig2: observed vs chance agreement.

    Each is written as PNG and SVG, drawn from `figure_numbers` only. The SVG element ids are fixed (svg.hashsalt), so two runs differ only in the SVG's date.
    """
    plotted = figure_numbers(res)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.fonttype": "none",
            "svg.hashsalt": SVG_HASH_SALT,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "text.color": INK,
        }
    )
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)

    # (a) the two 2x2 tables
    def luminance(rgba):
        lin = [
            c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
            for c in rgba[:3]
        ]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8))
    for ax, q in zip(axes, ("Q1", "Q2"), strict=True):
        t = plotted[FIGURES[0]][q]["table_2x2"]
        grid = np.array(
            [
                [t["first Yes / second Yes"], t["first Yes / second No"]],
                [t["first No / second Yes"], t["first No / second No"]],
            ]
        )
        n = grid.sum()
        shade = np.sqrt(grid / grid.max())
        ax.imshow(shade, cmap=cmap, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                v = grid[i, j]
                on_dark = luminance(cmap(shade[i, j])) < 0.3
                ax.text(
                    j,
                    i - 0.13,
                    f"{v:,}",
                    ha="center",
                    va="center",
                    fontsize=17,
                    fontweight="bold",
                    color="white" if on_dark else INK,
                )
                lab = "same answer" if i == j else "different answers"
                ax.text(
                    j,
                    i + 0.22,
                    f"{lab}\n{100 * v / n:.1f}% of records",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="white" if on_dark else INK,
                    linespacing=1.4,
                )
        ax.set_xticks([0, 1], ["Yes", "No"])
        ax.set_yticks([0, 1], ["Yes", "No"])
        ax.set_xlabel("Answer of the annotator who submitted second", color=INK2)
        ax.set_ylabel("Answer of the annotator who submitted first", color=INK2)
        ax.tick_params(which="both", length=0, colors=INK)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.set_xticks([0.5], minor=True)
        ax.set_yticks([0.5], minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=4)
        tq = plotted[FIGURES[0]][q]
        title = (
            "Q1  Is it a definition?"
            if q == "Q1"
            else "Q2  Is it relevant to the taxonomy?"
        )
        ax.set_title(
            f"{title}\n{n:,} records · same answer on {100 * tq['observed_agreement']:.1f}%",
            fontsize=11,
            color=INK,
            loc="left",
        )
    fig.suptitle(
        "How the first two annotators answered each concept record",
        x=0.02,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    lacking = plotted[FIGURES[0]]["Q1_records_missing_an_answer"]
    lack = f"{lacking} record lacks" if lacking == 1 else f"{lacking} records lack"
    fig.text(
        0.02,
        0.015,
        f"Each record was judged by two annotators (Q1: {lack} one answer, see text). "
        "Top-left and bottom-right = same answer.\nWhich of the two counts as 'first' is arbitrary "
        "(here: earlier submission); the agreement and alpha values do not depend on it. "
        "Darker = more records.",
        fontsize=8,
        color=INK2,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"{FIGURES[0]}.{ext}", dpi=DPI)
    plt.close(fig)

    # (b) observed vs chance agreement
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    rows = [
        ("Q2", "Q2  Relevant to\nthe taxonomy?"),
        ("Q1", "Q1  Is it a\ndefinition?"),
    ]
    for y, (q, _label) in enumerate(rows):
        t = plotted[FIGURES[1]][q]
        ch = 100 * t["chance_agreement"]
        ob = 100 * t["observed_agreement"]
        a = t["alpha"]
        ax.barh(y, ch, color=GRAY, height=0.46, edgecolor=SURFACE, linewidth=2)
        ax.barh(
            y,
            ob - ch,
            left=ch,
            color=BLUE,
            height=0.46,
            edgecolor=SURFACE,
            linewidth=2,
        )
        ax.barh(
            y,
            100 - ob,
            left=ob,
            color="none",
            height=0.46,
            edgecolor=MUTED,
            linewidth=1,
            linestyle=(0, (2, 2)),
        )
        ax.text(
            ch / 2,
            y,
            f"expected by luck alone\n{ch:.1f}%",
            ha="center",
            va="center",
            fontsize=8.5,
            color=INK,
        )
        ax.annotate(
            f"observed {ob:.1f}%  (+{ob - ch:.1f} points beyond luck)",
            xy=(ob, y + 0.23),
            xytext=(ob - 1, y + 0.42),
            ha="right",
            va="center",
            fontsize=8.5,
            color=INK,
            arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.8},
        )
        if 100 - ob > 10:
            ax.text(
                ob + (100 - ob) / 2,
                y,
                "never\nreached",
                ha="center",
                va="center",
                fontsize=7.5,
                color=INK2,
            )
        ax.text(
            102,
            y + 0.06,
            f"alpha = {ob - ch:.1f} / {100 - ch:.1f} = {a:.2f}",
            ha="left",
            va="center",
            fontsize=9.5,
            color=INK,
            fontweight="bold",
        )
        ax.text(
            102,
            y - 0.14,
            "beyond luck / room beyond luck",
            ha="left",
            va="center",
            fontsize=7.5,
            color=INK2,
        )
    ax.set_yticks([0, 1], [r[1] for r in rows], fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.45, 1.65)
    ax.set_xticks(range(0, 101, 20), [f"{x}%" for x in range(0, 101, 20)])
    ax.set_xlabel(
        "Share of records on which the two annotators gave the same answer", color=INK2
    )
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=INK2)
    fig.suptitle(
        "Why alpha is low for Q1", x=0.01, ha="left", fontsize=13, fontweight="bold"
    )
    q1, q2 = plotted[FIGURES[1]]["Q1"], plotted[FIGURES[1]]["Q2"]
    ch1 = 100 * q1["chance_agreement"]
    ob1 = 100 * q1["observed_agreement"]
    ch2 = 100 * q2["chance_agreement"]
    fig.text(
        0.01,
        0.885,
        f"About {100 * plotted[FIGURES[1]]['Q1_share_yes_answers']:.0f}% of all Q1 answers are 'Yes', so two "
        f"annotators would already give the same answer {ch1:.1f}% of the time by luck.\n"
        f"They reached {ob1:.1f}%: only {ob1 - ch1:.1f} of the {100 - ch1:.1f} possible points "
        f"above luck. For Q2 the answers are mixed, so luck gives only {ch2:.1f}%.",
        fontsize=8.5,
        color=INK2,
        ha="left",
    )
    fig.text(
        0.01,
        0.02,
        "Gray = agreement two annotators would reach by luck if each said 'Yes' as often as the "
        "annotators really did (Krippendorff's chance term).\nBlue = agreement beyond luck. "
        "Alpha = blue length divided by the distance from the gray end to 100%. 1 = perfect, "
        "0 = no better than luck.",
        fontsize=8,
        color=INK2,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.1, 0.8, 0.86))
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"{FIGURES[1]}.{ext}", dpi=DPI)
    plt.close(fig)


# The command


def agreement_command(
    out: str,
    annotations: str | None,
    export: str | None,
    data: str,
) -> int:
    """Write the agreement and curation numbers, their figures and a receipt into OUT."""
    target, data_dir = Path(out), Path(data)
    (export_path, export_name), (annotations_path, annotations_name) = (
        groundtruth.exports(export, annotations)
    )
    if target.exists() and any(target.iterdir()):
        raise SystemExit(f"{target} is not empty; choose a new folder for --out.")

    result: dict[str, Any] = {
        "generated_by": GENERATED_BY,
        "inputs": {
            "raw_exports": annotations_name,
            "v3": str(data_dir / V3),
            "form": str(data_dir / LABEL_STUDIO_XML),
            "released_labels": f"{export_name}: relevance-labels.json",
        },
        **agreement(annotations_path, export_path, data_dir),
    }
    curated = curation(export_path, export_name)
    target.mkdir(parents=True, exist_ok=True)
    files.write_json(target / AGREEMENT, result, **JSON_FORMAT)
    files.write_json(target / CURATION, curated, **JSON_FORMAT)
    figures(result, target)

    outputs = {p.name: p for p in sorted(target.iterdir()) if p.name != files.RECEIPT}
    inputs = {
        "annotations": annotations_path,
        "export": export_path,
        "v3": data_dir / V3,
        "form": data_dir / LABEL_STUDIO_XML,
        **stored_files(data_dir),
    }
    receipt = files.receipt(
        "evaluate agreement",
        settings={
            "bootstrap": {
                "replicates": BOOTSTRAP_REPLICATES,
                "rng": "numpy default_rng",
                "seeds": BOOTSTRAP_SEEDS,
            },
            "kappa_orderings": {
                "orderings": KAPPA_ORDERINGS,
                "rng": "Python random.Random",
                "seed": KAPPA_SEED,
            },
            "large_pair_records": LARGE_PAIR,
            "organiser_accounts": list(ORGANISERS),
            "json_format": JSON_FORMAT,
            "svg_hash_salt": SVG_HASH_SALT,
            "packages": {
                name: version(name) for name in ("krippendorff", "matplotlib", "numpy")
            },
        },
        inputs=inputs,
        outputs=outputs,
    )
    files.write_json(target / files.RECEIPT, receipt)
    for q in ("Q1", "Q2"):
        t = result["table"][q]
        print(
            q,
            t["N"],
            t["yes_yes"],
            t["no_no"],
            t["disagree"],
            t["pct_agree_2dp"],
            round(t["krippendorff"]["alpha"], 4),
        )
    print(json.dumps(result["reproduction_status"], indent=1))
    print("wrote", target)
    return 0
