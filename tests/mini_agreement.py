"""A miniature annotation round for the agreement numbers, made up for tests.

It uses the two papers and nine records of tests/mini_groundtruth.py, with four made-up annotators. Paper A was annotated in the weekly stage by 101 and 102, paper B in the completion stage by 102 and 103, and 104 answered both questions again on every record the first two disagreed on (the third round). 101 submitted paper A twice with the same answers, and once cancelled; both extra submissions are dropped.

First-round answers (Q1 = is it a definition, Q2 = is it relevant), and the third answer:

    record                  Q1 first two   Q2 first two   third (Q1, Q2)   disputed on
    A0 anchoring bias       Yes Yes        Yes Yes
    A1 framing              Yes Yes        Yes No         Yes No           Q2
    A2 Behavioral Bias      Yes No         Yes Yes        Yes Yes          Q1
    B0 media bias           Yes Yes        Yes Yes
    B1 the dataset          No  No         No  No
    B2 gatekeeping          Yes Yes        No  Yes        Yes No           Q2
    B3 selection bias       Yes (missing)  Yes Yes
    B4 tone                 No  Yes        Yes Yes        No  Yes          Q1
    B5 fuzzy trace theory   Yes Yes        Yes Yes

So Table 3 is, by hand:

    Q1: 8 complete pairs (B3 lacks an answer): 5 Yes/Yes, 1 No/No, 2 disagree.
        16 answers, 12 Yes and 4 No. Observed disagreement 2/8 = 0.25; expected
        2 x 12 x 4 / (16 x 15) = 0.4; alpha = 1 - 0.25 / 0.4 = 0.375.
    Q2: 9 complete pairs: 6 Yes/Yes, 1 No/No, 2 disagree. 18 answers, 14 Yes and 4 No.
        Observed 2/9; expected 2 x 14 x 4 / (18 x 17) = 112/306; alpha = 11/28 = 0.3929.
    4 records disputed (A1, A2, B2, B4).

The curator's export: the released labels are the majority answers (B3's Q1 set by the curator), five records are kept (A0, A2, B0, B3, B5), and two of them were edited (A2's definition with the assistant's text after an annotator's note; B5's citation, by a final touch). Paper ids are the real ones of tests/mini_groundtruth.py; annotator ids are made up.
As in the real exports, 101 also left an unsubmitted draft that names the person (a made-up name and e-mail), which no number reads, and the curator's export records the sha256 of the raw exports it was made from.
"""

import json
from pathlib import Path

from mini_groundtruth import (
    EDITS,
    FIELDS,
    LABELS,
    PAPER_A,
    PAPER_B,
    QUESTIONS,
    V3_RECORDS,
    app_dump,
    display,
    draft,
    dump,
    label_studio_dump,
    sha,
    v3,
    zip_bytes,
)

Q1, Q2 = QUESTIONS
FIRST_ROUND = {  # paper -> (stage folder, stage, project, task, annotators, answers per record)
    PAPER_A: (
        "Annotations_Weekly",
        "weekly",
        3,
        1,
        (101, 102),
        [
            (("Yes", "Yes"), ("Yes", "Yes")),
            (("Yes", "Yes"), ("Yes", "No")),
            (("Yes", "No"), ("Yes", "Yes")),
        ],
    ),
    PAPER_B: (
        "Annotation_rest",
        "rest",
        18,
        2,
        (102, 103),
        [
            (("Yes", "Yes"), ("Yes", "Yes")),
            (("No", "No"), ("No", "No")),
            (("Yes", "Yes"), ("No", "Yes")),
            (("Yes", None), ("Yes", "Yes")),
            (("No", "Yes"), ("Yes", "Yes")),
            (("Yes", "Yes"), ("Yes", "Yes")),
        ],
    ),
}
THIRD = 104
THIRD_ANSWERS = {  # (paper, record index) -> (disputed questions, third Q1, third Q2)
    (PAPER_A, 1): ([Q2], "Yes", "No"),
    (PAPER_A, 2): ([Q1], "Yes", "Yes"),
    (PAPER_B, 2): ([Q2], "Yes", "No"),
    (PAPER_B, 4): ([Q1], "No", "Yes"),
}
LABEL_STUDIO_XML = (
    "<View>\n"
    '  <Choices name="definition_is_definition_{{idx}}" toName="d" required="true"/>\n'
    '  <Choices name="term_relevant_{{idx}}" toName="d"/>\n'
    "</View>\n"
)


def fragment(paper: str) -> str:
    return f"{paper}__part1of1"


def shown(records: list[dict]) -> list[dict]:
    """The records as a Label Studio task shows them (V3 fields as display text)."""
    rows = []
    for idx, r in enumerate(records):
        row = {"idx": idx, "term": r["term"]}
        for field in FIELDS:
            if field == "examples":
                row[field] = [
                    x["text"] + "  [citation: " + ", ".join(x["citation"]) + "]"
                    if x["citation"]
                    else x["text"]
                    for x in r[field]
                ]
            else:
                row[field + "_display"] = display(r[field])
        rows.append(row)
    return rows


def answer(question: str, idx: int, value: str) -> dict:
    return {
        "from_name": f"{question}_{idx}",
        "to_name": f"{question}_label_{idx}",
        "type": "choices",
        "value": {"choices": [value]},
    }


def submission(number: int, annotator: int, result: list[dict], cancelled=False):
    return {
        "id": number,
        "completed_by": annotator,
        "was_cancelled": cancelled,
        "created_at": f"2026-08-0{number % 9 + 1}T10:00:00Z",
        "result": result,
    }


def annotations_zip() -> bytes:
    """The raw Label Studio exports: one first-round task per paper, one third-round task per paper with a disagreement."""
    files: dict[str, list] = {}
    number = 0
    for paper, (folder, _, project, task, people, answers) in FIRST_ROUND.items():
        subs = []
        for who, annotator in enumerate(people):
            result = [
                answer(q, idx, value)
                for idx, record in enumerate(answers)
                for q, pair in zip(QUESTIONS, record, strict=True)
                if (value := pair[who]) is not None
            ]
            number += 1
            subs.append(submission(number, annotator, result))
            if paper == PAPER_A and who == 0:  # a repeated and a cancelled submission
                number += 1
                subs.append(submission(number, annotator, result))
                number += 1
                subs.append(submission(number, annotator, [], cancelled=True))
        data = {
            "paper_id": fragment(paper),
            "original_paper_id": paper,
            "num_records": len(V3_RECORDS[paper]),
            "definition_records": shown(V3_RECORDS[paper]),
        }
        name = f"{folder}/project-{project}-at-2026-08-18.json"
        files[name] = [
            {
                "id": task,
                "project": project,
                "data": data,
                "annotations": subs,
                # 101 left an unsubmitted draft on paper A that names the person
                "drafts": [draft(task)] if paper == PAPER_A else [],
            }
        ]
    third_tasks = []
    for task, paper in enumerate((PAPER_A, PAPER_B), 3):
        disputed = sorted(i for p, i in THIRD_ANSWERS if p == paper)
        rows, result = [], []
        for queue, source in enumerate(disputed):
            questions, q1, q2 = THIRD_ANSWERS[(paper, source)]
            row = shown([V3_RECORDS[paper][source]])[0]
            rows.append(
                {
                    **row,
                    "idx": queue,
                    "source_record_index": source,
                    "disagreement_questions": questions,
                }
            )
            result += [answer(Q1, queue, q1), answer(Q2, queue, q2)]
        number += 1
        data = {
            "paper_id": fragment(paper) + "__third",
            "source_paper_id": fragment(paper),
            "original_paper_id": paper,
            "num_records": len(rows),
            "definition_records": rows,
            "disagreement_record_map": [
                {
                    "queue_record_index": r["idx"],
                    "source_record_index": r["source_record_index"],
                    "disagreement_questions": r["disagreement_questions"],
                }
                for r in rows
            ],
        }
        third_tasks.append(
            {
                "id": task,
                "project": 23,
                "data": data,
                "annotations": [submission(number, THIRD, result)],
            }
        )
    files["Annotation_Disagreements/project-23-at-2026-08-31.json"] = third_tasks
    return zip_bytes({name: label_studio_dump(tasks) for name, tasks in files.items()})


def votes(paper: str, idx: int) -> list[dict]:
    """Every vote on one record, as the curator's export lists them."""
    folder, stage, project, task, people, answers = FIRST_ROUND[paper]
    out = []
    for who, annotator in enumerate(people):
        given = {
            q: [pair[who]]
            for q, pair in zip(QUESTIONS, answers[idx], strict=True)
            if pair[who] is not None
        }
        out.append({"annotator": annotator, "stage": stage, "answers": given})
    if (paper, idx) in THIRD_ANSWERS:
        _, q1, q2 = THIRD_ANSWERS[(paper, idx)]
        out.append(
            {"annotator": THIRD, "stage": "third", "answers": {Q1: [q1], Q2: [q2]}}
        )
    return out


A2, B3, B5 = f"{PAPER_A}:2", f"{PAPER_B}:3", f"{PAPER_B}:5"
NOTE = "the definition stops mid-sentence"


def export_zip(v3_bytes: bytes, annotations: bytes) -> bytes:
    """The curator's export members that the agreement and curation numbers read."""
    base = v3()
    decisions, events, history, labels = {}, {}, {}, []
    kept = {p["paper_id"]: [] for p in base["papers"]}
    for paper in base["papers"]:
        pid = paper["paper_id"]
        for i, original in enumerate(paper["definition_records"]):
            key = f"{pid}:{i}"
            final = EDITS.get(key, original)
            answers = dict(zip(QUESTIONS, LABELS[pid][i], strict=True))
            decision = {
                "labels": answers,
                "record": final,
                "revision": 2 if key == A2 else 1,
                "supporting_fields_reviewed": key in (A2, B3),
                "label_choice": None if key in (A2, B3) else "human",
                "status": "reviewed",
                "updated_at": "2026-09-08T10:00:00Z",
            }
            if key == A2:
                decision["note_actions"] = {
                    "2:term_definition_error_text": "Accepted assistant recommendation"
                }
                decision["field_choices"] = {"definitions": "assistant"}
            if key == B3:
                decision["field_choices"] = {"synonyms": "v3"}
            decisions[key] = decision
            history[key] = [
                {**decision, "revision": r} for r in range(1, decision["revision"] + 1)
            ]
            events[key] = votes(pid, i)
            if key == A2:
                events[key][1]["answers"]["term_definition_error_text"] = [NOTE]
            retained = answers == {Q1: "Yes", Q2: "Yes"}
            if retained:
                kept[pid].append(final)
            labels.append(
                {
                    "source_key": key,
                    "paper_id": pid,
                    "output_record_index": len(kept[pid]) - 1 if retained else None,
                    "term": final["term"],
                    "retained": retained,
                    "approval_basis": "curator"
                    if key in (A2, B3)
                    else "annotation_policy",
                    **answers,
                }
            )
    validated = {
        **base,
        "papers": [
            {**p, "definition_records": kept[p["paper_id"]]} for p in base["papers"]
        ],
    }
    suggestions = {
        "author": "Assistant (mini)",
        "method": "Proposals only",
        "records": {A2: {"definitions": "proposed text"}, B3: {"synonyms": "none"}},
        "review_batches": [{"record_keys": [B3]}],
    }
    touches = {
        "policy": "Accept the remaining majority outcomes.",
        "summary": {
            "human_reviewed": 2,
            "changed_records": 1,  # before the final touch on B5
            "source_checks": {"pdf_normalized_match": 1},
            "final_touch_records": 1,
            "final_touch_categories": {"citation": 1},
            "pending_single_or_missing_fit_units": 0,
            "pending_single_or_missing_fit_records": 0,
            "policy_acceptances": 7,
            "primary_fields": {"definitions": 6, "examples": 1},
        },
        "records": {B5: {"base_record": V3_RECORDS[PAPER_B][5]}},
        "review_limits": ["Every kept record was checked."],
    }
    provenance = {
        "fingerprint": {"archive_sha256": sha(annotations), "v3_sha256": sha(v3_bytes)},
        "decisions": decisions,
        "annotations": events,
        "history": history,
    }
    return zip_bytes(
        {
            "validated.json": dump(validated),
            "original-v3.json": v3_bytes,
            "curation-provenance.json": app_dump(provenance),
            "relevance-labels.json": dump(labels),
            "curation-suggestions.json": dump(suggestions),
            "final-touches.json": dump(touches),
        }
    )


TABLE = {  # Table 3 of the miniature, worked out by hand above
    "Q1": {"N": 8, "yes_yes": 5, "no_no": 1, "disagree": 2, "alpha": "0.3750"},
    "Q2": {"N": 9, "yes_yes": 6, "no_no": 1, "disagree": 2, "alpha": "0.3929"},
}


def stored_statistics() -> dict:
    """The agreement table as the ground-truth statistics store it."""
    return {
        question: {
            "complete_pairs": t["N"],
            "yes_yes": t["yes_yes"],
            "no_no": t["no_no"],
            "disagree": t["disagree"],
            "alpha_nominal": float(t["alpha"]),
        }
        for question, t in zip(QUESTIONS, TABLE.values(), strict=True)
    }


def make_mini_agreement(work: Path, data_root: Path) -> None:
    """Write the repository inputs into WORK/data and the outside-git inputs into DATA_ROOT."""
    v3_bytes = dump(v3())
    annotations = annotations_zip()
    files = {
        work / "data/ground_truth/v3.json": v3_bytes,
        work / "data/frozen/labelStudioXML.md": LABEL_STUDIO_XML.encode(),
        work / "data/newAnnotations/annotation-audit.json": dump(
            {"statistics": {"questions": stored_statistics()}}
        ),
        data_root / "MBDef-Gold/validated_gt_stats.json": dump(
            {"annotation_archive": {"questions": stored_statistics()}}
        ),
        data_root / "groundtruth/newAnnotations/Annotations.zip": annotations,
        data_root / "groundtruth/validated-v3.zip": export_zip(v3_bytes, annotations),
    }
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
