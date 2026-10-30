"""A miniature ground-truth export, made up for tests: 2 papers, 9 records, 2 annotators.

It has the shape of the real export behind MBDef-Gold (V3, the raw Label Studio exports and the curator's export zip), small enough to read. The annotator ids are made up, because real exports may name the people who annotated.

What the curators decided, per record (both annotators agree on every record):

    paper 3457607                                  definition  relevant  ends up in
      0 anchoring bias                              Yes         Yes       relevant
      1 framing                                     Yes         No        non-relevant
      2 Behavioral Bias (definition completed)      Yes         Yes       relevant
    paper 0209e602...
      0 media bias                                  Yes         Yes       relevant
      1 the dataset                                 No          No        neither
      2 gatekeeping                                 Yes         No        non-relevant
      3 selection bias (two definitions)            Yes         Yes       relevant
      4 tone                                        No          Yes       neither
      5 fuzzy trace theory (citation repaired)      Yes         Yes       relevant

So the relevant set has 2 papers, 5 records and 6 definitions, 2 of them edited; the non-relevant set has 2 papers and 2 records. Records 2 and 5 are the ones the release report shows as worked examples, which is why the paper ids are the real ones.

Like the real exports, it also holds what `groundtruth pseudonymise` must replace: an unsubmitted draft that names its annotator (a made-up name and e-mail), the local paths of the papers' PDFs (a made-up home folder), and a paper passage in the assistant's proposals (made-up text). Record 5 has a final touch, sealed by hashes the way the curation app seals it.
"""

import hashlib
import io
import json
import zipfile
from pathlib import Path

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
PAPER_A = "3457607"
PAPER_B = "0209e602acaeab882fee84e244caf574cf345ef9"
ANNOTATORS = (101, 102)  # made up
DRAFT_NAME, DRAFT_EMAIL = "Annie Example", "annie@example.org"  # made up: 101's draft
PDF_FOLDER = "/home/annie/mini-repo/papers"  # made up: where the PDFs were
PASSAGE = "A made-up paragraph of the paper, around the definition."

# A stand-in for Tectonic: it writes a small "PDF" that records the date it was given.
FAKE_TECTONIC = """\
import hashlib, os, sys
from pathlib import Path
args = sys.argv[1:]
outdir, tex = Path(args[args.index("--outdir") + 1]), Path(args[-1])
body = "%PDF-fake date={} tex={}\\n".format(
    os.environ.get("SOURCE_DATE_EPOCH", "unset"), hashlib.sha256(tex.read_bytes()).hexdigest()
)
(outdir / (tex.stem + ".pdf")).write_text(body)
"""


def item(text, *citation):
    return {"text": text, "citation": list(citation)}


def record(term, definitions, **fields):
    return {
        "term": term,
        **{f: fields.get(f, []) for f in FIELDS},
        "definitions": definitions,
    }


def metadata(title):
    return {
        "title": title,
        "authors": ["A. Author"],
        "publication_year": 2020,
        "doi": "",
        "venue": "Mini Venue",
    }


V3_RECORDS = {
    PAPER_A: [
        record(
            "anchoring bias",
            [item("relying too much on the first number one sees", "[1]")],
            examples=[item("an opening price in a negotiation", "[1]")],
        ),
        record(
            "framing",
            [item("presenting the same facts in different ways", "[2]")],
            synonyms=[item("frame", "[2]")],
        ),
        record("Behavioral Bias", [item("a systematic error in", "[3]")]),
    ],
    PAPER_B: [
        record("media bias", [item("slant in news coverage", "[1]")]),
        record("the dataset", [item("a set of news articles", "[2]")]),
        record("gatekeeping", [item("choosing which stories are published", "[3]")]),
        record(
            "selection bias",
            [item("bias in what is selected", "[4]"), item("a skewed sample", "[4]")],
        ),
        record("tone", [item("the attitude a text expresses", "[5]")]),
        record(
            "fuzzy trace theory",
            [item("people keep both gist and verbatim traces", "[6]")],
        ),
    ],
}
LABELS = {  # (is a definition, is relevant), as both annotators and the curator answered
    PAPER_A: [("Yes", "Yes"), ("Yes", "No"), ("Yes", "Yes")],
    PAPER_B: [
        ("Yes", "Yes"),
        ("No", "No"),
        ("Yes", "No"),
        ("Yes", "Yes"),
        ("No", "Yes"),
        ("Yes", "Yes"),
    ],
}
EDITS = {  # the curator's approved edits: source key -> final record
    f"{PAPER_A}:2": record(
        "Behavioral Bias",
        [
            item(
                "a systematic error in judgement that follows from how people think",
                "[3]",
            )
        ],
    ),
    f"{PAPER_B}:5": record(
        "fuzzy trace theory",
        [item("people keep both gist and verbatim traces", "[6]", "[7]")],
    ),
}


def dump(data) -> bytes:
    return (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode()


def label_studio_dump(data) -> bytes:
    """JSON as Label Studio exports it: compact, non-ASCII kept, every "/" escaped."""
    text = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    return text.replace("/", "\\/").encode()


def app_dump(data) -> bytes:
    """JSON as the curation app writes the members of its export."""
    return json.dumps(data, indent=2, ensure_ascii=False).encode()


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def zip_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(zipfile.ZipInfo(name, (2026, 9, 8, 12, 0, 0)), content)
    return buffer.getvalue()


def display(items):
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


def v3() -> dict:
    papers = [
        {
            "paper_id": PAPER_A,
            "split": "train",
            "metadata": metadata("Biases in mini papers"),
            "definition_records": V3_RECORDS[PAPER_A],
        },
        {
            "paper_id": PAPER_B,
            "split": "dev",
            "metadata": metadata("Media bias in a mini corpus"),
            "definition_records": V3_RECORDS[PAPER_B],
        },
    ]
    return {
        "schema_version": "3",
        "dataset_id": "mini_v3",
        "paper_count": 2,
        "papers": papers,
    }


def annotations_zip() -> bytes:
    """The raw Label Studio exports: one weekly task per paper, two annotators each."""
    tasks = []
    for task_id, (paper_id, records) in enumerate(V3_RECORDS.items(), 1):
        shown = []
        for idx, r in enumerate(records):
            row = {"idx": idx, "term": r["term"]}
            for field in FIELDS:
                if field == "examples":
                    row[field] = [
                        x["text"]
                        + (
                            "  [citation: " + ", ".join(x["citation"]) + "]"
                            if x["citation"]
                            else ""
                        )
                        for x in r[field]
                    ]
                else:
                    row[field + "_display"] = display(r[field])
            shown.append(row)
        annotations = []
        for number, annotator in enumerate(ANNOTATORS, 1):
            result = []
            for idx, answers in enumerate(LABELS[paper_id]):
                for question, answer in zip(QUESTIONS, answers, strict=True):
                    result.append(
                        {
                            "from_name": f"{question}_{idx}",
                            "to_name": f"{question}_label_{idx}",
                            "type": "choices",
                            "value": {"choices": [answer]},
                        }
                    )
            annotations.append(
                {
                    "id": task_id * 10 + number,
                    "completed_by": annotator,
                    "was_cancelled": False,
                    "created_at": "2026-08-01T10:00:00Z",
                    "result": result,
                }
            )
        tasks.append(
            {
                "id": task_id,
                "project": 1,
                "data": {
                    "paper_id": paper_id,
                    "num_records": len(records),
                    "definition_records": shown,
                },
                "annotations": annotations,
                "drafts": [draft(task_id)] if task_id == 1 else [],
                "updated_by": ANNOTATORS[0],
            }
        )
    return zip_bytes(
        {"Annotations_Weekly/project-1-at-2026-08-01.json": label_studio_dump(tasks)}
    )


def draft(task_id: int) -> dict:
    """An unsubmitted draft of annotator 101, as Label Studio exports it: it names the person."""
    return {
        "id": 7,
        "user": DRAFT_EMAIL,
        "created_username": f"{DRAFT_NAME} {DRAFT_EMAIL}, {ANNOTATORS[0]}",
        "result": [
            {
                "from_name": "term_definition_error_text_0",
                "to_name": "definitions_value_0",
                "type": "textarea",
                "value": {"text": ["first number seen"]},
            }
        ],
        "lead_time": 10.5,
        "created_at": "2026-08-01T09:00:00Z",
        "task": task_id,
        "annotation": None,
    }


def pdf_sources() -> dict:
    return {
        paper: {
            "available": True,
            "same_input": True,
            "path": f"{PDF_FOLDER}/{paper}.pdf",
            "sha256": sha(paper.encode()),
        }
        for paper in (PAPER_A, PAPER_B)
    }


def export_zip(v3_bytes: bytes, annotations: bytes, label_studio_xml: bytes) -> bytes:
    """The curator's export, with the twelve members the real one has."""
    base = v3()
    decisions, history, events, labels = {}, {}, {}, []
    primary = {p["paper_id"]: [] for p in base["papers"]}
    separate = {p["paper_id"]: [] for p in base["papers"]}
    for paper in base["papers"]:
        pid = paper["paper_id"]
        for i, original in enumerate(paper["definition_records"]):
            key = f"{pid}:{i}"
            answers = dict(zip(QUESTIONS, LABELS[pid][i], strict=True))
            final = EDITS.get(key, original)
            decision = {
                "labels": answers,
                "record": final,
                "scope": "both",
                "status": "reviewed",
                "revision": 1,
                "updated_at": "2026-09-08T10:00:00Z",
            }
            decisions[key], history[key] = decision, [decision]
            events[key] = [
                {
                    "annotator": annotator,
                    "stage": "weekly",
                    "project": 1,
                    "task": 1 if pid == PAPER_A else 2,
                    "annotation": (1 if pid == PAPER_A else 2) * 10 + number,
                }
                for number, annotator in enumerate(ANNOTATORS, 1)
            ]
            retained = answers == {
                "definition_is_definition": "Yes",
                "term_relevant": "Yes",
            }
            irrelevant = answers == {
                "definition_is_definition": "Yes",
                "term_relevant": "No",
            }
            if retained:
                primary[pid].append(final)
            if irrelevant:
                separate[pid].append(final)
            labels.append(
                {
                    "source_key": key,
                    "paper_id": pid,
                    "output_record_index": len(primary[pid]) - 1 if retained else None,
                    "term": final["term"],
                    "retained": retained,
                    "supporting_fields_reviewed": key in EDITS,
                    "approval_basis": "human" if key in EDITS else "annotation_policy",
                    "in_irrelevant_candidates": irrelevant,
                    **answers,
                }
            )
    fingerprint = {"archive_sha256": sha(annotations), "v3_sha256": sha(v3_bytes)}
    touched = f"{PAPER_B}:5"  # its citation was repaired by a final touch
    touch = {
        "paper_id": PAPER_B,
        "term": EDITS[touched]["term"],
        "base_record": V3_RECORDS[PAPER_B][5],
        "issues": [],
        "pdf": pdf_sources()[PAPER_B],
        "was_curator_reviewed": True,
    }
    decision = decisions[touched]  # also its history's latest entry
    decision["final_review_sha256"] = sha(canonical(touch).encode())
    decision["final_record_sha256"] = sha(
        canonical({"record": decision["record"], "labels": decision["labels"]}).encode()
    )
    touches = {"fingerprint": fingerprint, "records": {touched: touch}}
    touches["sha256"] = sha(canonical(touches).encode())
    suggestions = {
        "fingerprint": fingerprint,
        "author": "Assistant (mini)",
        "method": "Proposals from the paper's text",
        "records": {
            touched: {
                "term": EDITS[touched]["term"],
                "recommendation": "Add the second citation.",
                "source": [{"block": 4, "text": PASSAGE}],
            }
        },
        "sha256": sha(b"the proposals file the curator saw"),
    }

    def dataset(chosen):
        papers = [
            {**p, "definition_records": chosen[p["paper_id"]]} for p in base["papers"]
        ]
        return {
            "schema_version": "3",
            "dataset_id": "mini_curated",
            "paper_count": 2,
            "papers": papers,
        }

    provenance = {
        "fingerprint": fingerprint,
        "scope": "both",
        "recommendations_sha256": suggestions["sha256"],
        "final_audit_sha256": touches["sha256"],
        "created_at": "2026-09-08T12:00:00+00:00",
        "decisions": decisions,
        "annotations": events,
        "history": history,
    }
    coverage = [
        {"key": f"{PAPER_A}:0", "field": "examples", "index": 0, "answer_count": 2},
        {"key": f"{PAPER_B}:3", "field": "definitions", "index": 1, "answer_count": 1},
        {"key": f"{PAPER_A}:1", "field": "synonyms", "index": 0, "answer_count": 0},
    ]
    return zip_bytes(
        {
            "validated.json": dump(dataset(primary)),
            "irrelevant-candidates.json": dump(dataset(separate)),
            "original-v3.json": v3_bytes,
            "annotations-original.zip": annotations,
            "curation-provenance.json": app_dump(provenance),
            "relevance-labels.json": dump(labels),
            "curation-suggestions.json": app_dump(suggestions),
            "final-touches.json": app_dump(touches),
            "review-coverage.json": dump(coverage),
            "pdf-sources.json": app_dump(pdf_sources()),
            "labelStudioXML.md": label_studio_xml,
            "README.txt": b"Miniature curator export for tests.\n",
        }
    )


def make_mini_groundtruth(work: Path, data_root: Path) -> str:
    """Write the repository inputs into WORK/data and the outside-git inputs into DATA_ROOT.

    Returns the sha256 of the curator's export, to pass as --export-sha256.
    """
    data = work / "data"
    files = {
        "ground_truth/v3.json": dump(v3()),
        "ground_truth/v2.json": dump(
            {
                "schema_version": "synthetic_gt_v1",
                "papers": [
                    {
                        "paper_id": PAPER_A,
                        "title": "Biases in mini papers",
                        "split": "train",
                        "definitions": [{"term": "anchoring bias"}],
                    }
                ],
            }
        ),
        "ground_truth/scidef.json": dump(
            {
                PAPER_A: {
                    "anchoring bias": {
                        "definition": "relying on the first number",
                        "context": "",
                        "type": "explicit",
                    }
                }
            }
        ),
        "frozen/labelStudioXML.md": b"<View>mini labelling interface</View>\n",
    }
    for name, content in files.items():
        (data / name).parent.mkdir(parents=True, exist_ok=True)
        (data / name).write_bytes(content)
    annotations = annotations_zip()
    export = export_zip(
        files["ground_truth/v3.json"], annotations, files["frozen/labelStudioXML.md"]
    )
    outside = {
        "groundtruth/validated-v3.zip": export,
        "groundtruth/newAnnotations/Annotations.zip": annotations,
        "groundtruth/newAnnotations/raw/IE-wiki-snapshot.txt": b"Mini snapshot of the annotation page.\n",
        # The release's fixed texts, files of MBDef-Gold
        "MBDef-Gold/validated_gt_README.md": b"# Mini guide\n",
        "MBDef-Gold/validated_gt_report.md": b"# Mini report\n",
        "MBDef-Gold/history/data-README.md": b"# Mini ground truth\n",
        "MBDef-Gold/figures/mini-chart.png": b"\x89PNG mini chart",
        "MBDef-Gold/figures/README.md": b"# Mini figures\n",
    }
    for name, content in outside.items():
        (data_root / name).parent.mkdir(parents=True, exist_ok=True)
        (data_root / name).write_bytes(content)
    return sha(export)
