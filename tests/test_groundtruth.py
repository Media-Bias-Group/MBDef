"""`mbdef groundtruth build` on a miniature export, with no network and no GPU.

The tests double as a worked example of the step: tests/mini_groundtruth.py writes a two-paper export, and the build turns it into a release (docs/step-3-extraction/groundtruth.md).
"""

import hashlib
import json
import zipfile
from pathlib import Path

from cli_helpers import (
    package_commit,
    real_mbdef_gold,
    run_cli,
    tool_dir,
    write_env,
)
from mini_groundtruth import (
    DRAFT_EMAIL,
    DRAFT_NAME,
    FAKE_TECTONIC,
    PAPER_A,
    PAPER_B,
    PASSAGE,
    PDF_FOLDER,
    make_mini_groundtruth,
)

REPO = Path(__file__).resolve().parents[1]
PINNED_EXPORT = "27ac4f2946deab4f3f8539b3a264b210f862b649cbd55353f6fe9f5bb29f9f46"
PINNED_COPY = "db6bc11b909a273d0356cace850c23d99be613245d842542b99a53282b693aaf"
COPY_EXPORT = "copies/validated-v3.zip"
COPY_ANNOTATIONS = "copies/newAnnotations/Annotations.zip"
GOLD, needs_gold = (
    real_mbdef_gold()
)  # MBDef-Gold, the released ground truth, in the data root
# The members of the export that pseudonymising leaves byte for byte as they are.
UNCHANGED = (
    "validated.json",
    "irrelevant-candidates.json",
    "original-v3.json",
    "relevance-labels.json",
    "review-coverage.json",
    "labelStudioXML.md",
)
# The release's files that hold its data and its report: the same from either export.
DATA_FILES = (
    "validated_gt.json",
    "validated_gt_non_relevant.json",
    "validated_gt_README.md",
    "validated_gt_report.tex",
    "validated_gt_report.pdf",
    "validated_gt_report.md",
)
DATA_MEMBERS = (*DATA_FILES, "figures/mini-chart.png", "figures/README.md")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mini_setup(tmp_path):
    work, root = tmp_path / "work", tmp_path / "data-root"
    export_sha = make_mini_groundtruth(work, root)
    write_env(work, DATA_ROOT=str(root))
    path = tool_dir(tmp_path / "bin", "git", fakes={"tectonic": FAKE_TECTONIC})
    return work, root, export_sha, path


def test_a_missing_env_key_fails_with_a_pointer_to_env_example(tmp_path):
    done = run_cli(tmp_path, "groundtruth", "build", "--out", str(tmp_path / "out"))

    assert done.returncode == 1
    assert done.stderr.strip() == (
        "DATA_ROOT is not set. Copy .env.example to .env and set DATA_ROOT there; "
        ".env.example says what it is."
    )
    assert not (tmp_path / "out").exists()


def test_build_writes_the_release_and_its_receipt(tmp_path):
    work, root, export_sha, path = mini_setup(tmp_path)

    done = run_cli(
        work,
        "groundtruth",
        "build",
        "--out",
        "out",
        "--export-sha256",
        export_sha,
        path=path,
    )

    assert done.returncode == 0, done.stderr
    out = work / "out"
    relevant = json.loads((out / "validated_gt.json").read_text(encoding="utf-8"))
    records = [r for p in relevant["papers"] for r in p["definition_records"]]
    assert relevant["paper_count"] == len(relevant["papers"]) == 2
    assert [r["term"] for r in records] == [
        "anchoring bias",
        "Behavioral Bias",
        "media bias",
        "selection bias",
        "fuzzy trace theory",
    ]
    assert sum(len(r["definitions"]) for r in records) == 6
    with zipfile.ZipFile(root / "groundtruth/validated-v3.zip") as export:
        assert (out / "validated_gt_non_relevant.json").read_bytes() == export.read(
            "irrelevant-candidates.json"
        )
        for name in export.namelist():  # the whole export, unpacked as evidence
            evidence = out / "newAnnotations/raw/final-export-20260908" / name
            assert evidence.read_bytes() == export.read(name)

    stats = json.loads((out / "validated_gt_stats.json").read_text(encoding="utf-8"))
    assert stats["changed_primary_records"] == 2
    assert stats["changed_record_keys"] == [
        "3457607:2",
        "0209e602acaeab882fee84e244caf574cf345ef9:5",
    ]
    assert stats["fates"] == {
        "primary": 5,
        "non_relevant": 2,
        "definition_no": 2,
        "definition_unknown_relevance_no": 0,
    }

    tex = (out / "validated_gt_report.tex").read_text(encoding="utf-8")
    assert "<<" not in tex
    assert "a systematic error in judgement that follows from how people think" in tex
    # Tectonic gets the release's pinned date, so the PDF is the same on every run.
    assert (
        (out / "validated_gt_report.pdf")
        .read_text()
        .startswith("%PDF-fake date=1788926168 ")
    )

    manifest = json.loads(
        (out / "validated_gt_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["counts"] == {
        "relevant_papers": 2,
        "non_relevant_paper_entries": 2,
        "non_relevant_papers_with_records": 2,
        "primary_records": 5,
        "non_relevant_records": 2,
    }
    with zipfile.ZipFile(out / "validated_gt_release.zip") as bundle:
        names = bundle.namelist()
        sums = bundle.read("SHA256SUMS").decode().splitlines()
        assert {"code/groundtruth.py", "code/groundtruth_report.tex"} <= set(names)
        assert "history/IE-wiki-snapshot.txt" in names
        assert "figures/mini-chart.png" in names
        for line in sums:
            digest, name = line.split("  ", 1)
            assert hashlib.sha256(bundle.read(name)).hexdigest() == digest
        assert len(sums) == len(names) - 1  # every member except SHA256SUMS itself
        for name in names[:-1]:
            if name != "validated_gt_manifest.json":
                assert (
                    manifest["files"][name]["sha256"]
                    == hashlib.sha256(bundle.read(name)).hexdigest()
                )

    receipt = json.loads((out / "receipt.json").read_text(encoding="utf-8"))
    head = package_commit()  # None outside a checkout
    assert receipt["command"] == "groundtruth build"
    assert receipt["code"]["commit"] == head
    assert receipt["settings"] == {
        "export_sha256": export_sha,
        "report_date_epoch": 1788926168,
        "zip_member_time": [2026, 9, 9, 12, 56, 8],
        "release_id": "MBDef-Gold-1.0.0",
    }
    assert receipt["inputs"]["export"]["sha256"] == export_sha
    assert receipt["inputs"]["annotations"]["sha256"] == sha(
        root / "groundtruth/newAnnotations/Annotations.zip"
    )
    assert receipt["inputs"]["v3"]["sha256"] == sha(work / "data/ground_truth/v3.json")
    assert receipt["outputs"]["validated_gt.json"]["sha256"] == sha(
        out / "validated_gt.json"
    )
    assert json.loads(done.stdout)["primary"] == 5


def test_build_refuses_an_export_other_than_the_pinned_one(tmp_path):
    work, _, export_sha, path = mini_setup(tmp_path)

    done = run_cli(work, "groundtruth", "build", "--out", "out", path=path)

    assert done.returncode == 1
    assert done.stderr.strip() == (
        f"The export's sha256 is {export_sha}, not a pinned one: "
        f"{PINNED_EXPORT} (the curator's approved export) or {PINNED_COPY} (its "
        "pseudonymised copy). For another export, pass its hash with --export-sha256 "
        "(docs/step-3-extraction/groundtruth.md)."
    )
    assert not (work / "out").exists()


def test_build_refuses_a_folder_that_is_not_empty(tmp_path):
    work, _, export_sha, path = mini_setup(tmp_path)
    (work / "out").mkdir()
    (work / "out/keep.txt").write_text("mine\n")

    done = run_cli(
        work,
        "groundtruth",
        "build",
        "--out",
        "out",
        "--export-sha256",
        export_sha,
        path=path,
    )

    assert done.returncode == 1
    assert "is not empty" in done.stderr
    assert sorted(p.name for p in (work / "out").iterdir()) == ["keep.txt"]


def members(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def everything(path: Path) -> str:
    """Every member of the zip at PATH as text, the zips inside it unpacked too."""
    texts = []
    for name, content in members(path).items():
        if name.endswith(".zip"):
            inner = path.parent / f".inner-{path.stem}"
            inner.write_bytes(content)
            texts.append(everything(inner))
            inner.unlink()
        else:
            texts.append(content.decode("utf-8"))
    return "\n".join(texts)


def test_pseudonymise_writes_copies_that_name_no_one(tmp_path):
    work, root, _, _ = mini_setup(tmp_path)
    export, annotations = (
        root / "groundtruth/validated-v3.zip",
        root / "groundtruth/newAnnotations/Annotations.zip",
    )
    before = everything(export) + everything(annotations)
    assert DRAFT_NAME in before and DRAFT_EMAIL in before and PDF_FOLDER in before

    done = run_cli(work, "groundtruth", "pseudonymise", "--out", "copies")

    assert done.returncode == 0, done.stderr
    export_copy, annotations_copy = work / COPY_EXPORT, work / COPY_ANNOTATIONS
    after = everything(export_copy) + everything(annotations_copy)
    for gone in (DRAFT_NAME, DRAFT_EMAIL, PDF_FOLDER, "/home/", PASSAGE):
        assert gone not in after

    # The draft stays, under the pseudonym of its Label Studio account; account numbers stay.
    raw_name = "Annotations_Weekly/project-1-at-2026-08-01.json"
    expected = json.loads(members(annotations)[raw_name])
    expected[0]["drafts"][0]["user"] = "account-101"
    expected[0]["drafts"][0]["created_username"] = "account-101, 101"
    assert json.loads(members(annotations_copy)[raw_name]) == expected
    assert [a["completed_by"] for a in expected[0]["annotations"]] == [101, 102]

    original, copy = members(export), members(export_copy)
    assert list(copy) == list(original)
    for name in UNCHANGED:
        assert copy[name] == original[name], name
    assert copy["annotations-original.zip"] == annotations_copy.read_bytes()
    pdfs = json.loads(copy["pdf-sources.json"])
    assert pdfs[PAPER_A]["path"] == f"papers/{PAPER_A}.pdf"
    assert pdfs[PAPER_B]["path"] == f"papers/{PAPER_B}.pdf"
    touches = json.loads(copy["final-touches.json"])
    touched = f"{PAPER_B}:5"
    assert touches["records"][touched]["pdf"] == pdfs[PAPER_B]
    proposals = json.loads(copy["curation-suggestions.json"])
    assert proposals["records"][touched]["source"] == [{"block": 4}]
    assert (
        proposals["sha256"]
        == json.loads(original["curation-suggestions.json"])["sha256"]
    )
    # The fingerprints name the copy of the raw exports; the final audit's seals hold again.
    provenance = json.loads(copy["curation-provenance.json"])
    new_fingerprint = {
        "archive_sha256": sha(annotations_copy),
        "v3_sha256": sha(work / "data/ground_truth/v3.json"),
    }
    for document in (provenance, touches, proposals):
        assert document["fingerprint"] == new_fingerprint
    assert (
        provenance["final_audit_sha256"]
        == touches["sha256"]
        != json.loads(original["final-touches.json"])["sha256"]
    )
    assert copy["README.txt"].startswith(original["README.txt"])
    assert b"pseudonymised copy" in copy["README.txt"]

    receipt = json.loads((work / "copies/receipt.json").read_text(encoding="utf-8"))
    assert receipt["command"] == "groundtruth pseudonymise"
    assert receipt["inputs"]["export"]["sha256"] == sha(export)
    assert receipt["outputs"]["validated-v3.zip"]["sha256"] == sha(export_copy)
    assert json.loads(done.stdout)["replaced"] == {
        "draft names and e-mails": 1,
        "local PDF paths": 3,
        "paper passages removed": 1,
        "fingerprints": 3,
        "final-audit hashes": 4,
    }


def test_pseudonymise_writes_the_same_bytes_twice(tmp_path):
    work, _, _, _ = mini_setup(tmp_path)

    first = run_cli(work, "groundtruth", "pseudonymise", "--out", "copies")
    second = run_cli(work, "groundtruth", "pseudonymise", "--out", "again")

    assert first.returncode == second.returncode == 0, first.stderr + second.stderr
    for name in ("validated-v3.zip", "newAnnotations/Annotations.zip"):
        assert sha(work / "copies" / name) == sha(work / "again" / name)


def test_pseudonymise_stops_at_a_trace_it_has_no_rule_for(tmp_path):
    work, root, _, _ = mini_setup(tmp_path)
    export = root / "groundtruth/validated-v3.zip"
    original = members(export)
    original["README.txt"] += (
        b"Exported on /home/annie/curation by annie@example.org.\n"
    )
    with zipfile.ZipFile(export, "w") as archive:
        for name, content in original.items():
            archive.writestr(name, content)

    done = run_cli(work, "groundtruth", "pseudonymise", "--out", "copies")

    assert done.returncode == 1
    assert done.stderr.strip() == (
        "Not written: the copies would still name a person or a local folder "
        "(an e-mail address in validated-v3.zip: README.txt; a local path in "
        "validated-v3.zip: README.txt). Add a rule for it (docs/step-3-extraction/groundtruth.md)."
    )
    assert not (work / "copies").exists()


def test_the_release_built_from_the_copies_has_the_same_data(tmp_path):
    work, root, export_sha, path = mini_setup(tmp_path)
    done = run_cli(work, "groundtruth", "pseudonymise", "--out", "copies")
    assert done.returncode == 0, done.stderr

    original = run_cli(
        work,
        "groundtruth",
        "build",
        "--out",
        "a",
        "--export-sha256",
        export_sha,
        path=path,
    )
    copied = run_cli(
        work,
        "groundtruth",
        "build",
        "--out",
        "b",
        "--export",
        COPY_EXPORT,
        "--annotations",
        COPY_ANNOTATIONS,
        "--export-sha256",
        sha(work / COPY_EXPORT),
        path=path,
    )

    assert original.returncode == copied.returncode == 0, (
        original.stderr + copied.stderr
    )
    for name in DATA_MEMBERS:
        assert (work / "a" / name).read_bytes() == (work / "b" / name).read_bytes(), (
            name
        )
    stats_a, stats_b = (
        json.loads((work / out / "validated_gt_stats.json").read_text(encoding="utf-8"))
        for out in ("a", "b")
    )
    assert stats_b.pop("export_sha256") == sha(work / COPY_EXPORT)
    assert stats_a.pop("export_sha256") == export_sha
    assert stats_a == stats_b


def test_build_refuses_a_copy_whose_hash_is_not_pinned(tmp_path):
    work, _, _, path = mini_setup(tmp_path)
    run_cli(work, "groundtruth", "pseudonymise", "--out", "copies")

    done = run_cli(
        work,
        "groundtruth",
        "build",
        "--out",
        "out",
        "--export",
        COPY_EXPORT,
        "--annotations",
        COPY_ANNOTATIONS,
        path=path,
    )

    assert done.returncode == 1
    assert done.stderr.strip() == (
        f"The export's sha256 is {sha(work / COPY_EXPORT)}, not a pinned one: "
        f"{PINNED_EXPORT} (the curator's approved export) or {PINNED_COPY} (its "
        "pseudonymised copy). For another export, pass its hash with --export-sha256 "
        "(docs/step-3-extraction/groundtruth.md)."
    )


@needs_gold
def test_the_pinned_copies_rebuild_mbdef_gold(tmp_path):
    """On the real data root (DATA_ROOT in the repository's .env), with Tectonic: rerunning `pseudonymise` on the original exports gives the copies data/data-root.tsv pins, and `build` accepts the copy without --export-sha256 and writes MBDef-Gold's data byte for byte as the data root holds it; only the export's hash in the statistics differs."""
    listed = {
        row.split("\t")[0]: row.split("\t")[1]
        for row in (REPO / "data/data-root.tsv")
        .read_text(encoding="utf-8")
        .splitlines()
    }
    copies, out = tmp_path / "copies", tmp_path / "out"

    made = run_cli(REPO, "groundtruth", "pseudonymise", "--out", str(copies))
    built = run_cli(
        REPO,
        "groundtruth",
        "build",
        "--export",
        str(copies / "validated-v3.zip"),
        "--annotations",
        str(copies / "newAnnotations/Annotations.zip"),
        "--out",
        str(out),
    )

    assert made.returncode == built.returncode == 0, made.stderr + built.stderr
    for name in ("validated-v3.zip", "newAnnotations/Annotations.zip"):
        assert sha(copies / name) == listed[f"groundtruth/pseudonymised/{name}"]
    assert sha(copies / "validated-v3.zip") == PINNED_COPY
    for name in DATA_FILES:
        assert (out / name).read_bytes() == (GOLD / name).read_bytes(), name
    stats = json.loads((out / "validated_gt_stats.json").read_text(encoding="utf-8"))
    released = json.loads(
        (GOLD / "validated_gt_stats.json").read_text(encoding="utf-8")
    )
    assert (stats.pop("export_sha256"), released.pop("export_sha256")) == (
        PINNED_COPY,
        PINNED_EXPORT,
    )
    assert stats == released


def test_without_the_originals_build_reads_the_copies_by_default(tmp_path):
    """A reader's data root holds only the pseudonymised copies (the originals are not public): `build` with no --export and --annotations reads the copies and writes the same data as from the originals."""
    work, root, export_sha, path = mini_setup(tmp_path)
    original = run_cli(
        work,
        *("groundtruth", "build", "--out", "a", "--export-sha256", export_sha),
        path=path,
    )
    copies = root / "groundtruth/pseudonymised"
    done = run_cli(work, "groundtruth", "pseudonymise", "--out", str(copies))
    assert done.returncode == 0, done.stderr
    (root / "groundtruth/validated-v3.zip").unlink()
    (root / "groundtruth/newAnnotations/Annotations.zip").unlink()

    copied = run_cli(
        work,
        *("groundtruth", "build", "--out", "b"),
        *("--export-sha256", sha(copies / "validated-v3.zip")),
        path=path,
    )

    assert original.returncode == copied.returncode == 0, (
        original.stderr + copied.stderr
    )
    for name in DATA_MEMBERS:
        assert (work / "a" / name).read_bytes() == (work / "b" / name).read_bytes(), (
            name
        )
