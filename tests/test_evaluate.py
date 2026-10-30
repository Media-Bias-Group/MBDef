"""`mbdef evaluate agreement` on a miniature annotation round, with no network.

The test doubles as a worked example of the step: tests/mini_agreement.py writes the round and works out its Table 3 by hand, and the command must print and store the same.
"""

import hashlib
import re
from pathlib import Path

from cli_helpers import package_commit, run_cli, write_env
from mini_agreement import TABLE, load, make_mini_agreement


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mini_setup(tmp_path):
    work, root = tmp_path / "work", tmp_path / "data-root"
    make_mini_agreement(work, root)
    write_env(work, DATA_ROOT=str(root))
    return work, root


def test_agreement_writes_table_3_of_the_round(tmp_path):
    work, _ = mini_setup(tmp_path)

    done = run_cli(work, "evaluate", "agreement", "--out", "out")

    assert done.returncode == 0, done.stderr
    agreement = load(work / "out/agreement.json")
    for q, expected in TABLE.items():
        row = agreement["table"][q]
        assert (row["N"], row["yes_yes"], row["no_no"], row["disagree"]) == (
            expected["N"],
            expected["yes_yes"],
            expected["no_no"],
            expected["disagree"],
        )
        assert f"{row['krippendorff']['alpha']:.4f}" == expected["alpha"]
    assert agreement["design"]["v3_records"] == 9
    assert (
        agreement["third_round"]["records_with_first_round_disagreement_on_Q1_or_Q2"]
        == 4
    )
    assert agreement["n_difference"]["records_without_a_complete_q1_pair"][0][
        "record"
    ] == ("0209e602acaeab882fee84e244caf574cf345ef9:3")


def test_agreement_writes_a_receipt_with_its_pinned_settings_and_inputs(tmp_path):
    work, root = mini_setup(tmp_path)

    done = run_cli(work, "evaluate", "agreement", "--out", "out")

    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines()[:2] == [
        "Q1 8 5 1 2 75.00 0.375",
        "Q2 9 6 1 2 77.78 0.3929",
    ]
    receipt = load(work / "out/receipt.json")
    assert receipt["command"] == "evaluate agreement"
    assert receipt["code"]["commit"] == package_commit()  # None outside a checkout
    assert receipt["settings"]["bootstrap"] == {
        "replicates": 10000,
        "rng": "numpy default_rng",
        "seeds": {"by_paper": 1, "by_batch": 2},
    }
    assert receipt["settings"]["kappa_orderings"] == {
        "orderings": 1000,
        "rng": "Python random.Random",
        "seed": 20261016,
    }
    annotations = root / "groundtruth/newAnnotations/Annotations.zip"
    assert receipt["inputs"]["annotations"]["sha256"] == sha(annotations)
    assert receipt["inputs"]["export"]["sha256"] == sha(
        root / "groundtruth/validated-v3.zip"
    )
    assert "paper_draft" not in receipt["inputs"]
    assert sorted(receipt["outputs"]) == [
        "agreement.json",
        "curation.json",
        "fig1_agreement_tables.png",
        "fig1_agreement_tables.svg",
        "fig2_observed_vs_chance.png",
        "fig2_observed_vs_chance.svg",
    ]
    agreement = load(work / "out/agreement.json")
    assert agreement["generated_by"] == "mbdef evaluate agreement"
    assert (
        agreement["paper_claims"]["first_round_accounts (paper: 11 annotator accounts)"]
        == 3
    )


def test_agreement_refuses_a_folder_that_is_not_empty(tmp_path):
    work, _ = mini_setup(tmp_path)
    (work / "out").mkdir()
    (work / "out/keep.txt").write_text("mine\n")

    done = run_cli(work, "evaluate", "agreement", "--out", "out")

    assert done.returncode == 1
    assert done.stderr.strip() == "out is not empty; choose a new folder for --out."
    assert sorted(p.name for p in (work / "out").iterdir()) == ["keep.txt"]


def test_agreement_needs_the_data_root(tmp_path):
    work, _ = mini_setup(tmp_path)
    (work / ".env").unlink()

    done = run_cli(work, "evaluate", "agreement", "--out", "out")

    assert done.returncode == 1
    assert done.stderr.strip().startswith(
        "DATA_ROOT is not set. Copy .env.example to .env"
    )
    assert not (work / "out").exists()


def test_agreement_draws_its_two_figures_the_same_way_every_time(tmp_path):
    work, _ = mini_setup(tmp_path)

    first = run_cli(work, "evaluate", "agreement", "--out", "out")
    second = run_cli(work, "evaluate", "agreement", "--out", "again")

    assert first.returncode == second.returncode == 0, first.stderr + second.stderr
    for name in ("fig1_agreement_tables", "fig2_observed_vs_chance"):
        png = (work / "out" / f"{name}.png").read_bytes()
        assert (
            png.startswith(b"\x89PNG")
            and png == (work / "again" / f"{name}.png").read_bytes()
        )
        svgs = [
            re.sub(
                r"<dc:date>[^<]*</dc:date>", "", (work / d / f"{name}.svg").read_text()
            )
            for d in ("out", "again")
        ]
        assert svgs[0] == svgs[1]  # only the date differs
    fig1 = (work / "out/fig1_agreement_tables.svg").read_text()
    assert "Q1: 1 record lacks one answer" in fig1
    fig2 = (work / "out/fig2_observed_vs_chance.svg").read_text()
    assert "alpha = 15.0 / 40.0 = 0.38" in fig2  # Q1: 75.0% observed, 60.0% by luck


def test_agreement_writes_what_the_curator_did(tmp_path):
    work, _ = mini_setup(tmp_path)

    done = run_cli(work, "evaluate", "agreement", "--out", "out")

    assert done.returncode == 0, done.stderr
    curation = load(work / "out/curation.json")
    assert curation["export"] == "$DATA_ROOT/groundtruth/validated-v3.zip"
    assert curation["records"] == 9
    assert curation["workload"]["individually_reviewed_by_curator"] == 2  # A2, B3
    assert curation["workload"]["biasdef130_records"] == 5
    assert curation["labels"]["label_set_by_curator_not_by_vote"] == 2
    assert curation["notes"]["free_text_notes"] == 1
    changes = curation["changes_vs_v3_in_biasdef130"]
    assert (changes["changed"], changes["unchanged"]) == (2, 3)  # A2 and B5 edited
    assert changes["changed_only_by_a_final_touch"] == 1  # B5
    assert curation["assistant"]["records_with_a_proposal"] == 2
    assert curation["assistant"]["fields_where_curator_accepted_assistant_text"] == 1
    assert [e["record"] for e in curation["examples"]] == ["3457607:2"]


def test_the_pseudonymised_copies_give_the_same_numbers(tmp_path):
    work, _ = mini_setup(tmp_path)
    made = run_cli(work, "groundtruth", "pseudonymise", "--out", "copies")
    assert made.returncode == 0, made.stderr

    original = run_cli(work, "evaluate", "agreement", "--out", "a")
    copied = run_cli(
        work,
        "evaluate",
        "agreement",
        "--out",
        "b",
        "--annotations",
        "copies/newAnnotations/Annotations.zip",
        "--export",
        "copies/validated-v3.zip",
    )

    assert original.returncode == copied.returncode == 0, (
        original.stderr + copied.stderr
    )
    assert copied.stdout.splitlines()[:2] == [
        "Q1 8 5 1 2 75.00 0.375",
        "Q2 9 6 1 2 77.78 0.3929",
    ]
    agreement_a, agreement_b = (
        load(work / "a/agreement.json"),
        load(work / "b/agreement.json"),
    )
    assert agreement_b.pop("inputs") == {
        **agreement_a.pop("inputs"),
        "raw_exports": "copies/newAnnotations/Annotations.zip",
        "released_labels": "copies/validated-v3.zip: relevance-labels.json",
    }
    assert agreement_a == agreement_b
    curation_a, curation_b = (
        load(work / "a/curation.json"),
        load(work / "b/curation.json"),
    )
    assert curation_b.pop("export") == "copies/validated-v3.zip"
    curation_a.pop("export")
    assert curation_a == curation_b
    for name in ("fig1_agreement_tables.png", "fig2_observed_vs_chance.png"):
        assert (work / "a" / name).read_bytes() == (work / "b" / name).read_bytes()


def test_without_the_originals_agreement_reads_the_copies_by_default(tmp_path):
    """A release's data root holds only the pseudonymised copies of the two exports: `agreement` reads them by default and names them as its inputs."""
    work, root = mini_setup(tmp_path)
    for name in ("validated-v3.zip", "newAnnotations/Annotations.zip"):
        target = root / "groundtruth/pseudonymised" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        (root / "groundtruth" / name).rename(target)

    done = run_cli(work, "evaluate", "agreement", "--out", "out")

    assert done.returncode == 0, done.stderr
    inputs = load(work / "out/agreement.json")["inputs"]
    assert inputs["raw_exports"] == (
        "$DATA_ROOT/groundtruth/pseudonymised/newAnnotations/Annotations.zip"
    )
    assert inputs["released_labels"].startswith(
        "$DATA_ROOT/groundtruth/pseudonymised/validated-v3.zip"
    )
