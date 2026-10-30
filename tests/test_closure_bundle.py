"""`mbdef closure bundle` on the miniature closure of tests/mini_bundle.py (worked out by hand there).

The bundle leaves out a collected paper whose text is a seed paper's, adds the work found only in the existing corpus, links every seed definition to its cited work, takes the identity check's decisions and builds the release folder, then verifies every manifest line. The release is the same every time apart from its build time.
"""

import csv
import hashlib
import json
import time
from pathlib import Path

import pytest
from cli_helpers import run_cli
from mini_bundle import World, without_build_time


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def run_bundle(world: World, out: Path, *extra: str):
    return run_cli(
        world.work,
        "closure",
        "bundle",
        "--out",
        str(out),
        "--ledgers",
        str(world.ledgers),
        "--closure-set",
        str(world.closure_set),
        "--closure-tei",
        str(world.closure_tei),
        "--left-out-tei",
        str(world.left_out_tei),
        "--audit-inputs",
        str(world.v11),
        "--seeds",
        str(world.seeds),
        "--seed-inputs",
        str(world.seed_inputs),
        "--identity",
        str(world.identity),
        "--loss-audit",
        str(world.loss_audit),
        *extra,
    )


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def listing(folder: Path) -> dict[str, str]:
    return {
        p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


def test_the_release_holds_the_collected_and_existing_papers_and_leaves_out_the_copy(
    world, tmp_path
):
    out = tmp_path / "bundle"

    done = run_bundle(world, out)

    assert done.returncode == 0, done.stderr
    release = out / "release"
    # The release: two seed papers; closure papers W1 (collected) and W4 (the existing corpus); W5 is left out.
    assert [r["paper_id"] for r in rows(release / "seeds/papers.csv")] == [
        "seed-a",
        "seed-b",
    ]
    closure = rows(release / "closure/papers.csv")
    assert sorted(r["title"] for r in closure) == [
        "A work only in the existing corpus",
        "Child work",
    ]
    assert {r["paper_id"] for r in closure} == {world.ids["w1"], world.ids["w4"]}
    by_title = {r["title"]: r for r in closure}
    assert by_title["Child work"]["pdf_identity_check"] == "doi"
    assert by_title["A work only in the existing corpus"]["pdf_identity_check"] == (
        "existing_corpus"
    )
    # W1 is cited by both seed papers' definitions; W4 by one.
    assert by_title["Child work"]["n_citing_seed_papers"] == "2"
    assert by_title["A work only in the existing corpus"]["n_citing_seed_papers"] == "1"

    exclusions = json.loads((out / "audit/exclusions.json").read_text())
    assert exclusions == [
        {
            "paper_id": world.ids["w5"],
            "reason": "same text as a seed paper",
            "seed_paper_id": "seed-b",
            "share": 1.0,
        }
    ]
    assert (release / "provenance/closure/exclusions.json").read_bytes() == (
        out / "audit/exclusions.json"
    ).read_bytes()
    assert not any(world.ids["w5"] in name for name in listing(release))

    # One link row per resolved citation: six; W5's rows name no closure paper (it is seed-b's own text).
    links = rows(release / "links/definition-links.csv")
    assert len(links) == 6
    status = {
        r["reference_title"]: (r["work_status"], r["closure_paper_id"]) for r in links
    }
    assert status["Child work"] == ("extracted", world.ids["w1"])
    assert status["A work only in the existing corpus"] == (
        "extracted",
        world.ids["w4"],
    )
    assert status["Seed paper A"][0] == "present_in_corpus"
    assert status["Another work without open access"] == ("resolved_no_oa_pdf", "")
    assert status["Seed paper B as a preprint"] == ("present_in_corpus", "")
    assert len(rows(release / "links/cited-works.csv")) == 5

    # The README's numbers are the release's.
    readme = (release / "README.md").read_text()
    assert "**Seed papers** (2)" in readme and "**Closure papers** (2)" in readme
    assert "1 collected papers are left out" in readme
    assert "2 of them are cited in a definition sentence itself" in readme

    # Every file is in the manifest, and every line checks out (the command verified them: 'N files').
    manifest = (release / "MANIFEST.sha256").read_text().splitlines()
    files = {k for k in listing(release) if k != "MANIFEST.sha256"}
    assert len(manifest) == len(files)
    assert {line.split("  ", 1)[1] for line in manifest} == files
    assert f"release: {release} ({len(manifest)} files verified)" in done.stdout


def test_the_identity_decisions_are_the_models_verdicts_checked_by_the_program(
    world, tmp_path
):
    out = tmp_path / "bundle"
    assert run_bundle(world, out).returncode == 0

    identity = out / "identity"
    first = json.loads((identity / "accepted.json").read_text())
    # Round 1: the model said yes and the program confirmed it. Round 2: yes, but the program refused it.
    # Round 3: the model said it is a different work. Only the first is accepted, and it stays accepted.
    assert [a["paperid"] for a in first["accepted"]] == [world.candidate_ids[0]]
    assert json.loads((identity / "r2/accepted.json").read_text())["accepted"] == []
    assert json.loads((identity / "r3/accepted.json").read_text())["accepted"] == []
    every = json.loads((identity / "accepted-all-r3.json").read_text())
    assert [a["paperid"] for a in every["accepted"]] == [world.candidate_ids[0]]
    assert list(every["rounds"]) == ["1", "2", "3"]
    summary = json.loads((out / "receipt.json").read_text())["summary"]["identity"]
    assert summary == {
        "round_1": 1,
        "round_2": 0,
        "after_round_2": 1,
        "round_3": 0,
        "after_round_3": 1,
    }


def test_the_receipt_names_the_inputs_the_settings_and_the_counts(world, tmp_path):
    out = tmp_path / "bundle"
    assert run_bundle(world, out).returncode == 0

    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["command"] == "mbdef closure bundle"
    assert receipt["settings"]["version"] == "v1.1"
    assert (
        receipt["settings"]["duplicate_share"],
        receipt["settings"]["shingle_words"],
    ) == (0.8, 8)
    assert (
        receipt["inputs"]["closure_tei"]["files"] == 6
    )  # three papers, each a TEI and a receipt
    assert (
        receipt["inputs"]["seed_rows"]["sha256"]
        == hashlib.sha256(
            (world.seed_inputs / "runner-rows.json").read_bytes()
        ).hexdigest()
    )
    summary = receipt["summary"]
    assert summary["duplicates"] == {
        "seeds": 2,
        "candidates": 3,
        "of_a_seed": 1,
        "of_another_candidate": 0,
    }
    assert summary["exclusions"] == {
        "left_out": 1,
        "seed_copies": 1,
        "closure_copies": 0,
        "not_cited": 0,
    }
    assert summary["existing"] == {
        "existing_only_works": 1,
        "added": 1,
        "equal_to_collected": 1,
    }
    assert summary["release"]["seed_papers"] == 2
    assert summary["release"]["closure_papers"] == 2
    assert summary["links"]["grouping_identical"] is True


def test_two_builds_differ_only_in_the_time_they_were_built(world, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    assert run_bundle(world, first).returncode == 0
    time.sleep(1.1)  # the build time is read to the second
    assert run_bundle(world, second).returncode == 0

    a, b = listing(first / "release"), listing(second / "release")
    differ = sorted(name for name in a if a[name] != b[name])
    assert differ == ["MANIFEST.sha256", "README.md", "provenance/release-facts.json"]
    # In the README and the facts it is one line, the build time ...
    readme = [
        (d / "release/README.md").read_text().splitlines() for d in (first, second)
    ]
    changed = [(x, y) for x, y in zip(*readme, strict=True) if x != y]
    assert len(changed) == 1 and changed[0][0].startswith("Built 20")
    facts = [
        json.loads((d / "release/provenance/release-facts.json").read_text())
        for d in (first, second)
    ]
    assert {k for k in facts[0] if facts[0][k] != facts[1][k]} == {"built_utc"}
    # ... and with the build time taken out, the two builds are equal.
    assert without_build_time(first / "release") == without_build_time(
        second / "release"
    )


def test_a_collected_paper_the_audit_reference_does_not_know_stops_the_run(
    world, tmp_path
):
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "duplicate-texts-v11.json").write_text("{}")
    (reference / "exclusions.json").write_text("[]\n")

    done = run_bundle(world, tmp_path / "bundle", "--audit-reference", str(reference))

    assert done.returncode == 1
    assert (
        "an audit result differs from the reference in: duplicate-texts.json"
        in done.stderr
    )


def test_an_input_that_is_missing_stops_with_the_way_to_fetch_it(world, tmp_path):
    done = run_bundle(
        world, tmp_path / "bundle", "--ledgers", str(tmp_path / "nowhere")
    )

    assert done.returncode == 1
    assert f"{tmp_path / 'nowhere'} does not exist (ledgers)" in done.stderr
    assert "data fetch frozen/" in done.stderr
    assert not (tmp_path / "bundle").exists()


def test_a_missing_input_that_is_not_public_is_named_as_such(world, tmp_path):
    """The closure papers' extraction is not public: the message names that input (not the TEIs), says it is not public and how to give your own."""
    done = run_bundle(
        world, tmp_path / "bundle", "--closure-set", str(tmp_path / "nowhere")
    )

    assert done.returncode == 1
    assert f"{tmp_path / 'nowhere'} does not exist (closure_set)" in done.stderr
    assert "the closure papers' extraction" in done.stderr
    assert "not public: its records are not in the public deposit" in done.stderr
    assert "--closure-set" in done.stderr
    assert "development" not in done.stderr
    assert "TEI" not in done.stderr
    assert not (tmp_path / "bundle").exists()
