"""Term merging after the relevance decisions: a merged row is a Label Group (Label Match C-20261021) restricted to its kept Extracted Concepts, with the provenance of every join.

tests/fixtures/term-merging-mini/ holds a miniature corpus view, its Label Groups and decisions, and the outputs of the original implementation on them, which the tests compare with byte for byte.
"""

import json
import shutil
from pathlib import Path

import pytest
from cli_helpers import run_cli
from test_postprocess_ensemble import frozen_or_skip

from mbdef.postprocess import label_groups, term_merging
from mbdef.postprocess.corpus import (
    CorpusView,
    Definition,
    ExtractedConcept,
)

MINI = Path(__file__).parent / "fixtures/term-merging-mini"
REPO = Path(__file__).resolve().parents[1]


def mini_view():
    rows = json.loads((MINI / "view.json").read_text())
    return CorpusView(
        frozenset(rows["papers"]),
        [
            ExtractedConcept(
                c["mention_id"],
                c["paper_id"],
                c["corpus"],
                c["label"],
                tuple(Definition(i, t) for i, t in c["definitions"]),
            )
            for c in rows["concepts"]
        ],
    )


def merge_terms(out, decisions, label_groups):
    """`postprocess merge-terms`; its summary.json (a failed command raises with its message)."""
    done = run_cli(
        REPO,
        *("postprocess", "merge-terms", "--decisions", str(decisions)),
        *("--label-groups", str(label_groups), "--out", str(out)),
    )
    if done.returncode != 0:
        raise SystemExit(done.stderr)
    return json.loads((out / "summary.json").read_text())


def run_mini(out, decisions=MINI / "decisions.csv"):
    return merge_terms(out, decisions, MINI / "label-groups")


@pytest.mark.parametrize("name", ["terms.jsonl", "groups.jsonl", "labels.jsonl"])
def test_the_label_groups_of_rule_c_20261021_equal_the_original_implementations(
    tmp_path, name
):
    label_groups.write_label_groups(mini_view(), tmp_path, "C-20261021")
    assert (tmp_path / name).read_bytes() == (MINI / "label-groups" / name).read_bytes()


def test_rule_c_20261015_writes_its_own_format_without_provenance(tmp_path):
    label_groups.write_label_groups(mini_view(), tmp_path, "C-20261015")
    assert not (tmp_path / "labels.jsonl").exists()
    first = json.loads((tmp_path / "groups.jsonl").read_text().splitlines()[0])
    assert "joined_by" not in first
    assert "C-20261015" in json.loads((tmp_path / "summary.json").read_text())["rule"]


@pytest.mark.parametrize(
    "name", ["concepts.jsonl", "split-labels.csv", "majority-disagreements.csv"]
)
def test_the_outputs_equal_the_work_branch_scripts_byte_for_byte(tmp_path, name):
    run_mini(tmp_path)
    assert (tmp_path / name).read_bytes() == (MINI / "expected" / name).read_bytes()


def test_the_summary_counts_equal_the_work_branch_scripts(tmp_path):
    summary = run_mini(tmp_path)
    for key in ("provenance", "rule"):
        summary.pop(key)
    summary["decisions"].pop("path")
    summary["label_groups"].pop("path")
    expected = json.loads((MINI / "expected/summary.json").read_text())
    assert summary == expected


def test_a_dropped_bridge_label_splits_its_group_when_rule_c_reruns_on_the_kept_labels(
    tmp_path,
):
    # "media bias" joins "bias in the media" and "MediaBias"; every "media bias" is dropped in the miniature
    summary = run_mini(tmp_path)
    check = summary["check_rule_c_on_kept_only"]
    assert check["kept_instances_in_a_different_grouping"] == 2
    assert check["groups"] == summary["after"]["concepts"] + 1
    rows = [json.loads(x) for x in (tmp_path / "concepts.jsonl").open()]
    bridged = next(r for r in rows if "MediaBias" in r["labels"])
    assert "joined through another label" in bridged["merge_flags"]


def test_every_kept_extracted_concept_carries_its_keys_and_normalisation(tmp_path):
    run_mini(tmp_path)
    rows = [json.loads(x) for x in (tmp_path / "concepts.jsonl").open()]
    refugees = next(i for r in rows for i in r["instances"] if i["term"] == "refugees")
    assert refugees["word_set_key"] == "refugee"
    assert any(
        s.startswith("plural 'refugees' -> 'refugee'")
        for s in refugees["normalisation"]
    )


def test_decisions_for_other_terms_than_the_label_groups_are_refused(tmp_path):
    lines = (MINI / "decisions.csv").read_text().splitlines(keepends=True)
    short = tmp_path / "decisions.csv"
    short.write_text("".join(lines[:-1]))
    with pytest.raises(SystemExit, match="not the same terms"):
        run_mini(tmp_path / "out", decisions=short)


@pytest.mark.slow
def test_the_frozen_term_merging_comes_back(tmp_path):
    """The real run: the ensemble's decisions and the Label Groups of Label Match C-20261021 (from the data root)."""
    frozen = frozen_or_skip("20261021/term-merging/concepts.jsonl")
    frozen_or_skip(f"{term_merging.LABEL_GROUPS}/groups.jsonl")
    shutil.rmtree(tmp_path)
    inputs = term_merging.default_inputs()
    summary = merge_terms(tmp_path, inputs["decisions"], inputs["label_groups"])
    assert (tmp_path / "concepts.jsonl").read_bytes() == frozen.read_bytes()
    stored = json.loads(
        frozen_or_skip("20261021/term-merging/summary.json").read_text()
    )
    for key in ("before", "kept", "after", "split_groups", "majority", "merge_flags"):
        assert summary[key] == stored[key]
