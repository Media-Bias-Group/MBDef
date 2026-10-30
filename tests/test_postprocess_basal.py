"""`mbdef postprocess basal`: basal's items, its client and its score tables equal the original scorer's.

tests/fixtures/basal-mini/ holds a made-up corpus store, the replies a stand-in server gives, and the outputs of the original scorer on them, which the tests compare with byte for byte: the package must write the same files. The tests on the real files compare with the frozen score tables of Step 5 (in git) and, marked slow, with the corpus items and the corpus table (they need the corpus store and the recorded corpus answers in the data root).
"""

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from cli_helpers import mini_gold, real_mbdef_gold, run_cli
from mini_basal import MINI, StandIn, make_store

from mbdef import files, postprocess

REPO = Path(__file__).resolve().parents[1]
EXPECTED = MINI / "expected"
# The ground-truth items of the recorded runs (all 2,328 records; in git)
RECORDED_GT_ITEMS = REPO / "data/frozen/results/core-satellite-decision-v1/items.jsonl"
FROZEN = REPO / postprocess.FROZEN
# The corpus items the recorded corpus run asked (its items.jsonl, 67,394 lines; not in git)
CORPUS_ITEMS_SHA256 = "9806c9e9d5c5d8b204cea982c972e42f1ccd1fb7ab35d663fd971ecf51efdcc3"
# MBDef-Gold, the ground truth whose items `items` also writes: in the data root, if it is there
GOLD, needs_gold = real_mbdef_gold()


@pytest.fixture(scope="module")
def mini_items(tmp_path_factory):
    """`postprocess basal items` on the mini store and MBDef-Gold (the real one in the data root if it is there, else a made-up one): the folder it wrote."""
    tmp = tmp_path_factory.mktemp("basal-items")
    store = make_store(tmp / "store.sqlite3")
    root = GOLD.parent if (GOLD / "validated_gt.json").is_file() else mini_gold(tmp)
    done = run_cli(
        REPO,
        *("postprocess", "basal", "items", "--store", str(store)),
        *("--links", str(MINI / "inputs/links.csv"), "--out", str(tmp / "run")),
        env={"DATA_ROOT": str(root)},
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return tmp / "run"


def test_items_are_each_term_with_its_first_three_definitions_cleaned_and_cut(
    mini_items,
):
    """Whitespace collapsed, U+FFFD removed, a definition over 600 characters cut to 599 + "…", at most 3 per term, only the P≥28 view: the original item builder's lines, byte for byte."""
    assert (mini_items / "items.jsonl").read_bytes() == (
        EXPECTED / "items.jsonl"
    ).read_bytes()


def test_the_definitions_shown_and_their_length_are_flags(tmp_path):
    """`items --max-definitions 1 --max-chars 10`: each state shows the first definition only, cut to 9 characters
    + "…" when longer than 10."""
    store = make_store(tmp_path / "store.sqlite3")
    done = run_cli(
        REPO,
        *("postprocess", "basal", "items", "--store", str(store)),
        *("--links", str(MINI / "inputs/links.csv"), "--out", str(tmp_path / "run")),
        *("--max-definitions", "1", "--max-chars", "10"),
        env={"DATA_ROOT": str(mini_gold(tmp_path))},
    )
    assert done.returncode == 0, done.stdout + done.stderr
    default = [
        json.loads(x) for x in (EXPECTED / "items.jsonl").read_text().splitlines()
    ]
    mine = [
        json.loads(x) for x in (tmp_path / "run/items.jsonl").read_text().splitlines()
    ]
    assert [x["id"] for x in mine] == [x["id"] for x in default]
    for short, full in zip(mine, default, strict=True):
        first = full["state"]["definitions in this paper"][0]
        assert short["state"]["definitions in this paper"] == [
            first if len(first) <= 10 else first[:9] + "…"
        ]


def test_items_writes_a_receipt_with_its_settings(mini_items):
    receipt = json.loads((mini_items / "receipt.json").read_text())
    assert receipt["command"] == "mbdef postprocess basal items"
    assert receipt["settings"] == {"max_definitions": 3, "max_chars": 600}
    assert set(receipt["outputs"]) == {"items.jsonl", "groundtruth-items.jsonl"}
    assert {"store", "links"} <= set(receipt["inputs"])


@needs_gold
def test_the_ground_truth_items_are_the_recorded_ones_of_train_and_dev(mini_items):
    """The reference items: every train + dev record of the ground truth, in its order, as the recorded runs asked them; no TEST record is written."""
    recorded = [
        line
        for line in RECORDED_GT_ITEMS.read_text(encoding="utf-8").splitlines(
            keepends=True
        )
        if json.loads(line)["split"] in ("train", "dev")
    ]
    written = (mini_items / "groundtruth-items.jsonl").read_text(encoding="utf-8")
    assert written == "".join(recorded)
    assert len(recorded) == 1700


def ask(server, items, answers, *extra):
    return run_cli(
        REPO,
        *("postprocess", "basal", "ask", "--server", server.url),
        *("--items", str(items), "--out", str(answers), "--concurrency", "1", *extra),
    )


def test_the_client_sends_the_original_requests_and_writes_the_original_answers(
    tmp_path,
):
    """The same request bodies in the same order (the recorded run's order seed), a malformed answer written as an error line and the run stopping with exit code 1, and a second run asking only the item without an answer."""
    items = Path(shutil.copy(EXPECTED / "items.jsonl", tmp_path / "items.jsonl"))
    answers = tmp_path / "answers.jsonl"

    with StandIn(items, MINI / "inputs/replies.jsonl") as server:
        first = ask(server, items, answers)
        second = ask(server, items, answers)
        received = list(server.received)

    assert first.returncode == 1, first.stdout + first.stderr
    assert "1 item(s) without an answer; run the same command again" in first.stderr
    assert second.returncode == 0, second.stdout + second.stderr
    assert "1 items to ask (8 already answered)" in second.stdout
    assert received == (EXPECTED / "requests.jsonl").read_text().splitlines()
    # answers are written as they come back, so only the set of lines is fixed
    assert sorted(answers.read_text().splitlines()) == sorted(
        (EXPECTED / "answers.jsonl").read_text().splitlines()
    )


def table(items, answers, out, *extra):
    return run_cli(
        REPO,
        *("postprocess", "basal", "table", "--items", str(items)),
        *("--answers", str(answers), "--out", str(out), *extra),
    )


def test_the_table_is_p_core_plus_p_satellite_of_the_first_answer_of_each_item(
    tmp_path,
):
    """One row per item in the items' order; an error line is skipped and a later second answer of an item is not used (two servers can answer the same item)."""
    answers = shutil.copy(EXPECTED / "answers.jsonl", tmp_path / "answers.jsonl")
    with Path(answers).open("a") as f:  # a second answer of m01, to be ignored
        f.write(
            json.dumps(
                {"id": "m01", "p": {"core": 0.0, "satellite": 0.0, "irrelevant": 1.0}}
            )
            + "\n"
        )

    done = table(
        EXPECTED / "items.jsonl",
        answers,
        tmp_path / "table.csv",
        "--method",
        "basal mini",
    )

    assert done.returncode == 0, done.stdout + done.stderr
    assert (tmp_path / "table.csv").read_bytes() == (
        EXPECTED / "table.csv"
    ).read_bytes()


def test_the_table_is_refused_while_an_item_has_no_answer(tmp_path):
    answers = tmp_path / "answers.jsonl"
    answers.write_text(
        "".join(
            line + "\n"
            for line in (EXPECTED / "answers.jsonl").read_text().splitlines()
            if '"m04"' not in line
        )
    )

    done = table(EXPECTED / "items.jsonl", answers, tmp_path / "table.csv")

    assert done.returncode == 1
    assert "1 of 9 items have no answer in" in done.stderr
    assert "m04" in done.stderr
    assert not (tmp_path / "table.csv").exists()


@pytest.mark.parametrize(
    ("answers", "frozen"),
    [
        (  # basal's reference for the ensemble: the H200 run over the ground truth
            "20261021/corpus-decision-models/out/basal-fp8-h200-gt-r1.jsonl",
            "20261021/corpus-decision-models/scores/basal-fp8-h200-gt-r1.csv",
        ),
        (  # the table the ensemble's weights were fitted on: the H100 run 1
            "20261020/decision-models/verdicts/basal-fp8-h100-r1.jsonl",
            "20261020/decision-models/scores/basal-fp8-h100-r1.csv",
        ),
    ],
)
def test_the_frozen_ground_truth_tables_come_back_byte_for_byte(
    tmp_path, answers, frozen
):
    """From the recorded answers and the recorded runs' item list, which also holds the TEST records: the table copies their ids, splits and scores like any other row, and nothing reads a label."""
    done = table(
        RECORDED_GT_ITEMS,
        FROZEN / answers,
        tmp_path / "table.csv",
        *("--method-of", str(FROZEN / frozen)),
    )

    assert done.returncode == 0, done.stdout + done.stderr
    assert (tmp_path / "table.csv").read_bytes() == (FROZEN / frozen).read_bytes()


@pytest.mark.slow
def test_the_frozen_corpus_items_and_corpus_table_come_back_byte_for_byte(
    tmp_path,
):
    """The real corpus store gives the recorded corpus items, and the recorded corpus answers (data root) give the frozen corpus table (CPU only, about a minute)."""
    files.load_env()
    try:  # over 10 MB, so in the data root (data/data-root.tsv)
        answers = postprocess.frozen_file(
            "20261021/corpus-decision-models/out/basal-fp8-h200-r1-s1of1.jsonl"
        )
    except SystemExit:
        pytest.skip("DATA_ROOT is not set")
    if not answers.exists():
        pytest.skip(f"{answers} is not there (mbdef data fetch)")
    frozen = FROZEN / "20261021/corpus-decision-models/scores/basal-fp8-h200-corpus.csv"

    done = run_cli(REPO, "postprocess", "basal", "items", "--out", str(tmp_path))
    assert done.returncode == 0, done.stdout + done.stderr
    items = tmp_path / "items.jsonl"
    assert hashlib.sha256(items.read_bytes()).hexdigest() == CORPUS_ITEMS_SHA256

    done = table(items, answers, tmp_path / "corpus.csv", "--method-of", str(frozen))
    assert done.returncode == 0, done.stdout + done.stderr
    assert (tmp_path / "corpus.csv").read_bytes() == frozen.read_bytes()


def test_no_step_writes_into_a_frozen_folder(tmp_path):
    """A frozen folder of a throwaway data root, so that a broken refusal could not write into `data/frozen/`."""
    root = tmp_path / "data-root"
    inside = root / "frozen/step5-post-processing/20261021/new-folder"
    items, answers = EXPECTED / "items.jsonl", EXPECTED / "answers.jsonl"
    for step in (
        ["items", "--out", str(inside)],
        ["ask", "--items", str(items), "--out", str(inside / "answers.jsonl")],
        ["table", "--items", str(items), "--answers", str(answers)]
        + ["--out", str(inside / "table.csv")],
    ):
        done = run_cli(
            REPO, "postprocess", "basal", *step, env={"DATA_ROOT": str(root)}
        )
        assert done.returncode == 1, step
        assert "is inside a frozen folder" in done.stderr, step
    assert not inside.exists()
