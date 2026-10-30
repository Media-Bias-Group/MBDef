"""`mbdef evaluate score` on two TEST papers, with recorded pair scores and no model.

The sample `tests/fixtures/score-sample/` (built by `tests/fixtures/make_score_sample.py`) holds the reference definitions of two held-out TEST papers, the saved outputs of two released TEST rows for them (Opus `Rdb54680` and Qwen `QWEN-POOL3`, byte copies of the released run), and the BGE similarities and ModernBERT entailment probabilities that the real models gave for every pair the scorer asks about (recorded on a GPU). The expected numbers are the released run's own per-paper results in its `final/results.json`, so the test checks the scorer against that run, not against itself.
"""

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from cli_helpers import package_commit, run_cli, write_env

SAMPLE = Path(__file__).resolve().parent / "fixtures/score-sample"
PAPERS = (
    "4243aa9bacd51bc02d4240378e7528912962b53e",  # 4 reference definitions
    "73bfad11b96a69cb882028ead115751adb55252d",  # 6 reference definitions
)
# final/results.json of the released TEST run: rows[].score.per_paper of the two papers, (tp, fp, fn, full-record score).
RELEASED = {
    "Rdb54680": {
        PAPERS[0]: (2, 17, 2, 0.049105839202481294),
        PAPERS[1]: (5, 3, 1, 0.1957301079946295),
    },
    "QWEN-POOL3": {
        PAPERS[0]: (4, 28, 0, 0.11111834912432772),
        PAPERS[1]: (6, 8, 0, 0.4357661951349362),
    },
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sample_setup(tmp_path):
    """A work folder with an empty `.env` and the fixture unpacked into it."""
    work = tmp_path / "work"
    work.mkdir()
    write_env(work)
    shutil.copytree(SAMPLE, work / "sample")
    return work


def score(work: Path, *extra: str):
    return run_cli(
        work,
        "evaluate",
        "score",
        "--scorings",
        "sample/scorings.json",
        "--replay-scores",
        "sample/pair-scores.json",
        "--out",
        "out",
        *extra,
    )


def test_each_scoring_is_scored_as_the_released_test_run_scored_it(tmp_path):
    work = sample_setup(tmp_path)

    done = score(work)

    assert done.returncode == 0, done.stderr
    scored = {s["id"]: s for s in read(work / "out/scores.json")["scorings"]}
    assert list(scored) == ["Rdb54680", "QWEN-POOL3"]
    for row, papers in RELEASED.items():
        result = scored[row]["score"]
        per_paper = {p["paper_id"]: p for p in result["per_paper"]}
        assert list(per_paper) == list(PAPERS)
        for paper, (tp, fp, fn, full) in papers.items():
            counts = per_paper[paper]["binary_one_to_one"]
            assert (counts["tp"], counts["fp"], counts["fn"]) == (tp, fp, fn)
            assert counts["gold"] == tp + fn and counts["predicted"] == tp + fp
            assert per_paper[paper]["full_record"]["score"] == pytest.approx(
                full, abs=1e-6
            )
    # Pooled over the two papers: 10 reference definitions.
    opus = scored["Rdb54680"]["score"]
    assert {k: opus["binary_one_to_one"][k] for k in ("tp", "fp", "fn")} == {
        "tp": 7,
        "fp": 20,
        "fn": 3,
    }
    assert f"{opus['binary_one_to_one']['precision']:.4f}" == "0.2593"  # 7 / 27
    assert f"{opus['binary_one_to_one']['recall']:.4f}" == "0.7000"  # 7 / 10
    assert f"{opus['full_record_macro_score']:.4f}" == "0.1224"
    qwen = scored["QWEN-POOL3"]["score"]
    assert f"{qwen['binary_one_to_one']['precision']:.4f}" == "0.2174"  # 10 / 46
    assert f"{qwen['binary_one_to_one']['recall']:.4f}" == "1.0000"
    assert f"{qwen['full_record_macro_score']:.4f}" == "0.2734"
    assert done.stdout.splitlines() == [
        "Rdb54680: 7 of 10 reference definitions found, 27 predicted "
        "(precision 25.9%, recall 70.0%), full-record score 0.122",
        "QWEN-POOL3: 10 of 10 reference definitions found, 46 predicted "
        "(precision 21.7%, recall 100.0%), full-record score 0.273",
        "wrote out",
    ]


def test_the_receipt_records_the_pinned_settings_and_every_input(tmp_path):
    work = sample_setup(tmp_path)

    done = score(work)

    assert done.returncode == 0, done.stderr
    receipt = read(work / "out/receipt.json")
    assert receipt["command"] == "evaluate score"
    assert receipt["code"]["commit"] == package_commit()  # None outside a checkout
    settings = receipt["settings"]
    assert settings["m3"] == {
        "slope": 1.4028410596332237,
        "intercept": -0.2973741431607951,
        "pair_score": "clip(slope * (token F1 + BGE cosine + mean of both "
        "entailment directions) / 3 + intercept, 0, 1)",
    }
    assert settings["threshold"] == 0.5
    assert settings["similarity_model"] == {
        "name": "BAAI/bge-base-en-v1.5",
        "revision": "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
        "max_length": 512,
        "batch_size": 32,
    }
    assert settings["entailment_model"] == {
        "name": "tasksource/ModernBERT-large-nli",
        "revision": "ca476cb923a8637073d4ceb0f19f7fc236e260d4",
        "max_length": 2048,
        "batch_size": 32,
    }
    assert settings["pair_scores"] == "replayed from sample/pair-scores.json"
    inputs = receipt["inputs"]
    assert inputs["scorings"]["sha256"] == sha(work / "sample/scorings.json")
    assert inputs["pair_scores"]["sha256"] == sha(work / "sample/pair-scores.json")
    assert inputs["Rdb54680/reference"]["sha256"] == sha(work / "sample/reference.json")
    assert inputs["Rdb54680/papers"]["files"] == 2
    assert receipt["outputs"]["scores.json"]["sha256"] == sha(work / "out/scores.json")


def test_a_scoring_that_fails_stops_the_command_and_writes_no_results(tmp_path):
    work = sample_setup(tmp_path)
    broken = work / "sample/rows/QWEN-POOL3/papers" / f"{PAPERS[1]}.json"
    paper = read(broken)
    paper["definition_records"][0]["definitions"][0]["text"] = "  "
    broken.write_text(json.dumps(paper), encoding="utf-8")

    done = score(work)

    assert done.returncode == 1
    assert done.stderr.strip().splitlines()[-1] == (
        f"Scoring QWEN-POOL3 failed: ValueError: {PAPERS[1]}: Invalid definition text. "
        "No results were written."
    )
    assert not (work / "out/scores.json").exists()
    assert not (work / "out/receipt.json").exists()


def test_a_pair_score_that_was_not_recorded_stops_the_scoring(tmp_path):
    """The replay guard. A changed definition asks for pairs nobody recorded."""
    work = sample_setup(tmp_path)
    changed = work / "sample/rows/Rdb54680/papers" / f"{PAPERS[0]}.json"
    paper = read(changed)
    paper["definition_records"][0]["definitions"][0]["text"] += " (edited)"
    changed.write_text(json.dumps(paper), encoding="utf-8")

    done = score(work)

    assert done.returncode == 1
    last = done.stderr.strip().splitlines()[-1]
    assert last.startswith(
        "Scoring Rdb54680 failed: NotRecorded: no recorded similarity for ("
    )
    assert last.endswith("No results were written.")
    assert not (work / "out/scores.json").exists()


def test_out_must_be_a_new_or_empty_folder(tmp_path):
    work = sample_setup(tmp_path)
    (work / "out").mkdir()
    (work / "out/old.json").write_text("{}", encoding="utf-8")

    done = score(work)

    assert done.returncode == 1
    assert done.stderr.strip() == "out is not empty; choose a new folder for --out."


def test_method_settings_in_env_are_ignored(tmp_path):
    """`.env` holds machine settings only: model keys such as DSPY_MODEL or SCIDEF_NLI_MODEL change nothing in a command of the package."""
    for name in ("plain", "keyed"):
        (tmp_path / name).mkdir()
    plain = sample_setup(tmp_path / "plain")
    keyed = sample_setup(tmp_path / "keyed")
    write_env(
        keyed,
        SCIDEF_NLI_MODEL="someone/another-nli-model",
        SCIDEF_NLI_BATCH_SIZE="8",
        DSPY_MODEL="someone/another-llm",
        DSPY_MAX_TOKENS="5",
    )

    assert score(plain).returncode == 0
    done = score(keyed)

    assert done.returncode == 0, done.stderr
    settings = read(keyed / "out/receipt.json")["settings"]
    assert settings["entailment_model"]["name"] == "tasksource/ModernBERT-large-nli"
    assert settings["entailment_model"]["batch_size"] == 32

    def scored(work):
        return [s["score"] for s in read(work / "out/scores.json")["scorings"]]

    assert scored(keyed) == scored(plain)


# --predictions: a predictions file (one record per line) against a split of MBDef-Gold


def gold_file(work: Path) -> Path:
    """An MBDef-Gold record file: the sample's two TEST papers, and one TRAIN paper that a TEST scoring never reads."""
    papers = read(work / "sample/reference.json")["papers"]
    train = {
        "paper_id": "train-paper",
        "split": "train",
        "definition_records": [{"term": "x", "definitions": [{"text": "y"}]}],
    }
    path = work / "validated_gt.json"
    path.write_text(json.dumps({"papers": [*papers, train]}), encoding="utf-8")
    return path


def predictions_file(work: Path, row: str, name: str, keep=None) -> str:
    """The sample row's records as a predictions file: one line per record; KEEP(record) rewrites each line."""
    lines = []
    for path in sorted((work / "sample/rows" / row / "papers").glob("*.json")):
        saved = read(path)
        for record in saved["definition_records"]:
            line = {"paper_id": saved["paper_id"], **record}
            lines.append(json.dumps(keep(line) if keep else line))
    (work / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return name


def score_predictions(work: Path, *args: str):
    return run_cli(
        work,
        *("evaluate", "score", "--gold", str(gold_file(work))),
        *("--replay-scores", "sample/pair-scores.json", "--out", "out", *args),
    )


def test_a_predictions_file_scores_as_the_same_records_in_paper_files(tmp_path):
    """The Opus row as a predictions file, on the TEST split, gives the released per-paper counts and full-record scores, like `--scorings`."""
    work = sample_setup(tmp_path)
    name = predictions_file(work, "Rdb54680", "opus.jsonl")

    done = score_predictions(work, "--predictions", name, "--split", "test")

    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == [
        "Reference: the test split of validated_gt.json, 2 papers with 10 definitions",
        "opus.jsonl: 7 of 10 reference definitions found, 27 predicted "
        "(precision 25.9%, recall 70.0%), full-record score 0.122",
        "  2 of the 2 papers have predictions; F1 37.8%",
        "wrote out",
    ]
    (scored,) = read(work / "out/scores.json")["scorings"]
    assert scored["reference"]["split"] == "test"
    assert scored["papers_outside_reference"] == 0
    for paper in scored["score"]["per_paper"]:
        tp, fp, fn, full = RELEASED["Rdb54680"][paper["paper_id"]]
        counts = paper["binary_one_to_one"]
        assert (counts["tp"], counts["fp"], counts["fn"]) == (tp, fp, fn)
        assert paper["full_record"]["score"] == pytest.approx(full, abs=1e-6)
    receipt = read(work / "out/receipt.json")
    assert set(receipt["inputs"]) == {"gold", "pair_scores", "opus.jsonl"}


def test_any_tool_s_predictions_need_only_a_term_and_definitions_as_strings(tmp_path):
    """A tool that writes only `{"paper_id", "term", "definitions": ["..."]}` is scored on the same definitions: the same matches; its full-record score changes (0.1498 instead of 0.1224), because only the fields a record has on either side are compared, and Opus's other lists matched little."""
    work = sample_setup(tmp_path)
    name = predictions_file(
        work,
        "Rdb54680",
        "plain.jsonl",
        lambda line: {
            "paper_id": line["paper_id"],
            "term": line["term"],
            "definitions": [d["text"] for d in line["definitions"]],
        },
    )

    done = score_predictions(work, "--predictions", name)

    assert done.returncode == 0, done.stderr
    assert "plain.jsonl: 7 of 10 reference definitions found, 27 predicted" in (
        done.stdout
    )
    (scored,) = read(work / "out/scores.json")["scorings"]
    assert f"{scored['score']['full_record_macro_score']:.4f}" == "0.1498"


def test_concatenated_files_are_pooled_and_papers_outside_the_split_are_left_out(
    tmp_path,
):
    """`cat a.jsonl b.jsonl` pools two extractors per paper (records merged by term); a paper that is not in the split is counted, not scored; `--paper` scores on some papers of the split only."""
    work = sample_setup(tmp_path)
    opus = predictions_file(work, "Rdb54680", "opus.jsonl")
    qwen = predictions_file(work, "QWEN-POOL3", "qwen.jsonl")
    other = json.dumps({"paper_id": "train-paper", "term": "x", "definitions": ["y"]})
    pooled = work / "pooled.jsonl"
    pooled.write_text(
        (work / opus).read_text() + (work / qwen).read_text() + other + "\n"
    )

    done = score_predictions(
        work, "--predictions", "pooled.jsonl", "--paper", PAPERS[1]
    )

    assert done.returncode == 0, done.stderr
    (scored,) = read(work / "out/scores.json")["scorings"]
    assert scored["reference"]["papers"] == 1
    assert scored["papers_with_output"] == 1
    assert scored["papers_outside_reference"] == 2  # train-paper, and PAPERS[0]
    assert "1 of the 1 papers have predictions" in done.stdout
    assert "2 other papers were not scored" in done.stdout
    counts = scored["score"]["binary_one_to_one"]
    assert counts["gold"] == 6 and counts["tp"] == 6  # Qwen alone found all 6


def test_a_bad_line_or_a_paper_outside_the_split_stops_with_where(tmp_path):
    work = sample_setup(tmp_path)
    (work / "bad.jsonl").write_text(
        json.dumps({"paper_id": PAPERS[0], "term": "t", "definitions": ["d"]})
        + "\n"
        + json.dumps({"paper_id": PAPERS[0], "definitions": ["d"]})
        + "\n"
    )

    bad = score_predictions(work, "--predictions", "bad.jsonl")
    outside = score_predictions(
        work, "--predictions", "bad.jsonl", "--paper", "train-paper"
    )
    both = score_predictions(
        work, "--predictions", "bad.jsonl", "--scorings", "sample/scorings.json"
    )

    assert bad.returncode == 1
    assert 'bad.jsonl, line 2: "term" must be a non-empty string' in bad.stderr
    assert outside.returncode == 1
    assert "Not papers of MBDef-Gold's test split: train-paper" in outside.stderr
    assert both.returncode == 1
    assert not (work / "out").exists()
