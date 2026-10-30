"""`mbdef extract train-filter`: fitting the relevance filters the way the frozen ones were.

No GPU and no model weights: `--stored-vectors` takes the BGE vectors from the frozen filter's own zip (`data/frozen/results/*.zip`, made with the pinned BGE), and the records and labels come from MBDef-Gold, the ground truth, in the data root (so the refits are slow tests; a miniature checks the rest). The expected results are the frozen zips' files.

The logistic regression's solver and the probabilities use OpenBLAS, which picks its kernels by CPU; the frozen filters were fitted with its `Haswell` kernels (also those of AMD's Zen), and other kernels add up in another order and change the last digits. The refits here force them with `OPENBLAS_CORETYPE=Haswell` (any x86-64 CPU with AVX2 runs them).
"""

import hashlib
import json
import math
import platform
import zipfile
from pathlib import Path

import numpy as np
import pytest
from cli_helpers import real_mbdef_gold, run_cli, write_env

from mbdef.extract.relevance import DIMENSIONS, ENCODER, ENCODER_FILES

REPO = Path(__file__).resolve().parents[1]
ZIPS = REPO / "data/frozen/results"
GOLD, needs_gold = real_mbdef_gold()
RELEVANT = GOLD / "validated_gt.json"
NON_RELEVANT = GOLD / "validated_gt_non_relevant.json"
# filter name: (its frozen zip, its frozen cutoff as relevance.FILTERS pins it)
FROZEN = {
    "ordinary": ("semantic-relevance-v1.zip", 0.3087965279332379),
    "title": ("semantic-title-relevance-v2.zip", 0.2371840437933272),
    "title-blocks": ("semantic-title-blocks-v3.zip", 0.24892858469583487),
}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def workdir(tmp_path: Path) -> Path:
    work = tmp_path / "work"
    work.mkdir()
    write_env(work, DATA_ROOT=str(tmp_path / "data-root"))
    return work


# The OpenBLAS kernels the frozen filters were fitted with (module docstring).
FROZEN_KERNELS = {"OPENBLAS_CORETYPE": "Haswell"}
X86 = ("x86_64", "AMD64")
x86_only = pytest.mark.skipif(
    platform.machine() not in X86, reason="OPENBLAS_CORETYPE names x86-64 kernels"
)


def train_filter(work: Path, out: Path, name: str, *extra: str, env=None):
    return run_cli(
        work,
        *("extract", "train-filter", "--filter", name, "--out", str(out)),
        *("--ground-truth", str(RELEVANT), "--non-relevant", str(NON_RELEVANT)),
        *("--filters", str(ZIPS), *extra),
        env=env,
    )


@pytest.fixture(scope="module")
def refits(tmp_path_factory) -> dict[str, Path]:
    """The three filters refitted from their zips' stored vectors: {name: output folder}, each with the command's printed summary in `summary.json` beside it."""
    if platform.machine() not in X86:
        pytest.skip("OPENBLAS_CORETYPE names x86-64 kernels")
    tmp_path = tmp_path_factory.mktemp("refit")
    work = workdir(tmp_path)
    folders = {}
    for name in FROZEN:
        out = tmp_path / name
        result = train_filter(work, out, name, "--stored-vectors", env=FROZEN_KERNELS)
        assert result.returncode == 0, result.stderr
        (tmp_path / f"{name}.summary.json").write_text(result.stdout, encoding="utf-8")
        folders[name] = out
    return folders


@pytest.mark.parametrize("name", FROZEN)
@needs_gold
def test_a_refit_from_the_stored_vectors_is_the_frozen_classifier_byte_for_byte(
    refits, name
):
    zip_name, cutoff = FROZEN[name]
    with zipfile.ZipFile(ZIPS / zip_name) as frozen:
        assert (refits[name] / "model.joblib").read_bytes() == frozen.read(
            "model.joblib"
        )
    assert read(refits[name] / "training.json")["cutoff"] == cutoff


# The keys each frozen training.json adds for its study's own cross-checks against earlier studies (fold assignments and vector coordinates equal to theirs); the refit writes everything else.
STUDY_CHECKS = {
    "identical_lexical_fold_assignment",
    "identical_previous_fold_assignments",
    "identical_previous_folds",
    "exact_original_concept_coordinates",
}


@x86_only
@needs_gold
def test_the_folds_seed_recall_target_and_regularisation_are_flags(tmp_path):
    """`--folds 3 --seed 1 --target-recall 0.9 --regularisation 0.5`: three folds, the cutoff that keeps ⌈0.9 × 592⌉ = 533 relevant records out of fold, and the settings recorded where the frozen ones are."""
    out = tmp_path / "out"
    result = train_filter(
        workdir(tmp_path),
        out,
        "ordinary",
        "--stored-vectors",
        *("--folds", "3", "--seed", "1", "--target-recall", "0.9"),
        *("--regularisation", "0.5"),
        env=FROZEN_KERNELS,
    )
    assert result.returncode == 0, result.stderr
    assert len(read(out / "folds.json")) == 3
    training = read(out / "training.json")
    assert training["selected"]["tp"] == math.ceil(0.9 * 592) == 533
    config = read(out / "config.json")
    assert (config["folds"], config["seed"], config["target_recall"]) == (3, 1, 0.9)
    assert config["classifier"] == (
        "LogisticRegression(C=0.5, solver='liblinear', max_iter=1000, random_state=1)"
    )
    settings = read(out / "receipt.json")["settings"]
    assert settings["classifier"] == config["classifier"]
    assert (settings["folds"], settings["fold_seed"], settings["target_recall"]) == (
        3,
        1,
        0.9,
    )
    assert read(out / "receipt.json")["frozen_filter"]["classifier_file"] == "differs"


@pytest.mark.parametrize("name", FROZEN)
@needs_gold
def test_the_refit_has_the_frozen_folds_out_of_fold_probabilities_and_training_summary(
    refits, name
):
    with zipfile.ZipFile(ZIPS / FROZEN[name][0]) as frozen:
        assert (refits[name] / "folds.json").read_bytes() == frozen.read("folds.json")
        training = json.loads(frozen.read("training.json"))
        oof = json.loads(frozen.read("oof.json"))
    assert read(refits[name] / "training.json") == {
        k: v for k, v in training.items() if k not in STUDY_CHECKS
    }
    refit = read(refits[name] / "oof.json")
    assert len(refit) == len(oof) == 1171
    for mine, theirs in zip(refit, oof, strict=True):
        # Theirs also hold each record's annotation-export row (not in git) and, for `title`, the classifier input; every key of ours is theirs, with the same value.
        assert mine == {k: theirs[k] for k in mine}


@needs_gold
def test_the_command_says_the_refit_is_the_frozen_filter_and_the_receipt_records_how(
    refits,
):
    out = refits["ordinary"]
    summary = read(out.parent / "ordinary.summary.json")
    # Worked example of the module docstring; tp 563 and fp 283 are the frozen training.json's.
    assert summary == {
        "filter": "ordinary",
        "records": 1171,
        "papers": 80,
        "relevant_records": 592,
        "cutoff": 0.3087965279332379,
        "kept_out_of_fold": {"relevant": 563, "not_relevant": 283},
        "vectors": "stored in the frozen zip",
        "frozen_filter": {
            "zip": "semantic-relevance-v1.zip",
            "classifier_file": "identical",
            "cutoff": "identical",
            "folds": "identical",
            "max_coefficient_difference": 0.0,
        },
    }
    receipt = read(out / "receipt.json")
    assert receipt["command"] == "mbdef extract train-filter"
    assert receipt["settings"] == {
        "filter": "ordinary",
        "classifier": "LogisticRegression(C=1, solver='liblinear', max_iter=1000, random_state=9)",
        "folds": 5,
        "fold_seed": 9,
        "target_recall": 0.95,
        "pickle_protocol": 4,
        "encoder": "BAAI/bge-base-en-v1.5 at a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
        "vectors": "stored in the frozen zip",
    }
    # The ground-truth files are the ones the frozen studies recorded as their training input.
    assert receipt["inputs"]["ground_truth"]["sha256"] == (
        "afce3c8b386e94d504b421f1802410b57bb63ebceae3199dabcd107592389b86"
    )
    assert receipt["inputs"]["non_relevant"]["sha256"] == (
        "02502de2c76ec02f0bc9d370f328b862ac780df7ede580ca65bfb934532f0072"
    )
    assert receipt["inputs"]["frozen_filter"]["path"].endswith(
        "semantic-relevance-v1.zip"
    )
    assert set(receipt["outputs"]) == {"model", "config", "training", "folds", "oof"}
    assert receipt["frozen_filter"] == summary["frozen_filter"]
    # Which OpenBLAS kernels ran (numpy's and scipy's copies): the forced ones.
    assert receipt["blas"]
    assert {b["architecture"] for b in receipt["blas"]} == {"Haswell"}


@x86_only
@needs_gold
def test_other_blas_kernels_change_the_refit_only_in_its_last_digits(tmp_path):
    """With OpenBLAS's older `Sandybridge` kernels (any x86-64 CPU with AVX runs them) the refit is not the frozen file, and the command says so; but no coefficient moves by more than 1e-12, the cutoff neither, and the same records pass out of fold (measured: largest difference 1.0e-14)."""
    work = workdir(tmp_path)
    out = tmp_path / "ordinary"
    result = train_filter(
        work,
        out,
        "ordinary",
        "--stored-vectors",
        env={"OPENBLAS_CORETYPE": "Sandybridge"},
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["frozen_filter"]["classifier_file"] == "differs"
    assert summary["frozen_filter"]["folds"] == "identical"
    assert 0 < summary["frozen_filter"]["max_coefficient_difference"] <= 1e-12
    assert abs(summary["cutoff"] - FROZEN["ordinary"][1]) <= 1e-12
    assert summary["kept_out_of_fold"] == {"relevant": 563, "not_relevant": 283}


# A miniature ground truth with synthetic text: 10 TRAIN papers with two relevant and two other records each, and one DEV paper that training must not read.
MINI_PAPERS = [f"paper-{n}" for n in range(10)]


def mini_record(paper: str, kind: str, n: int) -> dict:
    return {
        "term": f"{kind} term {n} of {paper}",
        "definitions": [{"text": f"A synthetic {kind} definition {n} of {paper}."}],
    }


def mini_ground_truths(folder: Path) -> tuple[Path, Path]:
    def paper(paper_id, split, kind):
        return {
            "paper_id": paper_id,
            "split": split,
            "metadata": {"title": f"Title of {paper_id}"},
            "definition_records": [mini_record(paper_id, kind, n) for n in range(2)],
        }

    files = {}
    for name, kind in (("relevant", "relevant"), ("other", "other")):
        papers = [paper(p, "train", kind) for p in MINI_PAPERS]
        papers.append(paper("dev-paper", "dev", kind))
        files[name] = folder / f"{name}.json"
        files[name].write_text(json.dumps({"papers": papers}), encoding="utf-8")
    return files["relevant"], files["other"]


def feature(record: dict) -> str:
    return record["term"] + "\n" + "\n".join(d["text"] for d in record["definitions"])


def write_recording(folder: Path, texts: list[str]) -> None:
    """An encoder recording (relevance.save_recording's format) with synthetic unit vectors:
    relevant texts lean one way, the others the other way."""
    rng = np.random.default_rng(9)
    lean = np.zeros(DIMENSIONS, dtype=np.float32)
    lean[0] = 1.0
    rows = []
    for text in texts:
        vector = rng.normal(size=DIMENSIONS).astype(np.float32) * 0.05
        vector += lean if "relevant" in text else -lean
        rows.append(vector / np.linalg.norm(vector))
    folder.mkdir(parents=True)
    keyed = sorted(
        (hashlib.sha256(t.encode()).hexdigest(), row)
        for t, row in zip(texts, rows, strict=True)
    )
    (folder / "encoded.json").write_text(
        json.dumps(
            {
                "model": ENCODER[0],
                "revision": ENCODER[1],
                "files_sha256": ENCODER_FILES,
                "dtype": "float32",
                "device": "cpu",
                "texts": [
                    {"sha256": key, "tokens": 20, "row": i}
                    for i, (key, _row) in enumerate(keyed)
                ],
            }
        ),
        encoding="utf-8",
    )
    np.save(folder / "vectors.npy", np.asarray([row for _key, row in keyed]))


def mini_texts() -> list[str]:
    return [
        feature(mini_record(p, kind, n))
        for p in MINI_PAPERS
        for kind in ("relevant", "other")
        for n in range(2)
    ]


def test_a_new_ground_truth_trains_from_a_recorded_encoder_on_its_train_papers_only(
    tmp_path,
):
    work = workdir(tmp_path)
    relevant, other = mini_ground_truths(tmp_path)
    write_recording(tmp_path / "recorded", mini_texts())
    out = tmp_path / "out"
    result = run_cli(
        work,
        *("extract", "train-filter", "--out", str(out), "--filters", str(ZIPS)),
        *("--ground-truth", str(relevant), "--non-relevant", str(other)),
        *("--replay", str(tmp_path / "recorded")),
    )
    assert result.returncode == 0, result.stderr
    oof = read(out / "oof.json")
    # 10 TRAIN papers × (2 relevant + 2 other); the DEV paper is not read.
    assert len(oof) == 40
    assert {r["paper_id"] for r in oof} == set(MINI_PAPERS)
    assert [r["relevance"] for r in oof] == [1] * 20 + [0] * 20
    assert {r["dataset"] for r in oof if r["relevance"] == 0} == {"non-relevant"}
    # Five folds, each paper held out in exactly one of them and never trained on there.
    folds = read(out / "folds.json")
    assert [f["fold"] for f in folds] == [0, 1, 2, 3, 4]
    held = [p for f in folds for p in f["validation_papers"]]
    assert sorted(held) == sorted(MINI_PAPERS)
    assert all(not set(f["train_papers"]) & set(f["validation_papers"]) for f in folds)
    for record in oof:
        fold = folds[record["fold"]]
        assert record["paper_id"] in fold["validation_papers"]
    # The cutoff keeps ⌈0.95 × 20⌉ = 19 relevant records out of fold, and is the largest that does.
    training = read(out / "training.json")
    cutoff = training["cutoff"]
    relevant_p = [r["probability"] for r in oof if r["relevance"] == 1]
    assert cutoff in relevant_p
    assert sum(p >= cutoff for p in relevant_p) >= math.ceil(0.95 * 20) == 19
    assert sum(p > cutoff for p in relevant_p) < 19
    assert training["records"] == 40 and training["papers"] == 10
    assert training["features"] == 768
    # The encoder's recording of what it was asked, and the comparison with the frozen filter.
    assert (out / "encoder/encoded.json").is_file()
    summary = json.loads(result.stdout)
    assert summary["vectors"] == "replay of recorded vectors"
    assert summary["frozen_filter"]["classifier_file"] == "differs"


def test_a_text_that_was_not_recorded_stops_the_run_before_anything_is_written(
    tmp_path,
):
    """The `title` filter reads the title and the feature text; only the feature texts are recorded."""
    work = workdir(tmp_path)
    relevant, other = mini_ground_truths(tmp_path)
    write_recording(tmp_path / "recorded", mini_texts())
    out = tmp_path / "out"
    result = run_cli(
        work,
        *("extract", "train-filter", "--filter", "title", "--out", str(out)),
        *("--ground-truth", str(relevant), "--non-relevant", str(other)),
        *("--filters", str(ZIPS), "--replay", str(tmp_path / "recorded")),
    )
    assert result.returncode == 1
    assert "was not recorded" in result.stderr
    assert not out.exists()


def test_training_refuses_papers_whose_titles_disagree_between_the_two_files(tmp_path):
    work = workdir(tmp_path)
    relevant, other = mini_ground_truths(tmp_path)
    changed = read(other)
    changed["papers"][3]["metadata"]["title"] = "Another title"
    other.write_text(json.dumps(changed), encoding="utf-8")
    write_recording(tmp_path / "recorded", mini_texts())
    result = run_cli(
        work,
        *("extract", "train-filter", "--filter", "title-blocks"),
        *("--out", str(tmp_path / "out"), "--filters", str(ZIPS)),
        *("--ground-truth", str(relevant), "--non-relevant", str(other)),
        *("--replay", str(tmp_path / "recorded")),
    )
    assert result.returncode == 1
    assert "different titles" in result.stderr


# A vector or probability recomputed with BGE on another device (as tests/test_extract_filter.py).
VECTOR_BAND = 1e-5


@pytest.mark.slow
@pytest.mark.gpu
@needs_gold
def test_with_bge_itself_the_refit_is_the_frozen_filter_within_the_band(tmp_path):
    """By hand, with BGE in the Hugging Face cache (`mbdef cluster download bge`), on a GPU if there is one (2 to 3 minutes on a 24-thread CPU): the vectors of all 1,171 training texts are within 1e-5 of the stored ones, the folds are identical, the same 563 relevant and 283 other records pass out of fold, and the cutoff and every coefficient are within 1e-5 of the frozen ones (on a 24-thread CPU: cutoff 0.3087962659 against 0.3087965279, largest coefficient difference 4.0e-6, largest vector difference 3.8e-6)."""
    torch = pytest.importorskip("torch")
    from mbdef.extract import relevance
    from mbdef.extract import train_filter as fitting

    work = workdir(tmp_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    write_env(work, DATA_ROOT=str(tmp_path / "data-root"), SCIDEF_NLI_DEVICE=device)
    out = tmp_path / "ordinary"
    result = train_filter(work, out, "ordinary")
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["kept_out_of_fold"] == {"relevant": 563, "not_relevant": 283}
    assert summary["frozen_filter"]["folds"] == "identical"
    assert summary["frozen_filter"]["max_coefficient_difference"] <= VECTOR_BAND
    assert abs(summary["cutoff"] - FROZEN["ordinary"][1]) <= VECTOR_BAND
    live = relevance.recorded([out / "encoder"])
    stored = fitting.stored_vectors(ZIPS / FROZEN["ordinary"][0])
    texts = [r["text"] for r in read(out / "oof.json")]
    for text in texts:
        vector = live[hashlib.sha256(text.encode()).hexdigest()][1]
        assert np.abs(vector - stored[text]).max() <= VECTOR_BAND


@pytest.mark.parametrize("name", FROZEN)
@needs_gold
def test_the_refit_records_the_encoder_settings_extract_filter_requires_of_a_zip(
    refits, name
):
    """`config.json` beside the classifier holds the encoder settings that `relevance.load_filter` checks in a filter's zip, with the frozen zip's values."""
    with zipfile.ZipFile(ZIPS / FROZEN[name][0]) as frozen:
        theirs = json.loads(frozen.read("config.json"))
    mine = read(refits[name] / "config.json")
    keys = (
        "model",
        "revision",
        "capacity",
        "batch_size",
        "dtype",
        "model_files_sha256",
    )
    assert {k: mine[k] for k in keys} == {k: theirs[k] for k in keys}
    assert mine["filter"] == name
    assert mine["features"] == read(refits[name] / "training.json")["features"]
