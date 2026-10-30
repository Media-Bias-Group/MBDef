"""`mbdef extract filter` on two ground-truth papers, with recorded BGE vectors.

CI has no BGE weights and no GPU, so the encoder's answers are replayed: `encoder/` in `tests/fixtures/filter-replay.zip` holds the token count and the BGE vector of every text these tests classify, recorded by `extract filter` itself on a GPU (the slow test below remakes them), and a text that was not recorded stops the run. The classifiers are the real frozen ones (`data/frozen/results/*.zip`), and the expected results are the released files: the published stages and the per-paper filter files of the released TEST run (byte-identical copies in the zip).

The filter reads records only, so its fixture holds no paper text: its members are byte-for-byte copies of the released TEST run's records, filter files and BGE recordings, without the windows.

The three record sets of the released stages, per part: the union (`-noverify`), Gemma's own verification (no suffix) and the Qwen verifier's (`-qv`).
"""

import hashlib
import json
import re
import zipfile
from pathlib import Path
from urllib.parse import quote

import pytest
from cli_helpers import package_commit, run_cli, write_env

REPO = Path(__file__).resolve().parents[1]
ZIPS = REPO / "data/frozen/results"
FIXTURES = Path(__file__).with_name("fixtures")
PAPERS = (
    "4243aa9bacd51bc02d4240378e7528912962b53e",
    "73bfad11b96a69cb882028ead115751adb55252d",
)
ARMS = ("TR-bootstrap-8k", "TR-bootstrap-16k32k")
COHORT = "test/gemma12b/{arm}/chat-default-think"
# Released stage suffix: (the run folder, the records filtered)
SETS = {
    "-noverify": ("run", "union"),
    "": ("run", "verified"),
    "-qv": ("run-verify-qwen", "verified"),
}
PROBABILITY_BAND = 1e-5  # a probability recomputed on a GPU


def paper_key(paper_id: str) -> str:
    return hashlib.sha256(paper_id.encode()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def recorded(tmp_path_factory) -> tuple[Path, Path]:
    """A work folder with an empty `.env`, and both fixtures unpacked into one recorded folder."""
    tmp_path = tmp_path_factory.mktemp("filter")
    work, root = tmp_path / "work", tmp_path / "recorded"
    work.mkdir()
    write_env(work, DATA_ROOT=str(tmp_path / "data-root"))
    with zipfile.ZipFile(FIXTURES / "filter-replay.zip") as bundle:
        bundle.extractall(root)
    return work, root


def run_filter(work: Path, run: Path, out: Path, *extra: str, env=None):
    return run_cli(
        work,
        *("extract", "filter", "--run", str(run), "--out", str(out)),
        *("--filters", str(ZIPS), *extra),
        env=env,
    )


def saved_result(cohort: Path, kind: str, suffix: str, paper: str) -> tuple[list, list]:
    """The released kept records and probabilities of one paper in one stage."""
    stage = read(
        cohort / "stages" / f"filtered-{kind}{suffix}" / f"{quote(paper, safe='')}.json"
    )
    if "relevance_probabilities" in stage:  # the -noverify stages hold them
        return stage["definition_records"], stage["relevance_probabilities"]
    run, _records = SETS[suffix]
    probabilities = read(
        cohort / run / "papers" / paper_key(paper) / f"filter-{kind}.json"
    )
    return stage["definition_records"], probabilities["relevance_probabilities"]


def assert_released(
    out: Path, cohort: Path, kind: str, suffix: str, papers=PAPERS
) -> None:
    for paper in papers:
        result = read(out / "papers" / paper_key(paper) / "filtered.json")
        kept, probabilities = saved_result(cohort, kind, suffix, paper)
        assert result["paper_id"] == paper
        assert result["definition_records"] == kept, (cohort.parent.name, suffix)
        new = result["relevance_probabilities"]
        assert len(new) == len(probabilities) == result["records_before_filter"]
        assert all(
            abs(a - b) <= PROBABILITY_BAND
            for a, b in zip(new, probabilities, strict=True)
        )
        assert result["capacity_excluded_record_indices"] == []


@pytest.mark.parametrize("suffix", list(SETS))
@pytest.mark.parametrize("arm", ARMS)
def test_the_ordinary_filter_keeps_the_released_records(
    recorded, tmp_path, arm, suffix
):
    work, root = recorded
    cohort = root / COHORT.format(arm=arm)
    run, records = SETS[suffix]

    done = run_filter(
        work,
        cohort / run,
        tmp_path / "out",
        "--records",
        records,
        "--replay",
        str(root / "encoder"),
    )

    assert done.returncode == 0, done.stderr
    assert_released(tmp_path / "out", cohort, "ordinary", suffix)


SURVEY = "1901.00596v4"  # "A Comprehensive Survey on Graph Neural Networks", part w8k
SURVEY_COHORT = "filter-only/" + COHORT.format(arm="TR-bootstrap-8k")


@pytest.mark.parametrize("suffix", list(SETS))
@pytest.mark.parametrize("kind", ["ordinary", "title"])
def test_both_published_filters_reject_what_they_rejected_when_released(
    recorded, tmp_path, kind, suffix
):
    work, root = recorded
    cohort = root / SURVEY_COHORT
    run, records = SETS[suffix]

    done = run_filter(
        work,
        cohort / run,
        tmp_path / "out",
        *("--records", records, "--filter", kind, "--replay", str(root / "encoder")),
    )

    assert done.returncode == 0, done.stderr
    assert_released(tmp_path / "out", cohort, kind, suffix, papers=[SURVEY])
    result = read(tmp_path / "out/papers" / paper_key(SURVEY) / "filtered.json")
    # The union has 28 records (Gemma's own verifier kept 17, Qwen's all 28); the ordinary filter keeps 4 (directed graph, undirected graph, diffusion process, attention mechanisms) and the title filter 2.
    assert result["records_before_filter"] == (17 if suffix == "" else 28)
    assert len(result["definition_records"]) == {"ordinary": 4, "title": 2}[kind]


@pytest.mark.parametrize("arm", ARMS)
def test_the_title_filter_keeps_the_released_records(recorded, tmp_path, arm):
    work, root = recorded
    cohort = root / COHORT.format(arm=arm)

    done = run_filter(
        work,
        cohort / "run-verify-qwen",
        tmp_path / "out",
        *("--filter", "title", "--replay", str(root / "encoder")),
    )

    assert done.returncode == 0, done.stderr
    assert_released(tmp_path / "out", cohort, "title", "-qv")


def test_another_cutoff_is_one_flag(recorded, tmp_path):
    work, root = recorded
    cohort = root / SURVEY_COHORT

    done = run_filter(
        work,
        cohort / "run",
        tmp_path / "out",
        *("--records", "union", "--cutoff", "0.15", "--replay", str(root / "encoder")),
    )

    assert done.returncode == 0, done.stderr
    result = read(tmp_path / "out/papers" / paper_key(SURVEY) / "filtered.json")
    _kept, released = saved_result(cohort, "ordinary", "-noverify", SURVEY)
    union = read(cohort / "run/papers" / paper_key(SURVEY) / "union.json")["records"]
    # 8 of the 28 released probabilities are at least 0.15 (4 at least the frozen cutoff).
    assert [r["term"] for r in result["definition_records"]] == [
        r["term"] for r, p in zip(union, released, strict=True) if p >= 0.15
    ]
    assert len(result["definition_records"]) == 8
    receipt = read(tmp_path / "out/receipt.json")
    assert (receipt["settings"]["cutoff"], receipt["settings"]["frozen_cutoff"]) == (
        0.15,
        0.3087965279332379,
    )


def test_the_receipt_records_the_frozen_filter_and_the_encoder(recorded, tmp_path):
    work, root = recorded
    cohort = root / COHORT.format(arm="TR-bootstrap-8k")

    done = run_filter(
        work,
        cohort / "run-verify-qwen",
        tmp_path / "out",
        "--replay",
        str(root / "encoder"),
    )

    assert done.returncode == 0, done.stderr
    receipt = read(tmp_path / "out/receipt.json")
    assert receipt["settings"] == {
        "filter": "ordinary",
        "filter_row": "R6e24b3d",
        "study": "semantic-relevance-v1",
        "cutoff": 0.3087965279332379,
        "frozen_cutoff": 0.3087965279332379,
        "records": "verified",
        "encoder": "BAAI/bge-base-en-v1.5 at a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
        "encoder_input": "replay of recorded vectors",
        "capacity_tokens": 512,
        "batch": 32,
    }
    zip_sha256 = hashlib.sha256(
        (ZIPS / "semantic-relevance-v1.zip").read_bytes()
    ).hexdigest()
    assert receipt["inputs"]["filter"]["sha256"] == zip_sha256
    assert sorted(receipt["inputs"]) == ["filter", "replay:1", "run"]
    assert receipt["code"]["commit"] == package_commit()  # None outside a checkout
    # Part w8k, Qwen verifier: 1 + 8 verified records, all relevant enough to keep.
    assert receipt["summary"] == {
        "papers": 2,
        "records_before_filter": 9,
        "kept_records": 9,
        "kept_definitions": 9,
        "capacity_excluded_records": 0,
        "predictions.jsonl": {"papers": 2, "records": 9, "definitions": 9},
    }
    assert json.loads(done.stdout) == receipt["summary"]
    # predictions.jsonl: the kept records of filtered.json, one line each.
    lines = (tmp_path / "out/predictions.jsonl").read_text().splitlines()
    kept = [
        record
        for path in sorted((tmp_path / "out/papers").glob("*/filtered.json"))
        for record in read(path)["definition_records"]
    ]
    assert [json.loads(line)["term"] for line in lines] == [r["term"] for r in kept]
    assert "predictions" in receipt["outputs"]


def test_a_run_records_its_encoder_and_a_text_not_recorded_stops_a_replay(
    recorded, tmp_path
):
    work, root = recorded
    small = root / COHORT.format(arm="TR-bootstrap-8k") / "run-verify-qwen"
    first, again, other = (tmp_path / name for name in ("first", "again", "other"))
    assert (
        run_filter(work, small, first, "--replay", str(root / "encoder")).returncode
        == 0
    )

    # The recording a run writes answers that run again, with the same results.
    done = run_filter(work, small, again, "--replay", str(first / "encoder"))
    assert done.returncode == 0, done.stderr
    for paper in PAPERS:
        name = f"papers/{paper_key(paper)}/filtered.json"
        assert (again / name).read_bytes() == (first / name).read_bytes()

    # It holds no text of the other part, so the first such text stops the run.
    larger = root / COHORT.format(arm="TR-bootstrap-16k32k") / "run-verify-qwen"
    done = run_filter(work, larger, other, "--replay", str(first / "encoder"))
    assert done.returncode == 1
    assert re.match(
        r"Stopped: the encoder input with sha256 [0-9a-f]{64} was not recorded",
        done.stderr,
    )
    assert not (other / "receipt.json").exists()


def test_a_filter_whose_files_changed_is_refused(recorded, tmp_path):
    work, root = recorded
    filters = tmp_path / "filters"
    filters.mkdir()
    with (
        zipfile.ZipFile(ZIPS / "semantic-relevance-v1.zip") as frozen,
        zipfile.ZipFile(filters / "semantic-relevance-v1.zip", "w") as changed,
    ):
        for member in frozen.namelist():
            data = frozen.read(member)
            changed.writestr(
                member, data + b"\n" if member == "training.json" else data
            )

    done = run_cli(
        work,
        *("extract", "filter", "--run", str(root / SURVEY_COHORT / "run")),
        *("--out", str(tmp_path / "out"), "--filters", str(filters)),
        *("--replay", str(root / "encoder")),
    )

    assert done.returncode == 1
    assert done.stderr.strip().startswith(
        f"{filters / 'semantic-relevance-v1.zip'}: training.json has sha256 "
    )
    assert done.stderr.strip().endswith(
        "not the frozen 406e8645cbc61706531ea4ebd55aba8a8aeb2e200ac9b421db6a4b13b6edaeec. "
        "Refusing to use a changed filter."
    )
    assert not (tmp_path / "out").exists()


@pytest.mark.slow
@pytest.mark.gpu
def test_with_bge_itself_the_vectors_are_the_recorded_ones(recorded, tmp_path):
    """By hand, with BGE in the Hugging Face cache (`mbdef cluster download bge`), on a GPU if there is one: every fixture run with the real encoder gives the released records and probabilities, and vectors within 1e-5 of the recording. The recording in the fixture is the merge of these runs' `encoder/` folders (made on a 24 GB GPU)."""
    torch = pytest.importorskip("torch")
    numpy = pytest.importorskip("numpy")
    from mbdef.extract import relevance

    work, root = recorded
    device = {"SCIDEF_NLI_DEVICE": "cuda" if torch.cuda.is_available() else "cpu"}
    runs = [(root / COHORT.format(arm=arm), PAPERS) for arm in ARMS] + [
        (root / SURVEY_COHORT, [SURVEY])
    ]
    folders = []
    for cohort, papers in runs:
        for suffix, (run, records) in SETS.items():
            for kind in ("ordinary", "title"):
                out = tmp_path / str(len(folders))
                done = run_filter(
                    work,
                    cohort / run,
                    out,
                    *("--records", records, "--filter", kind),
                    env=device,
                )
                assert done.returncode == 0, done.stderr
                assert_released(out, cohort, kind, suffix, papers=papers)
                folders.append(out / "encoder")
    live = relevance.recorded(folders)
    stored = relevance.recorded([root / "encoder"])
    assert set(live) == set(stored)
    for key, (tokens, vector) in live.items():
        stored_tokens, stored_vector = stored[key]
        assert tokens == stored_tokens
        if vector is None or stored_vector is None:  # over the encoder's 512 tokens
            assert vector is None and stored_vector is None, key
        else:
            assert numpy.abs(vector - stored_vector).max() <= PROBABILITY_BAND
