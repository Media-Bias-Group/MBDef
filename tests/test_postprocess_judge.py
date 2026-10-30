"""`mbdef postprocess judge`: the GT-RAG judge's packages, requests and tables equal the original implementation's.

tests/fixtures/judge-mini/ holds a made-up ground truth, corpus store and embedding cache, the replies of a stand-in server, and the outputs of the original implementation on them, which the tests compare with byte for byte. The package must write the same packages, send requests with the same recorded hash (model, messages, settings, answer schema), gather the same decisions and write the same tables, byte for byte. The slow tests do the same on the real files: the recorded corpus package, the recorded runs' request hashes and the two frozen score tables (they need the data root).
"""

import hashlib
import json
import zipfile
from pathlib import Path

import pytest
from cli_helpers import run_cli
from mini_basal import make_store
from mini_judge import MINI, ChatStandIn

from mbdef import files, postprocess

REPO = Path(__file__).resolve().parents[1]
EXPECTED = MINI / "expected"
FROZEN = REPO / postprocess.FROZEN


def member(package: Path) -> bytes:
    with zipfile.ZipFile(package) as z:
        return z.read("inputs.json")


@pytest.fixture(scope="module")
def mini_packages(tmp_path_factory):
    """`postprocess judge package` on the made-up inputs: the folder it wrote."""
    tmp = tmp_path_factory.mktemp("judge-package")
    store = make_store(tmp / "store.sqlite3", MINI / "inputs/store.json")
    done = run_cli(
        REPO,
        *("postprocess", "judge", "package", "--store", str(store)),
        *(
            "--links",
            str(MINI / "inputs/links.csv"),
            "--cache",
            str(MINI / "inputs/cache"),
        ),
        *("--ground-truth-folder", str(MINI / "inputs"), "--out", str(tmp / "out")),
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return tmp / "out"


def test_the_corpus_package_is_the_original_byte_for_byte(mini_packages):
    """Every corpus definition with its 32 nearest train records (never its own paper, nor the train paper of the same title), the papers in 8 shards, and one request line per 4 statements of a paper."""
    assert (
        member(mini_packages / "package.zip")
        == (EXPECTED / "corpus-inputs.json").read_bytes()
    )
    assert (mini_packages / "requests.jsonl").read_bytes() == (
        EXPECTED / "corpus-requests.jsonl"
    ).read_bytes()


def test_the_ground_truth_package_holds_train_and_dev_only(mini_packages):
    """The ground truth's statements shuffled per paper, each with its neighbours; the TEST paper is left out."""
    assert (
        member(mini_packages / "groundtruth-package.zip")
        == (EXPECTED / "groundtruth-inputs.json").read_bytes()
    )
    papers = json.loads(member(mini_packages / "groundtruth-package.zip"))["union"][
        "papers"
    ]
    assert {p["split"] for p in papers} == {"train", "dev"}


def recorded_hash(body) -> str:
    """The recorded runs' request hash, from the body the server received."""
    hashed = {
        "model": "openai/" + body["model"],
        "messages": body["messages"],
        "temperature": body["temperature"],
        "max_tokens": body["max_tokens"],
        "seed": body["seed"],
        "extra_body": {"chat_template_kwargs": body["chat_template_kwargs"]},
        "response_format": body["response_format"],
    }
    return hashlib.sha256(
        json.dumps(hashed, sort_keys=True, default=str).encode()
    ).hexdigest()


@pytest.fixture(scope="module")
def mini_runs(mini_packages, tmp_path_factory):
    """`postprocess judge ask` for both packages against the stand-in server: the run folders and the bodies."""
    tmp = tmp_path_factory.mktemp("judge-runs")
    received = {}
    with ChatStandIn() as server:
        for kind, package in (
            ("corpus", "package.zip"),
            ("groundtruth", "groundtruth-package.zip"),
        ):
            done = run_cli(
                REPO,
                *(
                    "postprocess",
                    "judge",
                    "ask",
                    "--package",
                    str(mini_packages / package),
                ),
                *(
                    "--out",
                    str(tmp / kind),
                    "--server",
                    server.url,
                    "--concurrency",
                    "2",
                ),
            )
            assert done.returncode == 0, done.stdout + done.stderr
            received[kind] = list(server.received)
            server.received.clear()
    return tmp, received


def test_the_requests_are_the_recorded_requests(mini_runs):
    """Each body carries the original messages, settings and answer schema (the recorded hash), and asks for 20 log-probabilities per token."""
    _, received = mini_runs
    expected = [
        json.loads(line)
        for line in (EXPECTED / "request-hashes.jsonl").read_text().splitlines()
    ]
    for kind, bodies in received.items():
        assert sorted(recorded_hash(b) for b in bodies) == sorted(
            h["request_sha256"] for h in expected if h["kind"] == kind
        )
        assert all(b["logprobs"] is True and b["top_logprobs"] == 20 for b in bodies)


def test_the_decisions_are_the_originals(mini_runs):
    """The verdicts under the statements' own ids, each with the read-out of its relevance probabilities."""
    runs, _ = mini_runs
    for kind in ("corpus", "groundtruth"):
        assert (runs / kind / "decisions.json").read_bytes() == (
            EXPECTED / f"{kind}-decisions.json"
        ).read_bytes()


def test_the_tables_are_the_originals(mini_packages, mini_runs, tmp_path):
    runs, _ = mini_runs
    for kind, package in (
        ("corpus", "package.zip"),
        ("groundtruth", "groundtruth-package.zip"),
    ):
        done = run_cli(
            REPO,
            *(
                "postprocess",
                "judge",
                "table",
                "--package",
                str(mini_packages / package),
            ),
            *("--runs", str(runs / kind), "--out", str(tmp_path / f"{kind}.csv")),
        )
        assert done.returncode == 0, done.stdout + done.stderr
        assert (tmp_path / f"{kind}.csv").read_bytes() == (
            EXPECTED / f"{kind}-table.csv"
        ).read_bytes()


def test_a_corpus_table_with_a_statement_missing_is_refused(
    mini_packages, mini_runs, tmp_path
):
    runs, _ = mini_runs
    decisions = json.loads((runs / "corpus/decisions.json").read_text())
    paper = next(p for p, c in decisions.items() if c)
    decisions[paper].pop(next(iter(decisions[paper])))
    (tmp_path / "run").mkdir()
    (tmp_path / "run/decisions.json").write_text(json.dumps(decisions))
    (tmp_path / "run/generation-complete.json").write_text("{}")

    done = run_cli(
        REPO,
        *(
            "postprocess",
            "judge",
            "table",
            "--package",
            str(mini_packages / "package.zip"),
        ),
        *("--runs", str(tmp_path / "run"), "--out", str(tmp_path / "t.csv")),
    )

    assert done.returncode == 1
    assert "1 statements have no verdict" in done.stderr
    assert not (tmp_path / "t.csv").exists()


def test_the_neighbour_count_and_batch_are_flags(tmp_path):
    """`package --neighbours 2 --batch 2`: each statement gets its 2 nearest train records, and the request list holds one line per 2 statements of a paper."""
    store = make_store(tmp_path / "store.sqlite3", MINI / "inputs/store.json")
    done = run_cli(
        REPO,
        *("postprocess", "judge", "package", "--store", str(store)),
        *("--links", str(MINI / "inputs/links.csv")),
        *(
            "--cache",
            str(MINI / "inputs/cache"),
            "--ground-truth-folder",
            str(MINI / "inputs"),
        ),
        *("--neighbours", "2", "--batch", "2", "--out", str(tmp_path / "out")),
    )
    assert done.returncode == 0, done.stdout + done.stderr
    package = json.loads(member(tmp_path / "out/package.zip"))
    lists = [
        arm
        for paper in package["neighbors"].values()
        for statement in paper.values()
        for arm in statement.values()
    ]
    assert lists and all(len(neighbours) == 2 for neighbours in lists)
    papers = package["union"]["papers"]
    lines = (tmp_path / "out/requests.jsonl").read_text().splitlines()
    assert all(len(json.loads(line)["candidates"]) <= 2 for line in lines)
    assert len(lines) == sum(-(-len(p["candidates"]) // 2) for p in papers)
    assert any(len(p["candidates"]) > 2 for p in papers)


def test_the_model_and_its_settings_are_flags(mini_packages, tmp_path):
    """`ask --model --temperature --max-tokens --seed`: the requests carry them, and each answer line's request hash is the hash of those settings."""
    with ChatStandIn() as server:
        done = run_cli(
            REPO,
            *("postprocess", "judge", "ask"),
            *("--package", str(mini_packages / "groundtruth-package.zip")),
            *("--out", str(tmp_path / "run"), "--server", server.url),
            *("--model", "made-up/model", "--temperature", "0.5"),
            *("--max-tokens", "100", "--seed", "3"),
        )
        bodies = list(server.received)
    assert done.returncode == 0, done.stdout + done.stderr
    assert {
        (b["model"], b["temperature"], b["max_tokens"], b["seed"]) for b in bodies
    } == {("made-up/model", 0.5, 100, 3)}
    lines = [
        json.loads(line)
        for line in (tmp_path / "run/answers.jsonl").read_text().splitlines()
    ]
    assert sorted(line["request_sha256"] for line in lines) == sorted(
        recorded_hash(b) for b in bodies
    )


def test_package_and_ask_write_receipts_with_their_settings(mini_packages, mini_runs):
    """Each step's folder records the command, its settings, the sha256 of its inputs and outputs and the code."""
    package = json.loads((mini_packages / "receipt.json").read_text())
    assert package["command"] == "mbdef postprocess judge package"
    assert package["settings"]["neighbours"] == 32 and package["settings"]["batch"] == 4
    assert set(package["outputs"]) == {
        "package.zip",
        "requests.jsonl",
        "groundtruth-package.zip",
        "groundtruth-requests.jsonl",
    }
    assert {"store", "links", "cache", "ground_truth_folder"} <= set(package["inputs"])
    runs, _ = mini_runs
    ask = json.loads((runs / "corpus/receipt.json").read_text())
    assert ask["command"] == "mbdef postprocess judge ask"
    assert ask["settings"]["model"] == "google/gemma-4-12B-it-qat-w4a16-ct"
    assert (ask["settings"]["temperature"], ask["settings"]["max_tokens"]) == (
        0.0,
        4096,
    )
    assert (
        ask["inputs"]["package"]["sha256"]
        == hashlib.sha256((mini_packages / "package.zip").read_bytes()).hexdigest()
    )
    assert set(ask["outputs"]) == {
        "answers.jsonl",
        "decisions.json",
        "generation-complete.json",
    }


def test_no_step_writes_into_a_frozen_folder(tmp_path):
    """A frozen folder of a throwaway data root, so that a broken refusal could not write into `data/frozen/`."""
    root = tmp_path / "data-root"
    inside = root / "frozen/step5-post-processing/20261021/new-folder"
    for step in (
        ["package", "--out", str(inside)],
        ["ask", "--package", str(tmp_path / "p.zip"), "--out", str(inside)],
        ["table", "--package", str(tmp_path / "p.zip"), "--runs", str(tmp_path)]
        + ["--out", str(inside / "t.csv")],
    ):
        done = run_cli(
            REPO, "postprocess", "judge", *step, env={"DATA_ROOT": str(root)}
        )
        assert done.returncode == 1, step
        assert "is inside a frozen folder" in done.stderr, step
    assert not inside.exists()


# --- the real files (slow: they need the data root) ------------------------------------------------------------------

RUNS = "20261020/judge-sweep/runs"  # the recorded runs, in the data root
TRAIN_DEV_RUN = RUNS + "/s1-qwen8b-k32-think0"
CORPUS_RUNS = [f"{RUNS}/corpus/s1-qwen8b-k32-think0-shard{s}" for s in range(8)]
# The recorded corpus package: its inputs.json and its request list
CORPUS_INPUTS_SHA256 = (
    "4c58203f7f69ddab2a5636618f16ccf20ee156c05272e242ca0cc3fca3270b64"
)
CORPUS_REQUESTS_SHA256 = (
    "ab3d6efeb25baee06bd357ab14d6f847273421ff0b67e340a0034e618264a7d9"
)


def frozen_or_skip(rel):
    files.load_env()
    try:
        path = postprocess.frozen_file(rel)
    except SystemExit:
        pytest.skip("DATA_ROOT is not set")
    if not path.exists():
        pytest.skip(f"{path} is not there (mbdef data fetch)")
    return path


@pytest.fixture(scope="module")
def real_packages(tmp_path_factory):
    frozen_or_skip(TRAIN_DEV_RUN)
    out = tmp_path_factory.mktemp("judge-real") / "packages"
    done = run_cli(REPO, "postprocess", "judge", "package", "--out", str(out))
    assert done.returncode == 0, done.stdout + done.stderr
    return out


def recorded_hashes(run: Path) -> dict[int, str]:
    """{request id: the hash its usage line recorded} of a recorded run."""
    usage = {}
    for line in (run / "requests.jsonl").read_text().splitlines():
        u = json.loads(line)
        usage[u["response_id"]] = u["request_sha256"]
    out = {}
    for line in (run / "raw-responses.jsonl").open():
        r = json.loads(line)
        out[r["request_id"]] = usage[r["response"]["response_id"]]
    return out


@pytest.mark.slow
def test_the_recorded_corpus_package_comes_back_byte_for_byte(real_packages):
    assert (
        hashlib.sha256(member(real_packages / "package.zip")).hexdigest()
        == CORPUS_INPUTS_SHA256
    )
    assert (
        hashlib.sha256((real_packages / "requests.jsonl").read_bytes()).hexdigest()
        == CORPUS_REQUESTS_SHA256
    )


@pytest.mark.slow
def test_the_recorded_requests_come_back(real_packages):
    """The train + dev run's 532 requests (their inputs and hashes) and the 30,016 corpus requests' hashes."""
    from mbdef.postprocess import judge

    run = frozen_or_skip(TRAIN_DEV_RUN)
    recorded = json.loads((run / "requests-inputs.json").read_text())
    mine = judge.requests_for(
        judge.read_package(real_packages / "groundtruth-package.zip")
    )
    assert [
        {k: v for k, v in r.items() if k != "prompt_tokens"} for r in recorded
    ] == mine
    hashes = recorded_hashes(run)
    assert {
        r["request_id"]: judge.request_sha256(*judge.render(r)[1:]) for r in mine
    } == hashes
    corpus = judge.read_package(real_packages / "package.zip")
    for s, rel in enumerate(CORPUS_RUNS):
        hashes = recorded_hashes(frozen_or_skip(rel))
        mine = judge.requests_for(corpus, f"shard{s}")
        assert {
            r["request_id"]: judge.request_sha256(*judge.render(r)[1:]) for r in mine
        } == hashes


@pytest.mark.slow
def test_the_frozen_judge_tables_come_back_byte_for_byte(real_packages, tmp_path):
    """From the recorded runs' decisions: the corpus table and the train + dev table the ensemble reads."""
    for package, runs, frozen in (
        (
            "package.zip",
            CORPUS_RUNS,
            "20261020/judge-sweep/corpus/20261021/judge-corpus-scores.csv",
        ),
        (
            "groundtruth-package.zip",
            [TRAIN_DEV_RUN],
            "20261020/judge-sweep/stage1/20261021/scores/s1-qwen8b-k32-think0-prob.csv",
        ),
    ):
        done = run_cli(
            REPO,
            *(
                "postprocess",
                "judge",
                "table",
                "--package",
                str(real_packages / package),
            ),
            *(
                "--runs",
                *map(str, map(frozen_or_skip, runs)),
                "--out",
                str(tmp_path / "t.csv"),
            ),
        )
        assert done.returncode == 0, done.stdout + done.stderr
        assert (tmp_path / "t.csv").read_bytes() == (FROZEN / frozen).read_bytes()
