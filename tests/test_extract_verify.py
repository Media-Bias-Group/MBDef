"""`mbdef extract verify` on synthetic papers, served from recorded responses.

No network, no GPU and no model: every verifier call is answered from a recording, and a call that was not recorded stops the run. The verifier is tested on the synthetic papers of `tests/synthetic_papers.py`: `verify/` in `tests/fixtures/synthetic-extract-replay.zip` holds its run on the synthetic extraction, recorded with a stand-in model that calls a quote a definition when it is one of the papers' defining sentences.

The verifier counts each batch's prompt tokens with the verifier model's tokenizer, which CI does not have. These tests count them with `tests/fixtures/word-tokenizer` (one token per word): the batches only depend on whether a batch fits the context, and every batch here is far inside it, so the batches, and with them the requests, are the recorded ones.
"""

import hashlib
import json
import re
import zipfile
from pathlib import Path

import pytest
import synthetic_papers as synthetic
from cli_helpers import run_cli, write_env

FIXTURES = Path(__file__).with_name("fixtures")
TOKENIZER = FIXTURES / "word-tokenizer"
SYNTHETIC = FIXTURES / "synthetic-extract-replay.zip"
SYNTHETIC_PARTS = ("a", "b+c")
GEMMA = (
    "google/gemma-4-12B-it-qat-w4a16-ct",
    "1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee",
)


def paper_key(paper_id: str) -> str:
    return hashlib.sha256(paper_id.encode()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def synthetic_verify(tmp_path: Path, part: str, out: Path, *extra: str):
    """`extract verify` of one synthetic part, replayed from the synthetic recording in TMP_PATH."""
    recorded = tmp_path / "recorded"
    if not recorded.exists():
        with zipfile.ZipFile(SYNTHETIC) as bundle:
            bundle.extractall(recorded)
        (tmp_path / "work").mkdir()
        write_env(tmp_path / "work", DATA_ROOT=str(tmp_path / "data-root"))
    return run_cli(
        tmp_path / "work",
        *("extract", "verify", "--extraction", str(recorded / "run" / part)),
        *("--replay", str(recorded / "verify" / part / "requests-verifier")),
        *("--tokenizer", str(TOKENIZER), "--out", str(out), *extra),
    )


@pytest.fixture(scope="module")
def synthetic_verified(tmp_path_factory):
    """Both synthetic parts verified from the recording: {part: (finished process, out, recording)}."""
    tmp_path = tmp_path_factory.mktemp("synthetic-verify")
    runs = {}
    for part in SYNTHETIC_PARTS:
        out = tmp_path / "verified" / part
        done = synthetic_verify(tmp_path, part, out)
        runs[part] = (done, out, tmp_path / "recorded" / "verify" / part)
    return runs


def test_on_synthetic_papers_a_replay_gives_the_recorded_verified_records(
    synthetic_verified,
):
    for part, (done, out, saved_run) in synthetic_verified.items():
        assert done.returncode == 0, done.stderr
        for paper in synthetic.TEST:
            saved = saved_run / "papers" / paper_key(paper)
            new = out / "papers" / paper_key(paper)
            for name in (
                "verified.json",
                "decisions.json",
                "paper-summary.json",
                "union.json",
            ):
                assert (new / name).read_bytes() == (saved / name).read_bytes(), (
                    part,
                    paper,
                    name,
                )
            names = sorted(p.name for p in (saved / "verification").iterdir())
            assert sorted(p.name for p in (new / "verification").iterdir()) == names
            for name in names:
                assert read(new / "verification" / name) == read(
                    saved / "verification" / name
                )


def test_on_synthetic_papers_the_verifier_keeps_the_definitions_and_drops_the_examples(
    synthetic_verified,
):
    """Each paper's union has its three defined terms and one example (test_extract.py); the stand-in verifier calls only the defining sentences definitions, so the example's record goes."""
    for part, (done, out, _saved) in synthetic_verified.items():
        assert done.returncode == 0, done.stderr
        for paper in synthetic.TEST:
            folder = out / "papers" / paper_key(paper)
            kept = read(folder / "verified.json")["definition_records"]
            assert {r["term"].lower() for r in kept} == set(synthetic.terms_of(paper))
            summary = read(folder / "paper-summary.json")
            assert (summary["candidates"], summary["failed_batches"]) == (4, 0)
            assert summary["missing_verdicts"] == 0
        receipt = read(out / "receipt.json")
        assert receipt["summary"]["kept_records"] == 6, part
        # predictions.jsonl: the kept records, one line each, with their paper.
        lines = [json.loads(line) for line in (out / "predictions.jsonl").open()]
        assert sorted(
            (line["paper_id"], line["term"].lower()) for line in lines
        ) == sorted(
            (paper, term)
            for paper in synthetic.TEST
            for term in synthetic.terms_of(paper)
        )
        assert receipt["summary"]["predictions.jsonl"]["records"] == 6
        assert receipt["settings"]["policy"] == "definitions-strict"
        assert receipt["settings"]["endpoint"] == "replay of recorded responses"


def test_on_synthetic_papers_an_answer_that_is_not_json_rejects_its_batch(tmp_path):
    """The released failure rule: a candidate without all its verdicts is rejected, never repaired. One answer of part a is replaced by text that is not JSON: its batch fails, its candidates are rejected, and the run goes on."""
    done = synthetic_verify(tmp_path, "a", tmp_path / "first")
    assert done.returncode == 0, done.stderr
    requests = tmp_path / "recorded/verify/a/requests-verifier"
    first = read(
        next(iter(sorted((tmp_path / "first/papers").glob("*/verification/*.json"))))
    )
    asked = next(
        path
        for path in requests.iterdir()
        if read(path)["response"]["id"] == first["response"]["response_id"]
    )
    receipt = read(asked)
    receipt["response"]["choices"][0]["message"]["content"] = "all of them are fine"
    asked.write_text(json.dumps(receipt), encoding="utf-8")

    done = synthetic_verify(tmp_path, "a", tmp_path / "second")

    assert done.returncode == 0, done.stderr
    summary = read(tmp_path / "second/receipt.json")["summary"]
    assert summary["failed_batches"] == 1
    # That batch asked about one candidate, a defining sentence: its record goes too.
    assert first["ids"] == ["d0001"]
    assert summary["candidates_rejected_for_missing_verdict"] == 1
    assert summary["kept_records"] == 5


def test_on_synthetic_papers_a_call_that_was_not_recorded_stops_the_run(tmp_path):
    # The recording holds the Qwen verifier's requests; the Gemma model's were not recorded.
    done = synthetic_verify(
        tmp_path, "a", tmp_path / "out", "--model", GEMMA[0], "--workers", "1"
    )

    assert done.returncode == 1
    assert "Stopped: a verifier call failed:" in done.stderr
    assert re.search(r"request_sha256 [0-9a-f]{64} was not recorded", done.stderr)
    assert not (tmp_path / "out/receipt.json").exists()
