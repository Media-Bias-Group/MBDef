"""`mbdef extract run` on synthetic papers, served from recorded responses.

No network, no GPU and no model: every model call is answered from a recording, and a call that was not recorded stops the run.
The papers are the synthetic papers of `tests/synthetic_papers.py`; `tests/fixtures/synthetic-extract-replay.zip` holds a run recorded on them with a stand-in model (`tests/fixtures/make_synthetic_replays.py` made it).
"""

import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

import pytest
import synthetic_papers as synthetic
from cli_helpers import run_cli, write_env

REPO = Path(__file__).resolve().parents[1]
PROGRAMS = REPO / "data/extractors/gemma12b-bootstrap"
PICKS = REPO / "data/extractors/picks.json"
SYNTHETIC = Path(__file__).with_name("fixtures") / "synthetic-extract-replay.zip"
SYNTHETIC_PARTS = (
    "a",
    "b+c",
)  # the synthetic programs a, b and c on 400, 650 and 900 characters


def paper_key(paper_id: str) -> str:
    return hashlib.sha256(paper_id.encode()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def synthetic_setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    """(a work folder with a `.env`, the synthetic recording unpacked, the synthetic TEST papers' TEIs)."""
    work = tmp_path / "work"
    work.mkdir()
    write_env(work, DATA_ROOT=str(tmp_path / "data-root"))
    recorded = tmp_path / "recorded"
    with zipfile.ZipFile(SYNTHETIC) as bundle:
        bundle.extractall(recorded)
    return work, recorded, synthetic.write_teis(tmp_path / "teis", synthetic.TEST)


def synthetic_args(
    recorded: Path, teis: Path, out: Path, *extra: str, papers=synthetic.TEST
) -> list[str]:
    args = ["extract", "run", "--teis", str(teis)]
    args += ["--programs", str(recorded / "programs")]
    for paper in papers:
        args += ["--paper", paper]
    for part in SYNTHETIC_PARTS:
        args += ["--part", part]
        args += ["--replay", str(recorded / "run" / part / "requests-extract")]
    return [*args, "--out", str(out), *extra]


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory):
    """One replayed run of both synthetic parts on t1 and t2: (finished process, out, recording)."""
    tmp_path = tmp_path_factory.mktemp("synthetic")
    work, recorded, teis = synthetic_setup(tmp_path)
    out = tmp_path / "out"
    return run_cli(work, *synthetic_args(recorded, teis, out)), out, recorded


def test_on_synthetic_papers_a_replay_gives_the_recorded_windows_and_unions(
    synthetic_run,
):
    done, out, recorded = synthetic_run

    assert done.returncode == 0, done.stderr
    for part in SYNTHETIC_PARTS:
        for paper in synthetic.TEST:
            saved = recorded / "run" / part / "papers" / paper_key(paper)
            new = out / part / "papers" / paper_key(paper)
            names = sorted(p.name for p in (saved / "extraction").iterdir())
            assert sorted(p.name for p in (new / "extraction").iterdir()) == names
            for name in names:
                assert read(new / "extraction" / name) == read(
                    saved / "extraction" / name
                ), (part, paper, name)
            assert (new / "union.json").read_bytes() == (
                saved / "union.json"
            ).read_bytes()


def test_on_synthetic_papers_each_union_holds_every_term_the_paper_states_once(
    synthetic_run,
):
    """The stand-in model proposes each definition or example sentence that stands whole in the window it reads; the windows overlap, so each paper's union has each of its three defined terms and its one example, once, with its sentence (tests/synthetic_papers.py)."""
    done, out, _recorded = synthetic_run

    assert done.returncode == 0, done.stderr
    for part in SYNTHETIC_PARTS:
        for paper in synthetic.TEST:
            union = read(out / part / "papers" / paper_key(paper) / "union.json")
            text = synthetic.text_of(paper)
            stated = synthetic.terms_of(paper) | {
                t: s for t, s in synthetic.EXAMPLES.items() if s in text
            }
            assert len(stated) == 4
            assert {
                r["term"].lower(): [d["text"] for d in r["definitions"]]
                for r in union["records"]
            } == {term: [sentence] for term, sentence in stated.items()}


def test_on_synthetic_papers_the_receipt_counts_the_windows_and_records(
    synthetic_run,
):
    done, out, recorded = synthetic_run

    assert done.returncode == 0, done.stderr
    receipt = read(out / "receipt.json")
    assert {
        part: [(b["program"], b["window_chars"], b["overlap_chars"]) for b in branches]
        for part, branches in receipt["settings"]["parts"].items()
    } == {"a": [("a", 400, 80)], "b+c": [("b", 650, 120), ("c", 900, 150)]}
    assert receipt["settings"]["endpoint"] == "replay of recorded responses"
    assert [p["paper_id"] for p in receipt["papers"]] == list(synthetic.TEST)
    # t1 has 1,070 characters and t2 859. A window ends at the last paragraph break in its second half, so they get 4 + 4 windows of at most 400 characters, 2 + 2 of 650 and 2 + 1 of 900.
    # Each paper's union has its 3 defined terms and 1 example: 8 records, one definition each.
    counts = {"records": 8, "definitions": 8, "candidates": 8, "failed_windows": 0}
    # Both parts find the same 4 terms with the same sentences, so pooled they are still 8 records.
    assert receipt["summary"] == {
        "a": {"papers": 2, "windows": 8, **counts, "model_calls": 8},
        "b+c": {"papers": 2, "windows": 7, **counts, "model_calls": 7},
        "predictions.jsonl": {"papers": 2, "records": 8, "definitions": 8},
    }
    assert json.loads(done.stdout) == receipt["summary"]
    # The usage log has the fields and values of the recorded one, but for the timing.
    for part in SYNTHETIC_PARTS:
        lines = (out / part / "usage-extract.jsonl").read_text().splitlines()
        saved = {
            row["request_sha256"]: row
            for row in map(
                json.loads,
                (recorded / "run" / part / "usage-extract.jsonl")
                .read_text()
                .splitlines(),
            )
        }
        assert len(lines) == len(saved) == receipt["summary"][part]["model_calls"]
        for row in map(json.loads, lines):
            assert list(row) == list(saved[row["request_sha256"]])
            for field in [f for f in row if f not in ("started", "seconds")]:
                assert row[field] == saved[row["request_sha256"]][field], field


def test_a_shard_of_extract_plan_names_the_papers_and_their_titles(
    synthetic_run, tmp_path
):
    """`--shard` reads the papers and titles of one shard that `extract plan` wrote (here one shard of the two synthetic papers): the run is the `--paper` run's, window for window."""
    _done, by_paper, _recorded = synthetic_run
    work, recorded, teis = synthetic_setup(tmp_path)
    plan = tmp_path / "plan"
    planned = run_cli(
        work,
        *("extract", "plan", "--teis", str(teis), "--picks", str(PICKS)),
        *("--out", str(plan)),
    )
    assert planned.returncode == 0, planned.stderr
    shard = plan / "shards" / "shard-000.json"
    out = tmp_path / "out"

    done = run_cli(
        work, *synthetic_args(recorded, teis, out, "--shard", str(shard), papers=())
    )

    assert done.returncode == 0, done.stderr
    for part in SYNTHETIC_PARTS:
        for paper in synthetic.TEST:
            folder = Path(part) / "papers" / paper_key(paper)
            assert (out / folder / "union.json").read_bytes() == (
                by_paper / folder / "union.json"
            ).read_bytes()
    receipt = read(out / "receipt.json")
    assert receipt["inputs"]["shard"]["sha256"] == (
        hashlib.sha256(shard.read_bytes()).hexdigest()
    )
    assert [p["title_source"] for p in receipt["papers"]] == ["shard", "shard"]


def test_a_shard_whose_tei_changed_is_refused(tmp_path):
    work, recorded, teis = synthetic_setup(tmp_path)
    paper = synthetic.TEST[0]
    shard = tmp_path / "shard-000.json"
    shard.write_text(
        json.dumps([{"paper_id": paper, "frozen_title": "x", "tei_sha256": "0" * 64}])
    )
    out = tmp_path / "out"

    done = run_cli(
        work, *synthetic_args(recorded, teis, out, "--shard", str(shard), papers=())
    )

    assert done.returncode == 1
    assert done.stderr.strip() == (
        f"{paper}: its TEI's sha256 is not the one {shard} names."
    )
    assert not out.exists()


def test_a_program_that_differs_from_its_manifest_is_refused(tmp_path):
    work, _recorded, teis = synthetic_setup(tmp_path)
    programs = tmp_path / "programs"
    shutil.copytree(PROGRAMS, programs)
    changed = programs / "w8k/program.json"
    changed.write_bytes(changed.read_bytes() + b"\n")
    out = tmp_path / "out"

    done = run_cli(
        work,
        *("extract", "run", "--teis", str(teis)),
        *("--programs", str(programs), "--out", str(out)),
    )

    assert done.returncode == 1
    assert done.stderr.strip() == (
        f"{changed}: its sha256 is {hashlib.sha256(changed.read_bytes()).hexdigest()}, not the "
        "8d75ce5f4cd5e379373d0891a2de14803f833cc55fb2ef5de8afbf1f7a48f4bd that "
        f"{programs / 'w8k/manifest.json'} records. Refusing to run a changed program."
    )
    assert not out.exists()


def test_a_live_run_needs_the_endpoint_from_env(tmp_path):
    work, _recorded, teis = synthetic_setup(tmp_path)
    out = tmp_path / "out"

    done = run_cli(
        work,
        *("extract", "run", "--teis", str(teis), "--programs", str(PROGRAMS)),
        *("--paper", synthetic.TEST[0], "--out", str(out)),
    )

    assert done.returncode == 1
    assert done.stderr.strip() == (
        "DSPY_API_BASE is not set. Copy .env.example to .env and set DSPY_API_BASE there; "
        ".env.example says what it is."
    )
    assert not out.exists()


def recorded_keys(recorded: Path) -> set[str]:
    return {
        read(path)["request_sha256"]
        for part in SYNTHETIC_PARTS
        for path in (recorded / "run" / part / "requests-extract").iterdir()
    }


def test_a_call_that_was_not_recorded_stops_the_run_and_names_its_request(tmp_path):
    work, recorded, teis = synthetic_setup(tmp_path)
    out = tmp_path / "out"
    titles = work / "titles.json"
    titles.write_text(
        json.dumps({synthetic.TEST[0]: "An altered title"}), encoding="utf-8"
    )

    done = run_cli(
        work,
        *synthetic_args(recorded, teis, out, "--titles", str(titles), "--workers", "1"),
    )

    assert done.returncode == 1
    assert "Stopped: a model call failed in part a" in done.stderr
    missing = re.search(r"request_sha256 ([0-9a-f]{64}) was not recorded", done.stderr)
    assert missing, done.stderr
    assert missing[1] not in recorded_keys(recorded)
    # The unrecorded request is kept, so it can be compared with the recorded ones.
    receipt = read(out / "a/requests-extract" / f"{missing[1]}.json")
    assert receipt["error"]["type"] == "NotRecorded"
    assert receipt["response"] is None
    assert "An altered title" in receipt["formatted_request"]["messages"][-1]["content"]
    assert not list(out.glob("*/papers/*/union.json"))
    assert not (out / "receipt.json").exists()


MINI_TEI = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc><titleStmt>
<title level="a" type="main">A paper about media bias</title></titleStmt></fileDesc></teiHeader>
<text><body><div><p>Media bias is the bias of journalists and news producers.</p></div></body></text>
</TEI>
"""


def missing_request(stderr: str) -> Path:
    """The receipt that a stopped replay wrote for its unrecorded call."""
    found = re.search(r"its request is in (\S+\.json)", stderr)
    assert found, stderr
    return Path(found[1])


def test_a_call_that_failed_when_recorded_fails_the_same_way_in_the_replay(tmp_path):
    """Four released windows hit `KeyError: 'temperature'` on their first call. In the chat format DSPy then asked again in JSON, and the window kept the JSON answer's records; in the JSON format the window gave no records. A replay must do the same, so it serves recorded failures too."""
    work, recorded, _teis = synthetic_setup(tmp_path)
    root = tmp_path / "data-root"
    teis, replay = tmp_path / "mini-teis", tmp_path / "replay"
    teis.mkdir()
    replay.mkdir()
    (teis / "mini.tei.xml").write_text(MINI_TEI, encoding="utf-8")

    def run(name: str, *extra: str):
        return run_cli(
            work,
            "extract",
            "run",
            "--teis",
            str(teis),
            "--programs",
            str(PROGRAMS),
            "--part",
            "w8k",
            "--replay",
            str(replay),
            "--out",
            str(root / name),
            *extra,
        )

    # Record the chat request as the released runs recorded the failed ones (the replay folder starts with one receipt of another run).
    shutil.copy(next((recorded / "run/a/requests-extract").glob("*.json")), replay)
    first = missing_request(run("first").stderr)
    failed = read(first) | {"error": {"type": "KeyError", "message": "'temperature'"}}
    (replay / first.name).write_text(json.dumps(failed), encoding="utf-8")

    # The replayed failure makes DSPy ask again in JSON; record that answer.
    second = missing_request(run("second").stderr)
    assert second.name != first.name
    answer = {
        "reasoning": "The window defines media bias.",
        "definition_records": [
            {
                "term": "MEDIA BIAS",
                "definitions": [
                    {"text": "media  bias is the bias of journalists", "citation": []}
                ],
            }
        ],
    }
    response = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": json.dumps(answer)},
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 30, "total_tokens": 40},
    }
    (replay / second.name).write_text(
        json.dumps(read(second) | {"response": response, "error": None}),
        encoding="utf-8",
    )

    chat = run("chat")
    assert chat.returncode == 0, chat.stderr
    window = read(next((root / "chat/w8k/papers").glob("*/extraction/0-00000.json")))
    assert window["requests"] == [first.stem, second.stem]
    assert window["definition_records"] == [
        {
            "term": "Media bias",
            "definitions": [
                {"text": "Media bias is the bias of journalists", "citation": []}
            ],
            "synonyms": [],
            "description": [],
            "related_terms": [],
            "subtypes": [],
            "supertypes": [],
            "examples": [],
        }
    ]
    assert read(root / "chat/receipt.json")["summary"]["w8k"]["model_calls"] == 1

    # In the JSON format the JSON request is the first call; recorded as failed, the window gives no records and the run goes on.
    failed = read(replay / second.name) | {"response": None}
    failed["error"] = {"type": "KeyError", "message": "'temperature'"}
    (replay / second.name).write_text(json.dumps(failed), encoding="utf-8")
    json_run = run("json", "--format", "json")
    assert json_run.returncode == 0, json_run.stderr
    papers_dir = root / "json/w8k/papers"
    assert not list(papers_dir.glob("*/extraction/*.json"))
    failure = read(next(papers_dir.glob("*/extraction-failures/0-00000.json")))
    assert failure["problem"] == (
        "RecordedFailure: the recorded call failed: KeyError: 'temperature'"
    )
    assert failure["requests"] == [second.stem]
    assert read(next(papers_dir.glob("*/union.json")))["records"] == []


def test_the_lm_logs_the_recorded_response_id_and_provider_through_dspy_s_engine_interface(
    tmp_path,
):
    """A DSPy program's call reaches the LM layer's engine (DSPy 3.4's `complete(Request)`), not a deprecated `forward()` override; the engine keys the request with the `extra_body` DSPy cannot carry, answers it from its recorded receipt, and logs the recorded response id and provider, as the released runs' usage logs did."""
    import warnings

    import dspy

    from mbdef import llm

    def lm(replay):
        return llm.make_lm(
            "m",
            api_base=None,
            api_key=None,
            max_tokens=50,
            seed=9,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            usage_log=tmp_path / "usage.jsonl",
            requests=tmp_path / "requests",
            replay=replay,
            stage="unit",
        )

    def ask(model):
        with warnings.catch_warnings():
            for deprecated in (
                "Implementing custom LMs",
                "Passing OpenAI-style message",
            ):
                warnings.filterwarnings("error", deprecated, DeprecationWarning)
            with dspy.context(lm=model, adapter=dspy.ChatAdapter()):
                return dspy.Predict("question -> answer")(question="Two plus two?")

    with pytest.raises(llm.NotRecorded):
        ask(lm({}))
    (path,) = (tmp_path / "requests").glob("*.json")
    receipt = read(path)
    assert receipt["formatted_request"]["model"] == "openai/m"
    assert {
        k: v for k, v in receipt["formatted_request"].items() if k != "messages"
    } == {
        "model": "openai/m",
        "max_tokens": 50,
        "seed": 9,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    answer = "[[ ## answer ## ]]\nFour\n\n[[ ## completed ## ]]"
    receipt["response"] = {
        "id": "gen-123",
        "provider": "DeepInfra",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": answer},
            }
        ],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
    }
    receipt["error"] = None
    (tmp_path / "requests").rename(tmp_path / "first-try")

    assert ask(lm({receipt["request_sha256"]: receipt})).answer == "Four"
    (row,) = [json.loads(line) for line in (tmp_path / "usage.jsonl").open()]
    assert (row["request_sha256"], row["response_id"], row["provider"]) == (
        receipt["request_sha256"],
        "gen-123",
        "DeepInfra",
    )
    assert (row["prompt_tokens"], row["completion_tokens"]) == (7, 3)
    assert row["settings"] == {
        k: v
        for k, v in receipt["formatted_request"].items()
        if k not in ("model", "messages")
    }


def test_without_the_ground_truth_teis_each_command_says_what_to_give_instead(tmp_path):
    """`data/paper_sources.zip` holds paper text and is not part of this repository: the commands that read it by default say so and what to give instead. Here the work folder has no `data/`."""
    frozen = REPO / "data/frozen/results/training-20260925/runner"
    papers = ("--dataset", str(frozen / "train-dataset.json"))
    papers += ("--split", str(frozen / "split.json"))
    commands = {
        "run": ("extract", "run", "--programs", str(PROGRAMS)),
        "train": ("extract", "train", "--window", "w8k", *papers),
        "check-programs": ("extract", "check-programs", "--programs", str(PROGRAMS)),
    }
    for name, args in commands.items():
        out = () if name == "check-programs" else ("--out", str(tmp_path / name))
        done = run_cli(
            tmp_path, *args, *out, *(papers if name == "check-programs" else ())
        )

        assert done.returncode == 1, (name, done.stdout, done.stderr)
        assert done.stderr.strip() == (
            "data/paper_sources.zip, the ground-truth papers' TEIs, holds paper text and is not "
            "part of this repository. Give --teis a folder or zip of <paper id>.tei.xml files, "
            "the GROBID TEIs of the papers you fetched (docs/step-3-extraction/extract.md)."
        ), name


def test_a_teis_path_that_does_not_exist_is_named(tmp_path):
    done = run_cli(
        tmp_path,
        *("extract", "run", "--programs", str(PROGRAMS), "--teis", "my-teis"),
        *("--out", str(tmp_path / "out")),
    )

    assert done.returncode == 1
    assert done.stderr.strip() == "my-teis: no such folder or zip of TEIs."


def test_a_shard_takes_the_place_of_paper_and_titles(tmp_path):
    """`--shard` names the papers and their titles, so `--paper` or `--titles` beside it is refused, not ignored."""
    for other in (["--paper", "p1"], ["--titles", "titles.json"]):
        done = run_cli(
            tmp_path,
            *("extract", "run", "--out", str(tmp_path / "out")),
            *("--shard", "shard-000.json", *other),
        )
        assert done.returncode == 1, done.stderr
        assert "drop --paper and --titles" in done.stderr
        assert not (tmp_path / "out").exists()


# The predictions file, --pdfs and --thinking: what a reader runs on their own papers


def test_predictions_jsonl_pools_each_paper_s_parts_into_one_record_per_term(
    synthetic_run,
):
    """`predictions.jsonl` holds each paper's records of both parts, merged by term: both parts find the same 4 terms with the same sentences, so each paper keeps 4 records, one line each, as `evaluate score --predictions` reads them."""
    from mbdef.extract.records import merge, read_predictions

    done, out, _recorded = synthetic_run

    assert done.returncode == 0, done.stderr
    lines = [json.loads(line) for line in (out / "predictions.jsonl").open()]
    assert [line["paper_id"] for line in lines] == ["t1"] * 4 + ["t2"] * 4
    assert list(lines[0]) == [
        "paper_id",
        "term",
        "definitions",
        "synonyms",
        "description",
        "related_terms",
        "subtypes",
        "supertypes",
        "examples",
    ]
    pooled = read_predictions(out / "predictions.jsonl")
    for paper in synthetic.TEST:
        unions = [
            read(out / part / "papers" / paper_key(paper) / "union.json")["records"]
            for part in SYNTHETIC_PARTS
        ]
        assert pooled[paper] == merge(unions[0] + unions[1])
    receipt = read(out / "receipt.json")
    assert (
        receipt["outputs"]["predictions"]["sha256"]
        == hashlib.sha256((out / "predictions.jsonl").read_bytes()).hexdigest()
    )


def fake_grobid(request):
    """GROBID's answer to a PDF named `<paper id>.pdf`: the synthetic paper's TEI."""
    from http_replay import Answer

    named = re.search(rb'filename="([^"]+)\.pdf"', request.body)
    assert named, request.body[:200]
    return Answer(body=synthetic.tei(named[1].decode()))


def test_pdfs_are_converted_by_grobid_and_give_the_same_records_as_their_teis(
    tmp_path, synthetic_run
):
    """`--pdfs`: each `<paper id>.pdf` goes to GROBID (a fake on a local port that answers with the synthetic paper's TEI), its TEI is kept in OUT/tei/, and the run on it asks exactly the recorded calls, so its records are those of the run on the TEIs."""
    from http_replay import NO_NETWORK, Answer, Replay

    _done, from_teis, _recorded = synthetic_run
    work, recorded, _teis = synthetic_setup(tmp_path)
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    for paper in synthetic.TEST:
        (pdfs / f"{paper}.pdf").write_bytes(b"%PDF-1.4 " + paper.encode())
    out = tmp_path / "out"
    args = synthetic_args(recorded, tmp_path / "unused", out, papers=())
    args[args.index("--teis") : args.index("--teis") + 2] = ["--pdfs", str(pdfs)]

    with Replay() as grobid:
        grobid.answer("GET /api/isalive", Answer(body="true"))
        grobid.answer("POST /api/processFulltextDocument", fake_grobid)
        grobid.answer("GET /api/version", Answer(body="0.9.1"))
        done = run_cli(work, *args, "--grobid-url", grobid.url, env=NO_NETWORK)

    assert done.returncode == 0, done.stderr
    assert grobid.misses == []
    assert len(grobid.asked("POST /api/processFulltextDocument")) == 2
    for paper in synthetic.TEST:
        assert (out / "tei" / f"{paper}.tei.xml").read_bytes() == synthetic.tei(paper)
    assert (out / "predictions.jsonl").read_bytes() == (
        from_teis / "predictions.jsonl"
    ).read_bytes()
    receipt = read(out / "receipt.json")
    assert receipt["settings"]["grobid"] == {
        "url": grobid.url,
        "version": "0.9.1",
        "consolidate_header": False,
        "pdfs": 2,
        "converted": 2,
        "failed": {},
    }
    assert "pdfs" in receipt["inputs"] and "tei" in receipt["outputs"]


def test_without_a_grobid_pdfs_stop_the_run_and_say_how_to_start_one(tmp_path):
    from http_replay import NO_NETWORK, Answer, Replay

    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    (pdfs / "p1.pdf").write_bytes(b"%PDF-1.4")
    with Replay() as grobid:
        grobid.answer("GET /api/isalive", Answer(503, "down"))
        done = run_cli(
            tmp_path,
            *("extract", "run", "--programs", str(PROGRAMS), "--pdfs", str(pdfs)),
            *("--grobid-url", grobid.url, "--out", str(tmp_path / "out")),
            env=NO_NETWORK,
        )

    assert done.returncode == 1
    assert f"No GROBID answers at {grobid.url}" in done.stderr
    assert "docker run" in done.stderr


def test_thinking_unset_sends_no_switch_so_the_recorded_calls_no_longer_match(
    tmp_path,
):
    """`--thinking unset` (for an API that refuses vLLM's `chat_template_kwargs`) changes every request, so a replay of the recorded run, made with the switch on, stops at the first call; the request it kept has no thinking switch."""
    work, recorded, teis = synthetic_setup(tmp_path)
    out = tmp_path / "out"

    done = run_cli(
        work,
        *synthetic_args(recorded, teis, out, "--thinking", "unset", "--workers", "1"),
    )

    assert done.returncode == 1
    request = read(missing_request(done.stderr))
    assert request["error"]["type"] == "NotRecorded"
    assert "enable_thinking" not in json.dumps(request)
