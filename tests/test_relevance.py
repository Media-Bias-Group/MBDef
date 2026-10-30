"""The relevance step (docs/step-1-2-papers/relevance.md): two classifiers, an arbiter for their disagreements, the relevant list.

The fast tests replay the made-up recordings of tests/fixtures/relevance-mini/ (its maker says which case each paper is) through the command line, as a user runs it. The slow test replays the released recordings in the data root and checks the counts the page reports and the link to the seed list.
"""

import csv
import hashlib
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from cli_helpers import run_cli, write_env

from mbdef.relevance.classify import CLASSIFIERS, label, render

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "relevance-mini"


def rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def ok(done) -> str:
    assert done.returncode == 0, done.stderr
    return done.stdout


def replay_all(tmp_path: Path) -> Path:
    """Run the whole step on the fixture's recordings; returns the output folder."""
    out = tmp_path / "out"
    papers = str(FIXTURE / "papers.csv")
    for name in ("qwen", "gemma"):
        ok(
            run_cli(
                tmp_path,
                "relevance",
                "classify",
                "--classifier",
                name,
                "--papers",
                papers,
                "--replay",
                str(FIXTURE / name),
                "--out",
                str(out / name),
            )
        )
    ok(
        run_cli(
            tmp_path,
            "relevance",
            "compare",
            "--papers",
            papers,
            "--qwen",
            str(out / "qwen"),
            "--gemma",
            str(out / "gemma"),
            "--out",
            str(out / "compare"),
        )
    )
    ok(
        run_cli(
            tmp_path,
            "relevance",
            "classify",
            "--classifier",
            "glm",
            "--papers",
            str(out / "compare" / "disagreements.csv"),
            "--examples",
            str(FIXTURE / "examples.csv"),
            "--replay",
            str(FIXTURE / "glm"),
            "--out",
            str(out / "glm"),
        )
    )
    ok(
        run_cli(
            tmp_path,
            "relevance",
            "export",
            "--compare",
            str(out / "compare"),
            "--glm",
            str(out / "glm"),
            "--out",
            str(out / "final"),
        )
    )
    return out


def test_a_replay_keeps_the_agreed_papers_and_those_the_arbiter_marks_relevant(
    tmp_path,
):
    out = replay_all(tmp_path)
    relevant = rows(out / "final" / "relevant-papers.csv")
    assert [
        (r["corpus_id"], r["route"], r["qwen_label"], r["gemma_label"], r["glm_label"])
        for r in relevant
    ] == [
        ("doi_10.1000_agree", "agreed", "relevant", "relevant", ""),
        ("doi_10.1000_tie", "tie-break", "not relevant", "relevant", "relevant"),
        # Qwen's answer to the other paper of its id: the released rule pairs answers with papers by order.
        ("hash_shared_b", "tie-break", "not relevant", "relevant", "relevant"),
    ]
    assert relevant[0]["title"] == "Framing of migration in national newspapers"
    assert relevant[0]["doi"] == "10.1000/agree"
    receipt = json.loads((out / "final" / "receipt.json").read_text())
    assert receipt["summary"] == {
        "agreed": 1,
        "tie-break": 2,
        "relevant": 3,
        "disagreements": 4,
        "arbiter answers": 4,
    }


def test_a_replay_stops_at_a_recorded_answer_whose_request_this_step_does_not_build(
    tmp_path,
):
    # One title edited: the recorded answer to the old title answers no request built from the new papers file.
    papers = (FIXTURE / "papers.csv").read_text(encoding="utf-8")
    changed = tmp_path / "papers.csv"
    changed.write_text(
        papers.replace(
            "Spreading false stories online", "Spreading true stories online"
        ),
        encoding="utf-8",
    )
    done = run_cli(
        tmp_path,
        "relevance",
        "classify",
        "--classifier",
        "gemma",
        "--papers",
        str(changed),
        "--replay",
        str(FIXTURE / "gemma"),
        "--out",
        str(tmp_path / "out"),
    )
    assert done.returncode != 0
    assert "recorded answer 3 (paper id doi_10.1000_tie, request_sha256 " in done.stderr
    assert "answers no request this step builds from the papers file" in done.stderr


def test_a_replay_stops_when_the_settings_differ_from_the_recorded_ones(tmp_path):
    done = run_cli(
        tmp_path,
        "relevance",
        "classify",
        "--classifier",
        "gemma",
        "--papers",
        str(FIXTURE / "papers.csv"),
        "--replay",
        str(FIXTURE / "gemma"),
        "--max-tokens",
        "512",
        "--out",
        str(tmp_path / "out"),
    )
    assert done.returncode != 0
    assert "'max_tokens': 2048" in done.stderr and "'max_tokens': 512" in done.stderr


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("relevant", "relevant"),
        (
            "The article is about rabbit farming.\n</think>\n\nnot relevant",
            "not relevant",
        ),
        ("<think>It may be relevant.</think> not relevant", "not relevant"),
        (
            "Final answer: not relevant. It is relevant only to soil science.",
            "not relevant",
        ),
        ("The label is Relevant.", "relevant"),
        ("This is irrelevant to media.", "not relevant"),
        ("not sure", "can't determine"),
        ("I can't say from this abstract.", "can't say"),
        ("", "can't determine"),
    ],
)
def test_an_answer_is_reduced_to_its_final_or_last_label(response, expected):
    assert label(response) == expected


PAPER = {"title": "A title", "abstract": "An abstract."}
EXAMPLE = {
    "title": "An example",
    "abstract": "Its abstract.",
    "manual_label": "relevant",
}


def test_the_similar_examples_prompt_without_examples_is_the_zero_shot_prompt():
    """Qwen's released run used FS-similar, whose examples no corpus paper had: the text it was sent is ZS's, the template `CLASSIFIERS` gives Qwen (docs/step-1-2-papers/relevance.md#the-prompts)."""
    assert CLASSIFIERS["qwen"].template == "zero_shot"
    zero_shot = render("zero_shot", PAPER)

    assert render("few_shot_similar", {**PAPER, "examples": []}) == zero_shot
    assert render("few_shot_diverse", {**PAPER, "examples": []}) == zero_shot
    assert render("few_shot_similar", {**PAPER, "examples": [EXAMPLE]}).startswith(
        "Use these similarly labeled examples as guidance:\n\n- Title: An example\n"
    )
    assert render("cot", PAPER) == (
        "Think step-by-step before providing the answer.\n\n" + zero_shot
    )


@pytest.fixture
def server():
    """A local OpenAI-compatible server that answers `relevant` for a paper whose prompt names news, else `not relevant`, and keeps every request it got (path, key and body)."""
    got: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            got.append(
                {
                    "path": self.path,
                    "authorization": self.headers["Authorization"],
                    "body": body,
                }
            )
            text = (
                "relevant"
                if "news" in body["messages"][1]["content"].lower()
                else "not relevant"
            )
            answer = {
                "choices": [
                    {"message": {"content": f"  {text}\n"}, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 2,
                    "total_tokens": 9,
                },
            }
            data = json.dumps(answer).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v1", got
    httpd.shutdown()


def test_a_live_run_asks_the_server_once_per_paper_with_text_and_its_folder_replays(
    tmp_path, server
):
    base, got = server
    write_env(tmp_path, DSPY_API_BASE=base, DSPY_API_KEY="test-key")
    out = tmp_path / "live"
    ok(
        run_cli(
            tmp_path,
            "relevance",
            "classify",
            "--classifier",
            "gemma",
            "--papers",
            str(FIXTURE / "papers.csv"),
            "--workers",
            "2",
            "--out",
            str(out),
        )
    )
    assert len(got) == 7  # eight papers, one without title and abstract
    first = sorted(got, key=lambda g: g["body"]["messages"][1]["content"])[0]
    assert first["path"] == "/v1/chat/completions"
    assert first["authorization"] == "Bearer test-key"
    assert sorted(first["body"]) == ["max_tokens", "messages", "model"]
    assert (
        first["body"]["model"] == "gemma_4_12b_it"
        and first["body"]["max_tokens"] == 2048
    )
    assert first["body"]["messages"][0]["content"].startswith(
        "You are an expert in media bias research."
    )
    assert first["body"]["messages"][1]["content"] == (
        "Article to classify:\nTitle: A faster sorting network\nAbstract: We give a sorting network of smaller depth."
        "\n\nReturn exactly one label and no explanation:\nrelevant\nnot relevant\nnot sure\n\nAnswer:"
    )
    answers = rows(out / "answers.csv")
    # In the papers' order, each naming its paper, so two papers with one id cannot be mixed up.
    assert [(a["corpus_id"], a["label"]) for a in answers] == [
        ("doi_10.1000_agree", "relevant"),
        ("hash_shared_a", "not relevant"),
        ("doi_10.1000_tie", "not relevant"),
        ("doi_10.1000_unsure", "relevant"),  # "newspaper"
        ("hash_shared_b", "relevant"),
        ("doi_10.1000_neither", "not relevant"),
        ("doi_10.1000_unasked", "not relevant"),
    ]
    assert answers[0]["response"] == "relevant" and answers[0]["total_tokens"] == "9"
    assert rows(out / "not-answered.csv") == [
        {
            "corpus_id": "hash_da39a3ee5e6b4b0d",
            "paperid": "hash_da39a3ee5e6b4b0d",
            "reason": "no title or abstract",
        }
    ]
    # The live run's folder is a recording: a replay of it sends nothing and gives the same answers.
    again = tmp_path / "again"
    ok(
        run_cli(
            tmp_path,
            "relevance",
            "classify",
            "--classifier",
            "gemma",
            "--papers",
            str(FIXTURE / "papers.csv"),
            "--replay",
            str(out),
            "--out",
            str(again),
        )
    )
    assert len(got) == 7
    assert (again / "answers.csv").read_bytes() == (out / "answers.csv").read_bytes()


# ---------------------------------------------------------------------------------------------
# The released recordings (data root, folder relevance/).
# ---------------------------------------------------------------------------------------------


def data_root() -> Path | None:
    env = REPO / ".env"
    for line in env.read_text().splitlines() if env.exists() else []:
        if line.startswith("DATA_ROOT="):
            return Path(line.split("=", 1)[1]).expanduser()
    value = os.environ.get("DATA_ROOT")
    return Path(value).expanduser() if value else None


REPO = Path(__file__).resolve().parent.parent
ROOT = data_root()
# The released list, row for row (corpus_id, paper id, labels, route); its sha256 is that of the released export.
RELEASED_LIST_SHA256 = (
    "750c3ac71c0256660a5158d7a123d2c26477582be6fce066e094258379a15978"
)


@pytest.mark.slow
@pytest.mark.skipif(
    ROOT is None or not (ROOT / "relevance" / "papers.csv").is_file(),
    reason="needs the data root's relevance/ folder (data/data-root.tsv)",
)
def test_a_replay_of_the_released_recordings_gives_the_released_list(tmp_path):
    assert ROOT is not None
    relevance = ROOT / "relevance"
    out = tmp_path / "out"
    env = {"DATA_ROOT": str(ROOT)}
    summaries = {}
    for name in ("qwen", "gemma"):
        ok(
            run_cli(
                tmp_path,
                "relevance",
                "classify",
                "--classifier",
                name,
                "--replay",
                str(relevance / "answers" / name),
                "--out",
                str(out / name),
                env=env,
            )
        )
    ok(
        run_cli(
            tmp_path,
            "relevance",
            "compare",
            "--qwen",
            str(out / "qwen"),
            "--gemma",
            str(out / "gemma"),
            "--out",
            str(out / "compare"),
            env=env,
        )
    )
    ok(
        run_cli(
            tmp_path,
            "relevance",
            "classify",
            "--classifier",
            "glm",
            "--papers",
            str(out / "compare" / "disagreements.csv"),
            "--replay",
            str(relevance / "answers" / "glm"),
            "--out",
            str(out / "glm"),
            env=env,
        )
    )
    ok(
        run_cli(
            tmp_path,
            "relevance",
            "export",
            "--compare",
            str(out / "compare"),
            "--glm",
            str(out / "glm"),
            "--out",
            str(out / "final"),
            env=env,
        )
    )
    for name in ("qwen", "gemma", "compare", "glm", "final"):
        summaries[name] = json.loads((out / name / "receipt.json").read_text())[
            "summary"
        ]
    # The counts docs/step-1-2-papers/relevance.md reports.
    assert summaries["qwen"]["labels"] == {
        "not relevant": 11457,
        "relevant": 2998,
        "can't determine": 27,
    }
    assert summaries["gemma"]["labels"] == {
        "not relevant": 9376,
        "relevant": 5074,
        "can't determine": 112,
    }
    assert summaries["qwen"]["without title or abstract"] == 300
    assert summaries["compare"] == {
        "answered by both": 14482,
        "both not relevant": 9305,
        "agreed": 2963,
        "disagreements": 2196,
        "both can't determine": 18,
    }
    assert summaries["glm"]["labels"] == {
        "not relevant": 1033,
        "relevant": 925,
        "can't determine": 234,
    }
    assert summaries["final"] == {
        "agreed": 2963,
        "tie-break": 925,
        "relevant": 3888,
        "disagreements": 2196,
        "arbiter answers": 2192,
    }
    relevant_list = out / "final" / "relevant-papers.csv"
    assert (
        hashlib.sha256(relevant_list.read_bytes()).hexdigest() == RELEASED_LIST_SHA256
    )
    # Every one of the 3,869 seed papers (data/seeds/) came from a row of this list with its route.
    listed = {(r["paperid"], r["route"]) for r in rows(relevant_list)}
    with (REPO / "data/seeds/seed-papers.tsv").open(
        newline="", encoding="utf-8"
    ) as handle:
        seeds = list(csv.DictReader(handle, delimiter="\t"))
    assert len(seeds) == 3869
    assert {s["step2_id"] for s in seeds} <= {paper_id for paper_id, _ in listed}
    routed = [s for s in seeds if s["step2_route"] != "not-selected"]
    assert len(routed) == 3865 and all(
        (s["step2_id"], s["step2_route"]) in listed for s in routed
    )
