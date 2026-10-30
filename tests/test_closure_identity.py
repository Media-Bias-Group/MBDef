"""`mbdef closure identity`: the PDF identity check, replayed from recorded answers, with no network.

Five synthetic checks (`tests/fixtures/synthetic-identity-replay.zip`, made by `tests/fixtures/make_synthetic_replays.py` with a stand-in model) are judged from their recorded answers, and `decide` reproduces the recorded decisions exactly. They reach the outcomes of the released checks: one candidate accepted, one the model accepts and the program checks refuse, one different work, a positive control accepted and a hard negative refused. (The released checks hold the PDFs' first pages, so they are not public.) A request that was not recorded stops the run. The evidence step reads a collect folder; the merge joins two rounds.
"""

import json
import shutil
import zipfile
from pathlib import Path

import pytest
from cli_helpers import run_cli
from http_replay import NO_NETWORK, Answer, Replay
from mini_collect import downloads, setup, write_rows

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic-identity-replay.zip"
ACCEPTED = "doi_10.5555_synthetic.1001"  # the candidate the checks accept


@pytest.fixture
def replay(tmp_path):
    """The five checks, unpacked into a folder."""
    folder = tmp_path / "synthetic"
    with zipfile.ZipFile(FIXTURE) as archive:
        archive.extractall(folder)
    return folder


def judge(work, replay, out, *extra, evidence=None):
    return run_cli(
        work,
        "closure",
        "identity",
        "judge",
        "--evidence",
        str(evidence or replay / "evidence-5.json"),
        "--out",
        str(out),
        "--replay",
        str(replay / "requests"),
        *extra,
        env=NO_NETWORK,
    )


def test_five_checks_are_judged_from_their_recorded_answers(tmp_path, replay):
    out = tmp_path / "judged"

    done = judge(tmp_path, replay, out)

    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith("judged 5 in ") and done.stdout.endswith(
        "unparsed 0\n"
    )
    verdicts = json.loads((out / "verdicts.json").read_text())
    expected = json.loads((replay / "expected-verdicts.json").read_text())
    assert verdicts["verdicts"] == expected
    assert verdicts["model"] == "Qwen/Qwen3.6-35B-A3B"
    assert verdicts["revision"] == "995ad96eacd98c81ed38be0c5b274b04031597b0"
    assert verdicts["prompt_version"] == "identity-llm/1" and verdicts["schema_mode"]
    # Nothing was sent, so there are no request receipts of this run; the receipt says it was a replay.
    assert not (out / "requests").exists()
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["command"] == "mbdef closure identity judge"
    assert receipt["settings"]["replay"] == [str(replay / "requests")]
    assert receipt["settings"]["base_url"] is None
    assert (receipt["settings"]["max_tokens"], receipt["settings"]["temperature"]) == (
        500,
        0,
    )
    assert receipt["summary"] == {"judged": 5, "unparsed": 0}


def test_a_request_that_was_not_recorded_stops_the_run(tmp_path, replay):
    evidence = json.loads((replay / "evidence-5.json").read_text())
    evidence["items"][0]["text"] += " One more sentence the recording never saw."
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(evidence))

    done = judge(tmp_path, replay, tmp_path / "judged", evidence=changed)

    assert done.returncode == 1
    assert "was not recorded" in done.stderr and "request_sha256" in done.stderr
    assert not (tmp_path / "judged" / "verdicts.json").exists()


def test_decide_reproduces_the_recorded_decisions_exactly(tmp_path, replay):
    judged = tmp_path / "judged"
    assert judge(tmp_path, replay, judged).returncode == 0
    out = tmp_path / "decided"

    done = run_cli(
        tmp_path,
        "closure",
        "identity",
        "decide",
        "--evidence",
        str(replay / "evidence-5.json"),
        "--verdicts",
        str(judged / "verdicts.json"),
        "--out",
        str(out),
    )

    assert done.returncode == 0, done.stderr
    decisions = json.loads((out / "identity-decisions.json").read_text())
    assert decisions == json.loads((replay / "expected-decisions.json").read_text())
    accepted = json.loads((out / "accepted.json").read_text())
    # Two accepted: the candidate the checks confirmed, and the positive control (not a candidate:
    # only candidates are listed). The model's yes on the second candidate failed the program checks.
    assert [a["paperid"] for a in accepted["accepted"]] == [ACCEPTED]
    assert accepted["summary"]["candidate"] == {
        "n": 3,
        "model_same_work": 2,
        "accepted_by_all_checks": 1,
        "unparsed": 0,
    }
    assert accepted["summary"]["control_same"]["accepted_by_all_checks"] == 1
    assert accepted["summary"]["control_different_hard"]["accepted_by_all_checks"] == 0
    # The released byte format: indent 1, no final newline.
    assert (out / "accepted.json").read_bytes().endswith(b"]\n}")
    assert (out / "identity-decisions.json").read_bytes().endswith(b"}\n]")
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["summary"]["accepted_candidates"] == 1


def accepted_file(summary_name, paper_ids):
    return {
        "schema": "identity-accepted/1",
        "summary": {"candidate": {"n": summary_name}},
        "accepted": [
            {"paperid": p, "verdict": "same_work", "round": summary_name}
            for p in paper_ids
        ],
    }


def test_merge_joins_the_rounds_and_a_later_verdict_replaces_an_earlier_one(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    third = tmp_path / "third.json"
    first.write_text(json.dumps(accepted_file(1, ["a", "b"])))
    second.write_text(json.dumps(accepted_file(2, ["c", "b"])))
    third.write_text(json.dumps(accepted_file(3, ["d"])))

    two = run_cli(
        tmp_path,
        "closure",
        "identity",
        "merge",
        "--earlier",
        str(first),
        "--new",
        str(second),
        "--round",
        "2",
        "--out",
        str(tmp_path / "all.json"),
    )
    three = run_cli(
        tmp_path,
        "closure",
        "identity",
        "merge",
        "--earlier",
        str(tmp_path / "all.json"),
        "--new",
        str(third),
        "--round",
        "3",
        "--out",
        str(tmp_path / "all-r3.json"),
    )

    assert two.returncode == three.returncode == 0, (two.stderr, three.stderr)
    both = json.loads((tmp_path / "all.json").read_text())
    assert both["rounds"] == {
        "1": {"candidate": {"n": 1}},
        "2": {"candidate": {"n": 2}},
    }
    # b keeps its place (second) and takes round 2's verdict.
    assert [(x["paperid"], x["round"]) for x in both["accepted"]] == [
        ("a", 1),
        ("b", 2),
        ("c", 2),
    ]
    every = json.loads((tmp_path / "all-r3.json").read_text())
    assert list(every["rounds"]) == ["1", "2", "3"]
    assert [x["paperid"] for x in every["accepted"]] == ["a", "b", "c", "d"]
    assert (
        two.stdout.strip() == "accepted verdicts, all rounds: 3 (before: 2, round 2: 2)"
    )


def test_a_live_judge_sends_the_pinned_request_and_records_it_for_replay(
    tmp_path, replay
):
    expected = json.loads((replay / "expected-verdicts.json").read_text())
    answers = {v["id"]: v["raw"] for v in expected}
    evidence = json.loads((replay / "evidence-5.json").read_text())
    by_text = {item["text"]: item["id"] for item in evidence["items"]}

    def serve(request):  # the server: answers the item whose PDF text the prompt holds
        body = json.loads(request.body)
        prompt = body["messages"][1]["content"]
        item = next(i for t, i in by_text.items() if t in prompt)
        return Answer(body={"choices": [{"message": {"content": answers[item]}}]})

    with Replay() as server:
        server.answer("POST /v1/chat/completions", serve)
        done = run_cli(
            tmp_path,
            "closure",
            "identity",
            "judge",
            "--evidence",
            str(replay / "evidence-5.json"),
            "--out",
            str(tmp_path / "live"),
            "--base",
            f"{server.url}/v1",
            "--workers",
            "2",
            env=NO_NETWORK,
        )

    assert done.returncode == 0, done.stderr
    assert server.misses == []
    body = json.loads(server.asked("POST /v1/chat/completions")[0].body)
    # What the released run sent: temperature 0, 500 tokens, thinking off, the JSON schema, strict.
    assert body["model"] == "Qwen/Qwen3.6-35B-A3B"
    assert (body["temperature"], body["max_tokens"]) == (0, 500)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["response_format"]["json_schema"]["schema"]["required"] == [
        "verdict",
        "pdf_title",
        "pdf_authors",
        "pdf_year",
        "evidence",
        "reasoning",
    ]
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    live = json.loads((tmp_path / "live" / "verdicts.json").read_text())
    assert live["verdicts"] == expected
    # Its five request receipts replay the run with no server: the same request keys.
    receipts = sorted((tmp_path / "live" / "requests").iterdir())
    assert sorted(p.name for p in receipts) == sorted(
        p.name for p in (replay / "requests").iterdir()
    )
    again = judge(
        tmp_path,
        tmp_path / "live",
        tmp_path / "again",
        evidence=replay / "evidence-5.json",
    )
    assert again.returncode == 0, again.stderr
    assert json.loads((tmp_path / "again" / "verdicts.json").read_text()) == live


def test_a_server_that_refuses_the_json_schema_is_asked_again_without_it(
    tmp_path, replay
):
    """The first item is answered with the schema; the server then refuses it, and from there on every item is asked without it (a switch that all workers share, as in the released run). The verdicts file says which items were asked with the schema."""
    evidence = json.loads((replay / "evidence-5.json").read_text())
    evidence["items"] = evidence["items"][:2]
    two = tmp_path / "two.json"
    two.write_text(json.dumps(evidence))
    raw = json.loads((replay / "expected-verdicts.json").read_text())[0]["raw"]
    schema_requests = []

    def serve(request):
        if "response_format" in json.loads(request.body):
            schema_requests.append(request)
            if len(schema_requests) > 1:
                return Answer(400, "response_format is not supported")
        return Answer(body={"choices": [{"message": {"content": raw}}]})

    with Replay() as server:
        server.answer("POST /v1/chat/completions", serve)
        done = run_cli(
            tmp_path,
            "closure",
            "identity",
            "judge",
            *("--evidence", str(two), "--out", str(tmp_path / "plain")),
            *("--base", f"{server.url}/v1", "--workers", "1"),
            env=NO_NETWORK,
        )

    assert done.returncode == 0, done.stderr
    asked = [json.loads(r.body) for r in server.asked("POST /v1/chat/completions")]
    assert ["response_format" in b for b in asked] == [True, True, False]
    verdicts = json.loads((tmp_path / "plain" / "verdicts.json").read_text())
    first, second = (item["id"] for item in evidence["items"])
    assert verdicts["schema_mode_by_item"] == {first: True, second: False}
    assert verdicts["schema_mode"] is False  # the switch at the end of the run
    assert verdicts["verdicts"][1]["answer"]["verdict"] in (
        "same_work",
        "different_work",
    )


# The evidence step -------------------------------------------------------------------------------


def pdf(text: str) -> bytes:
    """A one-page PDF whose only content is `text` (pdftotext reads it; the xref is rebuilt)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    for number, body in enumerate(objects, 1):
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    return out + b"trailer << /Root 1 0 R /Size 6 >>\n%%EOF\n"


def collect_folder(tmp_path):
    """A collect folder: six PDFs; p1, p2 and p6 verified by DOI, p3 by its title in the text, p4 and p5 refused."""
    work, folder, _ = setup(tmp_path)
    papers = {
        f"p{n}": {
            "doi": f"10.1234/p{n}",
            "title": f"Title number {n}",
            "authors": f"Author{n}, A.",
            "year": "2020",
        }
        for n in range(1, 7)
    }
    downloads(
        folder, papers, {p: pdf(f"Title number {p[1]} by Author{p[1]}") for p in papers}
    )
    status = []
    for paperid, row in papers.items():
        xml = folder / "grobid" / paperid[1] / f"work_{paperid}.grobid.tei.xml"
        verified = paperid in ("p1", "p2", "p3", "p6")
        if verified:
            xml.parent.mkdir(parents=True, exist_ok=True)
            xml.write_text("<TEI/>")
            xml.with_suffix(".closure.json").write_text(
                json.dumps(
                    {"identity_match": "doi" if paperid != "p3" else "title_in_text"}
                )
            )
        status.append(
            {
                "paperid": paperid,
                "doi": row["doi"],
                "grobid_xml_path": str(xml) if verified else "",
                "converted": str(verified),
                "error": ""
                if verified
                else "grobid_identity_unverified: no matching DOI or title/year/author",
            }
        )
    write_rows(
        folder / "grobid_status.csv",
        status,
        [
            "paperid",
            "doi",
            "pdf_archive_path",
            "pdf_archive_member",
            "grobid_xml_path",
            "converted",
            "status_code",
            "error",
            "elapsed_seconds",
            "attempted_at",
        ],
    )
    return work, folder


@pytest.mark.skipif(
    shutil.which("pdftotext") is None, reason="needs poppler's pdftotext"
)
def test_evidence_takes_the_refused_pdfs_as_candidates_and_the_verified_ones_as_controls(
    tmp_path,
):
    work, folder = collect_folder(tmp_path)
    out = tmp_path / "round-1"

    done = run_cli(
        work,
        "closure",
        "identity",
        "evidence",
        "--collect-folder",
        str(folder),
        "--out",
        str(out),
        "--controls",
        "2",
    )

    assert done.returncode == 0, done.stderr
    evidence = json.loads((out / "evidence.json").read_text())
    kinds = {}
    for item in evidence["items"]:
        kinds.setdefault(item["kind"], []).append(item)
    assert sorted(i["paperid"] for i in kinds["candidate"]) == ["p4", "p5"]
    # Only DOI and title-year-author matches can be positive controls (p1, p2, p6), never p3 (title in text); two are drawn, and the third is the pool the negative controls take their PDFs from.
    drawn = sorted(i["paperid"] for i in kinds["control_same"])
    assert len(drawn) == 2 and set(drawn) <= {"p1", "p2", "p6"}
    assert len(kinds["control_different_hard"]) == 1
    assert len(kinds["control_different_random"]) == 1
    first = kinds["candidate"][0]
    assert first["ref"] == {
        "title": "Title number 4",
        "authors": "Author4, A.",
        "year": "2020",
        "doi": "10.1234/p4",
    }
    assert first["text"] == "Title number 4 by Author4"
    assert len(first["pdf_sha256"]) == 64
    # A hard negative pairs a verified reference with another verified paper's PDF; none is its own.
    (other,) = {"p1", "p2", "p6"} - set(drawn)
    for item in kinds["control_different_hard"] + kinds["control_different_random"]:
        assert item["paperid"] in drawn and item["pdf_paperid"] == other

    # A second round with the first as `--earlier` repeats no candidate and no control.
    again = run_cli(
        work,
        "closure",
        "identity",
        "evidence",
        "--collect-folder",
        str(folder),
        "--out",
        str(tmp_path / "round-2"),
        "--controls",
        "2",
        "--earlier",
        str(out / "evidence.json"),
    )
    assert again.returncode == 0, again.stderr
    later = json.loads((tmp_path / "round-2" / "evidence.json").read_text())
    assert later["items"] == [] and later["round"] == 2
    assert (tmp_path / "round-2" / "evidence-full.json").is_file()
