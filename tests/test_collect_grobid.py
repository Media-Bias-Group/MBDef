"""`mbdef collect grobid`: convert downloaded PDFs to TEI, keeping only the paper asked for.

GROBID is a fake on a local port that answers each uploaded PDF with a fixture TEI; nothing else is reachable.
"""

import json

import identity_cases
import pytest
from cli_helpers import run_cli
from http_replay import Answer, Replay
from mini_collect import (
    downloads,
    form_parts,
    read_receipts,
    read_rows,
    setup,
    sha,
    tei,
)

ALIVE = Answer(body="true")


@pytest.fixture
def replay():
    with Replay() as server:
        yield server


def grobid(work, replay, env, *extra):
    return run_cli(
        work, "collect", "grobid", "--grobid-url", replay.url, *extra, env=env
    )


def counts(done) -> dict:
    return json.loads(done.stdout.splitlines()[0])


def test_a_grobid_that_is_not_alive_stops_the_run_before_any_conversion(
    tmp_path, replay
):
    work, folder, env = setup(tmp_path)
    downloads(folder, {"pending": {"doi": "10.1234/p"}}, {"pending": b"%PDF-1.4 a"})
    replay.answer("GET /api/isalive", Answer(503, "down"))

    done = grobid(work, replay, env)

    assert done.returncode == 1
    assert done.stderr.strip().startswith(f"GROBID is not alive at {replay.url}.")
    assert [r.path for r in replay.requests] == ["/api/isalive"]
    assert not (folder / "grobid_status.csv").exists()


def test_a_pdf_is_posted_as_a_form_and_its_tei_and_receipt_are_kept(tmp_path, replay):
    doi = "10.1002/(sici)1099-0771(200004/06)13:2<145::aid-bdm320>3.0.co;2-4"
    paperid = "doi_10.1002_sici_1099__digest"
    work, folder, env = setup(tmp_path)
    downloads(folder, {paperid: {"doi": doi}}, {paperid: b"%PDF-1.4 fixture"})
    converted = tei(title="Example title", surnames=("Smith",), year="2020", doi=doi)
    replay.answer("GET /api/isalive", ALIVE)
    replay.answer("POST /api/processFulltextDocument", Answer(body=converted))
    replay.answer("GET /api/version", Answer(body="0.9.1"))

    first = grobid(work, replay, env, "--consolidate-header")
    second = grobid(work, replay, env, "--consolidate-header")

    assert first.returncode == second.returncode == 0, first.stderr
    assert counts(first) == {
        "downloaded": 1,
        "converted": 1,
        "failed": 0,
        "skipped": 0,
        "deferred": 0,
    }
    assert counts(second) | {"skipped": 0} == counts(first) | {"converted": 0}
    assert replay.misses == []
    assert [r.path for r in replay.requests] == [
        "/api/isalive",
        "/api/processFulltextDocument",
        "/api/version",
    ]
    assert form_parts(replay.requests[1]) == {
        "input": (f"{paperid}.pdf", "application/pdf", b"%PDF-1.4 fixture"),
        "consolidateHeader": (None, "text/plain", b"1"),
        "consolidateCitations": (None, "text/plain", b"0"),
    }
    # The TEI is named by the sha256 of its paper id and DOI, never by the DOI itself.
    name = "work_9bf04aedc66501bc97a6215e666a01bcf05e54e810ec955e37393fcdb0d7c904"
    written = folder / "grobid/04" / f"{name}.grobid.tei.xml"
    assert written.read_bytes() == converted
    receipt = json.loads(
        (written.parent / f"{name}.grobid.tei.closure.json").read_text()
    )
    assert receipt == {
        "schema": "corpus-closure/1",
        "depth": 1,
        "paperid": paperid,
        "requested_doi": doi,
        "observed_doi": doi,
        "title": "",
        "authors": "",
        "year": "",
        "identity_match": "doi",
        "observed_title": "Example title",
        "observed_year": "2020",
        "pdf_sha256": "c5c2e8be6ad0825a56ded1f1a153ceaf11dc060ab44b120a94406a08207babc2",
        "tei_sha256": sha(written),
        "pdf_archive_path": str(folder / "pdf_batches/pdf_batch_000001.zip"),
        "pdf_archive_member": f"do/{paperid}.pdf",
        "grobid_url": replay.url,
    }
    assert len(read_rows(folder / "grobid_status.csv")) == 1
    runs = read_receipts(folder)
    assert [r["settings"]["grobid_version"] for r in runs] == ["0.9.1", None]
    assert runs[0]["settings"]["consolidate_header"] is True
    assert runs[0]["outputs"]["tei"]["files"] == 2
    assert runs[0]["inputs"]["downloads"]["sha256"] == sha(
        folder / "pdf_download_status.csv"
    )


def test_an_invalid_tei_is_retried_and_a_success_clears_the_failure_list(
    tmp_path, replay
):
    work, folder, env = setup(tmp_path)
    paper = {"title": "Example title", "authors": "Smith, A.", "year": "2020"}
    downloads(
        folder, {"hash_1234567890abcdef": paper}, {"hash_1234567890abcdef": b"%PDF-1"}
    )
    replay.answer("GET /api/isalive", ALIVE, ALIVE)
    replay.answer(
        "POST /api/processFulltextDocument",
        Answer(body=b"<not-tei/>"),
        Answer(body=tei(title="Example title", surnames=("Smith",), year="2020")),
    )
    replay.answer("GET /api/version", Answer(body="0.9.1"), Answer(body="0.9.1"))

    first = grobid(work, replay, env)
    failed_after_first = read_rows(folder / "failed_grobid.csv")
    second = grobid(work, replay, env)

    assert first.returncode == 1
    assert first.stderr.strip() == (
        f"1 PDFs were not converted; {folder / 'failed_grobid.csv'} says why. "
        "Rerun to retry them."
    )
    assert [r["error"] for r in failed_after_first] == ["invalid_tei_xml"]
    assert second.returncode == 0, second.stderr
    assert counts(second)["converted"] == 1
    assert read_rows(folder / "failed_grobid.csv") == []
    assert replay.misses == []


def test_a_missing_archive_is_a_failed_conversion_without_a_request(tmp_path, replay):
    work, folder, env = setup(tmp_path)
    downloads(
        folder, {"missing-pdf": {"doi": "10.1234/missing"}}, {"missing-pdf": b"%PDF"}
    )
    (folder / "pdf_batches/pdf_batch_000001.zip").unlink()
    replay.answer("GET /api/isalive", ALIVE)
    replay.answer("GET /api/version", Answer(body="0.9.1"))

    done = grobid(work, replay, env)

    assert done.returncode == 1
    assert counts(done) == {
        "downloaded": 1,
        "converted": 0,
        "failed": 1,
        "skipped": 0,
        "deferred": 0,
    }
    assert "/api/processFulltextDocument" not in [r.path for r in replay.requests]
    [failure] = read_rows(folder / "failed_grobid.csv")
    assert failure["paperid"] == "missing-pdf"
    assert "pdf_batch_000001.zip" in failure["error"]


def test_each_identity_rule_keeps_or_refuses_the_converted_pdf(tmp_path, replay):
    """Each case of tests/identity_cases.py: the rule that matched, or why the PDF was refused."""
    work, folder, env = setup(tmp_path)
    verdicts = identity_cases.prepare(folder, replay)

    done = grobid(work, replay, env, "--identity-verdicts", str(verdicts))

    assert replay.misses == []
    assert done.returncode == 1  # some PDFs are refused
    found = {}
    for row in read_rows(folder / "grobid_status.csv"):
        if row["converted"] == "True":
            receipt = json.loads(
                (folder / row["grobid_xml_path"])
                .with_suffix(".closure.json")
                .read_text()
            )
            found[row["paperid"]] = receipt["identity_match"]
        else:
            found[row["paperid"]] = row["error"]
    assert found == {p: case[3] for p, case in identity_cases.CASES.items()}
    kept = json.loads(
        next((folder / "grobid").rglob("*.closure.json")).read_text()
    )  # any receipt; the verdict's is checked below
    assert kept["schema"] == "corpus-closure/1"
    [verified] = [
        json.loads(p.read_text())
        for p in (folder / "grobid").rglob("*.closure.json")
        if json.loads(p.read_text())["identity_match"] == "llm_verified"
    ]
    assert verified["identity_llm"] == {
        "model": "m",
        "revision": "r",
        "prompt_version": "p",
        "verdict": "same_work",
        "pdf_title": None,
        "evidence": None,
        "reasoning": None,
        "checks": None,
    }
    [receipt] = read_receipts(folder)
    assert receipt["inputs"]["identity_verdicts"]["sha256"] == sha(verdicts)


def test_the_pinned_defaults_are_the_released_ones(tmp_path):
    done = run_cli(tmp_path, "collect", "grobid", "--help")

    assert done.returncode == 0
    text = " ".join(done.stdout.split())
    for default in (
        "the running GROBID service (default: http://localhost:8070)",
        "parallel GROBID requests (default: 10)",
        "time per PDF (default: 180.0)",
        "online sources (default: off)",
    ):
        assert default in text
