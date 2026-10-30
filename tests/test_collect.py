"""The end-to-end test of the collect step: a three-paper queue from lookup to TEI, with no network.

Unpaywall, OpenAlex, the publishers and GROBID are one replay server on a local port that answers only what is recorded below (tests/http_replay.py); every other address goes to a closed proxy port.
The test doubles as a worked example of the step (docs/step-1-2-papers/collect.md).
"""

import hashlib
import json
import zipfile

from cli_helpers import run_cli
from http_replay import Answer, Replay
from mini_collect import form_parts, head, read_receipts, read_rows, setup, sha, tei

from mbdef.collect.identity import read_receipt

QUEUE = [
    {  # found by Unpaywall through its DOI
        "queue_status": "missing",
        "title": "Measuring Media Bias",
        "authors": "Lovelace, A.; Hopper, G.",
        "year": "2020",
        "doi": "https://doi.org/10.1234/MMB",
    },
    {  # no DOI: found by OpenAlex through its title, year and an author
        "queue_status": "missing",
        "title": "Framing in the News",
        "authors": "Noether, E.",
        "year": "2019",
    },
    {
        "queue_status": "review",
        "title": "Two candidate works",
        "review_reason": "ambiguous",
    },
]


def test_the_step_takes_a_queue_to_identity_checked_tei_files(tmp_path):
    work, folder, env = setup(tmp_path, QUEUE)
    with Replay() as replay:
        api = replay.url
        replay.answer(
            "GET /v2/10.1234%2Fmmb",
            Answer(
                body={
                    "is_oa": True,
                    "best_oa_location": {
                        "url_for_pdf": f"{api}/pdf/mmb.pdf",
                        "host_type": "repository",
                    },
                }
            ),
        )
        replay.answer(  # OpenAlex knows the first paper but has no PDF of it
            "GET /works/doi:10.1234%2Fmmb",
            Answer(
                body={
                    "id": "https://openalex.org/W1",
                    "doi": "https://doi.org/10.1234/mmb",
                    "title": "Measuring Media Bias",
                    "publication_year": 2020,
                    "authorships": [],
                    "open_access": {"is_oa": False},
                    "best_oa_location": None,
                    "locations": [],
                }
            ),
        )
        replay.answer(
            "GET /works",
            Answer(
                body={
                    "meta": {"count": 1},
                    "results": [
                        {
                            "id": "https://openalex.org/W2",
                            "doi": "https://doi.org/10.5678/fitn",
                            "title": "Framing in the news",
                            "publication_year": 2019,
                            "authorships": [
                                {"author": {"display_name": "Emmy Noether"}}
                            ],
                            "open_access": {"is_oa": True},
                            "best_oa_location": None,
                            "locations": [
                                {
                                    "pdf_url": f"{api}/pdf/fitn.pdf",
                                    "is_oa": True,
                                    "version": "publishedVersion",
                                    "source": {"type": "repository"},
                                }
                            ],
                        }
                    ],
                }
            ),
        )
        mmb_pdf, fitn_pdf = b"%PDF-1.7 measuring media bias", b"%PDF-1.7 framing"
        replay.answer("GET /pdf/mmb.pdf", Answer(body=mmb_pdf))
        replay.answer("GET /pdf/fitn.pdf", Answer(body=fitn_pdf))
        replay.answer("GET /api/isalive", Answer(body="true"))
        replay.answer("GET /api/version", Answer(body="0.9.1"))
        teis = {
            mmb_pdf: tei(
                title="Measuring Media Bias",
                surnames=("Lovelace", "Hopper"),
                year="2020",
                doi="10.1234/mmb",
            ),
            fitn_pdf: tei(
                title="Framing in the News", surnames=("Noether",), year="2019"
            ),
        }

        def grobid_answer(request):  # converts whichever PDF it is sent
            _, _, pdf = form_parts(request)["input"]
            return Answer(body=teis[pdf])

        replay.answer("POST /api/processFulltextDocument", grobid_answer)

        # OpenAlex first; Unpaywall then gets what OpenAlex left without a PDF (the released closure's order).
        steps = [
            ["openalex", "--api-url", api, "--sleep-seconds", "0"],
            ["open-access", "--api-url", f"{api}/v2", "--sleep-seconds", "0"],
            ["download"],
            ["download", "--candidates", str(folder / "openalex_candidates.csv")],
            ["grobid", "--grobid-url", api],
        ]
        done = [run_cli(work, "collect", *step, env=env) for step in steps]

    assert [d.returncode for d in done] == [0] * 5, [d.stderr for d in done]
    assert replay.misses == []
    summaries = [json.loads(d.stdout.splitlines()[0]) for d in done]
    assert (summaries[0]["resolved"], summaries[0]["unresolved"]) == (1, 1)
    assert summaries[1] == {
        "eligible": 1,  # the paper OpenAlex has no PDF of
        "resolved": 1,
        "unresolved": 0,
        "errors": 0,
        "skipped_review": 1,
    }
    assert [s["downloaded"] for s in summaries[2:4]] == [1, 1]
    assert summaries[4] == {
        "downloaded": 2,
        "converted": 2,
        "failed": 0,
        "skipped": 0,
        "deferred": 0,
    }

    # The two PDFs, one per resolver, in one zip batch.
    [batch] = (folder / "pdf_batches").iterdir()
    with zipfile.ZipFile(batch) as archive:
        assert sorted(archive.read(n) for n in archive.namelist()) == sorted(
            [mmb_pdf, fitn_pdf]
        )

    # Two TEI files, each with a receipt the closure step accepts, bound to its PDF's bytes.
    converted = read_rows(folder / "grobid_status.csv")
    by_title = {}
    for row in converted:
        path = folder / row["grobid_xml_path"]
        receipt = read_receipt(path)
        assert receipt is not None
        by_title[receipt["observed_title"]] = receipt
    assert {t: r["identity_match"] for t, r in by_title.items()} == {
        "Measuring Media Bias": "doi",
        "Framing in the News": "title_year_author",
    }
    assert by_title["Measuring Media Bias"]["requested_doi"] == "10.1234/mmb"
    assert (
        by_title["Framing in the News"]["requested_doi"] == "10.5678/fitn"
    )  # found by OpenAlex
    assert by_title["Measuring Media Bias"]["pdf_sha256"] == (
        hashlib.sha256(mmb_pdf).hexdigest()
    )
    assert read_rows(folder / "failed_grobid.csv") == []

    # One receipt per run: the command, the pinned defaults, the input hashes and the commit.
    receipts = read_receipts(folder)
    assert [r["command"] for r in receipts] == [
        "collect openalex",
        "collect open-access",
        "collect download",
        "collect download",
        "collect grobid",
    ]
    assert {r["code"]["commit"] for r in receipts} == {head()}
    assert receipts[1]["settings"]["timeout_seconds"] == 20.0
    assert receipts[0]["settings"] | {"api_url": None} == {
        "api_url": None,
        "api_key": "not set",
        "sleep_seconds": 0.0,
        "timeout_seconds": 20.0,
        "max_retries": 3,
        "limit": None,
        "title_rule": "2026-09-29",
    }
    assert receipts[2]["settings"] == {
        "concurrency": 4,
        "timeout_seconds": 45.0,
        "max_pdf_mb": 80.0,
        "pdf_batch_size": 5000,
        "impersonate": "chrome",
        "limit": None,
        "europe_pmc_url": "https://europepmc.org",
        "paced_hosts": {"europepmc.org": 3.0, "www.ebi.ac.uk": 3.0},
    }
    assert receipts[4]["settings"] == {
        "grobid_url": api,
        "grobid_version": "0.9.1",
        "concurrency": 10,
        "timeout_seconds": 180.0,
        "consolidate_header": False,
        "limit": None,
    }
    queue_sha = sha(folder / "citation-papers-to-collect.csv")
    assert [r["inputs"]["collection"]["sha256"] for r in receipts[:2]] == [
        queue_sha
    ] * 2
    assert receipts[4]["inputs"]["downloads"]["sha256"] == sha(
        folder / "pdf_download_status.csv"
    )
    assert (
        receipts[4]["outputs"]["tei"]["files"] == 4
    )  # two TEI files and their receipts
