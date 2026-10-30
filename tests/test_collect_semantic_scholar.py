"""`mbdef collect semantic-scholar`: a title match for the papers Crossref found no DOI for.

Crossref and Semantic Scholar are replayed from recorded answers on a local port; nothing else is reachable.
"""

import json

import pytest
from cli_helpers import run_cli
from http_replay import Answer, Replay
from mini_collect import downloads, read_receipts, read_rows, setup

FIELDS = "title,year,authors,externalIds,openAccessPdf"
ABUSE = {
    "queue_status": "missing",
    "title": "Detecting and tracking political abuse in social media",
    "authors": "Conover, M. D.; Ratkiewicz, J.",
    "year": "2011",
}


@pytest.fixture
def replay():
    with Replay() as server:
        yield server


def run(work, replay, env, action, *extra):
    pause = ["--check-sleep-seconds", "0"] if action == "semantic-scholar" else []
    crossref = ["--crossref-url", replay.url] if action == "semantic-scholar" else []
    return run_cli(
        work,
        "collect",
        action,
        "--api-url",
        replay.url,
        *crossref,
        "--sleep-seconds",
        "0",
        *pause,
        "--backoff-seconds",
        "0",
        *extra,
        env=env,
    )


def counts(done) -> dict:
    return json.loads(done.stdout.splitlines()[0])


def crossref_record(title, doi, year, family):
    return {
        "DOI": doi,
        "title": [title],
        "author": [{"family": family}],
        "issued": {"date-parts": [[year]]},
        "type": "journal-article",
    }


def paper(title, *, year=2011, authors=("Michael Conover",), ids=None, pdf=None):
    return {
        "paperId": "s2-" + title[:6].lower().replace(" ", "-"),
        "title": title,
        "year": year,
        "authors": [{"authorId": "1", "name": name} for name in authors],
        "externalIds": ids or {},
        "openAccessPdf": {"url": pdf, "status": "GREEN"} if pdf else None,
        "matchScore": 194.25104,
    }


def answers_by(key, table, default):
    """A replayed service that answers by one query parameter (the title or the bibliographic query)."""

    def answer(request):
        return table.get(request.query[key], default)

    return answer


def test_a_title_match_needs_title_year_and_author_and_its_doi_must_pass_crossref_too(
    tmp_path, replay
):
    arxiv = ABUSE | {"title": "A survey of preprints on media bias", "year": "2020"}
    review = ABUSE | {"title": "Partisan news and the attention of readers"}
    no_crossref = ABUSE | {"title": "Bots in the run-up to an election"}
    different = ABUSE | {"title": "Measuring the slant of newspaper coverage"}
    unknown = ABUSE | {"title": "An unpublished manuscript on framing"}
    found_by_crossref = ABUSE | {"title": "Framing theory and its uses in research"}
    work, folder, env = setup(
        tmp_path,
        [ABUSE, arxiv, review, no_crossref, different, unknown, found_by_crossref],
        SEMANTIC_SCHOLAR_API_KEY="s2-secret-key",
    )
    framing = crossref_record(
        found_by_crossref["title"], "10.1234/framing", 2011, "Conover"
    )
    empty = Answer(body={"message": {"items": []}})
    replay.answer(
        "GET /works",
        answers_by(
            "query.bibliographic",
            {
                f"{found_by_crossref['title']} Conover Ratkiewicz": Answer(
                    body={"message": {"items": [framing]}}
                )
            },
            empty,
        ),
    )
    assert counts(run(work, replay, env, "crossref"))["no_match"] == 6
    crossref_requests = len(replay.requests)
    s2 = {
        ABUSE["title"]: paper(
            "Detecting and Tracking Political Abuse in Social Media",
            ids={"DOI": "10.1609/ICWSM.V5I1.14127", "ArXiv": None},
            pdf="https://ojs.example/abuse.pdf",
        ),
        arxiv["title"]: paper(
            "A survey of preprints on media bias",
            year=2020,
            ids={"DOI": "10.48550/arXiv.2001.01234", "ArXiv": "2001.01234"},
        ),
        review["title"]: paper(
            review["title"],
            ids={"DOI": "10.5555/review"},
            pdf="https://r.example/r.pdf",
        ),
        no_crossref["title"]: paper(no_crossref["title"], ids={"DOI": "10.5555/nocr"}),
        different["title"]: paper("Measuring slant in newspapers"),
    }
    replay.answer(
        "GET /paper/search/match",
        lambda request: (
            Answer(body={"data": [s2[request.query["query"]]]})
            if request.query["query"] in s2
            else Answer(404, {"error": "Title match not found"})
        ),
    )
    replay.answer(
        "GET /works/10.1609/icwsm.v5i1.14127",
        Answer(
            body={
                "message": crossref_record(
                    "Detecting and tracking political abuse in social media",
                    "10.1609/icwsm.v5i1.14127",
                    2011,
                    "Ratkiewicz",
                )
            }
        ),
    )
    replay.answer(
        "GET /works/10.5555/review",
        Answer(
            body={
                "message": crossref_record(
                    review["title"], "10.5555/review", 2013, "Conover"
                )
            }
        ),
    )
    replay.answer("GET /works/10.5555/nocr", Answer(404, "Resource not found."))

    first = run(work, replay, env, "semantic-scholar")
    second = run(work, replay, env, "semantic-scholar")

    assert first.returncode == second.returncode == 0, first.stderr
    assert (
        counts(first)
        == counts(second)
        == {
            "eligible": 6,
            "matched": 4,
            "with_doi": 2,
            "with_pdf": 3,
            "no_match": 2,
            "insufficient_evidence": 0,
            "errors": 0,
            "unattempted": 0,
            "skipped_review": 0,
        }
    )
    assert replay.misses == []
    asked = replay.requests[crossref_requests:]  # the second run asks nothing
    matches = [r for r in asked if r.path == "/paper/search/match"]
    assert len(matches) == 6
    assert matches[0].query == {"query": ABUSE["title"], "fields": FIELDS}
    assert matches[0].headers["x-api-key"] == "s2-secret-key"
    checks = [r.path for r in asked if r.path.startswith("/works/")]
    assert sorted(checks) == [
        "/works/10.1609/icwsm.v5i1.14127",
        "/works/10.5555/nocr",
        "/works/10.5555/review",
    ]
    rows = {r["title"]: r for r in read_rows(folder / "semantic_scholar_status.csv")}
    assert len(rows) == 6
    abuse = rows[ABUSE["title"]]
    assert abuse["paperid"].startswith("hash_") and abuse["attempted_at"]
    del abuse["paperid"], abuse["attempted_at"]
    assert abuse == {
        "title": ABUSE["title"],
        "authors": ABUSE["authors"],
        "year": "2011",
        "s2_status": "matched",
        "reason": "exact title+year+author (repo rule)",
        "s2_paper_id": "s2-detect",
        "doi": "10.1609/icwsm.v5i1.14127",
        "doi_check": "crossref_record_passes",
        "arxiv": "",
        "pdf_url": "https://ojs.example/abuse.pdf",
        "matched_title": "Detecting and Tracking Political Abuse in Social Media",
        "matched_year": "2011",
        "match_score": "194.25104",
        "status_code": "200",
        "error": "",
    }
    summary = {
        title: (r["s2_status"], r["reason"], r["doi"], r["doi_check"], r["arxiv"])
        for title, r in rows.items()
        if title != ABUSE["title"]
    }
    assert summary == {
        arxiv["title"]: (
            "matched",
            "exact title+year+author (repo rule)",
            "10.48550/arxiv.2001.01234",
            "arxiv_doi",
            "2001.01234",
        ),
        review["title"]: (
            "matched",
            "exact title+year+author (repo rule)",
            "",
            "dropped 10.5555/review: Crossref record fails the rule "
            "(title but no matching year)",
            "",
        ),
        no_crossref["title"]: (
            "matched",
            "exact title+year+author (repo rule)",
            "",
            "dropped 10.5555/nocr: not a Crossref DOI",
            "",
        ),
        different["title"]: (
            "no_match",
            "best title match is a different title",
            "",
            "",
            "",
        ),
        unknown["title"]: ("no_match", "no Semantic Scholar title match", "", "", ""),
    }
    candidates = {
        r["title"]: (r["pdf_url"], json.loads(r["alternate_pdf_urls"]), r["doi"])
        for r in read_rows(folder / "semantic_scholar_candidates.csv")
    }
    assert candidates == {
        ABUSE["title"]: (
            "https://ojs.example/abuse.pdf",
            [],
            "10.1609/icwsm.v5i1.14127",
        ),
        arxiv["title"]: (
            "https://arxiv.org/pdf/2001.01234",
            [],
            "10.48550/arxiv.2001.01234",
        ),
        review["title"]: ("https://r.example/r.pdf", [], ""),
    }
    receipts = read_receipts(folder)
    assert receipts[-1]["command"] == "collect semantic-scholar"
    assert receipts[-1]["settings"]["api_key"] == "set"
    assert "s2-secret-key" not in json.dumps(receipts)


def openalex_work(doi, *, pdf=None, is_oa=False):
    locations = (
        [
            {
                "pdf_url": pdf,
                "is_oa": True,
                "version": "publishedVersion",
                "source": {"type": "repository"},
            }
        ]
        if pdf
        else []
    )
    return {
        "id": "https://openalex.org/W" + doi[-4:],
        "doi": "https://doi.org/" + doi,
        "title": "A paper",
        "publication_year": 2020,
        "authorships": [],
        "open_access": {"is_oa": is_oa or bool(pdf)},
        "best_oa_location": None,
        "locations": locations,
    }


def test_dois_whose_pdf_is_missing_get_semantic_scholars_open_arxiv_and_europe_pmc_copies(
    tmp_path, replay
):
    def row(doi):
        return {
            "queue_status": "missing",
            "title": f"Paper {doi}",
            "doi": doi,
            "year": "2020",
        }

    dois = [
        "10.1234/closed",
        "10.1234/haspdf",
        "10.1234/downloaded",
        "10.1234/unknown",
        "10.1234/null",
    ]
    work, folder, env = setup(tmp_path, [row(doi) for doi in dois])
    openalex = {
        "10.1234/closed": Answer(body=openalex_work("10.1234/closed")),
        "10.1234/haspdf": Answer(
            body=openalex_work("10.1234/haspdf", pdf="https://pub.example/h.pdf")
        ),
        "10.1234/downloaded": Answer(
            body=openalex_work("10.1234/downloaded", pdf="https://pub.example/d.pdf")
        ),
        "10.1234/unknown": Answer(404, {}),
        "10.1234/null": Answer(body=openalex_work("10.1234/null", is_oa=True)),
    }
    for doi, answer in openalex.items():
        replay.answer(f"GET /works/doi:{doi.replace('/', '%2F')}", answer)
    replay.answer("GET /works", Answer(body={"results": [], "meta": {"count": 0}}))
    done = run_cli(
        work,
        "collect",
        "openalex",
        "--api-url",
        replay.url,
        "--sleep-seconds",
        "0",
        env=env,
    )
    assert done.returncode == 0, done.stderr
    ids = {r["doi"]: r["paperid"] for r in read_rows(folder / "openalex_status.csv")}
    downloads(
        folder,
        {ids["10.1234/downloaded"]: {"doi": "10.1234/downloaded"}},
        {ids["10.1234/downloaded"]: b"%PDF-1.4"},
    )
    batch = {
        "DOI:10.1234/closed": {
            "paperId": "s2-closed",
            "isOpenAccess": True,
            "openAccessPdf": {
                "url": "https://repo.example/closed.pdf",
                "status": "GREEN",
                "license": "CCBY",
            },
            "externalIds": {
                "ArXiv": "1905.00001",
                "PubMedCentral": "6543210",
                "DOI": "10.1234/closed",
            },
        },
        "DOI:10.1234/haspdf": {
            "paperId": "s2-haspdf",
            "isOpenAccess": False,
            "openAccessPdf": {
                "url": "https://sci-hub.se/haspdf.pdf",
                "status": None,
                "license": None,
            },
            "externalIds": {"PubMedCentral": "PMC123456"},
        },
        "DOI:10.1234/null": None,
    }
    replay.answer(
        "POST /paper/batch",
        lambda request: Answer(
            body=[batch[i] for i in json.loads(request.body)["ids"]]
        ),
    )
    asked_before = len(replay.requests)

    first = run(work, replay, env, "semantic-scholar-doi")
    second = run(work, replay, env, "semantic-scholar-doi")

    assert first.returncode == second.returncode == 0, first.stderr
    assert counts(first) == {
        "dois_asked": 3,
        "rows_recorded": 3,
        "rows_failed": 0,
        "with_copies": 2,
    }
    assert counts(second) == {
        "dois_asked": 0,
        "rows_recorded": 0,
        "rows_failed": 0,
        "with_copies": 2,
    }
    [asked] = replay.requests[asked_before:]
    assert asked.query == {"fields": "openAccessPdf,externalIds,isOpenAccess"}
    assert json.loads(asked.body) == {
        "ids": ["DOI:10.1234/closed", "DOI:10.1234/haspdf", "DOI:10.1234/null"]
    }
    rows = {r["doi"]: r for r in read_rows(folder / "semantic_scholar_doi_status.csv")}
    assert set(rows) == {"10.1234/closed", "10.1234/haspdf", "10.1234/null"}
    closed = rows["10.1234/closed"]
    assert closed["attempted_at"]
    del closed["attempted_at"]
    assert closed == {
        "paperid": ids["10.1234/closed"],
        "doi": "10.1234/closed",
        "s2_paper_id": "s2-closed",
        "is_oa": "True",
        "oa_status": "GREEN",
        "license": "CCBY",
        "pdf_url": "https://repo.example/closed.pdf",
        "arxiv": "1905.00001",
        "pmc": "6543210",
        "status_code": "200",
    }
    assert (rows["10.1234/haspdf"]["is_oa"], rows["10.1234/haspdf"]["oa_status"]) == (
        "",
        "",
    )
    assert rows["10.1234/null"]["status_code"] == "404"
    candidates = {
        r["doi"]: [r["pdf_url"], *json.loads(r["alternate_pdf_urls"])]
        for r in read_rows(folder / "semantic_scholar_doi_candidates.csv")
    }
    assert candidates == {
        "10.1234/closed": [
            "https://repo.example/closed.pdf",
            "https://arxiv.org/pdf/1905.00001",
            "https://europepmc.org/articles/PMC6543210?pdf=render",
        ],
        # A host that copies papers without the publisher's permission is never used.
        "10.1234/haspdf": ["https://europepmc.org/articles/PMC123456?pdf=render"],
    }
    assert read_receipts(folder)[-1]["command"] == "collect semantic-scholar-doi"


def test_the_pinned_defaults_are_the_v1_1_ones(tmp_path):
    match = " ".join(
        run_cli(tmp_path, "collect", "semantic-scholar", "--help").stdout.split()
    )
    batch = " ".join(
        run_cli(tmp_path, "collect", "semantic-scholar-doi", "--help").stdout.split()
    )

    api = "the Semantic Scholar Graph API (default: https://api.semanticscholar.org/graph/v1)"
    assert api in match and api in batch
    assert "doubled after an HTTP 429 (default: 1.111)" in match
    assert "pause before each Crossref check (default: 2.5)" in match
    assert "HTTP 429 or 5xx (default: 6)" in match
    assert "this much longer each time (default: 5.0)" in match
    assert "pause after each batch (default: 1.2)" in batch
    assert "request timeout (default: 120.0)" in batch
