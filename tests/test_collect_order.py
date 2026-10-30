"""The order in which the released closure asked its services, across the collect commands.

- A row without a DOI whose Semantic Scholar title match has a PDF is not searched in OpenAlex while that PDF may still come: only once its download failed for good (a 404, say; not a 429 or a timeout, which a later run retries).
- Unpaywall is asked about what OpenAlex left without a PDF: a queue DOI OpenAlex lacks, and the DOI of an OpenAlex title match without a PDF. A row OpenAlex found a PDF for is not asked.

Every service is replayed from recorded answers on a local port; nothing else is reachable.
"""

import json

import pytest
from cli_helpers import run_cli
from http_replay import Answer, Replay
from mini_collect import read_rows, setup

ROW = {"queue_status": "missing", "authors": "Lovelace, A.", "year": "2020"}
WITH_S2_PDF = ROW | {"title": "A paper Semantic Scholar has a copy of"}


@pytest.fixture
def replay():
    with Replay() as server:
        yield server


def run(work, replay, env, action, *extra):
    return run_cli(work, "collect", action, *extra, env=env)


def counts(done) -> dict:
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.splitlines()[0])


def openalex_work(title, doi="", pdf=None, year=2020):
    return {
        "id": "https://openalex.org/W1",
        "doi": f"https://doi.org/{doi}" if doi else "",
        "title": title,
        "publication_year": year,
        "authorships": [{"author": {"display_name": "Ada Lovelace"}}],
        "open_access": {"is_oa": bool(pdf)},
        "best_oa_location": None,
        "locations": [
            {
                "pdf_url": pdf,
                "is_oa": True,
                "version": "publishedVersion",
                "source": {"type": "repository"},
            }
        ]
        if pdf
        else [],
    }


@pytest.mark.parametrize(("failure", "searched"), [(404, True), (429, False)])
def test_a_semantic_scholar_pdf_is_tried_before_the_openalex_title_search(
    tmp_path, replay, failure, searched
):
    work, folder, env = setup(tmp_path, [WITH_S2_PDF])
    pdf = f"{replay.url}/s2/copy.pdf"

    def works(
        request,
    ):  # Crossref's search, and OpenAlex's title search on the same path
        if "query.bibliographic" in request.query:
            return Answer(body={"message": {"items": []}})
        return Answer(body={"results": [], "meta": {"count": 0}})

    replay.answer("GET /works", works)
    replay.answer(
        "GET /paper/search/match",
        Answer(
            body={
                "data": [
                    {
                        "paperId": "s2",
                        "title": WITH_S2_PDF["title"],
                        "year": 2020,
                        "authors": [{"name": "Ada Lovelace"}],
                        "externalIds": {},
                        "openAccessPdf": {"url": pdf},
                    }
                ]
            }
        ),
    )
    replay.answer("GET /s2/copy.pdf", Answer(failure, "no"))
    url = ("--api-url", replay.url, "--sleep-seconds", "0")
    counts(run(work, replay, env, "crossref", *url, "--backoff-seconds", "0"))
    s2 = run(
        work, replay, env, "semantic-scholar", *url,
        "--crossref-url", replay.url, "--check-sleep-seconds", "0", "--backoff-seconds", "0",
    )  # fmt: skip
    assert counts(s2)["with_pdf"] == 1

    waiting = counts(run(work, replay, env, "openalex", *url))
    searched_before = [r for r in replay.asked("GET /works") if "filter" in r.query]
    download = run(
        work, replay, env, "download", "--candidates",
        str(folder / "semantic_scholar_candidates.csv"),
    )  # fmt: skip
    after = counts(run(work, replay, env, "openalex", *url))

    assert (waiting["waiting_for_a_download"], searched_before) == (1, [])
    assert counts(download)["failed"] == 1
    searches = [r for r in replay.asked("GET /works") if "filter" in r.query]
    assert len(searches) == (1 if searched else 0)
    assert after["waiting_for_a_download"] == (0 if searched else 1)


NOT_IN_OPENALEX = ROW | {
    "title": "A paper whose DOI OpenAlex lacks",
    "doi": "10.1234/absent",
}
IN_OPENALEX = ROW | {"title": "A paper OpenAlex has a PDF of", "doi": "10.1234/present"}
BY_TITLE = ROW | {"title": "A paper OpenAlex finds by its title"}


def test_unpaywall_is_asked_about_what_openalex_left_without_a_pdf(tmp_path, replay):
    work, folder, env = setup(tmp_path, [NOT_IN_OPENALEX, IN_OPENALEX, BY_TITLE])
    replay.answer("GET /works/doi:10.1234%2Fabsent", Answer(404, {}))
    replay.answer(
        "GET /works/doi:10.1234%2Fpresent",
        Answer(
            body=openalex_work(
                IN_OPENALEX["title"], "10.1234/present", "https://repo.example/p.pdf"
            )
        ),
    )
    replay.answer(
        "GET /works",
        Answer(
            body={
                "results": [openalex_work(BY_TITLE["title"], "10.1234/bytitle")],
                "meta": {"count": 1},
            }
        ),
    )
    openalex = run(
        work, replay, env, "openalex", "--api-url", replay.url, "--sleep-seconds", "0"
    )
    assert counts(openalex)["resolved"] == 1
    for doi in ("10.1234%2Fabsent", "10.1234%2Fbytitle"):
        replay.answer(f"GET /v2/{doi}", Answer(body={"is_oa": False}))

    unpaywall = run(
        work,
        replay,
        env,
        "open-access",
        "--api-url",
        f"{replay.url}/v2",
        "--sleep-seconds",
        "0",
    )

    assert counts(unpaywall)["errors"] == 0
    assert replay.misses == []
    asked = {r.path for r in replay.requests if r.path.startswith("/v2/")}
    assert asked == {"/v2/10.1234%2Fabsent", "/v2/10.1234%2Fbytitle"}
    statuses = {
        r["title"]: (r["doi"], r["oa_status"])
        for r in read_rows(folder / "open_access_status.csv")
    }
    assert statuses == {
        NOT_IN_OPENALEX["title"]: ("10.1234/absent", "not_open_access"),
        BY_TITLE["title"]: ("10.1234/bytitle", "not_open_access"),
    }
