"""A DOI that `collect crossref` or `collect semantic-scholar` found for a queued paper without one is looked up by `collect openalex` and `collect open-access` as if the queue had it.

Every service is replayed from recorded answers on a local port; nothing else is reachable.
"""

import json

import pytest
from cli_helpers import run_cli
from http_replay import Answer, Replay
from mini_collect import read_rows, setup

ROW = {"queue_status": "missing", "authors": "Lovelace, A.", "year": "2020"}
BY_CROSSREF = ROW | {"title": "A paper Crossref knows by its title"}
BY_S2 = ROW | {"title": "A paper only Semantic Scholar matches"}
NOT_IN_OPENALEX = ROW | {"title": "A paper whose DOI OpenAlex lacks"}
UNFOUND = ROW | {"title": "A paper nobody finds by its title"}


@pytest.fixture
def replay():
    with Replay() as server:
        yield server


def run(work, replay, env, action, *extra):
    return run_cli(
        work,
        "collect",
        action,
        "--api-url",
        replay.url,
        "--sleep-seconds",
        "0",
        *extra,
        env=env,
    )


def counts(done) -> dict:
    return json.loads(done.stdout.splitlines()[0])


def crossref_record(row, doi):
    return {
        "DOI": doi,
        "title": [row["title"]],
        "author": [{"family": "Lovelace"}],
        "issued": {"date-parts": [[2020]]},
        "type": "journal-article",
    }


def work_record(row, doi, pdf=None):
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
        "id": "https://openalex.org/W1",
        "doi": "https://doi.org/" + doi,
        "title": row["title"],
        "publication_year": 2020,
        "authorships": [{"author": {"display_name": "Ada Lovelace"}}],
        "open_access": {"is_oa": bool(pdf)},
        "best_oa_location": None,
        "locations": locations,
    }


def test_found_dois_are_looked_up_under_the_original_row(tmp_path, replay):
    work, folder, env = setup(tmp_path, [BY_CROSSREF, BY_S2, NOT_IN_OPENALEX, UNFOUND])
    crossref_pages = {
        BY_CROSSREF["title"]: [crossref_record(BY_CROSSREF, "10.1234/found")],
        NOT_IN_OPENALEX["title"]: [crossref_record(NOT_IN_OPENALEX, "10.1234/absent")],
    }

    def works(
        request,
    ):  # Crossref's search, and OpenAlex's title search on the same path
        if "query.bibliographic" not in request.query:
            return Answer(body={"results": [], "meta": {"count": 0}})
        title = request.query["query.bibliographic"].removesuffix(" Lovelace")
        return Answer(body={"message": {"items": crossref_pages.get(title, [])}})

    replay.answer("GET /works", works)
    assert (
        counts(run(work, replay, env, "crossref", "--backoff-seconds", "0"))["matched"]
        == 2
    )
    replay.answer(
        "GET /paper/search/match",
        lambda request: (
            Answer(
                body={
                    "data": [
                        {
                            "paperId": "s2",
                            "title": BY_S2["title"],
                            "year": 2020,
                            "authors": [{"name": "Ada Lovelace"}],
                            "externalIds": {"DOI": "10.1234/S2FOUND"},
                        }
                    ]
                }
            )
            if request.query["query"] == BY_S2["title"]
            else Answer(404, {"error": "Title match not found"})
        ),
    )
    replay.answer(
        "GET /works/10.1234/s2found",
        Answer(body={"message": crossref_record(BY_S2, "10.1234/s2found")}),
    )
    s2 = run(
        work,
        replay,
        env,
        "semantic-scholar",
        "--crossref-url",
        replay.url,
        "--check-sleep-seconds",
        "0",
        "--backoff-seconds",
        "0",
    )
    assert counts(s2)["with_doi"] == 1
    replay.answer(
        "GET /works/doi:10.1234%2Ffound",
        Answer(
            body=work_record(BY_CROSSREF, "10.1234/found", "https://repo.example/f.pdf")
        ),
    )
    replay.answer(
        "GET /works/doi:10.1234%2Fs2found",
        Answer(body=work_record(BY_S2, "10.1234/s2found")),
    )
    replay.answer("GET /works/doi:10.1234%2Fabsent", Answer(404, {}))
    title_searches = len(replay.asked("GET /works"))

    first = run(work, replay, env, "openalex")
    second = run(work, replay, env, "openalex")

    assert first.returncode == second.returncode == 0, first.stderr
    assert counts(first) == counts(second)
    assert counts(first)["resolved"] == 1
    assert replay.misses == []
    # Only the paper without a found DOI is searched by its title, once.
    searches = replay.asked("GET /works")[title_searches:]
    assert [r.query["filter"] for r in searches] == [
        f"title.search:{UNFOUND['title']},publication_year:2019|2020|2021"
    ]
    rows = {r["title"]: r for r in read_rows(folder / "openalex_status.csv")}
    assert len(read_rows(folder / "openalex_status.csv")) == 4
    found = {
        title: (
            r["query_doi"],
            r["doi"],
            r["matched_by"],
            r["oa_status"],
            r["status_code"],
        )
        for title, r in rows.items()
    }
    assert found == {
        BY_CROSSREF["title"]: ("", "10.1234/found", "crossref_doi", "has_pdf", "200"),
        BY_S2["title"]: ("", "10.1234/s2found", "s2_doi", "not_open_access", "200"),
        NOT_IN_OPENALEX["title"]: (
            "",
            "10.1234/absent",
            "crossref_doi_not_in_openalex",
            "not_found",
            "404",
        ),
        UNFOUND["title"]: ("", "", "title_filter", "not_found", "200"),
    }
    [candidate] = read_rows(folder / "openalex_candidates.csv")
    assert (candidate["title"], candidate["pdf_url"]) == (
        BY_CROSSREF["title"],
        "https://repo.example/f.pdf",
    )

    # Unpaywall gets what OpenAlex left without a PDF, by the DOI OpenAlex names.
    for doi in ("10.1234%2Fs2found", "10.1234%2Fabsent"):
        replay.answer(f"GET /v2/{doi}", Answer(body={"is_oa": False}))
    unpaywall = run(work, replay, env, "open-access", "--api-url", f"{replay.url}/v2")

    assert unpaywall.returncode == 0, unpaywall.stderr
    assert replay.misses == []
    statuses = {
        r["title"]: (r["doi"], r["oa_status"])
        for r in read_rows(folder / "open_access_status.csv")
    }
    assert statuses == {
        BY_S2["title"]: ("10.1234/s2found", "not_open_access"),
        NOT_IN_OPENALEX["title"]: ("10.1234/absent", "not_open_access"),
    }
