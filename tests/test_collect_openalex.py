"""`mbdef collect openalex`: find PDFs of queued papers with OpenAlex, by DOI or title.

OpenAlex is replayed from recorded answers on a local port; nothing else is reachable.
"""

import json

import pytest
from cli_helpers import run_cli
from http_replay import Answer, Replay
from mini_collect import read_receipts, read_rows, setup

SELECT = (
    "id,doi,title,publication_year,authorships,open_access,best_oa_location,locations"
)


@pytest.fixture
def replay():
    with Replay() as server:
        yield server


def openalex(work, replay, env, *extra):
    return run_cli(
        work,
        "collect",
        "openalex",
        "--api-url",
        replay.url,
        "--sleep-seconds",
        "0",
        *extra,
        env=env,
    )


def counts(done) -> dict:
    return json.loads(done.stdout.splitlines()[0])


def work_record(
    *,
    title="Matched paper",
    year=2020,
    doi="https://doi.org/10.1234/matched",
    author="Ada Lovelace",
    locations=(),
    best=None,
    is_oa=True,
):
    return {
        "id": "https://openalex.org/W123",
        "doi": doi,
        "title": title,
        "publication_year": year,
        "authorships": [{"author": {"display_name": author}}],
        "open_access": {"is_oa": is_oa},
        "best_oa_location": best,
        "locations": list(locations),
    }


def location(url, source_type, version="publishedVersion"):
    return {
        "pdf_url": url,
        "is_oa": True,
        "version": version,
        "source": {"type": source_type},
    }


MATCHED = {
    "queue_status": "missing",
    "title": "Matched paper",
    "authors": "Lovelace, A.",
    "year": "2020",
    "doi": "10.1234/matched",
}


def test_a_doi_lookup_keeps_every_pdf_location_repositories_first_and_resumes(
    tmp_path, replay
):
    work, folder, env = setup(
        tmp_path,
        [MATCHED, {"queue_status": "review", "title": "Manual"}],
        OPENALEX_API_KEY="secret-key",
    )
    publisher = location("https://publisher.example/paper.pdf", "journal")
    repository = location(
        "https://repository.example/paper.pdf", "repository", "acceptedVersion"
    )
    replay.answer(
        "GET /works/doi:10.1234%2Fmatched",
        Answer(
            body=work_record(
                locations=[publisher, repository, publisher], best=publisher
            )
        ),
    )

    first = openalex(work, replay, env)
    second = openalex(work, replay, env)

    assert first.returncode == second.returncode == 0, first.stderr
    assert (
        counts(first)
        == counts(second)
        == {
            "eligible": 1,
            "resolved": 1,
            "unresolved": 0,
            "ambiguous": 0,
            "errors": 0,
            "rate_limited": 0,
            "skipped_review": 1,
            "waiting_for_a_download": 0,
            "unattempted": 0,
        }
    )
    assert replay.misses == []
    [asked] = replay.requests  # the second run asks nothing
    assert asked.query == {"select": SELECT}
    assert asked.headers["authorization"] == "Bearer secret-key"
    [candidate] = read_rows(folder / "openalex_candidates.csv")
    assert candidate["pdf_url"] == "https://repository.example/paper.pdf"
    assert candidate["oa_host_type"] == "repository"
    assert json.loads(candidate["alternate_pdf_urls"]) == [
        "https://publisher.example/paper.pdf"
    ]
    assert (candidate["matched_by"], candidate["doi"]) == ("doi", "10.1234/matched")
    assert len(read_rows(folder / "openalex_status.csv")) == 1
    receipts = read_receipts(folder)
    assert [r["settings"]["api_key"] for r in receipts] == ["set", "set"]
    assert "secret-key" not in json.dumps(receipts)


def test_the_earlier_title_search_needs_title_year_and_author_to_agree(
    tmp_path, replay
):
    row = MATCHED | {"doi": ""}
    work, folder, env = setup(tmp_path, [row])
    exact = work_record(
        locations=[location("https://repository.example/matched.pdf", "repository")]
    )
    wrong_year = work_record(year=2019)
    wrong_author = work_record(author="Grace Hopper")
    replay.answer(
        "GET /works", Answer(body={"results": [wrong_year, wrong_author, exact]})
    )

    done = openalex(work, replay, env, "--title-rule", "2026-09-28")

    assert done.returncode == 0, done.stderr
    assert counts(done)["resolved"] == 1
    [asked] = replay.requests
    assert asked.query == {
        "search": "Matched paper",
        "per-page": "200",
        "select": SELECT,
        "filter": "publication_year:2020",
    }
    [candidate] = read_rows(folder / "openalex_candidates.csv")
    assert candidate["paperid"].startswith("hash_")
    assert (candidate["doi"], candidate["query_doi"]) == ("10.1234/matched", "")
    assert candidate["matched_by"] == "title"


def test_two_matching_results_of_the_earlier_search_are_ambiguous(tmp_path, replay):
    row = MATCHED | {"doi": "", "title": "Duplicate title"}
    work, folder, env = setup(tmp_path, [row])
    one = work_record(title="Duplicate title", doi="https://doi.org/10.1234/one")
    two = work_record(title="Duplicate title", doi="https://doi.org/10.1234/two")
    replay.answer("GET /works", Answer(body={"results": [one, two]}))

    done = openalex(work, replay, env, "--title-rule", "2026-09-28")

    assert done.returncode == 0, done.stderr
    assert counts(done)["ambiguous"] == 1
    assert read_rows(folder / "openalex_candidates.csv") == []
    assert [r["oa_status"] for r in read_rows(folder / "openalex_status.csv")] == [
        "ambiguous"
    ]


def test_an_error_is_retried_on_the_next_run_and_a_found_pdf_is_not_asked_again(
    tmp_path, replay
):
    work, folder, env = setup(tmp_path, [MATCHED | {"title": "Retry paper"}])
    found = work_record(
        title="Retry paper",
        locations=[location("https://repository.example/retry.pdf", "repository")],
    )
    replay.answer(
        "GET /works/doi:10.1234%2Fmatched", Answer(503, "temporary"), Answer(body=found)
    )

    runs = [openalex(work, replay, env, "--max-retries", "0") for _ in range(3)]

    assert [(counts(r)["errors"], counts(r)["resolved"]) for r in runs] == [
        (1, 0),
        (0, 1),
        (0, 1),
    ]
    assert len(replay.requests) == 2
    assert [r["oa_status"] for r in read_rows(folder / "openalex_status.csv")] == [
        "error",
        "has_pdf",
    ]


def test_titles_with_search_syntax_are_sent_to_the_earlier_search_as_plain_words(
    tmp_path, replay
):
    title = 'How old do you think I am?": A study of language and age'
    row = MATCHED | {"doi": "", "title": title, "year": "2013"}
    work, folder, env = setup(tmp_path, [row])
    replay.answer("GET /works", Answer(body={"results": []}))

    done = openalex(work, replay, env, "--title-rule", "2026-09-28")

    assert done.returncode == 0, done.stderr
    assert counts(done)["unresolved"] == 1
    assert [r.query["search"] for r in replay.requests] == [
        "How old do you think I am A study of language and age"
    ]


def test_a_spent_rate_limit_stops_the_run_keeps_earlier_progress_and_exits_1(
    tmp_path, replay
):
    queue = [
        MATCHED,
        MATCHED | {"title": "Second paper", "doi": "10.1234/second", "year": "2021"},
        MATCHED | {"title": "Third paper", "doi": "10.1234/third", "year": "2022"},
    ]
    work, folder, env = setup(tmp_path, queue)
    replay.answer(
        "GET /works/doi:10.1234%2Fmatched",
        Answer(
            body=work_record(
                locations=[location("https://repo.example/p.pdf", "repository")]
            )
        ),
    )
    replay.answer(
        "GET /works/doi:10.1234%2Fsecond",
        *[Answer(429, {"error": "rate limit"})] * 3,
    )

    done = openalex(work, replay, env, "--max-retries", "2")

    assert done.returncode == 1
    assert done.stderr.strip() == (
        "OpenAlex rate limit exhausted; progress is saved. "
        "Set OPENALEX_API_KEY or rerun later to resume."
    )
    assert counts(done)["rate_limited"] == 1
    assert counts(done)["resolved"] == 1
    # One success, then three attempts for the throttled row, then the run stops.
    assert len(replay.requests) == 4 and replay.misses == []
    assert [r["oa_status"] for r in read_rows(folder / "openalex_status.csv")] == [
        "has_pdf"
    ]
    assert len(read_rows(folder / "openalex_candidates.csv")) == 1
    assert read_receipts(folder)[0]["settings"]["api_key"] == "not set"


# The rules the released closure found its papers with ------------------------------------------
# A queue DOI that OpenAlex lacks is not searched by title: Unpaywall gets the DOI. A row without a DOI is searched with the title rule `2026-09-29`: OpenAlex's title filter, years one around the row's, the first author among the cited ones, and a year one off only when no record has the row's year. `--title-rule 2026-09-28` keeps the earlier full-text search.


def test_a_queue_doi_openalex_lacks_is_not_found_without_a_title_search(
    tmp_path, replay
):
    work, folder, env = setup(tmp_path, [MATCHED])
    replay.answer("GET /works/doi:10.1234%2Fmatched", Answer(404, {}))

    done = openalex(work, replay, env)

    assert done.returncode == 0, done.stderr
    assert replay.asked("GET /works") == []
    [status] = read_rows(folder / "openalex_status.csv")
    assert (status["oa_status"], status["matched_by"], status["doi"]) == (
        "not_found",
        "doi_not_in_openalex",
        "10.1234/matched",
    )


def title_search(replay, *works, count=None):
    meta = {"count": len(works) if count is None else count}
    replay.answer("GET /works", Answer(body={"results": list(works), "meta": meta}))


def test_the_title_rule_filters_on_the_title_and_the_years_around_the_row(
    tmp_path, replay
):
    work, folder, env = setup(
        tmp_path, [MATCHED | {"doi": "", "title": "Matched, paper"}]
    )
    exact = work_record(
        title="Matched, paper",
        locations=[location("https://repository.example/m.pdf", "repository")],
    )
    title_search(replay, exact)

    done = openalex(work, replay, env)

    assert done.returncode == 0, done.stderr
    [asked] = replay.requests
    assert asked.query == {
        "filter": "title.search:Matched  paper,publication_year:2019|2020|2021",
        "per-page": "200",
        "select": SELECT,
    }
    [candidate] = read_rows(folder / "openalex_candidates.csv")
    assert (candidate["matched_by"], candidate["doi"]) == (
        "title_filter",
        "10.1234/matched",
    )


def test_a_record_whose_first_author_is_not_cited_is_ignored(tmp_path, replay):
    # A book review lists its reviewer first and the book's authors after.
    work, folder, env = setup(tmp_path, [MATCHED | {"doi": ""}])
    review = work_record() | {
        "authorships": [
            {"author": {"display_name": "Grace Hopper"}},
            {"author": {"display_name": "Ada Lovelace"}},
        ]
    }
    title_search(replay, review)

    done = openalex(work, replay, env)

    assert done.returncode == 0, done.stderr
    [status] = read_rows(folder / "openalex_status.csv")
    assert (status["oa_status"], status["matched_by"]) == ("not_found", "title_filter")


@pytest.mark.parametrize(
    ("years", "outcome"),
    [
        ((2021,), ("has_pdf", "title_filter_year_off_by_one", "2021")),
        ((2021, 2020), ("has_pdf", "title_filter", "2020")),
        ((2021, 2019), ("ambiguous", "title_filter_year_off_by_one", "")),
        ((2020, 2020), ("ambiguous", "title_filter", "")),
    ],
)
def test_a_year_one_off_counts_only_when_no_record_has_the_row_year(
    tmp_path, replay, years, outcome
):
    work, folder, env = setup(tmp_path, [MATCHED | {"doi": ""}])
    records = [
        work_record(
            year=year,
            doi=f"https://doi.org/10.1234/y{year}{i}",
            locations=[
                location(f"https://repository.example/{year}{i}.pdf", "repository")
            ],
        )
        for i, year in enumerate(years)
    ]
    title_search(replay, *records)

    done = openalex(work, replay, env)

    assert done.returncode == 0, done.stderr
    [status] = read_rows(folder / "openalex_status.csv")
    assert (
        status["oa_status"],
        status["matched_by"],
        status["matched_year"],
    ) == outcome


def test_a_page_with_more_records_than_it_shows_is_ambiguous(tmp_path, replay):
    work, folder, env = setup(tmp_path, [MATCHED | {"doi": ""}])
    title_search(replay, work_record(), count=201)

    done = openalex(work, replay, env)

    assert done.returncode == 0, done.stderr
    [status] = read_rows(folder / "openalex_status.csv")
    assert (status["oa_status"], status["matched_by"]) == ("ambiguous", "title_filter")


def test_a_row_without_a_year_is_searched_by_title_and_needs_the_year_to_match(
    tmp_path, replay
):
    work, folder, env = setup(tmp_path, [MATCHED | {"doi": "", "year": ""}])
    title_search(replay, work_record())

    done = openalex(work, replay, env)

    assert done.returncode == 0, done.stderr
    [asked] = replay.requests
    assert asked.query["filter"] == "title.search:Matched paper"
    [status] = read_rows(folder / "openalex_status.csv")
    assert (status["oa_status"], status["matched_by"]) == ("not_found", "title_filter")


def test_the_title_rule_asks_again_what_the_earlier_rule_left_without_a_work(
    tmp_path, replay
):
    # As in the released closure: a row the earlier search left ambiguous or not found is searched again under rule `2026-09-29`; a row it settled with a work keeps that work.
    left = MATCHED | {"doi": "", "title": "Left without a work"}
    settled = MATCHED | {"doi": ""}
    work, folder, env = setup(tmp_path, [left, settled])
    found = {  # the earlier search finds one work; the later filter finds the other
        "Matched paper": work_record(),
        "title.search:Left": work_record(title="Left without a work", is_oa=False),
    }

    def works(request):
        asked = request.query.get("search") or request.query["filter"]
        hits = [w for text, w in found.items() if asked.startswith(text)]
        return Answer(body={"results": hits, "meta": {"count": len(hits)}})

    replay.answer("GET /works", works)
    earlier = openalex(work, replay, env, "--title-rule", "2026-09-28")

    later = openalex(work, replay, env)

    assert (earlier.returncode, later.returncode) == (0, 0), later.stderr
    searches = [r.query for r in replay.asked("GET /works")]
    assert [("search" in q, "filter" in q) for q in searches] == [
        (True, True),
        (True, True),
        (False, True),
    ]
    assert searches[2]["filter"].startswith("title.search:Left without a work,")
    latest = {r["title"]: r for r in read_rows(folder / "openalex_status.csv")}
    assert (
        latest[left["title"]]["oa_status"],
        latest[left["title"]]["matched_by"],
    ) == (
        "not_open_access",
        "title_filter",
    )
    assert latest[settled["title"]]["matched_by"] == "title"
    assert counts(later)["resolved"] == 0
