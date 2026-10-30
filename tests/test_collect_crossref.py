"""`mbdef collect crossref`: find the DOI of a queued paper that has none (Crossref).

Crossref is replayed from recorded answers on a local port; nothing else is reachable.
"""

import json

import pytest
from cli_helpers import run_cli
from http_replay import Answer, Replay
from mini_collect import EMAIL, read_receipts, read_rows, setup

SELECT = (
    "DOI,title,subtitle,author,issued,published,published-print,published-online,"
    "posted,type"
)
TRUE = {
    "queue_status": "missing",
    "title": "Deciding what's true: The rise of political fact-checking",
    "authors": "Graves, L.; Amazeen, M. A.",
    "year": "2016",
}


@pytest.fixture
def replay():
    with Replay() as server:
        yield server


def crossref(work, replay, env, *extra):
    return run_cli(
        work,
        "collect",
        "crossref",
        "--api-url",
        replay.url,
        "--sleep-seconds",
        "0",
        "--backoff-seconds",
        "0",
        *extra,
        env=env,
    )


def counts(done) -> dict:
    return json.loads(done.stdout.splitlines()[0])


def record(title, doi, *, years=(2016,), authors=("Graves",), **extra):
    return {
        "DOI": doi,
        "title": [title],
        "author": [{"family": name, "given": "A."} for name in authors],
        "issued": {"date-parts": [[year] for year in years]},
        "type": "journal-article",
    } | extra


def page(*items):
    return Answer(body={"status": "ok", "message": {"items": list(items)}})


def test_a_row_without_a_doi_takes_the_one_doi_whose_title_year_and_author_agree(
    tmp_path, replay
):
    queue = [
        TRUE,
        TRUE | {"title": "A paper with a DOI", "doi": "10.1234/has"},
        TRUE | {"queue_status": "review", "title": "An uncertain link"},
    ]
    work, folder, env = setup(tmp_path, queue)
    book = record(
        "Deciding What's True",
        "10.7312/GRAV17506",
        subtitle=["The Rise of Political Fact-Checking"],
        type="edited-book",
        **{"published-online": {"date-parts": [[2017, 1, 3]]}},
    )
    review = record(
        "Deciding what's true: the rise of political fact-checking",
        "10.1/review",
        years=(2019,),
    )
    other = record("Something else entirely", "10.1/other")
    replay.answer("GET /works", page(other, book, review))

    first = crossref(work, replay, env)
    second = crossref(work, replay, env)

    assert first.returncode == second.returncode == 0, first.stderr
    assert (
        counts(first)
        == counts(second)
        == {
            "eligible": 1,
            "matched": 1,
            "no_match": 0,
            "ambiguous": 0,
            "insufficient_evidence": 0,
            "errors": 0,
            "with_doi": 1,
            "skipped_review": 1,
            "unattempted": 0,
        }
    )
    assert replay.misses == []
    [asked] = replay.requests  # the second run asks nothing
    assert asked.query == {
        "query.bibliographic": (
            "Deciding what's true: The rise of political fact-checking Graves Amazeen"
        ),
        "rows": "20",
        "select": SELECT,
        "mailto": EMAIL,
    }
    assert asked.headers["user-agent"].startswith("mbdef")
    [row] = read_rows(folder / "crossref_status.csv")
    assert row["paperid"].startswith("hash_")
    assert row["attempted_at"]
    del row["paperid"], row["attempted_at"]
    assert row == {
        "title": TRUE["title"],
        "authors": TRUE["authors"],
        "year": "2016",
        "first_author": "Graves",
        "query": (
            "Deciding what's true: The rise of political fact-checking Graves Amazeen"
        ),
        "cr_status": "matched",
        "reason": "exact title+year+author, unique DOI",
        "doi": "10.7312/grav17506",
        "matched_title": "Deciding What's True",
        "matched_year": "2016;2017",
        "matched_first_author": "Graves",
        "matched_type": "edited-book",
        "n_results": "3",
        "n_title": "2",
        "n_title_year": "1",
        "n_exact": "1",
        "candidate_dois": "10.7312/grav17506",
        "status_code": "200",
        "error": "",
    }
    receipts = read_receipts(folder)
    assert [r["command"] for r in receipts] == ["collect crossref"] * 2
    assert receipts[0]["settings"]["mailto"] == "set"
    assert EMAIL not in json.dumps(receipts)


def test_two_dois_no_agreeing_record_or_too_little_evidence_give_no_doi(
    tmp_path, replay
):
    title = "Echo chambers on social media platforms"
    rows = {
        "ambiguous": TRUE | {"title": title},
        "no exact title": TRUE | {"title": title + " revisited"},
        "title but no matching year": TRUE | {"title": title, "year": "2001"},
        "title+year but no shared author surname": TRUE
        | {"title": title, "authors": "Hopper, G."},
        "citation lacks title+year": TRUE | {"title": "Echo", "year": "n.d."},
        "citation lacks title": TRUE | {"title": " "},
    }
    work, folder, env = setup(tmp_path, list(rows.values()))
    one = record(title, "10.1234/one")
    two = record(title.upper(), "https://doi.org/10.1234/TWO")
    replay.answer("GET /works", lambda request: page(one, two))

    done = crossref(work, replay, env)

    assert done.returncode == 0, done.stderr
    assert counts(done) | {"eligible": 6} == counts(done)
    assert (counts(done)["ambiguous"], counts(done)["no_match"]) == (1, 3)
    assert counts(done)["insufficient_evidence"] == 2
    # The empty title is never sent; the short one is (the rule decides after the answer).
    assert len(replay.requests) == 5
    found = {
        (r["title"], r["year"], r["authors"]): r
        for r in read_rows(folder / "crossref_status.csv")
    }
    got = {
        reason: found[(row["title"], row["year"], row["authors"])]
        for reason, row in rows.items()
    }
    ambiguous = got.pop("ambiguous")
    assert (ambiguous["cr_status"], ambiguous["reason"], ambiguous["doi"]) == (
        "ambiguous",
        "2 DOIs pass exact title+year+author",
        "",
    )
    assert ambiguous["candidate_dois"] == "10.1234/one;10.1234/two"
    assert ambiguous["n_exact"] == "2"
    for reason in (
        "no exact title",
        "title but no matching year",
        "title+year but no shared author surname",
    ):
        assert (got[reason]["cr_status"], got[reason]["reason"]) == ("no_match", reason)
    short, empty = got["citation lacks title+year"], got["citation lacks title"]
    assert (short["cr_status"], short["n_results"], short["status_code"]) == (
        "insufficient_evidence",
        "2",
        "200",
    )
    assert (empty["reason"], empty["n_results"], empty["status_code"]) == (
        "citation lacks title",
        "0",
        "",
    )


def test_failed_searches_are_retried_and_a_paper_failing_three_runs_is_left(
    tmp_path, replay
):
    work, folder, env = setup(tmp_path, [TRUE])
    replay.answer(
        "GET /works",
        Answer(503, "busy"),
        page(),  # the second attempt of the first run is answered
    )
    first = crossref(work, replay, env, "--max-attempts", "2")
    assert first.returncode == 0, first.stderr
    assert counts(first)["no_match"] == 1
    assert len(replay.requests) == 2

    # A paper whose answers fail: an error row per run, and after three of them no more requests.
    other = TRUE | {"title": "A paper Crossref keeps refusing"}
    (tmp_path / "second").mkdir()
    work, folder, env = setup(tmp_path / "second", [other])
    replay.answer("GET /works", *[Answer(429, "slow down")] * 4, Answer(404, "gone"))
    runs = [crossref(work, replay, env, "--max-attempts", "2") for _ in range(4)]

    assert [counts(r)["errors"] for r in runs] == [1, 1, 1, 1]
    assert len(replay.requests) == 2 + 2 + 2 + 1  # 429 twice, twice, then 404 at once
    assert [r["error"] for r in read_rows(folder / "crossref_status.csv")] == [
        "HTTP Error 429 after 2 attempts with backoff",
        "HTTP Error 429 after 2 attempts with backoff",
        "HTTP Error 404",
    ]
    assert [r["status_code"] for r in read_rows(folder / "crossref_status.csv")] == [
        "429",
        "429",
        "404",
    ]


def test_the_pinned_defaults_are_the_v1_1_ones(tmp_path):
    done = run_cli(tmp_path, "collect", "crossref", "--help")

    assert done.returncode == 0
    text = " ".join(done.stdout.split())
    assert "the Crossref API (default: https://api.crossref.org)" in text
    assert "longer if Crossref asks (default: 0.2)" in text
    assert "request timeout (default: 30.0)" in text
    assert "HTTP 429 or 5xx (default: 9)" in text
    assert "doubling each time (default: 1.0)" in text


def test_the_records_per_search_and_the_shortest_title_are_flags(tmp_path, replay):
    """`--rows 5` asks Crossref for 5 records; `--min-title-chars 80` makes the 56-character title too little evidence. The receipt records both."""
    work, folder, env = setup(tmp_path, [TRUE])
    replay.answer("GET /works", page(record(TRUE["title"], "10.1/x")))

    done = crossref(work, replay, env, "--rows", "5", "--min-title-chars", "80")

    assert done.returncode == 0, done.stderr
    [asked] = replay.requests
    assert asked.query["rows"] == "5"
    assert counts(done)["insufficient_evidence"] == 1
    [row] = read_rows(folder / "crossref_status.csv")
    assert row["reason"] == "citation lacks title"
    settings = read_receipts(folder)[0]["settings"]
    assert (settings["rows"], settings["min_title_chars"]) == (5, 80)
