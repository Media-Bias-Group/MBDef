"""`mbdef collect open-access`: find an open-access PDF for each queued paper (Unpaywall).

Unpaywall is replayed from recorded answers on a local port; nothing else is reachable.
"""

import json

import pytest
from cli_helpers import run_cli
from http_replay import Answer, Replay
from mini_collect import EMAIL, head, read_receipts, read_rows, setup, sha


@pytest.fixture
def replay():
    with Replay() as server:
        yield server


def open_access(work, replay, env, *extra):
    return run_cli(
        work,
        "collect",
        "open-access",
        "--api-url",
        f"{replay.url}/v2",
        "--sleep-seconds",
        "0",
        *extra,
        env=env,
    )


def test_a_missing_unpaywall_email_fails_with_a_pointer_to_env_example(tmp_path):
    work, folder, env = setup(tmp_path, [], UNPAYWALL_EMAIL="")

    done = run_cli(work, "collect", "open-access", env=env)

    assert done.returncode == 1
    assert done.stderr.strip() == (
        "UNPAYWALL_EMAIL is not set. Copy .env.example to .env and set UNPAYWALL_EMAIL "
        "there; .env.example says what it is."
    )
    assert not (folder / "open_access_status.csv").exists()


def test_a_missing_queue_is_refused_with_where_it_comes_from(tmp_path):
    work, folder, env = setup(tmp_path)

    done = run_cli(work, "collect", "open-access", env=env)

    assert done.returncode == 1
    assert done.stderr.strip() == (
        f"The collection queue {folder / 'citation-papers-to-collect.csv'} does not exist. "
        "The closure step's corpus check writes it (docs/step-1-2-papers/collect.md); "
        "pass another one with --collection."
    )
    assert not folder.exists()


def test_with_reference_urls_a_row_falls_back_to_the_address_the_reference_gave(
    tmp_path, replay
):
    work, folder, env = setup(
        tmp_path,
        [
            {
                "queue_status": "missing",
                "title": "Published work",
                "authors": "Lovelace, A.",
                "year": "2020",
                "doi": "10.48550/arXiv.1234; 10.1234/Published",
                "url": "https://publisher.example/paper.pdf",
            },
            {
                "queue_status": "missing",
                "title": "No source",
                "authors": "Hopper, G.",
                "year": "2021",
                "url": "publisher.example/not-absolute",
            },
            {"queue_status": "review", "title": "Manual review"},
        ],
    )
    replay.answer("GET /v2/10.1234%2Fpublished", Answer(body={"is_oa": False}))

    done = open_access(work, replay, env, "--reference-urls")

    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.splitlines()[0]) == {
        "eligible": 2,
        "resolved": 1,
        "unresolved": 1,
        "errors": 0,
        "skipped_review": 1,
    }
    assert replay.misses == []
    [asked] = replay.requests
    assert asked.query == {"email": EMAIL}
    status = sorted(
        read_rows(folder / "open_access_status.csv"), key=lambda r: r["title"]
    )
    assert [(r["title"], r["doi"], r["oa_status"], r["pdf_url"]) for r in status] == [
        ("No source", "", "no_download_source", ""),
        (
            "Published work",
            "10.1234/published",
            "provided_url",
            "https://publisher.example/paper.pdf",
        ),
    ]
    assert [r["title"] for r in read_rows(folder / "open_access_candidates.csv")] == [
        "Published work"
    ]


def test_the_best_location_wins_then_the_first_other_location_with_a_pdf(
    tmp_path, replay
):
    queue = [
        {"queue_status": "missing", "title": f"Paper {n}", "doi": f"10.1234/{n}"}
        for n in ("best", "other", "closed")
    ]
    work, folder, env = setup(tmp_path, queue)
    best = {"url_for_pdf": "https://best.example/pdf", "host_type": "publisher"}
    other = {"url_for_pdf": "https://other.example/pdf", "host_type": "repository"}
    replay.answer(
        "GET /v2/10.1234%2Fbest",
        Answer(body={"is_oa": True, "best_oa_location": best, "oa_locations": [other]}),
    )
    replay.answer(
        "GET /v2/10.1234%2Fother",
        Answer(body={"is_oa": True, "best_oa_location": None, "oa_locations": [other]}),
    )
    replay.answer("GET /v2/10.1234%2Fclosed", Answer(body={"is_oa": False}))

    done = open_access(work, replay, env)

    assert done.returncode == 0, done.stderr
    assert replay.misses == []
    status = {r["title"]: r for r in read_rows(folder / "open_access_status.csv")}
    assert [
        (status[f"Paper {n}"]["pdf_url"], status[f"Paper {n}"]["oa_host_type"])
        for n in ("best", "other", "closed")
    ] == [
        ("https://best.example/pdf", "publisher"),
        ("https://other.example/pdf", "repository"),
        ("", ""),
    ]
    assert status["Paper closed"]["oa_status"] == "not_open_access"


def test_paper_ids_follow_the_normalised_doi_and_never_collide(tmp_path, replay):
    queue = [
        {
            "queue_status": "missing",
            "title": "Demo",
            "doi": "https://doi.org/10.1234/Demo",
        },
        {"queue_status": "missing", "title": "Colon", "doi": "10.1234/a:b"},
        {"queue_status": "missing", "title": "Question", "doi": "10.1234/a?b"},
        {
            "queue_status": "missing",
            "title": "Paper",
            "year": "2020",
            "authors": "Lovelace, A.",
        },
        {
            "queue_status": "missing",
            "title": " Paper ",
            "year": "2020",
            "authors": "lovelace, a.",
        },
    ]
    work, folder, env = setup(tmp_path, queue)
    for doi in ("10.1234%2Fdemo", "10.1234%2Fa%3Ab", "10.1234%2Fa%3Fb"):
        replay.answer(f"GET /v2/{doi}", Answer(404, {"error": "not found"}))

    done = open_access(work, replay, env, "--reference-urls")  # rows without a DOI too

    assert done.returncode == 0, done.stderr
    assert replay.misses == []
    ids = [r["paperid"] for r in read_rows(folder / "open_access_status.csv")]
    assert len(set(ids)) == 5
    # The DOI part is its safe form and its sha1; then comes a hash of the whole queue row.
    assert ids[0].startswith(
        "doi_10.1234_demo__51facb5bc98ddd6f6b88a82e4f5a99f3700e0512__"
    )
    assert ids[1].split("__")[0] == ids[2].split("__")[0] == "doi_10.1234_a_b"
    assert ids[1].split("__")[1] != ids[2].split("__")[1]
    # Without a DOI: a hash of title, year and authors, blind to case and outer spaces.
    assert ids[3].startswith("hash_")
    assert ids[3].split("__")[0] == ids[4].split("__")[0]


def test_an_error_is_retried_on_the_next_run_and_a_found_pdf_is_not_asked_again(
    tmp_path, replay
):
    work, folder, env = setup(
        tmp_path,
        [{"queue_status": "missing", "title": "Retry me", "doi": "10.1234/retry"}],
    )
    found = {
        "is_oa": True,
        "best_oa_location": {
            "url_for_pdf": "https://repository.example/retry.pdf",
            "host_type": "repository",
        },
    }
    replay.answer(
        "GET /v2/10.1234%2Fretry", Answer(503, "temporary failure"), Answer(body=found)
    )

    runs = [open_access(work, replay, env) for _ in range(3)]

    counts = [json.loads(run.stdout.splitlines()[0]) for run in runs]
    assert [(c["errors"], c["resolved"]) for c in counts] == [(1, 0), (0, 1), (0, 1)]
    assert len(replay.requests) == 2
    assert [r["oa_status"] for r in read_rows(folder / "open_access_status.csv")] == [
        "error",
        "has_pdf",
    ]
    assert [r["pdf_url"] for r in read_rows(folder / "open_access_candidates.csv")] == [
        "https://repository.example/retry.pdf"
    ]
    assert len(read_receipts(folder)) == 3


def test_the_receipt_records_the_settings_the_hashes_and_the_commit(tmp_path, replay):
    work, folder, env = setup(
        tmp_path, [{"queue_status": "missing", "title": "Paper", "doi": "10.1234/p"}]
    )
    replay.answer("GET /v2/10.1234%2Fp", Answer(body={"is_oa": False}))

    done = open_access(work, replay, env, "--timeout-seconds", "7")

    assert done.returncode == 0, done.stderr
    [receipt] = read_receipts(folder)
    assert receipt["command"] == "collect open-access"
    assert receipt["code"]["commit"] == head()
    assert receipt["settings"] == {
        "api_url": f"{replay.url}/v2",
        "sleep_seconds": 0.0,
        "timeout_seconds": 7.0,
        "limit": None,
        "reference_urls": False,
    }
    queue = folder / "citation-papers-to-collect.csv"
    assert receipt["inputs"]["collection"]["sha256"] == sha(queue)
    for name in ("status", "candidates"):
        written = folder / f"open_access_{name}.csv"
        assert receipt["outputs"][name]["sha256"] == sha(written)
    assert receipt["summary"]["unresolved"] == 1
    assert EMAIL not in json.dumps(
        receipt
    )  # the contact address is personal, not a setting


def test_the_pinned_defaults_are_the_released_ones(tmp_path):
    done = run_cli(tmp_path, "collect", "open-access", "--help")

    assert done.returncode == 0
    text = " ".join(done.stdout.split())
    assert "(default: https://api.unpaywall.org/v2)" in text
    assert "pause after each request (default: 1.0)" in text
    assert "request timeout (default: 20.0)" in text
