"""`mbdef collect download`: download the PDFs a resolver found into zip batches.

The publishers are replayed from recorded answers on a local port; nothing else is reachable.
"""

import json
import zipfile

import pytest
from cli_helpers import run_cli
from http_replay import Answer, Replay
from mini_collect import pdf_archive, read_receipts, read_rows, setup, sha, write_rows

# The columns of a resolver's candidates file (open access; OpenAlex adds the rest).
CANDIDATE_FIELDS = [
    "paperid",
    "queue_status",
    "title",
    "authors",
    "year",
    "doi",
    "source_url",
    "pdf_url",
    "oa_status",
    "oa_host_type",
    "status_code",
    "error",
    "attempted_at",
]
OPENALEX_FIELDS = CANDIDATE_FIELDS + [
    "alternate_pdf_urls",
    "query_doi",
    "openalex_id",
    "matched_by",
    "matched_title",
    "matched_year",
]


@pytest.fixture
def replay():
    with Replay() as server:
        yield server


def pdf(content: bytes) -> Answer:
    return Answer(body=content, headers={"Content-Type": "application/pdf"})


def html(content: bytes, status: int = 200) -> Answer:
    return Answer(status, content, {"Content-Type": "text/html"})


def download(work, env, *extra):
    return run_cli(work, "collect", "download", *extra, env=env)


def counts(done) -> dict:
    return json.loads(done.stdout.splitlines()[0])


def test_pdfs_are_archived_and_html_and_oversized_answers_are_refused(tmp_path, replay):
    work, folder, env = setup(tmp_path)
    write_rows(
        folder / "open_access_candidates.csv",
        [
            {"paperid": "valid-a", "doi": "10.1234/a", "pdf_url": f"{replay.url}/x/a"},
            {"paperid": "html-b", "doi": "10.1234/b", "pdf_url": f"{replay.url}/x/b"},
            {"paperid": "large-c", "doi": "10.1234/c", "pdf_url": f"{replay.url}/x/c"},
        ],
        CANDIDATE_FIELDS,
    )
    replay.answer("GET /x/a", pdf(b"preamble\n%PDF-1.4\nvalid"))
    # An HTML answer: its page is read once more for a declared PDF (none here).
    replay.answer("GET /x/b", *[html(b"<html>not a PDF</html>")] * 2)
    replay.answer("GET /x/c", pdf(b"%PDF-1.4" + b"x" * 2000))

    done = download(
        work,
        env,
        "--concurrency",
        "1",
        "--max-pdf-mb",
        "0.001",
        "--pdf-batch-size",
        "1",
    )

    assert done.returncode == 0, done.stderr
    assert counts(done) == {
        "candidates": 3,
        "downloaded": 1,
        "failed": 2,
        "skipped": 0,
        "deferred": 0,
    }
    assert replay.misses == []
    assert all("Chrome" in r.headers["user-agent"] for r in replay.requests)
    with zipfile.ZipFile(folder / "pdf_batches/pdf_batch_000001.zip") as archive:
        assert archive.namelist() == ["va/valid-a.pdf"]
        assert archive.read("va/valid-a.pdf") == b"preamble\n%PDF-1.4\nvalid"
        comment = json.loads(archive.getinfo("va/valid-a.pdf").comment)
        assert comment == {"paperid": "valid-a", "doi": "10.1234/a"}
    status = {r["paperid"]: r for r in read_rows(folder / "pdf_download_status.csv")}
    assert status["valid-a"]["pdf_archive_member"] == "va/valid-a.pdf"
    assert status["html-b"]["error"] == "not_pdf content_type=text/html"
    assert status["large-c"]["error"] == "pdf_too_large max_mb=0.001"
    failed = read_rows(folder / "failed_pdf_downloads.csv")
    assert {r["paperid"] for r in failed} == {"html-b", "large-c"}
    [receipt] = read_receipts(folder)
    assert receipt["settings"] == {
        "concurrency": 1,
        "timeout_seconds": 45.0,
        "max_pdf_mb": 0.001,
        "pdf_batch_size": 1,
        "impersonate": "chrome",
        "limit": None,
        "europe_pmc_url": "https://europepmc.org",
        "paced_hosts": {"europepmc.org": 3.0, "www.ebi.ac.uk": 3.0},
    }
    assert receipt["inputs"]["candidates"]["sha256"] == sha(
        folder / "open_access_candidates.csv"
    )
    assert receipt["outputs"]["archives"]["files"] == 1


def test_an_archived_pdf_is_recorded_without_downloading_it_again(tmp_path, replay):
    """A run that stopped between writing the zip and the ledger is completed from the zip."""
    work, folder, env = setup(tmp_path)
    write_rows(
        folder / "open_access_candidates.csv",
        [{"paperid": "recovered", "doi": "10.1234/r", "pdf_url": f"{replay.url}/x/r"}],
        CANDIDATE_FIELDS,
    )
    pdf_archive(
        folder / "pdf_batches/pdf_batch_000001.zip",
        {"re/recovered.pdf": b"%PDF-1.4 recovered"},
    )

    done = download(work, env)

    assert done.returncode == 0, done.stderr
    assert counts(done) == {
        "candidates": 1,
        "downloaded": 0,
        "failed": 0,
        "skipped": 1,
        "deferred": 0,
    }
    assert replay.requests == []
    [recovered] = read_rows(folder / "pdf_download_status.csv")
    assert (recovered["downloaded"], recovered["doi"]) == ("True", "10.1234/r")
    assert recovered["pdf_archive_member"] == "re/recovered.pdf"


def test_a_failed_download_is_retried_on_the_next_run(tmp_path, replay):
    work, folder, env = setup(tmp_path)
    write_rows(
        folder / "open_access_candidates.csv",
        [
            {
                "paperid": "retry",
                "doi": "10.1234/retry",
                "pdf_url": f"{replay.url}/x/retry",
            }
        ],
        CANDIDATE_FIELDS,
    )
    # The first run reads the landing page twice (the second time for a declared PDF).
    replay.answer(
        "GET /x/retry",
        html(b"landing page"),
        html(b"landing page"),
        pdf(b"%PDF-1.4 success"),
    )

    first = download(work, env)
    failed_after_first = read_rows(folder / "failed_pdf_downloads.csv")
    second = download(work, env)

    assert counts(first)["failed"] == 1
    assert [r["paperid"] for r in failed_after_first] == ["retry"]
    assert counts(second)["downloaded"] == 1
    assert read_rows(folder / "failed_pdf_downloads.csv") == []
    assert [r["downloaded"] for r in read_rows(folder / "pdf_download_status.csv")] == [
        "False",
        "True",
    ]


def test_openalex_alternates_are_tried_in_order_until_one_is_a_pdf(tmp_path, replay):
    work, folder, env = setup(tmp_path)
    urls = [
        f"{replay.url}/publisher.pdf",
        f"{replay.url}/repo-a.pdf",
        f"{replay.url}/repo-b.pdf",
    ]
    write_rows(
        folder / "openalex_candidates.csv",
        [
            {
                "paperid": "multi-source",
                "doi": "10.1234/multi",
                "pdf_url": urls[0],
                "alternate_pdf_urls": json.dumps(urls[1:]),
            }
        ],
        OPENALEX_FIELDS,
    )
    replay.answer("GET /publisher.pdf", html(b"<html>paywall</html>", 403))
    replay.answer("GET /repo-a.pdf", html(b"<html>landing</html>"))
    replay.answer("GET /repo-b.pdf", pdf(b"%PDF-1.4 archived"))

    done = download(work, env, "--candidates", str(folder / "openalex_candidates.csv"))

    assert done.returncode == 0, done.stderr
    assert counts(done)["downloaded"] == 1
    assert [r.path for r in replay.requests] == [
        "/publisher.pdf",
        "/repo-a.pdf",
        "/repo-b.pdf",
    ]
    [status] = read_rows(folder / "pdf_download_status.csv")
    assert (status["pdf_url"], status["downloaded"]) == (urls[2], "True")
    with zipfile.ZipFile(folder / "pdf_batches/pdf_batch_000001.zip") as archive:
        assert archive.namelist() == ["mu/multi-source.pdf"]


def test_missing_candidates_are_refused_with_where_they_come_from(tmp_path):
    work, folder, env = setup(tmp_path)

    done = download(work, env)

    assert done.returncode == 1
    assert done.stderr.strip() == (
        f"The candidates file {folder / 'open_access_candidates.csv'} does not exist. "
        "`collect open-access` or `collect openalex` writes it (docs/step-1-2-papers/collect.md)."
    )
    assert not folder.exists()


def test_the_pinned_defaults_are_the_released_ones(tmp_path):
    done = run_cli(tmp_path, "collect", "download", "--help")

    assert done.returncode == 0
    text = " ".join(done.stdout.split())
    for default in (
        "parallel downloads (default: 4)",
        "the body (default: 45.0)",
        "in MiB (default: 80.0)",
        "PDFs per zip (default: 5000)",
    ):
        assert default in text


# What the released closure did after a failed download -----------------------------------------
# A landing page that declares its PDF (`citation_pdf_url`) is followed, and a PubMed Central paper is fetched from Europe PMC's rendering; Europe PMC is asked at most once every 3 seconds.


def one_candidate(folder, url):
    write_rows(
        folder / "open_access_candidates.csv",
        [{"paperid": "recovered", "doi": "10.1234/r", "pdf_url": url}],
        CANDIDATE_FIELDS,
    )


def test_a_landing_page_that_declares_its_pdf_is_followed(tmp_path, replay):
    work, folder, env = setup(tmp_path)
    one_candidate(folder, f"{replay.url}/article/1")
    page = b'<html><meta name="citation_pdf_url" content="/files/1.pdf"></html>'
    replay.answer("GET /article/1", html(page), html(page))
    replay.answer("GET /files/1.pdf", pdf(b"%PDF-1.4 the paper"))

    done = download(work, env)

    assert done.returncode == 0, done.stderr
    assert counts(done)["downloaded"] == 1
    assert replay.misses == []
    [status] = read_rows(folder / "pdf_download_status.csv")
    assert (status["pdf_url"], status["downloaded"]) == (
        f"{replay.url}/files/1.pdf",
        "True",
    )


def test_a_pubmed_central_paper_comes_from_europe_pmcs_rendering(tmp_path, replay):
    work, folder, env = setup(tmp_path)
    one_candidate(folder, f"{replay.url}/pmc/articles/PMC1234567/pdf/")
    replay.answer(
        "GET /pmc/articles/PMC1234567/pdf/", html(b"checking your browser", 403)
    )
    replay.answer("GET /epmc/articles/PMC1234567", pdf(b"%PDF-1.4 rendered"))

    done = download(work, env, "--europe-pmc-url", f"{replay.url}/epmc")

    assert done.returncode == 0, done.stderr
    assert counts(done)["downloaded"] == 1
    [asked] = replay.asked("GET /epmc/articles/PMC1234567")
    assert asked.query == {"pdf": "render"}


def test_a_failed_recovery_is_named_in_the_error(tmp_path, replay):
    work, folder, env = setup(tmp_path)
    one_candidate(folder, f"{replay.url}/pmc/articles/PMC7654321/pdf/")
    replay.answer("GET /pmc/articles/PMC7654321/pdf/", html(b"no", 403))
    replay.answer("GET /epmc/articles/PMC7654321", html(b"gone", 404))

    done = download(work, env, "--europe-pmc-url", f"{replay.url}/epmc")

    assert done.returncode == 0, done.stderr
    [status] = read_rows(folder / "pdf_download_status.csv")
    assert status["error"] == (
        f"download_status_403 | recovery: {replay.url}/epmc/articles/PMC7654321?pdf=render: "
        "download_status_404"
    )


def test_europe_pmc_is_asked_once_every_3_seconds_and_waits_after_a_429():
    from mbdef.collect import download as d

    now, slept = [100.0], []

    def clock():
        return now[0]

    def sleep(seconds):
        slept.append(seconds)
        now[0] += seconds

    pace = d.Pacing(d.PACED_HOSTS, clock=clock, sleep=sleep)
    for host in (
        "europepmc.org",
        "europepmc.org",
        "publisher.example",
        "europepmc.org",
    ):
        pace.wait(host)
    pace.after("europepmc.org", 429, "")
    pace.wait("europepmc.org")

    assert slept == [0.0, 3.0, 3.0, 60.0]
