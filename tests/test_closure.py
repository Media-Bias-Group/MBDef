"""The end-to-end test of the closure step: a two-paper closure from the seed papers to the collection queue and back.

Two seed papers cite one work between them, a child nobody has collected; one of them also cites the other. `closure run` finds the child missing and queues it; `collect grobid` (GROBID is a fake on a local port, nothing else is reachable) converts the child's PDF; `closure run` again, with the collect folder, finds nothing left to collect. The test doubles as a worked example of the step (docs/step-4-closure/closure.md).
"""

import csv
import json

from cli_helpers import run_cli, write_env
from http_replay import NO_NETWORK, Answer, Replay
from mini_closure import QUEUE_FIELDS, two_seed_papers
from mini_collect import downloads, fake_grobid, head, sha, tei


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_a_two_paper_closure_queues_the_missing_work_and_finds_it_collected(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    write_env(work, DATA_ROOT=str(tmp_path / "data-root"))
    dataset, seed_tei = two_seed_papers(tmp_path / "seeds")
    out = tmp_path / "closure"

    first = run_cli(
        work,
        "closure",
        "run",
        "--dataset",
        str(dataset),
        "--out",
        str(out),
        "--corpus",
        str(seed_tei),
        "--tei-dir",
        str(seed_tei),
        env=NO_NETWORK,
    )

    assert first.returncode == 0, first.stderr
    summary = json.loads((out / "summary.json").read_text())
    # Three citation markers resolved: the child twice (once per seed), seed-a once (from seed-b).
    assert (summary["markers"], summary["linked_markers"]) == (3, 3)
    # Two cited works: the child (missing) and seed-a (a seed paper, so present).
    assert summary["before"] == {
        "reference_occurrences": 3,
        "resolved_markers_without_references": 0,
        "unique_cited_papers": 2,
        "present": 1,
        "missing": 1,
        "review": 0,
    }
    assert summary["after"] is None and not summary["operational_failure"]
    queue = read_rows(out / "before-queue.csv")
    assert [(r["queue_status"], r["title"], r["doi"]) for r in queue] == [
        ("missing", "Child work", "10.9876/child")
    ]
    assert queue[0]["citing_paper_ids"] == "seed-a; seed-b"
    assert queue[0]["occurrence_count"] == "2"
    assert list(queue[0]) == QUEUE_FIELDS

    # The receipt names the command, the pinned settings, the commit and the dataset's hash.
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["command"] == "mbdef closure run"
    assert receipt["code"]["commit"] == head()
    assert receipt["settings"]["depth"] == 1 and receipt["settings"]["workers"] == 4
    assert receipt["inputs"]["dataset"]["sha256"] == sha(dataset)
    assert receipt["summary"]["before"]["missing"] == 1

    # `collect` takes the queue: the child's PDF is converted and its title page shows the same DOI.
    folder = tmp_path / "collect"
    paperid = "doi_10.9876_child__1"
    pdf = b"%PDF-1.7 child work"
    downloads(
        folder,
        {
            paperid: {
                "doi": "10.9876/child",
                "title": "Child work",
                "authors": "Author",
                "year": "2020",
            }
        },
        {paperid: pdf},
    )
    child = tei(
        title="Child work", surnames=("Author",), year="2020", doi="10.9876/child"
    )
    with Replay() as replay:
        replay.answer("GET /api/isalive", Answer(body="true"))
        replay.answer("GET /api/version", Answer(body="0.9.1"))
        replay.answer(
            "POST /api/processFulltextDocument", fake_grobid({paperid: child})
        )
        grobid = run_cli(
            work,
            "collect",
            "grobid",
            "--folder",
            str(folder),
            "--grobid-url",
            replay.url,
            env=NO_NETWORK,
        )
    assert grobid.returncode == 0, grobid.stderr
    assert replay.misses == []

    second = run_cli(
        work,
        "closure",
        "run",
        "--dataset",
        str(dataset),
        "--out",
        str(out),
        "--corpus",
        str(seed_tei),
        "--tei-dir",
        str(seed_tei),
        "--collect-folder",
        str(folder),
        env=NO_NETWORK,
    )

    assert second.returncode == 0, second.stderr
    summary = json.loads((out / "summary.json").read_text())
    assert summary["before"]["missing"] == 1
    assert (summary["after"]["present"], summary["after"]["missing"]) == (2, 0)
    assert summary["remaining_reasons"] == {}
    assert read_rows(out / "after-queue.csv") == []
