"""`closure run`: the one-level closure of a set of seed papers, up to the queue of works to collect.

From the seed papers' extracted definitions (a dataset of papers whose items carry citation strings) and their TEI files, the run

1. resolves every citation string to a reference-list entry of its paper (`extend_gt.extend`);
2. decides, for every cited work, whether a paper with that identity is already in the corpus roots (`corpus_check`): present, missing, or `review` (it could be two different works);
3. writes the works still `missing` or `review` as `before-queue.csv`, the collection queue of `mbdef collect` (docs/step-1-2-papers/collect.md); and, when `--collect-folder` holds TEI files that `collect grobid` kept, checks again with them as one more corpus root (`after*`), so the summary says what is left to collect.

So the closure run alternates with the collect step: run it, collect the queue, run it again with the collect folder. Downloaded papers never enter the seed list: depth is one. Reruns with the same inputs resume in the same output folder; changed inputs need a new one. Outputs have no timestamps.
"""

import fcntl
import hashlib
import json
from collections import Counter
from pathlib import Path

from mbdef import files
from mbdef.closure import RESOLVER
from mbdef.closure.corpus_check import (
    CHECKER,
    _validate_dataset,
    check_citation_corpus,
)
from mbdef.closure.extend_gt import extend
from mbdef.closure.tei_sources import Sources
from mbdef.collect.grobid import TEI_FOLDER

WORKERS = 4  # corpus files scanned at the same time


def run_closure(
    *,
    dataset_path: Path,
    output_dir: Path,
    corpus_paths: list[Path],
    tei_dirs: tuple[Path, ...] = (),
    source_zip: Path | None = None,
    collect_folder: Path | None = None,
    workers: int = WORKERS,
) -> dict:
    """Resolve and check one seed set; return the summary (also written to `summary.json`)."""
    if workers < 1:
        raise ValueError("workers must be positive")
    if not corpus_paths:
        raise ValueError("At least one explicit corpus root is required")
    dataset_path, output_dir = Path(dataset_path).resolve(), Path(output_dir).resolve()
    if dataset_path.is_relative_to(output_dir):
        raise ValueError("Seed dataset must be outside the output directory")
    payload = dataset_path.read_bytes()
    dataset = json.loads(payload)
    papers = _validate_dataset(dataset)
    manifest = {
        "schema": "corpus-closure-run/1",
        "depth": 1,
        "input_sha256": hashlib.sha256(payload).hexdigest(),
        "seed_ids": sorted(paper["paper_id"] for paper in papers),
        "tei_dirs": [str(Path(path).resolve()) for path in tei_dirs],
        "source_zip": str(source_zip.resolve()) if source_zip else None,
        "corpus_paths": [str(Path(path).resolve()) for path in corpus_paths],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another closure run owns this output directory"
            ) from exc
        manifest_path = output_dir / "seeds.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError(
                "Seed input or corpus configuration changed; use a new output directory"
            )
        files.write_atomic({manifest_path: files.json_text(manifest)})
        sources = Sources(
            archive=output_dir / "unused-archive",
            grobid_files=output_dir / "unused-grobid",
            source_zip=source_zip or output_dir / "unused.zip",
            cache=output_dir / "tei-cache",
            directories=tei_dirs,
        )
        resolved, links, unresolved, _ = extend(dataset, sources)
        links["ground_truth"] = str(dataset_path)
        files.write_atomic(
            {
                output_dir / "citations.json": files.json_text(resolved),
                output_dir / "links.json": files.json_text(links),
                output_dir / "unresolved.json": files.json_text(unresolved),
            }
        )

        def check(label, roots):
            return check_citation_corpus(
                output_dir / "citations.json",
                roots,
                output_dir / f"{label}.json",
                output_dir / f"{label}-queue.csv",
                output_dir / f"{label}-citations.json",
                workers=workers,
            )

        before = check("before", list(corpus_paths))
        collected = Path(collect_folder) / TEI_FOLDER if collect_folder else None
        after = None
        if collected and collected.exists() and any(collected.rglob("*.xml")):
            after = check("after", [*corpus_paths, collected])
        last = after or before
        result = {
            "schema": "corpus-closure-run/2",
            "input_sha256": manifest["input_sha256"],
            "seed_count": len(papers),
            "depth": 1,
            "markers": links["markers"],
            "linked_markers": links["linked_markers"],
            "seed_source_failures": len(links["papers_without_tei"]),
            "before": before["summary"],
            "after": after["summary"] if after else None,
            "remaining_reasons": dict(
                Counter(
                    paper["corpus_match"]["reason"] or paper["corpus_match"]["status"]
                    for paper in last["papers"]
                    if paper["corpus_match"]["status"] != "present"
                )
            ),
        }
        result["operational_failure"] = bool(
            result["seed_source_failures"]
            or any(source["error_count"] for source in last["sources"])
        )
        files.write_atomic({output_dir / "summary.json": files.json_text(result)})
        return result


def command(
    dataset: str,
    out: str,
    corpus_paths: list[str],
    tei_dirs: list[str],
    source_zip: str | None,
    collect_folder: str | None,
    workers: int,
) -> int:
    """`closure run`: resolve, check against the corpus, write the queue; print the summary."""
    result = run_closure(
        dataset_path=Path(dataset),
        output_dir=Path(out),
        corpus_paths=[Path(p) for p in corpus_paths],
        tei_dirs=tuple(Path(p) for p in tei_dirs),
        source_zip=Path(source_zip) if source_zip else None,
        collect_folder=Path(collect_folder) if collect_folder else None,
        workers=workers,
    )
    folder = Path(out)
    inputs = {"dataset": Path(dataset)}
    if collect_folder:
        inputs["collected_tei"] = Path(collect_folder) / TEI_FOLDER
    receipt = files.receipt(
        "mbdef closure run",
        {
            "resolver": RESOLVER,
            "checker": CHECKER,
            "depth": 1,
            "corpus_paths": corpus_paths,
            "tei_dirs": tei_dirs,
            "source_zip": source_zip,
            "collect_folder": collect_folder,
            "workers": workers,
        },
        {k: v for k, v in inputs.items() if v.exists()},
        {
            name: folder / name
            for name in (
                "citations.json",
                "links.json",
                "before-queue.csv",
                "summary.json",
            )
        },
    )
    receipt["summary"] = result
    files.write_json(folder / files.RECEIPT, receipt)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"queue for `collect`: {folder / 'before-queue.csv'}")
    if result["operational_failure"]:
        raise SystemExit(
            "Some seed papers have no TEI or some corpus files could not be read; see summary.json."
        )
    return 0
