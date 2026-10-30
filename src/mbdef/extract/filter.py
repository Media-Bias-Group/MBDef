"""`mbdef extract filter`: the frozen relevance filter on every paper of a run.

The input is a folder with `papers/<sha256 of the paper id>/union.json`: an extraction run (one part of `extract run`) or a verifier run (`extract verify`, which also holds `verified.json`). The filter reads each paper's title from `union.json` and filters either the verified records (`--records verified`, the default) or the union's records (`--records union`, the released "no verifier" variants). `relevance.py` says what the filter does. The defaults are the released ones: the `ordinary` filter at its frozen cutoff. BGE runs on `SCIDEF_NLI_DEVICE` from `.env` (`cpu`, `cuda` or `auto`), as the scoring models do. Everything goes into `--out` (new or empty):

    papers/<sha256 of the paper id>/filtered.json   the kept records, every record's probability,
                                                    the records over the encoder's capacity
    encoder/encoded.json, encoder/vectors.npy       every text's token count and BGE vector, a
                                                    recording that `--replay` answers from
    predictions.jsonl                               the kept records of every paper (records.py)
    receipt.json                                    settings, input hashes, commit

With `--replay FOLDER` (a recorded `encoder/` folder; repeatable) no model is loaded, no `.env` key is read, and a text that was not recorded stops the run. The per-paper files have the released runs' JSON format.
"""

import json
from pathlib import Path

from mbdef import files
from mbdef.extract.records import PREDICTIONS, write_predictions

NAMES = ("ordinary", "title", "title-blocks")  # the keys of relevance.FILTERS
FILTER = NAMES[0]
RECORDS = ("verified", "union")  # the first is the default
JSON_FORMAT = {"indent": 2, "sort_keys": True}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def filter_papers(
    run: Path,
    out: Path,
    name: str = FILTER,
    records: str = RECORDS[0],
    cutoff: float | None = None,
    bge: Path | None = None,
    replay: list[Path] | None = None,
    filters: Path | None = None,
) -> int:
    """The step (module docstring). Returns 0."""
    # NumPy, scikit-learn and Torch are imported here, not when the command line starts.
    from mbdef.extract import relevance

    run, out = Path(run), Path(out)
    filters = Path(filters) if filters else relevance.ZIPS
    if name not in relevance.FILTERS:
        raise SystemExit(
            f"Unknown filter {name}; use one of {', '.join(relevance.FILTERS)}."
        )
    if records not in RECORDS:
        raise SystemExit(f"--records is one of {', '.join(RECORDS)}, not {records}.")
    paper_dirs = sorted(p.parent for p in (run / "papers").glob("*/union.json"))
    if not paper_dirs:
        raise SystemExit(f"{run} has no papers/*/union.json to filter.")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; give a new or empty folder.")
    loaded = relevance.load_filter(name, filters)
    if replay:
        bge, encoder = None, relevance.Encoder(replay=relevance.recorded(replay))
    else:
        bge = Path(bge) if bge else relevance.bge_folder()
        encoder = relevance.Encoder(bge, files.env("SCIDEF_NLI_DEVICE"))
    summary: dict = dict.fromkeys(
        ("papers", "records_before_filter", "kept_records", "kept_definitions"), 0
    )
    summary["capacity_excluded_records"] = 0
    kept_by_paper = []
    for paper_dir in paper_dirs:
        union = read(paper_dir / "union.json")
        if records == "union":
            given = union["records"]
        elif (paper_dir / "verified.json").is_file():
            given = read(paper_dir / "verified.json")["definition_records"]
        else:
            raise SystemExit(
                f"{paper_dir} has no verified.json; verify the run first "
                "(extract verify) or filter the union (--records union)."
            )
        result = relevance.apply_filter(encoder, loaded, union["title"], given, cutoff)
        files.write_json(
            out / "papers" / paper_dir.name / "filtered.json",
            {"paper_id": union["paper_id"], "title": union["title"], **result},
            **JSON_FORMAT,
        )
        kept = result["definition_records"]
        kept_by_paper.append((union["paper_id"], kept))
        summary["papers"] += 1
        summary["records_before_filter"] += len(given)
        summary["kept_records"] += len(kept)
        summary["kept_definitions"] += sum(len(r["definitions"]) for r in kept)
        summary["capacity_excluded_records"] += len(
            result["capacity_excluded_record_indices"]
        )
    encoder.save(out / "encoder")
    summary[PREDICTIONS] = write_predictions(out / PREDICTIONS, kept_by_paper)
    inputs = {"run": run, "filter": filters / f"{loaded['study']}.zip"}
    if replay:
        for number, folder in enumerate(replay, 1):
            inputs[f"replay:{number}"] = Path(folder)
    elif bge is not None:
        for file in relevance.ENCODER_FILES:
            inputs[f"encoder:{file}"] = bge / file
    settings = {
        "filter": name,
        "filter_row": loaded["row"],
        "study": loaded["study"],
        "cutoff": loaded["cutoff"] if cutoff is None else cutoff,
        "frozen_cutoff": loaded["cutoff"],
        "records": records,
        "encoder": f"{relevance.ENCODER[0]} at {relevance.ENCODER[1]}",
        "encoder_input": "replay of recorded vectors"
        if replay
        else f"computed on {encoder.device}",
        "capacity_tokens": relevance.CAPACITY,
        "batch": relevance.BATCH,
    }
    receipt = files.receipt(
        "mbdef extract filter",
        settings,
        inputs,
        {
            "papers": out / "papers",
            "encoder": out / "encoder",
            "predictions": out / PREDICTIONS,
        },
    )
    receipt["summary"] = summary
    files.write_json(out / files.RECEIPT, receipt)
    print(files.json_text(summary), end="")
    return 0
