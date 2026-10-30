"""`mbdef extract train-dataset`: the prompt-training dataset and its split.

The released files are `data/frozen/results/training-20260925/runner/{train-dataset,split}.json` (the inputs of the released training runs, which `extract train` reads). Rebuilding them needs the ground-truth papers' TEIs, which hold paper text and are not part of this repository. A miniature with synthetic papers checks each field and the refusals.
"""

import hashlib
import json
from pathlib import Path

from cli_helpers import run_cli, write_env


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# A miniature with synthetic text: three TRAIN papers, one DEV paper, and one TEST paper in each of the three ground-truth files.
TEI = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc><titleStmt>
<title level="a">{title}</title></titleStmt></fileDesc><profileDesc><abstract>
<p>{abstract}</p></abstract></profileDesc></teiHeader>
<text><body><div><p>{body}</p></div></body></text></TEI>
"""


def mini_inputs(root: Path) -> dict[str, Path]:
    def paper(paper_id, split, title=None, records=()):
        return {
            "paper_id": paper_id,
            "split": split,
            "metadata": {"title": title} if title else {},
            "definition_records": list(records),
        }

    record = {
        "term": "synthetic term",
        "definitions": [{"text": "A made-up definition."}],
    }
    ground_truth = [
        paper("train-b", "train", "Paper B", [record]),
        paper("dev-a", "dev", "Paper A"),
        paper("train-a", "train", "Paper A2"),
        paper("doi:10.1/x", "train", "Paper X"),
        paper("test-1", "original_test"),
    ]
    non_relevant = [paper("train-b", "train"), paper("test-2", "new_test")]
    # The metadata file has no entry for doi:10.1/x, so that paper takes its TEI's title.
    metadata = [
        paper("train-b", "train", "Paper B"),
        paper("dev-a", "dev", "Paper A"),
        paper("train-a", "train", "Paper A2"),
        paper("test-3", "test"),
    ]
    paths = {}
    for name, papers in (
        ("ground_truth", ground_truth),
        ("non_relevant", non_relevant),
        ("metadata", metadata),
    ):
        paths[name] = root / f"{name}.json"
        paths[name].write_text(json.dumps({"papers": papers}), encoding="utf-8")
    teis = root / "teis"
    teis.mkdir()
    for paper_id, file_name in (
        ("train-b", "train-b"),
        ("dev-a", "dev-a"),
        ("train-a", "train-a"),
        ("doi:10.1/x", "doi%3A10.1%2Fx"),
    ):
        (teis / f"{file_name}.tei.xml").write_text(
            TEI.format(
                title=f"TEI title of {paper_id}",
                abstract=f"An abstract of {paper_id}.",
                body="A body paragraph.",
            ),
            encoding="utf-8",
        )
    paths["teis"] = teis
    paths["plan"] = root / "plan.json"
    paths["plan"].write_text('{"schema": "a plan"}\n', encoding="utf-8")
    plan_sha256 = hashlib.sha256(paths["plan"].read_bytes()).hexdigest()
    paths["amendment"] = root / "amendment.json"
    paths["amendment"].write_text(
        json.dumps({"amends_plan_sha256": plan_sha256}), encoding="utf-8"
    )
    return paths


def build(tmp_path: Path, paths: dict[str, Path]):
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    write_env(work, DATA_ROOT=str(tmp_path / "data-root"))
    options = [f"--{key.replace('_', '-')}={value}" for key, value in paths.items()]
    return run_cli(
        work, "extract", "train-dataset", "--out", str(tmp_path / "out"), *options
    )


def test_a_miniature_ground_truth_gives_each_paper_its_tei_text_windows_and_references(
    tmp_path,
):
    paths = mini_inputs(tmp_path)
    result = build(tmp_path, paths)
    assert result.returncode == 0, result.stderr
    dataset = read(tmp_path / "out/train-dataset.json")
    split = read(tmp_path / "out/split.json")
    papers = dataset["papers"]
    # The ground truth's order, TRAIN and DEV only.
    assert [p["paper_id"] for p in papers] == [
        "train-b",
        "dev-a",
        "train-a",
        "doi:10.1/x",
    ]
    assert [p["position"] for p in papers] == [0, 1, 2, 3]
    first = papers[0]
    tei = (paths["teis"] / "train-b.tei.xml").read_bytes()
    text = "An abstract of train-b.\n\nA body paragraph."  # abstract, then body, one blank line
    assert first == {
        "paper_id": "train-b",
        "position": 0,
        "split": "train",
        "title": "Paper B",
        "local_tei": "papers/train-b.tei.xml",
        "tei_file": "train-b.tei.xml",
        "tei_sha256": hashlib.sha256(tei).hexdigest(),
        "tei_bytes": len(tei),
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text_chars": len(text),
        "windows": {"w8k": 1, "w16k": 1, "w32k": 1},
        "gold_records": [
            {
                "term": "synthetic term",
                "definitions": [{"text": "A made-up definition."}],
            }
        ],
    }
    # A paper missing from the metadata file takes its TEI's title; its file name is quoted.
    assert papers[3]["title"] == "TEI title of doi:10.1/x"
    assert papers[3]["tei_file"] == "doi%3A10.1%2Fx.tei.xml"
    assert dataset["contents"].startswith(
        "3 TRAIN + 1 DEV papers (no TEST); gold from "
    )
    plan_sha256 = hashlib.sha256(paths["plan"].read_bytes()).hexdigest()
    amendment_sha256 = hashlib.sha256(paths["amendment"].read_bytes()).hexdigest()
    assert (dataset["plan_sha256"], dataset["amendment_sha256"]) == (
        plan_sha256,
        amendment_sha256,
    )
    assert split["trainset"] == ["doi:10.1/x", "train-a", "train-b"]
    assert split["valset"] == ["dev-a"]
    # TEST papers of all three files are refused by id; nothing else of them is in the files.
    assert split["forbidden_test_ids"] == ["test-1", "test-2", "test-3"]
    assert split["forbidden_sources"] == [
        str(paths[k]) for k in ("ground_truth", "non_relevant", "metadata")
    ]


def test_a_missing_tei_stops_the_build_and_writes_nothing(tmp_path):
    paths = mini_inputs(tmp_path)
    (paths["teis"] / "dev-a.tei.xml").unlink()
    result = build(tmp_path, paths)
    assert result.returncode == 1
    assert "has no TEI for: dev-a" in result.stderr
    assert not (tmp_path / "out").exists()


def test_a_paper_whose_split_differs_between_the_files_is_refused(tmp_path):
    paths = mini_inputs(tmp_path)
    metadata = read(paths["metadata"])
    metadata["papers"][2]["split"] = "test"  # train-a is TRAIN in the ground truth
    paths["metadata"].write_text(json.dumps(metadata), encoding="utf-8")
    result = build(tmp_path, paths)
    assert result.returncode == 1
    assert "Paper train-a is test in" in result.stderr


def test_an_amendment_of_another_plan_is_refused(tmp_path):
    paths = mini_inputs(tmp_path)
    paths["amendment"].write_text('{"amends_plan_sha256": "0"}', encoding="utf-8")
    result = build(tmp_path, paths)
    assert result.returncode == 1
    assert "does not amend" in result.stderr
