"""`mbdef explorer data` on a miniature MBDef dataset: page data for the released mentions only.

The miniature's earlier page data holds three term mentions: "Agenda" in paper A, which the relevance filter dropped and the tables do not hold, then "framing" in paper A and "Framing" in paper B, the one concept "framing".
Because the dropped mention comes first, every index of the released ones moves (mention 1 becomes 0, label group 1 becomes 0), which the tests follow through every file.
The map and topics are replaced by a stand-in: UMAP and k-means need far more than two points.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from test_explorer import sha256

from mbdef import explorer, explorer_data
from mbdef.dataset.schema import frame

TEXTS = {  # each mention's definitions
    "m0": ["Agenda is a list of items."],
    "m1": ["Framing selects some aspects of a perceived reality."],
    "m2": ["Framing makes some aspects more salient.", "A frame is a schema."],
}
TERMS = {"m0": "Agenda", "m1": "framing", "m2": "Framing"}
# a 4-dimensional stand-in for each definition's embedding vector
VECTORS = {
    "Agenda is a list of items.": [0.0, 0.0, 1.0, 0.0],
    "Framing selects some aspects of a perceived reality.": [1.0, 0.0, 0.0, 0.0],
    "Framing makes some aspects more salient.": [0.8, 0.6, 0.0, 0.0],
    "A frame is a schema.": [0.6, 0.0, 0.0, 0.8],
}


def mention_vector(m: str) -> np.ndarray:
    v = np.sum([np.array(VECTORS[t], np.float32) for t in TEXTS[m]], axis=0)
    return v / np.linalg.norm(v)


COS = float(mention_vector("m1") @ mention_vector("m2"))


def write_cache(folder: Path, vectors: dict[str, list[float]] = VECTORS) -> Path:
    """An embedding cache in the layout of embed.py: L2-normalised float16 rows and the sha1 of each text."""
    folder.mkdir(parents=True)
    texts = [(m, t) for m in ("m1", "m2") for t in TEXTS[m]]
    rows = np.array([vectors[t] for _, t in texts], np.float32)
    np.save(
        folder / "vectors.npy",
        (rows / np.linalg.norm(rows, axis=1, keepdims=True)).astype(np.float16),
    )
    keys = [hashlib.sha1(f"{TERMS[m]}: {t}".encode()).hexdigest() for m, t in texts]
    (folder / "keys.txt").write_text("\n".join(keys))
    return folder


def page_data() -> dict:
    """The earlier page data: all three mentions, one neighbour each, the dropped mention's label group first."""
    merged = [
        "media effects",
        2,
        -1,
        [],
        [1, 2],
    ]  # a supertype both papers state, from mentions 1 and 2
    return {
        "mentions/00.json": {
            "n": 3,
            "label": ["Agenda", "framing", "Framing"],
            "paper": [0, 0, 1],
            "group": [0, 1, 1],
            "ndef": [1, 1, 2],
            "c": [1000, 9000, 8000],
            "b": [1000, 9500, 7000],
            "j": [500, 10000, 6000],
            "s": [833, 10000, 7500],
            "keep": [0, 1, 1],
            "strict": [0, 1, 1],
            "xyz": [0] * 9,
            "xy": [0] * 6,
            "nb": [1, 2, 1],
            "nbs": [100, round(COS * 1000), round(COS * 1000)],
            "steps": [0, 0, 1],
            "mention_id": ["m0", "m1", "m2"],
            "topic": [5, 7, 7],
        },
        "defs/00.json": [TEXTS[m] for m in ("m0", "m1", "m2")],
        "fields/00.json": [
            {},
            {"sp": [["media effects", []]]},
            {"sp": [["media effects", []]]},
        ],
        "terms/00.json": [None, {"nd": 3, "df": [], "sp": [merged]}],
        "groups.json": {"name": ["Agenda", "Framing"], "cnb": [-1, -1], "cns": [0, 0]},
        "papers.json": {
            "id": ["paper-a", "paper-b"],
            "title": ["Framing: Toward Clarification", "Media frames"],
            "corpus": ["seeds", "closure"],
        },
        "merges.json": {
            "1": {
                "labels": [
                    ["framing", "framing", "framing", [], 1],
                    ["Framing", "framing", "framing", ["case-folded"], 1],
                ],
                "joined_by": [],
            }
        },
        "relations.json": {"fields": ["sy", "rt", "sb", "sp"], "edges": []},
        "ensemble-ref.json": {
            "n_ref": 2,
            "ref": {"c": [0.1, 0.9], "b": [0.1, 0.9], "j": [0.1, 0.9]},
            "weights": {"classifier": 1.0, "basal": 1.0, "judge": 1.0},
            "cut": 0.415,
            "best_f1_cut": 0.585,
            "ranks": {"c": [0, 4, 3], "b": [0, 4, 3], "j": [1, 4, 3]},
            "ids_sha256": "",
            "checks": {
                "page_recompute": {
                    "keep_flips_from_rounded_raw": 0,
                    "reference_ties": {"basal": 0},
                }
            },
        },
        "meta.json": {
            "cut": 0.415,
            "best_f1_cut": 0.585,
            "weights": {"classifier": 1.0, "basal": 1.0, "judge": 1.0},
            "counts": {"terms": 3, "kept": 2, "excluded": 1},
            "shard": 3,
            "concepts": {"concepts": 1, "single_term_concepts": 0},
            "fields": {
                "codes": {"supertypes": "sp"},
                "names": ["sp"],
                "counts": {"sp": [3, 3]},
                "terms_shard": 2000,
                "relations": 0,
                "note": "",
            },
            "provenance": {"commit": "bcee3cc"},
        },
    }


def write_dataset(folder: Path, page: dict) -> Path:
    """A miniature MBDef dataset: its tables in data/ (the two released mentions) and the page data in explorer/."""
    data = folder / "data"
    data.mkdir(parents=True)

    def score(c, b, j, s):
        return dict(classifier_score=c, basal_score=b, judge_score=j, ensemble_score=s)

    tables = {
        "instances": [
            dict(
                mention_id="m1",
                concept_id="g1",
                term="framing",
                paper_id="paper-a",
                strict=True,
                **score(0.9, 0.95, 1.0, 1.0),
            ),
            dict(
                mention_id="m2",
                concept_id="g1",
                term="Framing",
                paper_id="paper-b",
                strict=True,
                **score(0.8, 0.7, 0.6, 0.75),
            ),
        ],
        "definitions": [
            dict(def_id=f"o{k}", mention_id=m, concept_id="g1", term=TERMS[m], text=t)
            for k, (m, t) in enumerate((m, t) for m in ("m1", "m2") for t in TEXTS[m])
        ],
        "concepts": [
            dict(
                concept_id="g1",
                label="framing",
                labels=["framing", "Framing"],
                mention_ids=["m1", "m2"],
                joined_by=[
                    dict(
                        key_type="word-set",
                        key="framing",
                        labels=["Framing", "framing"],
                    )
                ],
            )
        ],
        "papers": [
            dict(
                paper_id="paper-a", title="Framing: Toward Clarification", n_instances=1
            ),
            dict(paper_id="paper-b", title="Media frames", n_instances=1),
        ],
    }
    for name, rows in tables.items():
        frame(rows, name).write_parquet(data / f"{name}.parquet")
    bundle = folder / "explorer"
    sums = []
    for name, content in sorted(page.items()):
        (bundle / name).parent.mkdir(parents=True, exist_ok=True)
        (bundle / name).write_bytes(json.dumps(content, ensure_ascii=False).encode())
        sums.append(f"{sha256((bundle / name).read_bytes())}  {name}\n")
    (bundle / "SHA256SUMS").write_text("".join(sums))
    return folder


@pytest.fixture
def small(monkeypatch):
    """One neighbour per mention, one topic, and a stand-in map: row i at (3i, 3i+1, 3i+2) and (2i, 2i+1)."""
    monkeypatch.setattr(explorer_data, "NEIGHBOURS", 1)
    monkeypatch.setattr(explorer_data, "TOPICS", 1)
    monkeypatch.setattr(
        explorer_data,
        "layout",
        lambda X: (
            np.arange(3 * len(X)).reshape(-1, 3),
            np.arange(2 * len(X)).reshape(-1, 2),
            np.zeros(len(X), int),
        ),
    )


def read(folder: Path) -> dict:
    return {
        p.relative_to(folder).as_posix(): p.read_bytes()
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


def test_only_the_released_mentions_remain_with_every_index_renumbered(
    tmp_path, small, capsys
):
    dataset = write_dataset(tmp_path / "MBDef", page_data())
    cache = write_cache(tmp_path / "cache")

    assert explorer_data.released(dataset, cache, tmp_path / "out") == 0

    out = {
        name: json.loads(data)
        for name, data in read(tmp_path / "out").items()
        if name.endswith(".json")
    }
    m = out["mentions/00.json"]
    assert m["mention_id"] == ["m1", "m2"] and m["n"] == 2 and "keep" not in m
    assert (m["group"], m["paper"], m["nb"], m["topic"]) == (
        [0, 0],
        [0, 1],
        [1, 0],
        [0, 0],
    )
    assert m["nbs"] == [round(COS * 1000)] * 2
    assert (m["xyz"], m["xy"]) == ([0, 1, 2, 3, 4, 5], [0, 1, 2, 3])
    assert out["defs/00.json"] == [TEXTS["m1"], TEXTS["m2"]]
    assert out["fields/00.json"] == [{"sp": [["media effects", []]]}] * 2
    assert out["terms/00.json"] == [
        {"nd": 3, "df": [], "sp": [["media effects", 2, -1, [], [0, 1]]]}
    ]
    groups = out["groups.json"]
    assert groups["name"] == [
        "framing"
    ]  # the dataset's concept label, not the page's old "Framing"
    assert (groups["n"], groups["papers"], groups["ndef"], groups["far"]) == (
        [2],
        [2],
        [3],
        [0],
    )
    assert groups["labels"] == [{"framing": 1, "Framing": 1}] and "kept" not in groups
    assert groups["pair"] == [round(COS * 1000)]
    assert out["merges.json"] == {
        "0": {
            "labels": [
                ["framing", "framing", "framing", [], 1],
                ["Framing", "framing", "framing", ["case-folded"], 1],
            ],
            "joined_by": [["word-set", "framing", ["Framing", "framing"]]],
        }
    }
    assert out["papers.json"] == {
        "id": ["paper-a", "paper-b"],
        "title": ["Framing: Toward Clarification", "Media frames"],
        "corpus": ["seeds", "closure"],
        "n": [1, 1],
        "ndef": [1, 2],
    }
    ref = out["ensemble-ref.json"]
    assert ref["ranks"] == {"c": [4, 3], "b": [4, 3], "j": [4, 3]}
    assert ref["ids_sha256"] == hashlib.sha256(b"m1\nm2").hexdigest()
    assert ref["checks"]["page_recompute"] == {"reference_ties": {"basal": 0}}
    assert (
        ref["checks"]["released"]["mentions"] == 2
        and ref["checks"]["released"]["below_cut"] == 0
    )
    meta = out["meta.json"]
    assert meta["counts"] == {
        "terms": 2,
        "dropped": 1,
        "definitions": 3,
        "concepts": 1,
        "papers": 2,
        "seeds": {"terms": 1, "papers": 1},
        "closure": {"terms": 1, "papers": 1},
    }
    assert meta["topics"] == [
        {"id": 0, "name": "framing", "size": 2, "top": ["framing"]}
    ]
    assert meta["method"] == explorer.METHOD and "provenance" not in meta
    assert meta["fields"]["counts"] == {"sp": [2, 2]}
    assert "Left out: 1 dropped mentions." in capsys.readouterr().out


def test_a_second_run_on_its_own_output_writes_the_same_files(tmp_path, small):
    dataset = write_dataset(tmp_path / "MBDef", page_data())
    cache = write_cache(tmp_path / "cache")
    explorer_data.released(dataset, cache, tmp_path / "first")
    again = tmp_path / "again"
    (again / "data").parent.mkdir(parents=True, exist_ok=True)
    (again / "data").symlink_to(dataset / "data")
    (again / "explorer").symlink_to(tmp_path / "first")

    explorer_data.released(again, cache, tmp_path / "second")

    assert read(tmp_path / "second") == read(tmp_path / "first")


def test_the_output_passes_the_checks_of_explorer_build(tmp_path, small):
    dataset = write_dataset(tmp_path / "MBDef", page_data())
    explorer_data.released(dataset, write_cache(tmp_path / "cache"), tmp_path / "out")

    problems, summary = explorer.disagreements(
        explorer.checked_bundle(tmp_path / "out"), dataset / "data"
    )

    assert problems == []
    assert summary == "2 mentions, 3 definitions, 1 concept, 2 papers"


def test_vectors_other_than_the_page_data_was_made_with_stop_it(tmp_path, small):
    dataset = write_dataset(tmp_path / "MBDef", page_data())
    cache = write_cache(
        tmp_path / "cache", VECTORS | {"A frame is a schema.": [0.0, 1.0, 0.0, 0.0]}
    )

    with pytest.raises(SystemExit, match="are not those the page data was made with"):
        explorer_data.released(dataset, cache, tmp_path / "out")
    assert not (tmp_path / "out").exists()
