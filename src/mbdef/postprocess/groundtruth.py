"""The labelled ground truth for Step 5 relevance: MBDef-Gold's relevant records and the non-relevant records.

A Record is one Extracted Concept of a ground-truth paper (a term in that paper, with its Definitions). Splits are fixed per paper and were never changed by the human verification: train, dev, and test (= original_test + new_test).
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from mbdef import files

from . import FROZEN, REPO


@dataclass(frozen=True)
class Record:
    id: str
    paper_id: str
    split: str  # train | dev | test
    term: str
    label: int  # 1 Relevant, 0 not
    definitions: tuple[str, ...]
    votes: tuple[int, int] | None = (
        None  # (relevant votes, not relevant votes), when known
    )

    @property
    def unanimous(self) -> bool | None:
        return None if self.votes is None else min(self.votes) == 0


# MBDef-Gold's two record files (in the data root: `mbdef data fetch MBDef-Gold/`) and their records' label
RELEVANT, NON_RELEVANT = "validated_gt.json", "validated_gt_non_relevant.json"
FILES = ((RELEVANT, 1), (NON_RELEVANT, 0))
FOLDER = f"$DATA_ROOT/{files.MBDEF_GOLD}"  # where they are, as the help texts name it


def paths() -> list[tuple[Path, int]]:
    """The two record files in the data root, each with its records' label."""
    return [(files.mbdef_gold(name), y) for name, y in FILES]


# The splits of the reference that the corpus's scores are ranked against: a TEST record is never part of it
REFERENCE_SPLITS = ("train", "dev")
# Vote tallies per record, joined one to one with the annotation export (a frozen input)
VOTES = REPO / FROZEN / "20261015/gt-label-quality/records.csv"


def load(splits=("train", "dev", "test"), votes=VOTES):
    """All ground-truth Records of the given splits, relevant file first, papers and records in file order.
    Record id = "<paper_id>:<1 relevant | 0 not>:<index in the paper's records>"."""
    tally = {}
    if votes and Path(votes).exists():
        with Path(votes).open() as f:
            for r in csv.DictReader(f):
                y = 1 if r["file"] == "relevant" else 0
                tally[f"{r['paper_id']}:{y}:{r['record_index']}"] = (
                    int(r["votes_relevant"]),
                    int(r["votes_not_relevant"]),
                )
    out = []
    for path, y in paths():
        for paper in json.loads(path.read_text())["papers"]:
            split = "test" if "test" in paper["split"] else paper["split"]
            if split not in splits:
                continue
            for j, rec in enumerate(paper["definition_records"]):
                rid = f"{paper['paper_id']}:{y}:{j}"
                defs = tuple(
                    d["text"] for d in rec["definitions"] if d.get("text", "").strip()
                )
                out.append(
                    Record(
                        id=rid,
                        paper_id=paper["paper_id"],
                        split=split,
                        term=rec["term"],
                        label=y,
                        definitions=defs,
                        votes=tally.get(rid),
                    )
                )
    return out
