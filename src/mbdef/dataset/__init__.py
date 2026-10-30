"""MBDef, the released dataset (docs/dataset/dataset.md): Step 5's kept terms as parquet tables with a dataset card, and a JSON copy; and MBDef-Gold, the ground truth, as a dataset of its own.

`mbdef dataset build` reads Step 5's decisions and term merging, the corpus store and release v1.1's paper lists, and writes `<out>/taxonomy-dataset/` (the folder a dataset host takes: README.md, the card; data/*.parquet; provenance.json) and `<out>/validation.{json,md}` beside it. `mbdef dataset export` writes the same tables as JSON into `<out>/taxonomy-dataset-json/`. `mbdef dataset build-gold` writes MBDef-Gold's split files and card into `<out>/MBDef-Gold/` (gold.py). All are plain functions over plain files (build.py, card.py, export.py, gold.py); store.py holds what the build reads from the corpus store.

Every input has a flag; its default (`Inputs`) is the file of the adopted run: Step 5's frozen files (`data/frozen/step5-post-processing/`, those over 10 MB in the data root), the corpus store and release v1.1 in the data root, MBDef-Gold (the ground truth; MBDef reads it only to count the labelled train and dev records) in the data root, and three recorded files in `data/frozen/dataset/`: the recorded OpenAlex answers, the merges the merge audit found likely wrong, and the Step 5 numbers the card quotes.
"""

from dataclasses import dataclass
from pathlib import Path

from mbdef import files, postprocess
from mbdef.postprocess import groundtruth

REPO = postprocess.REPO
RECORDED = REPO / "data/frozen/dataset"
OPENALEX = "openalex-works-20261021.jsonl.gz"  # in the recorded folder: the recorded OpenAlex answers
LIKELY_WRONG = "likely-wrong-word-order.json"  # in the recorded folder: merges of the same words in another order
CARD_NUMBERS = "card-numbers.json"  # in the recorded folder: Step 5's evaluation numbers that the card quotes
CORPUS = "frozen/closure/definition-corpus-v1.1"  # release v1.1, in the data root
STORE = "frozen/artifacts/experiments/taxonomy-curation-v2/library.sqlite3"  # the corpus store, in the data root
DECISIONS = "20261021/decisions/ensemble"  # the ensemble's keep or drop (.csv in the data root, .json in git)
LABEL_GROUPS = "20261021/label-groups"
TERM_MERGING = "20261021/term-merging"
PICK = "20261020/ensembles/pick.json"
DATASET = "taxonomy-dataset"  # the folder a dataset host takes
JSON_COPY = "taxonomy-dataset-json"
LICENCE = "cc-by-4.0"
# The dataset's repository on Hugging Face, which the card's loading example names (files.py names both datasets)
REPO_ID = files.MBDEF_REPO
# The version this code builds; each version is a commit of the dataset's repository on Hugging Face.
# Its data are byte-identical whatever command builds it (docs/dataset/dataset.md, "How it was checked").
VERSION = "1.0.0"
# The image at the top of each card, copied into the dataset's folder as BANNER_PATH (`--banner` names another): MBDef's shows the MBDef Explorer's map, MBDef-Gold's its own records
BANNER = REPO / "docs/images/mbdef-banner.png"
GOLD_BANNER = REPO / "docs/images/mbdef-gold-banner.png"
BANNER_PATH = "assets/banner.png"
# The picture of sample records on MBDef-Gold's card (made from `dataset.gold.samples`, so it shows the records of the card's table)
GOLD_SAMPLES = REPO / "docs/images/mbdef-gold-samples.png"
SAMPLES_PATH = "assets/samples.png"
# The links both cards give; the paper's and the code's are placeholders until they are public.
PAPER_URL = "https://arxiv.org/abs/XXXX.XXXXX"
CODE_URL = "https://github.com/XXXX/XXXX"
EXPLORER_URL = "https://media-bias-group.github.io/MBDef/"
PROJECT_URL = "https://media-bias-group.github.io/MBDef-ProjectPage/"


@dataclass(frozen=True)
class Inputs:
    """Every file the build reads."""

    decisions: Path  # the keep or drop of every corpus term (csv)
    decisions_json: Path  # its cut, best-F1 cut and counts
    # the Label Groups over all corpus terms: terms.jsonl, run.json
    label_groups: Path
    term_merging: Path  # the merging of the kept terms: concepts.jsonl, summary.json
    pick: Path  # the ensemble's frozen weights
    store: Path  # the corpus store (sqlite)
    corpus: Path  # release v1.1: seeds/papers.csv, closure/papers.csv, links/{cited-works,definition-links}.csv
    recorded: Path  # OPENALEX, LIKELY_WRONG and CARD_NUMBERS
    ground_truth: Path  # the relevant ground-truth records
    non_relevant: Path  # the non-relevant ones
    votes: Path  # each record's votes (MBDef-Gold only)

    @property
    def links(self) -> Path:
        return self.corpus / "links/definition-links.csv"


def inputs(
    decisions: str | Path | None = None,
    label_groups: str | Path | None = None,
    term_merging: str | Path | None = None,
    pick: str | Path | None = None,
    store: str | Path | None = None,
    corpus: str | Path | None = None,
    recorded: str | Path | None = None,
    ground_truth: str | Path | None = None,
    non_relevant: str | Path | None = None,
    votes: str | Path | None = None,
) -> Inputs:
    """The inputs as given, each missing one the adopted run's file. A decisions csv given has its json beside it."""
    frozen = REPO / postprocess.FROZEN
    gold = groundtruth.paths() if not (ground_truth and non_relevant) else []
    if decisions:
        decisions, decisions_json = (
            Path(decisions),
            Path(decisions).with_suffix(".json"),
        )
    else:
        decisions = postprocess.frozen_file(DECISIONS + ".csv")
        decisions_json = postprocess.frozen_file(DECISIONS + ".json")
    return Inputs(
        decisions=decisions,
        decisions_json=decisions_json,
        label_groups=Path(label_groups or frozen / LABEL_GROUPS),
        term_merging=Path(term_merging or frozen / TERM_MERGING),
        pick=Path(pick or frozen / PICK),
        store=Path(store) if store else files.data_root() / STORE,
        corpus=Path(corpus) if corpus else files.data_root() / CORPUS,
        recorded=Path(recorded or RECORDED),
        ground_truth=Path(ground_truth) if ground_truth else gold[0][0],
        non_relevant=Path(non_relevant) if non_relevant else gold[1][0],
        votes=Path(votes or groundtruth.VOTES),
    )
