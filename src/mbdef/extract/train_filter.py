"""`mbdef extract train-filter`: fit a relevance filter the way the frozen ones were.

The three frozen filters of `relevance.py` (`ordinary`, `title`, `title-blocks`) were fitted one study each, on the same records, folds and settings:

1. **Records.** Every definition record of the ground truth's TRAIN papers: the relevant ones from MBDef-Gold's `validated_gt.json` (label 1), then the not relevant ones from its `validated_gt_non_relevant.json` (label 0), both in the data root unless flags name other files, file by file, paper by paper, record by record (`training_records`). The TRAIN papers are the relevant file's papers with split `train`. The released ground truth gives 1,171 records of 80 papers, 592 of them relevant.
2. **Classifier input**, as the filter reads it when it is applied (`relevance.py`): `ordinary` the feature text (the term and its definitions, one per line); `title` the paper's title, a line break and the feature text; `title-blocks` the feature text's BGE vector next to the title's (1,536 numbers). A paper's title is its `metadata.title`, which must be the same in both files.
3. **Folds.** The papers go into 5 folds (`StratifiedGroupKFold`, shuffled with seed 9; a paper's records are always in one fold). Each record gets an *out-of-fold* probability from a classifier fitted on the other four folds. The classifier is a logistic regression (`classifier`) on the vectors as they are; a fit that does not converge stops the run.
4. **Cutoff.** The relevant records' out-of-fold probabilities, sorted from high to low; the cutoff is the ⌈0.95 × relevant⌉-th of them, the largest cutoff that keeps 95 % of the relevant records (`recall_cutoff`).
5. **The filter** is one more classifier, fitted on all records, saved as the frozen ones were (joblib with pickle protocol 4, so the file can be compared byte for byte).

The solver and the probabilities use OpenBLAS, which picks its kernels by CPU. The frozen filters were fitted with its `Haswell` kernels (also AMD Zen's); other kernels sum in another order, so the coefficients differ by about 1e-14 and the cutoff in its last digits. `OPENBLAS_CORETYPE=Haswell` forces those kernels on any x86-64 CPU with AVX2; the receipt records which kernels ran.

The BGE vectors come from the pinned encoder (`relevance.Encoder`, on `SCIDEF_NLI_DEVICE` from `.env`), from a recording of it (`--replay`), or, with `--stored-vectors`, from the frozen filter's own zip, which holds the vector of every text its study encoded: then the refit needs no model and no GPU. Everything goes into `--out` (new or empty):

    model.joblib       the classifier
    config.json        the encoder settings `relevance.load_filter` checks, and the fit's settings
    training.json      records, papers, relevant and other records, features, the cutoff, and the
                       out-of-fold counts at the cutoff (`selected`) and at 0.5 (`default`)
    folds.json         each fold's training and held-out papers
    oof.json           every record with its label, out-of-fold probability and fold
    encoder/           the encoder's recording (not with --stored-vectors)
    receipt.json       settings, input hashes, the comparison with the frozen filter, the BLAS
                       kernels, commit

Worked example (`ordinary`, refitted from its stored vectors): the first record, *dual-process approaches* (paper 0209e602…, relevant), is in fold 2 and gets 0.4802 out of fold. 563 of the 592 relevant records (⌈0.95 × 592⌉ = 563) have at least 0.3087965279332379, so that is the cutoff; at it, 283 of the 579 other records pass too. The classifier file is the frozen one byte for byte.
"""

import io
import json
import math
import warnings
import zipfile
from pathlib import Path

import numpy as np

from mbdef import files
from mbdef.extract import relevance
from mbdef.extract.train import gold_files

SPLIT = "train"
# A record's `dataset`: from the relevant file, from the other one.
DATASETS = ("biasdef", "non-relevant")
# The fit's settings, each a flag of `extract train-filter`
FOLDS = 5  # --folds
SEED = 9  # the folds' shuffle and the classifier's (--seed)
TARGET_RECALL = 0.95  # of the relevant records, out of fold (--target-recall)
REGULARISATION = 1  # the logistic regression's C, the inverse of its regularisation strength (--regularisation)
# The logistic regression's other settings, as the frozen filters were fitted
SOLVER, MAX_ITER = "liblinear", 1000
DEFAULT_CUTOFF = 0.5  # reported next to the chosen cutoff
PICKLE_PROTOCOL = 4  # the frozen classifiers' (Python 3.13's default)
# The members of a frozen zip that hold its study's BGE vectors: (texts, vectors in that order).
STORED = (
    ("embedding-texts.json", "embeddings.npy"),
    ("concept-embedding-texts.json", "concept-embeddings.npy"),
    ("title-encoder/embedding-texts.json", "title-encoder/embeddings.npy"),
)


# Records


def training_records(relevant: dict, non_relevant: dict) -> list[dict]:
    """The TRAIN papers' records with their label (module docstring, step 1)."""
    train = {p["paper_id"] for p in relevant["papers"] if p["split"] == SPLIT}
    if not train:
        raise SystemExit("The ground truth has no TRAIN paper; nothing to train on.")
    rows = []
    for dataset, label, ground_truth in (
        (DATASETS[0], 1, relevant),
        (DATASETS[1], 0, non_relevant),
    ):
        for paper in ground_truth["papers"]:
            if paper["paper_id"] not in train:
                continue
            if paper["split"] != SPLIT:
                raise SystemExit(
                    f"Paper {paper['paper_id']} is {paper['split']} in the {dataset} file but "
                    f"{SPLIT} in the relevant one."
                )
            for index, record in enumerate(paper["definition_records"]):
                rows.append(
                    {
                        "paper_id": paper["paper_id"],
                        "split": SPLIT,
                        "dataset": dataset,
                        "record_index": index,
                        "term": record["term"],
                        "definitions": record["definitions"],
                        "text": relevance.feature_text(record),
                        "relevance": label,
                    }
                )
    return rows


def paper_titles(ground_truths: list[dict], paper_ids: set[str]) -> dict[str, str]:
    """{paper id: title}: each file has each paper once, with the same non-empty title."""
    found = []
    for ground_truth in ground_truths:
        papers = [p for p in ground_truth["papers"] if p["paper_id"] in paper_ids]
        titles = {p["paper_id"]: p["metadata"]["title"] for p in papers}
        if len(papers) != len(titles) or set(titles) != paper_ids:
            raise SystemExit(
                "A TRAIN paper is missing or repeated in a ground-truth file."
            )
        if any(not isinstance(t, str) or not t.strip() for t in titles.values()):
            raise SystemExit("A TRAIN paper has an empty title.")
        found.append(titles)
    if any(titles != found[0] for titles in found[1:]):
        raise SystemExit("The ground-truth files give a TRAIN paper different titles.")
    return found[0]


# Vectors and features


def stored_vectors(path: Path) -> dict[str, np.ndarray]:
    """{text: BGE vector} of every text a frozen filter's zip stores (`STORED`)."""
    store: dict[str, np.ndarray] = {}
    with zipfile.ZipFile(path) as bundle:
        names = set(bundle.namelist())
        for texts_name, vectors_name in STORED:
            if texts_name not in names:
                continue
            texts = json.loads(bundle.read(texts_name))
            vectors = np.load(io.BytesIO(bundle.read(vectors_name)))
            if vectors.shape != (len(texts), relevance.DIMENSIONS):
                raise SystemExit(f"{path}: {vectors_name} does not fit {texts_name}.")
            for text, vector in zip(texts, vectors, strict=True):
                if text in store and not np.array_equal(store[text], vector):
                    raise SystemExit(f"{path}: two different vectors for one text.")
                store[text] = vector
    if not store:
        raise SystemExit(f"{path} stores no vectors.")
    return store


def from_store(store: dict[str, np.ndarray], source: Path):
    """An encoder that answers from stored vectors and stops at a text that is not stored."""

    def encode(texts: list[str]) -> np.ndarray:
        missing = [t for t in texts if t not in store]
        if missing:
            raise SystemExit(
                f"Stopped: {source} stores no vector for {missing[0][:60]!r}… "
                f"({len(missing)} text(s) missing)."
            )
        return np.asarray([store[t] for t in texts], dtype=np.float32).reshape(
            len(texts), relevance.DIMENSIONS
        )

    return encode


def matrix(encode, texts: list[str]) -> np.ndarray:
    """The vectors of `texts` in order, each distinct text encoded once."""
    distinct = list(dict.fromkeys(texts))
    rows = dict(zip(distinct, encode(distinct), strict=True))
    return np.asarray([rows[t] for t in texts], dtype=np.float32).reshape(
        len(texts), relevance.DIMENSIONS
    )


def features(name: str, rows: list[dict], titles: dict[str, str], encode) -> np.ndarray:
    """The classifier input of every record for filter `name` (module docstring, step 2)."""
    if name == "ordinary":
        return matrix(encode, [r["text"] for r in rows])
    if name == "title":
        texts = [relevance.title_feature(titles[r["paper_id"]], r) for r in rows]
        return matrix(encode, texts)
    concept = matrix(encode, [r["text"] for r in rows])
    title = matrix(encode, [titles[r["paper_id"]] for r in rows])
    return np.concatenate([concept, title], axis=1)


# The fit


def classifier_settings(regularisation=REGULARISATION, seed=SEED) -> dict:
    return dict(C=regularisation, solver=SOLVER, max_iter=MAX_ITER, random_state=seed)


def classifier_text(regularisation=REGULARISATION, seed=SEED) -> str:
    """How config.json and the receipt name the classifier, as the frozen filters' config does:
    `LogisticRegression(C=1, solver='liblinear', max_iter=1000, random_state=9)` by default."""
    settings = classifier_settings(regularisation, seed)
    return f"LogisticRegression({', '.join(f'{k}={v!r}' for k, v in settings.items())})"


def classifier(regularisation=REGULARISATION, seed=SEED):
    """The frozen filters' logistic regression (`classifier_text`)."""
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(**classifier_settings(regularisation, seed))


def recall_cutoff(
    labels: np.ndarray, probabilities: np.ndarray, target_recall=TARGET_RECALL
) -> float:
    """The largest cutoff that keeps ⌈target_recall × relevant⌉ relevant records."""
    relevant = np.sort(np.asarray(probabilities)[np.asarray(labels) == 1])[::-1]
    if not len(relevant) or not np.isfinite(relevant).all():
        raise SystemExit("No relevant record with a probability: no cutoff can be set.")
    return float(relevant[math.ceil(target_recall * len(relevant)) - 1])


def binary_summary(
    labels: np.ndarray, probabilities: np.ndarray, cutoff: float
) -> dict:
    """The records kept at `cutoff` against their labels: counts, precision, recall, F1, MCC."""
    from sklearn.metrics import (
        confusion_matrix,
        matthews_corrcoef,
        precision_recall_fscore_support,
    )

    kept = np.asarray(probabilities) >= cutoff
    tn, fp, fn, tp = confusion_matrix(labels, kept, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, kept, average="binary", zero_division=0
    )
    return {
        "threshold": float(cutoff),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mcc": float(matthews_corrcoef(labels, kept)),
    }


def fit(
    x: np.ndarray,
    labels: np.ndarray,
    papers: np.ndarray,
    folds_count=FOLDS,
    seed=SEED,
    target_recall=TARGET_RECALL,
    regularisation=REGULARISATION,
) -> dict:
    """Folds, out-of-fold probabilities, cutoff and the final classifier (steps 3 to 5)."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.model_selection import StratifiedGroupKFold

    oof = np.full(len(labels), np.nan)
    fold_of = np.full(len(labels), -1)
    folds = []
    splitter = StratifiedGroupKFold(
        n_splits=folds_count, shuffle=True, random_state=seed
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for fold, (train, held) in enumerate(splitter.split(x, labels, papers)):
            model = classifier(regularisation, seed).fit(x[train], labels[train])
            oof[held] = model.predict_proba(x[held])[:, 1]
            fold_of[held] = fold
            folds.append(
                {
                    "fold": fold,
                    "train_papers": sorted({str(p) for p in papers[train]}),
                    "validation_papers": sorted({str(p) for p in papers[held]}),
                }
            )
        if not np.isfinite(oof).all():
            raise SystemExit("A record got no out-of-fold probability.")
        cutoff = recall_cutoff(labels, oof, target_recall)
        model = classifier(regularisation, seed).fit(x, labels)
    return {
        "model": model,
        "oof": oof,
        "fold_of": fold_of,
        "folds": folds,
        "cutoff": cutoff,
    }


# The command


def train_filter(
    out: Path,
    name: str = "ordinary",
    relevant: Path | None = None,
    non_relevant: Path | None = None,
    filters: Path = relevance.ZIPS,
    stored: bool = False,
    bge: Path | None = None,
    replay: list[Path] | None = None,
    folds: int = FOLDS,
    seed: int = SEED,
    target_recall: float = TARGET_RECALL,
    regularisation: float = REGULARISATION,
) -> int:
    """The step (module docstring). Returns 0."""
    import joblib

    out = Path(out)
    if name not in relevance.FILTERS:
        raise SystemExit(
            f"Unknown filter {name}; use one of {', '.join(relevance.FILTERS)}."
        )
    if stored and replay:
        raise SystemExit("Give --stored-vectors or --replay, not both.")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; give a new or empty folder.")
    frozen_zip = Path(filters) / f"{relevance.FILTERS[name]['study']}.zip"
    (relevant, _), (non_relevant, _) = gold_files(relevant, non_relevant)
    ground_truths = [
        json.loads(p.read_text(encoding="utf-8")) for p in (relevant, non_relevant)
    ]
    rows = training_records(*ground_truths)
    papers = np.asarray([r["paper_id"] for r in rows])
    titles = (
        paper_titles(ground_truths, set(papers.tolist())) if name != "ordinary" else {}
    )
    encoder = None
    if stored:
        encode = from_store(stored_vectors(frozen_zip), frozen_zip)
    elif replay:
        encoder = relevance.Encoder(replay=relevance.recorded(replay))
        encode = encoder.vectors
    else:
        bge = Path(bge) if bge else relevance.bge_folder()
        encoder = relevance.Encoder(bge, files.env("SCIDEF_NLI_DEVICE"))
        encode = encoder.vectors
    try:
        x = features(name, rows, titles, encode)
    except ValueError as error:
        # relevance.Encoder: an input over the encoder's capacity
        raise SystemExit(f"Stopped: {error}.") from error
    labels = np.asarray([r["relevance"] for r in rows])
    result = fit(x, labels, papers, folds, seed, target_recall, regularisation)
    named = classifier_text(regularisation, seed)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["model"], out / "model.joblib", protocol=PICKLE_PROTOCOL)
    oof, cutoff = result["oof"], result["cutoff"]
    training = {
        "records": len(rows),
        "papers": len(set(papers.tolist())),
        "positive_records": int(labels.sum()),
        "negative_records": int(len(labels) - labels.sum()),
        "features": int(x.shape[1]),
        "cutoff": cutoff,
        "selected": binary_summary(labels, oof, cutoff),
        "default": binary_summary(labels, oof, DEFAULT_CUTOFF),
    }
    files.write_json(out / "training.json", training)
    files.write_json(
        out / "config.json",
        {
            "filter": name,
            "model": relevance.ENCODER[0],
            "revision": relevance.ENCODER[1],
            "capacity": relevance.CAPACITY,
            "batch_size": relevance.BATCH,
            "dtype": "float32",
            "model_files_sha256": relevance.ENCODER_FILES,
            "classifier": named,
            "folds": folds,
            "seed": seed,
            "target_recall": target_recall,
            "features": training["features"],
        },
    )
    files.write_json(out / "folds.json", result["folds"])
    files.write_json(
        out / "oof.json",
        [
            {**row, "probability": float(p), "fold": int(f)}
            for row, p, f in zip(rows, oof, result["fold_of"], strict=True)
        ],
    )
    outputs = {
        "model": out / "model.joblib",
        "training": out / "training.json",
        "config": out / "config.json",
        "folds": out / "folds.json",
        "oof": out / "oof.json",
    }
    inputs = {
        "ground_truth": Path(relevant),
        "non_relevant": Path(non_relevant),
        "frozen_filter": frozen_zip,
    }
    if encoder is not None:
        encoder.save(out / "encoder")
        outputs["encoder"] = out / "encoder"
    for number, folder in enumerate(replay or [], 1):
        inputs[f"replay:{number}"] = Path(folder)
    if bge is not None:
        for file in relevance.ENCODER_FILES:
            inputs[f"encoder:{file}"] = Path(bge) / file
    vectors = (
        "stored in the frozen zip"
        if stored
        else "replay of recorded vectors"
        if replay
        else f"computed on {encoder.device if encoder else None}"
    )
    frozen = compare_frozen(out, frozen_zip)
    settings = {
        "filter": name,
        "classifier": named,
        "folds": folds,
        "fold_seed": seed,
        "target_recall": target_recall,
        "pickle_protocol": PICKLE_PROTOCOL,
        "encoder": f"{relevance.ENCODER[0]} at {relevance.ENCODER[1]}",
        "vectors": vectors,
    }
    receipt = files.receipt("mbdef extract train-filter", settings, inputs, outputs)
    receipt["frozen_filter"] = frozen
    receipt["blas"] = blas_kernels()
    files.write_json(out / files.RECEIPT, receipt)
    summary = {
        "filter": name,
        "records": training["records"],
        "papers": training["papers"],
        "relevant_records": training["positive_records"],
        "cutoff": cutoff,
        "kept_out_of_fold": {
            "relevant": training["selected"]["tp"],
            "not_relevant": training["selected"]["fp"],
        },
        "vectors": vectors,
        "frozen_filter": frozen,
    }
    print(files.json_text(summary), end="")
    return 0


def compare_frozen(out: Path, frozen_zip: Path) -> dict:
    """How the fitted filter in `out` compares with the frozen filter in `frozen_zip`."""
    import joblib

    with zipfile.ZipFile(frozen_zip) as bundle:
        model, training, folds = (
            bundle.read(m) for m in ("model.joblib", "training.json", "folds.json")
        )
    mine = joblib.load(out / "model.joblib")
    theirs = joblib.load(io.BytesIO(model))
    same_shape = mine.coef_.shape == theirs.coef_.shape
    return {
        "zip": frozen_zip.name,
        "classifier_file": same(model, (out / "model.joblib").read_bytes()),
        "cutoff": same(
            json.loads(training)["cutoff"],
            json.loads((out / "training.json").read_text(encoding="utf-8"))["cutoff"],
        ),
        "folds": same(folds, (out / "folds.json").read_bytes()),
        "max_coefficient_difference": float(
            max(
                np.abs(mine.coef_ - theirs.coef_).max(),
                np.abs(mine.intercept_ - theirs.intercept_).max(),
            )
        )
        if same_shape
        else None,
    }


def blas_kernels() -> list[dict]:
    """The BLAS libraries loaded (numpy's and scipy's OpenBLAS) and the kernels they picked."""
    from threadpoolctl import threadpool_info

    return [
        {key: info.get(key) for key in ("prefix", "version", "architecture")}
        for info in threadpool_info()
        if info.get("user_api") == "blas"
    ]


def same(frozen, refit) -> str:
    return "identical" if frozen == refit else "differs"
