"""The frozen relevance filter: is a definition record about media bias?

A record's **feature text** is its term and its definitions, one per line (no other list). The BGE encoder turns the text into a vector, and a small classifier (logistic regression) gives the probability that the record is relevant. A record is kept if that probability is at least the filter's cutoff; a kept record is copied unchanged. There are three frozen filters, each trained on the 1,171 records of the ground truth's 80 training papers (592 of them relevant), with the cutoff set so that 95 % of the relevant records pass in five-fold cross-validation:

| `FILTERS` name | Row | Study zip in `data/frozen/results/` | Classifier input | Cutoff |
|---|---|---|---|---|
| `ordinary` | R6e24b3d | `semantic-relevance-v1.zip` | the feature text | 0.3087965279332379 |
| `title` | R0bea6bc | `semantic-title-relevance-v2.zip` | the title, a line break, the feature text | 0.2371840437933272 |
| `title-blocks` | Ra1da1ae | `semantic-title-blocks-v3.zip` | the feature text's vector next to the title's | 0.24892858469583487 |

The released extractors filter with `ordinary`; the selection grid also published `title` stages, and `title-blocks` is a row of the held-out TEST rescoring. `load_filter` reads a filter from its zip and refuses it unless the classifier, training and configuration files have their frozen sha256 and the configuration names the pinned encoder.

**The encoder** (`Encoder`) is BGE base v1.5 (`ENCODER` at its pinned revision, `ENCODER_FILES`), in float32, with the first token's (CLS) vector L2-normalised, 32 texts per batch, as the studies trained it. Nothing is truncated: a record whose input has more than 512 tokens cannot be classified, so it is dropped and its index listed in `capacity_excluded_record_indices`. The encoder can also answer from a recording (`recorded`, the `encoder/` folder an earlier run wrote): every text must then have been recorded, or the run stops (the guard of `llm.recorded`, for vectors).

`apply_filter` is the one function that applies a filter to a paper's records; `filter.py` is the command.

Worked example (released TEST run, part w8k, the Qwen verifier's records): paper 4243aa9b… has one verified record, *nation branding*; `ordinary` gives it 0.4995 ≥ 0.3088, so it is kept.
"""

import copy
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from mbdef import files, models

ZIPS = Path("data/frozen/results")
ENCODER = models.PINNED["bge"]
ENCODER_FILES = {
    "model.safetensors": "c7c1988aae201f80cf91a5dbbd5866409503b89dcaba877ca6dba7dd0a5167d7",
    "config.json": "bc00af31a4a31b74040d73370aa83b62da34c90b75eb77bfa7db039d90abd591",
    "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    "tokenizer_config.json": "9261e7d79b44c8195c1cada2b453e55b00aeb81e907a6664974b4d7776172ab3",
    "1_Pooling/config.json": "c9bef85e8bbf4b2eab4941b3fb62bd33f88686748b478f2e264d256472d9643b",
}
CAPACITY = 512  # tokens, special tokens included
BATCH = 32
DIMENSIONS = 768
FILTERS: dict[str, dict[str, Any]] = {
    "ordinary": {
        "row": "R6e24b3d",
        "study": "semantic-relevance-v1",
        "cutoff": 0.3087965279332379,
        "features": 768,
        "sha256": {
            "model.joblib": "c01a88ffe2b5bb0fd5ad6c7644d686c4de1ff89a9341271bd42ded9c5a804dd2",
            "training.json": "406e8645cbc61706531ea4ebd55aba8a8aeb2e200ac9b421db6a4b13b6edaeec",
            "config.json": "f31d900b7493c46ecb3e15f9b3b91eaf3a26eaf057e73be3bb1eae54d0adc29c",
        },
    },
    "title": {
        "row": "R0bea6bc",
        "study": "semantic-title-relevance-v2",
        "cutoff": 0.2371840437933272,
        "features": 768,
        "sha256": {
            "model.joblib": "392cd5b13478201e4af17a1727043d1bd3c2d8e7957a03ecb7ec3e4d6b4eb752",
            "training.json": "f65c28e94e4f7c01348d4195c4ce06acb764cf718bbf45f00592f8d61c810f8d",
            "config.json": "cb3b8be43962170b1a77cd0e2fd827c0bd41f324b24461f1858c6dd135d13696",
        },
    },
    "title-blocks": {
        "row": "Ra1da1ae",
        "study": "semantic-title-blocks-v3",
        "cutoff": 0.24892858469583487,
        "features": 1536,
        "sha256": {
            "model.joblib": "e1d7f42a799e2eddc4294d55f44d5c135e8cb2656d4d4f4246d98b2be59be965",
            "training.json": "30656cf6d33b93f72a52a42b2d1b5e6425ead3b753445d97cd1cb9b7321c987a",
            "config.json": "aead927ebcf7365578de7560ae2e3e5ce210907b372ed4ed91bf0d45d270e2b0",
        },
    },
}
RECORDING = ("encoded.json", "vectors.npy")  # the files of a recorded encoder folder


def feature_text(record: dict) -> str:
    """The term and every definition, one per line; a record needs non-empty definitions."""
    if not record["definitions"] or not all(
        d["text"].strip() for d in record["definitions"]
    ):
        raise ValueError("Classifier requires nonempty definition statements")
    return record["term"] + "\n" + "\n".join(d["text"] for d in record["definitions"])


def title_feature(title: str, record: dict) -> str:
    """The `title` filter's input: the paper's title, a line break and the feature text."""
    if not title.strip():
        raise ValueError("Empty title")
    return title + "\n" + feature_text(record)


def load_filter(name: str, zips: Path = ZIPS) -> dict:
    """A frozen filter (`FILTERS[name]` plus its `classifier`), after checking its zip."""
    import joblib

    spec = FILTERS[name]
    path = zips / f"{spec['study']}.zip"
    with zipfile.ZipFile(path) as bundle:
        file_set = {member: bundle.read(member) for member in spec["sha256"]}
    for member, digest in spec["sha256"].items():
        found = files.sha256(file_set[member])
        if found != digest:
            raise SystemExit(
                f"{path}: {member} has sha256 {found}, not the frozen {digest}. "
                "Refusing to use a changed filter."
            )
    training = json.loads(file_set["training.json"])
    config = json.loads(file_set["config.json"])
    pinned = {
        "model": ENCODER[0],
        "revision": ENCODER[1],
        "capacity": CAPACITY,
        "batch_size": BATCH,
        "dtype": "float32",
        "model_files_sha256": ENCODER_FILES,
    }
    if (training["cutoff"], training["features"]) != (
        spec["cutoff"],
        spec["features"],
    ) or any(config[key] != value for key, value in pinned.items()):
        raise SystemExit(
            f"{path}: its cutoff or encoder settings are not the frozen ones."
        )
    classifier = joblib.load(io.BytesIO(file_set["model.joblib"]))
    if getattr(classifier, "n_features_in_", None) != spec["features"]:
        raise SystemExit(
            f"{path}: the classifier does not read {spec['features']} features."
        )
    return {"name": name, **spec, "classifier": classifier}


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class Encoder:
    """BGE as the filters were trained with it, or a replay of a recording (module docstring).

    With `replay` (from `recorded`), no model is loaded. Every text asked for is kept in `seen` (token count and, once encoded, the vector), and `save` writes it as a recording.
    """

    def __init__(
        self,
        folder: Path | None = None,
        device: str = "auto",
        replay: dict[str, tuple[int, np.ndarray | None]] | None = None,
    ):
        self.replay, self.device = replay, None
        self.seen: dict[str, tuple[int, np.ndarray | None]] = {}
        if replay is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        # The scoring models' rule (`auto` is CUDA when there is a GPU), shared with evaluate.
        from mbdef.evaluate.metric import pick_device

        torch, device = pick_device(device)
        self.device = device
        folder = Path(folder) if folder else bge_folder()
        for name, digest in ENCODER_FILES.items():
            found = files.digest(folder / name)
            if found != digest:
                raise SystemExit(
                    f"{folder / name}: its sha256 is {found}, not the pinned {digest}."
                )
        pooling = json.loads((folder / "1_Pooling/config.json").read_text())
        if not pooling["pooling_mode_cls_token"] or pooling["pooling_mode_mean_tokens"]:
            raise SystemExit(f"{folder}: BGE pooling changed.")
        self.torch = torch
        self.tokenizer: Any = AutoTokenizer.from_pretrained(
            folder, local_files_only=True
        )
        self.model: Any = (
            AutoModel.from_pretrained(
                folder, local_files_only=True, dtype=torch.float32
            )
            .to(device)
            .eval()
        )

    def recording(self, text: str) -> tuple[int, np.ndarray | None]:
        key = text_key(text)
        assert self.replay is not None
        if key not in self.replay:
            raise SystemExit(
                f"Stopped: the encoder input with sha256 {key} was not recorded "
                f"({text[:60]!r}…)."
            )
        return self.replay[key]

    def lengths(self, texts: list[str]) -> list[int]:
        """Each text's number of tokens, special tokens included, without truncation."""
        if not texts:
            return []
        if self.replay is not None:
            counts = [self.recording(text)[0] for text in texts]
        else:
            ids = self.tokenizer(
                texts, truncation=False, padding=False, add_special_tokens=True
            )["input_ids"]
            counts = [len(row) for row in ids]
        for text, count in zip(texts, counts, strict=True):
            self.seen.setdefault(text_key(text), (count, None))
        return counts

    def vectors(self, texts: list[str]) -> np.ndarray:
        """The texts' L2-normalised CLS vectors, 32 texts per batch, in the order given."""
        if not texts:
            return np.empty((0, DIMENSIONS), dtype=np.float32)
        lengths = self.lengths(texts)
        if any(n > CAPACITY for n in lengths):
            raise ValueError("Encoder capacity exceeded")
        if self.replay is not None:
            rows = [self.recording(text)[1] for text in texts]
            if any(row is None for row in rows):
                raise SystemExit("Stopped: a recorded encoder input has no vector.")
            matrix = np.asarray(rows, dtype=np.float32)
        else:
            torch, batches = self.torch, []
            for start in range(0, len(texts), BATCH):
                batch = self.tokenizer(
                    texts[start : start + BATCH],
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                )
                if (
                    batch["attention_mask"].sum(dim=1).tolist()
                    != lengths[start : start + BATCH]
                ):
                    raise ValueError("Classifier input token count changed")
                with torch.inference_mode():
                    hidden = self.model(
                        **{k: v.to(self.device) for k, v in batch.items()}
                    ).last_hidden_state[:, 0]
                    normalized = torch.nn.functional.normalize(hidden, p=2, dim=1)
                    batches.append(normalized.cpu().numpy())
            matrix = np.concatenate(batches)
        if matrix.shape != (len(texts), DIMENSIONS) or not np.isfinite(matrix).all():
            raise ValueError("Invalid BGE embeddings")
        for text, count, row in zip(texts, lengths, matrix, strict=True):
            self.seen[text_key(text)] = (count, row)
        return matrix

    def save(self, folder: Path) -> None:
        """Write what this encoder saw as a recording (`RECORDING`)."""
        save_recording(self.seen, folder, self.device)


def save_recording(
    seen: dict[str, tuple[int, np.ndarray | None]], folder: Path, device: str | None
) -> None:
    """`encoded.json` (each text's sha256, token count and vector row) and `vectors.npy`."""
    keys = sorted(seen)
    rows = [seen[k][1] for k in keys if seen[k][1] is not None]
    texts, row = [], 0
    for key in keys:
        tokens, vector = seen[key]
        texts.append(
            {"sha256": key, "tokens": tokens, "row": None if vector is None else row}
        )
        row += vector is not None
    folder.mkdir(parents=True, exist_ok=True)
    files.write_json(
        folder / RECORDING[0],
        {
            "model": ENCODER[0],
            "revision": ENCODER[1],
            "files_sha256": ENCODER_FILES,
            "dtype": "float32",
            "device": device,
            "texts": texts,
        },
    )
    np.save(
        folder / RECORDING[1],
        np.asarray(rows, dtype=np.float32).reshape(len(rows), DIMENSIONS),
    )


def recorded(folders: list[Path]) -> dict[str, tuple[int, np.ndarray | None]]:
    """{text sha256: (tokens, vector or None)} from recorded encoder folders (`RECORDING`)."""
    seen: dict[str, tuple[int, np.ndarray | None]] = {}
    for folder in folders:
        meta = json.loads((Path(folder) / RECORDING[0]).read_text(encoding="utf-8"))
        if (meta["model"], meta["revision"], meta["files_sha256"]) != (
            *ENCODER,
            ENCODER_FILES,
        ):
            raise SystemExit(
                f"{folder}: recorded with another encoder than {ENCODER[0]}."
            )
        matrix = np.load(Path(folder) / RECORDING[1])
        for text in meta["texts"]:
            vector = None if text["row"] is None else matrix[text["row"]]
            old = seen.get(text["sha256"])
            if old is None or old[1] is None:
                seen[text["sha256"]] = (text["tokens"], vector)
    return seen


def bge_folder() -> Path:
    """The pinned BGE snapshot in the Hugging Face cache (or stop, saying how to get it)."""
    from mbdef.cluster import cached_snapshot

    return cached_snapshot(*ENCODER, tuple(ENCODER_FILES))


def probabilities(
    encoder: Encoder, loaded: dict, title: str, records: list[dict]
) -> tuple[list[float | None], list[int]]:
    """Each record's probability (None if its input is over `CAPACITY`) and the excluded indices."""
    name, classifier = loaded["name"], loaded["classifier"]
    if name == "title":
        texts = [title_feature(title, r) for r in records]
    else:
        texts = [feature_text(r) for r in records]
    fits = [n <= CAPACITY for n in encoder.lengths(texts)]
    if name == "title-blocks" and records and encoder.lengths([title])[0] > CAPACITY:
        fits = [False] * len(records)
    eligible = [text for text, ok in zip(texts, fits, strict=True) if ok]
    if not eligible:
        values = np.empty((0,), dtype=float)
    elif name == "title-blocks":
        concept = encoder.vectors(eligible)
        title_vector = encoder.vectors([title])[0]
        matrix = np.asarray(
            [np.concatenate([c, title_vector]) for c in concept], dtype=np.float32
        ).reshape(len(eligible), 2 * DIMENSIONS)
        values = classifier.predict_proba(matrix)[:, 1]
    else:
        values = classifier.predict_proba(encoder.vectors(eligible))[:, 1]
    if len(values) != len(eligible) or not np.isfinite(values).all():
        raise ValueError("Invalid classifier output")
    given = iter(float(v) for v in values)
    return [next(given) if ok else None for ok in fits], [
        i for i, ok in enumerate(fits) if not ok
    ]


def apply_filter(
    encoder: Encoder,
    loaded: dict,
    title: str,
    records: list[dict],
    cutoff: float | None = None,
) -> dict:
    """One paper's records filtered: the kept records (copied), every probability, the excluded.

    `loaded` is a `load_filter` result; `cutoff` replaces its frozen cutoff.
    """
    cutoff = loaded["cutoff"] if cutoff is None else cutoff
    values, excluded = probabilities(encoder, loaded, title, records)
    return {
        "filter": loaded["name"],
        "filter_row": loaded["row"],
        "cutoff": cutoff,
        "records_before_filter": len(records),
        "relevance_probabilities": values,
        "capacity_excluded_record_indices": excluded,
        "definition_records": [
            copy.deepcopy(r)
            for r, p in zip(records, values, strict=True)
            if p is not None and p >= cutoff
        ],
    }
