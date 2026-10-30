"""DefScore, the definition matcher: when does a predicted definition match a reference definition?

Read it in four parts, top to bottom:

1. **The pair score.** For a (reference, predicted) pair of definition texts, DefScore is the mean of three similarities, then an affine calibration, clipped to [0, 1]:

       DefScore = clip(DEFSCORE_SLOPE * (token F1 + BGE cosine + entailment) / 3 + DEFSCORE_INTERCEPT, 0, 1)

   where token F1 compares the lower-case ASCII words of the two texts, BGE cosine is the cosine of the two texts' BGE-base vectors, and entailment is the mean of ModernBERT's entailment probability in both directions. Worked example (TEST paper 73bfad11…, in the scorer tests' sample `tests/fixtures/score-sample/`): the reference "fabricated information that mimics the output of the news media in form, but not in organizational process or intent" and the prediction "We define fake news to be fabricated information that mimics …" have token F1 0.8636, cosine 0.8761 and entailment (0.3992 + 0.9757) / 2 = 0.6875, so DefScore = 1.40284 * 0.8091 - 0.29737 = 0.8376: a match.
2. **Matching.** All reference definitions of a paper against all its predicted ones: the one-to-one assignment that accepts the most pairs at or above the cutoff (0.5 in the released scoring runs), ties broken by the total score (`binary_diagnostics`). Accepted pairs are the matches (tp); the rest of the predictions are fp, the rest of the reference fn.
   `definition_set_details` also gives the soft many-to-one credit that training optimises.
3. **The full-record score** (`full_record`, stored as "M3-full-v1"): the secondary score. Records are aligned one-to-one by their best definition pair; then every field of the aligned records (terms, the seven text fields, citations) is matched the same way, and the score is the mean F1 over the fields that are not empty on both sides.
4. **The models and recorded scores.** `models` loads BGE and ModernBERT at their pinned revisions on first use. `recorded` answers every similarity and entailment from a file that `recording` wrote, so a scoring runs without the models (the tests); a pair that was not recorded raises `NotRecorded`, so a changed input fails instead of passing.

This is the matcher of the released scoring runs; it was checked against the original implementation's numbers (docs/step-3-extraction/evaluate.md). The pinned settings below are those of the released scoring runs (the held-out TEST rescoring and the extractor selection); the fit that produced DEFSCORE_SLOPE and DEFSCORE_INTERCEPT is data/frozen/artifacts/defsim_benchmark.json.
torch, transformers, numpy and scipy are imported where they are used.
"""

import json
import math
import os
import threading
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from mbdef import models
from mbdef.groundtruth import FIELDS, ascii_words

Pairs = Sequence[tuple[str, str]]
PairScorer = Callable[[Pairs], list[float]]

# The DefScore pair score (the fit "calibrated_base_equal_mean").
DEFSCORE_SLOPE = 1.4028410596332237
DEFSCORE_INTERCEPT = -0.2973741431607951
DEFSCORE_FORMULA = (
    "clip(slope * (token F1 + BGE cosine + mean of both entailment directions) / 3 "
    "+ intercept, 0, 1)"
)
# The cutoffs: one-to-one matching in the released scoring runs, and the soft credit of training.
MATCH_THRESHOLD = 0.5
SOFT_THRESHOLD = 0.25
# The two models, at the revisions every released score was made with.
BGE_MODEL, BGE_REVISION = models.PINNED["bge"]
BGE_MAX_LENGTH = 512
BGE_BATCH_SIZE = 32
NLI_MODEL, NLI_REVISION = models.PINNED["nli"]
NLI_MAX_LENGTH = 2048  # tokens of premise and hypothesis together
NLI_BATCH_SIZE = 32  # SCIDEF_NLI_BATCH_SIZE of the released scoring runs
FULL_RECORD_VERSION = "M3-full-v1"
FULL_RECORD_COMPONENTS = ("terms", *FIELDS, "citations")
PAIR_SCORES_SCHEMA = (
    "taxonomy_pipeline/pair-scores-1"  # a recorded id: it keeps its name (files.py)
)


# 1. The pair score


def token_f1(left: str, right: str) -> float:
    """F1 of the shared words (with multiplicity) of two texts, after `normalize`."""
    left_tokens = ascii_words(left).split()
    right_tokens = ascii_words(right).split()
    if not left_tokens or not right_tokens:
        return float(left_tokens == right_tokens)
    overlap = sum((Counter(left_tokens) & Counter(right_tokens)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(right_tokens)
    recall = overlap / len(left_tokens)
    return 2 * precision * recall / (precision + recall)


def defscore_pair_scores(
    pairs: Pairs, similarity: PairScorer, entailment: PairScorer
) -> list[float]:
    """DefScore of each (reference, predicted) pair. `similarity` gives the BGE cosine of each pair, `entailment` the probability that the first text entails the second."""
    if not pairs:
        return []
    lexical = [token_f1(left, right) for left, right in pairs]
    embedding = similarity(pairs)
    both = entailment([*pairs, *((right, left) for left, right in pairs)])
    n = len(pairs)
    mutual = [(both[i] + both[n + i]) / 2 for i in range(n)]
    return [
        min(
            1.0, max(0.0, DEFSCORE_SLOPE * ((lex + bge + nli) / 3) + DEFSCORE_INTERCEPT)
        )
        for lex, bge, nli in zip(lexical, embedding, mutual, strict=True)
    ]


# 2. Matching


def f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def binary_prf(gold, predicted, matched_gold, matched_predictions) -> dict:
    precision = matched_predictions / predicted if predicted else float(gold == 0)
    recall = matched_gold / gold if gold else float(predicted == 0)
    return {
        "gold": gold,
        "predicted": predicted,
        "matched_gold": matched_gold,
        "matched_predictions": matched_predictions,
        "precision": precision,
        "recall": recall,
        "f1": f1(precision, recall),
    }


def binary_diagnostics(values, gold: int, predicted: int, threshold: float) -> dict:
    """Accepted matches of a gold-major flattened score matrix (`gold` rows, `predicted` columns).

    The one-to-one matching maximises the number of accepted pairs first, then their total score. "Best-gold collisions" (several predictions whose best reference is the same one) are reported separately: they are not proof of duplicates.
    """
    if len(values) != gold * predicted:
        raise ValueError("Score matrix shape does not match definition counts")
    if not 0 <= threshold <= 1 or any(not 0 <= s <= 1 for s in values):
        raise ValueError("Expected finite scores and threshold in [0, 1]")
    groups: dict[int, list] = {}
    pairs: list[dict[str, Any]] = []
    matched_gold = matched_predictions = 0
    if gold and predicted:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        matrix = np.asarray(values).reshape(gold, predicted)
        accepted = matrix >= threshold
        matched_gold = int(accepted.any(axis=1).sum())
        matched_predictions = int(accepted.any(axis=0).sum())
        for j in range(predicted):
            i = int(matrix[:, j].argmax())
            if accepted[i, j]:
                groups.setdefault(i, []).append(
                    {"prediction_index": j, "score": float(matrix[i, j])}
                )
        # One extra accepted edge outweighs every possible similarity tie-break.
        weights = accepted * (min(gold, predicted) + 1 + matrix)
        left, right = linear_sum_assignment(weights, maximize=True)
        pairs = [
            {
                "gold_index": int(i),
                "prediction_index": int(j),
                "score": float(matrix[i, j]),
            }
            for i, j in zip(left, right, strict=True)
            if accepted[i, j]
        ]
    collisions: list[dict[str, Any]] = [
        {"gold_index": i, "predictions": found}
        for i, found in groups.items()
        if len(found) > 1
    ]
    tp = len(pairs)
    return {
        "binary_threshold": threshold,
        "binary_many_to_one": binary_prf(
            gold, predicted, matched_gold, matched_predictions
        ),
        "binary_one_to_one": {
            **binary_prf(gold, predicted, tp, tp),
            "mcc": None,
            "mcc_reason": "Open extraction has no defined true-negative population",
            "tp": tp,
            "fp": predicted - tp,
            "fn": gold - tp,
        },
        "best_gold_collisions": {
            "gold_targets": len(collisions),
            "excess_predictions": sum(len(g["predictions"]) - 1 for g in collisions),
            "groups": collisions,
        },
        "one_to_one_pairs": pairs,
    }


def texts(record: dict, field: str) -> list[str]:
    return [
        str(item.get("text") or "")
        for item in record.get(field, [])
        if item.get("text")
    ]


def definitions(records: list[dict]) -> list[str]:
    return [text for record in records for text in texts(record, "definitions")]


def definition_set_details(
    gold: list[dict],
    predicted: list[dict],
    pair_scorer: PairScorer,
    threshold: float = SOFT_THRESHOLD,
    include_diagnostics: bool = False,
) -> dict:
    """All definitions of the reference records against all predicted ones.

    The soft credit (SciDef's many-to-one matching): each reference definition earns its best pair score if that is at least `threshold` (recall), and so does each predicted one (precision); `optimizer_score` pools both, as training optimises it. With `include_diagnostics`, the one-to-one matches at `threshold` and at 0.5 are added (`binary_diagnostics`).
    """
    gold_definitions = definitions(gold)
    predicted_definitions = definitions(predicted)
    if not gold_definitions or not predicted_definitions:
        value = float(not gold_definitions and not predicted_definitions)
        result: dict[str, Any] = {
            "gold": len(gold_definitions),
            "predicted": len(predicted_definitions),
            "gold_credit": value * len(gold_definitions),
            "prediction_credit": value * len(predicted_definitions),
            "precision": value,
            "recall": value,
            "f1": value,
            "optimizer_score": value,
        }
        values = []
    else:
        pairs = [(g, p) for g in gold_definitions for p in predicted_definitions]
        values = pair_scorer(pairs)
        width = len(predicted_definitions)
        gold_best = [
            max(values[start : start + width]) for start in range(0, len(values), width)
        ]
        prediction_best = [
            max(values[index] for index in range(offset, len(values), width))
            for offset in range(width)
        ]
        gold_credit = sum(value for value in gold_best if value >= threshold)
        prediction_credit = sum(
            value for value in prediction_best if value >= threshold
        )
        recall = gold_credit / len(gold_definitions)
        precision = prediction_credit / len(predicted_definitions)
        result = {
            "gold": len(gold_definitions),
            "predicted": len(predicted_definitions),
            "gold_credit": gold_credit,
            "prediction_credit": prediction_credit,
            "precision": precision,
            "recall": recall,
            "f1": f1(precision, recall),
            "optimizer_score": (gold_credit + prediction_credit)
            / (len(gold_definitions) + len(predicted_definitions)),
        }
    if include_diagnostics:
        sizes = (len(gold_definitions), len(predicted_definitions))
        result.update(binary_diagnostics(values, *sizes, threshold))
        result["binary_at_0_50"] = binary_diagnostics(values, *sizes, 0.5)
    return result


def one_to_one(
    gold: list[dict],
    predicted: list[dict],
    pair_scorer: PairScorer,
    threshold: float = MATCH_THRESHOLD,
) -> dict:
    """DefScore's one-to-one matches of the definitions at `threshold` ({tp, fp, fn, gold, predicted, precision, recall, f1, ...})."""
    return definition_set_details(
        gold, predicted, pair_scorer, threshold, include_diagnostics=True
    )["binary_one_to_one"]


# 3. The full-record score


def field_counts(gold, predicted, credit, matched) -> dict:
    p = credit / predicted if predicted else float(gold == 0)
    r = credit / gold if gold else float(predicted == 0)
    bp = matched / predicted if predicted else float(gold == 0)
    br = matched / gold if gold else float(predicted == 0)
    return dict(
        gold=gold,
        predicted=predicted,
        credit=credit,
        precision=p,
        recall=r,
        f1=2 * p * r / (p + r) if p + r else 0.0,
        binary_precision=bp,
        binary_recall=br,
        binary_f1=2 * bp * br / (bp + br) if bp + br else 0.0,
        tp=matched,
        fp=predicted - matched,
        fn=gold - matched,
    )


def full_record(
    gold: list[dict],
    predicted: list[dict],
    pair_scorer: PairScorer,
    threshold: float = MATCH_THRESHOLD,
    allow_partial: bool = False,
) -> dict:
    """The full-record score (stored as `M3-full-v1`): concept-bound, one-to-one scoring of the complete records.

    Definitions establish which records are the same concept; supporting fields cannot match records. Texts equal after `normalize` count 1 without a model call. `allow_partial` (only for localized training and dev windows) also aligns a reference record without definitions by its exact term; paper-level scoring never uses it.
    """
    if not 0 < threshold <= 1:
        raise ValueError("Full-record threshold must be in (0, 1]")
    norm = ascii_words
    semantic: dict[tuple[str, str], float] = {}

    def fill(pairs):
        requests = sorted(
            {(a, b) for a, b in pairs if norm(a) != norm(b) and (a, b) not in semantic}
        )
        values = pair_scorer(requests) if requests else []
        if len(values) != len(requests) or any(
            not math.isfinite(v) or not 0 <= v <= 1 for v in values
        ):
            raise ValueError("Invalid DefScore pair scores")
        semantic.update(zip(requests, values, strict=True))

    def similarity(a, b):
        if norm(a) == norm(b):
            return float(bool(norm(a)))
        return semantic[a, b]

    fill(
        (a["text"], b["text"])
        for g in gold
        for p in predicted
        for a in g.get("definitions", [])
        for b in p.get("definitions", [])
    )
    matrix, partial_edges = [], set()
    for i, g in enumerate(gold):
        for j, p in enumerate(predicted):
            value = max(
                (
                    similarity(a["text"], b["text"])
                    for a in g.get("definitions", [])
                    for b in p.get("definitions", [])
                ),
                default=0.0,
            )
            if (
                allow_partial
                and not g.get("definitions")
                and norm(g.get("term"))
                and norm(g.get("term")) == norm(p.get("term"))
            ):
                value = 1.0
                partial_edges.add((i, j))
            matrix.append(value)
    alignment = binary_diagnostics(matrix, len(gold), len(predicted), threshold)
    matches = alignment["one_to_one_pairs"]
    edges = {(m["gold_index"], m["prediction_index"]) for m in matches}
    matched_terms = sum(
        bool(norm(gold[i].get("term")))
        and norm(gold[i].get("term")) == norm(predicted[j].get("term"))
        for i, j in edges
    )
    components = {
        "terms": field_counts(len(gold), len(predicted), matched_terms, matched_terms)
    }
    prepared = {
        field: (
            [(i, a) for i, g in enumerate(gold) for a in g.get(field, [])],
            [(j, b) for j, p in enumerate(predicted) for b in p.get(field, [])],
        )
        for field in FIELDS
    }
    fill(
        (a["text"], b["text"])
        for left, right in prepared.values()
        for i, a in left
        for j, b in right
        if (i, j) in edges
    )
    citation_gold = citation_predicted = citation_matched = 0
    collisions = alignment["best_gold_collisions"]["excess_predictions"]
    for field, (left, right) in prepared.items():
        cells = [
            similarity(a["text"], b["text"]) if (i, j) in edges else 0.0
            for i, a in left
            for j, b in right
        ]
        matching = binary_diagnostics(cells, len(left), len(right), threshold)
        pairs = matching["one_to_one_pairs"]
        components[field] = field_counts(
            len(left), len(right), sum(p["score"] for p in pairs), len(pairs)
        )
        citation_gold += sum(len(a.get("citation", [])) for _, a in left)
        citation_predicted += sum(len(a.get("citation", [])) for _, a in right)
        for pair in pairs:
            a = Counter(
                norm(c) for c in left[pair["gold_index"]][1].get("citation", [])
            )
            b = Counter(
                norm(c) for c in right[pair["prediction_index"]][1].get("citation", [])
            )
            citation_matched += sum(v for k, v in (a & b).items() if k)
    components["citations"] = field_counts(
        citation_gold, citation_predicted, citation_matched, citation_matched
    )
    active = [v for v in components.values() if v["gold"] + v["predicted"]]
    return {
        "version": FULL_RECORD_VERSION,
        "threshold": threshold,
        "score": sum(v["f1"] for v in active) / len(active) if active else 1.0,
        "binary_macro_f1": sum(v["binary_f1"] for v in active) / len(active)
        if active
        else 1.0,
        "active_components": len(active),
        "fields": components,
        "record_matches": matches,
        "partial_term_matches": len(edges & partial_edges),
        "best_gold_collision_excess": collisions,
    }


# 4. The models and recorded scores


def pick_device(device: str):
    import torch  # ty: ignore[unresolved-import]

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA scoring requested but CUDA is unavailable")
    return torch, device


def entailment_class(labels: dict) -> int:
    """The output index of the entailment class: the one label that says "entail" (the pinned model's is `entailment`). A model whose labels name none is refused, never guessed."""
    found = [int(i) for i, label in labels.items() if "entail" in str(label).casefold()]
    if len(found) != 1:
        names = ", ".join(str(label) for label in labels.values())
        raise SystemExit(
            f"The entailment model's labels ({names}) name no single entailment class, so its "
            "entailment probability cannot be read."
        )
    return found[0]


class NLIScorer:
    """ModernBERT entailment: `directional(pairs)` gives P(first text entails the second).

    Pairs are scored in batches of `batch_size`, shortest first, and cached by their text pair.
    Step 5 calls `NLIScorer().directional(pairs)` (its own pinned model and revision are NLI_MODEL and NLI_REVISION). Another model must be given with its own pinned `revision`.
    The device defaults to SCIDEF_NLI_DEVICE, else "auto".
    """

    def __init__(
        self,
        model_name: str = NLI_MODEL,
        device: str | None = None,
        batch_size: int = NLI_BATCH_SIZE,
        revision: str | None = None,
    ):
        if revision is None and model_name != NLI_MODEL:
            raise SystemExit(
                f"The entailment model {model_name} needs a pinned revision (a commit of its "
                f"Hugging Face repository): pass revision=…. Only {NLI_MODEL} has one by "
                f"default ({NLI_REVISION})."
            )
        revision = revision or NLI_REVISION
        from transformers import (  # ty: ignore[unresolved-import]
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self.torch, self.device = pick_device(
            device or os.environ.get("SCIDEF_NLI_DEVICE") or "auto"
        )
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                model_name, revision=revision
            )
            .to(self.device)
            .eval()
        )
        self.entailment = entailment_class(
            getattr(self.model.config, "id2label", {}) or {}
        )
        self.cache: dict[tuple[str, str], float] = {}
        self.lock = threading.Lock()

    def directional(self, pairs: Pairs) -> list[float]:
        with self.lock:
            missing = sorted(
                {pair for pair in pairs if pair not in self.cache},
                key=lambda pair: len(pair[0]) + len(pair[1]),
            )
            for start in range(0, len(missing), self.batch_size):
                batch = missing[start : start + self.batch_size]
                inputs = self.tokenizer(  # ty: ignore[call-non-callable]
                    [left for left, _ in batch],
                    [right for _, right in batch],
                    padding=True,
                    truncation=True,
                    max_length=min(
                        getattr(
                            self.model.config, "max_position_embeddings", NLI_MAX_LENGTH
                        ),
                        NLI_MAX_LENGTH,
                    ),
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                with self.torch.no_grad():
                    scores = self.torch.softmax(self.model(**inputs).logits, dim=-1)[
                        :, self.entailment
                    ]
                for pair, score in zip(
                    batch, scores.detach().cpu().tolist(), strict=True
                ):
                    self.cache[pair] = float(score)
        return [self.cache[pair] for pair in pairs]


class BGEScorer:
    """BGE-base: `scores(pairs)` gives the cosine of the two texts' CLS vectors."""

    def __init__(self, device: str | None = None):
        from transformers import (  # ty: ignore[unresolved-import]
            AutoModel,
            AutoTokenizer,
        )

        self.torch, self.device = pick_device(
            device or os.environ.get("SCIDEF_NLI_DEVICE") or "auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL, revision=BGE_REVISION)
        self.model = (
            AutoModel.from_pretrained(BGE_MODEL, revision=BGE_REVISION)
            .to(self.device)
            .eval()
        )
        self.cache: dict[str, object] = {}
        self.lock = threading.Lock()

    def scores(self, pairs: Pairs) -> list[float]:
        torch = self.torch
        with self.lock:
            missing = sorted(
                {text for pair in pairs for text in pair if text not in self.cache},
                key=len,
            )
            for start in range(0, len(missing), BGE_BATCH_SIZE):
                batch = missing[start : start + BGE_BATCH_SIZE]
                inputs = self.tokenizer(  # ty: ignore[call-non-callable]
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=BGE_MAX_LENGTH,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                with torch.no_grad():
                    vectors = self.model(**inputs).last_hidden_state[:, 0]
                    vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
                # Similarity lookup stays on CPU; avoid one CUDA sync per pair.
                self.cache.update(zip(batch, vectors.cpu(), strict=True))
            return [
                float(torch.dot(self.cache[left], self.cache[right]))
                for left, right in pairs
            ]


def models(device: str, batch_size: int = NLI_BATCH_SIZE):
    """(similarity, entailment) of the real models, each loaded on its first call."""
    loaded: dict[str, object] = {}

    def similarity(pairs: Pairs) -> list[float]:
        if "bge" not in loaded:
            loaded["bge"] = BGEScorer(device)
        return loaded["bge"].scores(pairs)  # ty: ignore[unresolved-attribute]

    def entailment(pairs: Pairs) -> list[float]:
        if "nli" not in loaded:
            loaded["nli"] = NLIScorer(device=device, batch_size=batch_size)
        return loaded["nli"].directional(pairs)  # ty: ignore[unresolved-attribute]

    return similarity, entailment


class NotRecorded(Exception):
    """A replayed scoring asked for a pair score that was not recorded."""


def model_names() -> dict[str, str]:
    return {
        "similarity_model": f"{BGE_MODEL}@{BGE_REVISION}",
        "entailment_model": f"{NLI_MODEL}@{NLI_REVISION}",
    }


def recorded(path: Path):
    """(similarity, entailment) answered from a file that `recording` wrote.

    The file must name the pinned models. A pair that is not in it raises `NotRecorded`.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != PAIR_SCORES_SCHEMA:
        raise SystemExit(f"{path}: not a pair-score file ({PAIR_SCORES_SCHEMA}).")
    for key, name in model_names().items():
        if data.get(key) != name:
            raise SystemExit(f"{path}: recorded with {data.get(key)}, not {name}.")
    text = data["texts"]

    def lookup(kind: str):
        table = {(text[i], text[j]): value for i, j, value in data[kind]}

        def answer(pairs: Pairs) -> list[float]:
            found = []
            for pair in pairs:
                if pair not in table:
                    left, right = (t[:40] + ("…" if len(t) > 40 else "") for t in pair)
                    raise NotRecorded(
                        f"no recorded {kind} for ({left!r}, {right!r}) in {path}"
                    )
                found.append(table[pair])
            return found

        return answer

    return lookup("similarity"), lookup("entailment")


def recording(similarity: PairScorer, entailment: PairScorer):
    """(similarity, entailment, write): the same scorers, remembering every pair they answer; `write(path)` saves them in the format `recorded` reads."""
    seen: dict[str, dict[tuple[str, str], float]] = {
        "similarity": {},
        "entailment": {},
    }

    def remember(kind: str, scorer: PairScorer) -> PairScorer:
        def answer(pairs: Pairs) -> list[float]:
            values = scorer(pairs)
            seen[kind].update(zip(pairs, values, strict=True))
            return values

        return answer

    def write(path: Path) -> None:
        index: dict[str, int] = {}
        for kind in seen:
            for pair in sorted(seen[kind]):
                for t in pair:
                    index.setdefault(t, len(index))
        data = {
            "schema": PAIR_SCORES_SCHEMA,
            **model_names(),
            "texts": list(index),
            **{
                kind: [
                    [index[a], index[b], value]
                    for (a, b), value in sorted(seen[kind].items())
                ]
                for kind in seen
            },
        }
        with Path(path).open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(data, ensure_ascii=False) + "\n")

    return remember("similarity", similarity), remember("entailment", entailment), write
