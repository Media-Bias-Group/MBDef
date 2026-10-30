"""Fast NLI scorer for Step 5: the entailment model run in the fastest setting that matched the exact scorer.

Model (unchanged): tasksource/ModernBERT-large-nli at revision ca476cb923a8637073d4ceb0f19f7fc236e260d4, the entailment model of the extraction score DefScore (the paper's pair score of word overlap, embedding similarity and entailment; ``mbdef.evaluate.metric.NLIScorer``, its constants NLI_MODEL and NLI_REVISION). The exact scorer, ``NLIScorer().directional(pairs)``, returns softmax(logits)[entailment] with pair[0] as the premise and pair[1] as the hypothesis; this one returns the same within about 0.01.

Execution setting = benchmark config ``fp16-sdpa-budget16k-compile-default-dyn-pipe`` (in the NLI benchmark: 1,572 against 279 pairs/s on a 24 GB GPU, largest |dP(entailment)| 8.4e-3 and no flip at 0.5 on 2,000 parity pairs):

- fp16 weights and activations, sdpa attention, softmax in fp32;
- pairs sorted by token length; batches filled up to 16,384 padded tokens (rows x longest row);
- ``torch.compile`` default mode, dynamic shapes; batches of one pair are padded to two rows (a one-row batch makes the dynamic-shape graph recompile);
- tokenisation off the GPU's critical path. Instead of tokenising every pair on a side thread (the benchmark's ``-pipe``), every distinct text is tokenised once and pair inputs are assembled as ``[CLS] premise [SEP] hypothesis [SEP]``, which is exactly what the tokenizer's pair template produces (checked at start-up; pairs longer than the model's 2,048-token cap are encoded by the tokenizer itself, with its truncation). Batch assembly is vectorised numpy and runs while the GPU works on the previous batch (no synchronisation inside the loop).

Interface::

    nli = CompiledNLI(device="cuda")
    P = nli.probs(pairs)      # np.ndarray [N, 3] float32, columns in nli.labels order (the model's id2label)
    e = nli.entail(pairs)     # np.ndarray [N] float32, P(entailment)
    P = nli.probs_indexed(texts, premise_idx, hypothesis_idx)   # same, pairs given as indices into a text list

Pairs are deduplicated internally; results come back in input order. For this model ``nli.labels`` is read from ``config.id2label`` (see ``nli.labels`` / ``nli.entail_index``).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from mbdef.evaluate.metric import NLI_MODEL as MODEL
from mbdef.evaluate.metric import NLI_REVISION as REVISION
from mbdef.evaluate.metric import entailment_class

TOKEN_BUDGET = 16384
MAX_LENGTH_CAP = 2048  # same cap as NLIScorer: min(max_position_embeddings, 2048)

_PROBE_PAIRS = [
    ("Media bias is a slanted report.", "Bias exists."),
    ("  leading and trailing spaces  ", "line\nbreak\tand tab"),
    ("Umlaute: Žluťoučký, naïve, “quotes” — dash", "日本語 and emoji 🙂"),
    ("", "empty premise"),
    ("x", ""),
    (
        "framing: the selection of some aspects of a perceived reality",
        "framing is a type of media bias.",
    ),
]


class CompiledNLI:
    """Fast fp16 + torch.compile scorer for DefScore's NLI model. See the module docstring."""

    def __init__(
        self,
        device: str = "cuda",
        *,
        budget: int = TOKEN_BUDGET,
        compile: bool | None = None,
        warmup: bool = True,
        model_name: str = MODEL,
        revision: str = REVISION,
    ):
        import torch  # ty: ignore[unresolved-import]
        from transformers import (  # ty: ignore[unresolved-import]
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self.torch = torch
        self.device = device
        on_gpu = device.startswith("cuda")
        if on_gpu and not torch.cuda.is_available():
            raise RuntimeError("CompiledNLI: CUDA requested but not available")
        # fp16 + compile is the benchmarked GPU setting; on CPU (quick trials and tests only) run fp32, eager.
        self.dtype = torch.float16 if on_gpu else torch.float32
        self.compiled = on_gpu if compile is None else compile
        self.budget = int(budget)
        torch.set_float32_matmul_precision("highest")
        self.tok: Any = AutoTokenizer.from_pretrained(
            model_name, revision=revision
        )  # typed loosely: HF returns a union
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, revision=revision, dtype=self.dtype, attn_implementation="sdpa"
        )
        self.model = model.to(device).eval()
        cfg = self.model.config
        id2label = {int(k): str(v) for k, v in (cfg.id2label or {}).items()}
        self.labels = [id2label[i] for i in range(len(id2label))]
        self.entail_index = entailment_class(id2label)
        self.max_len = min(
            getattr(cfg, "max_position_embeddings", MAX_LENGTH_CAP), MAX_LENGTH_CAP
        )
        self.cls, self.sep, self.pad = (
            self.tok.cls_token_id,
            self.tok.sep_token_id,
            self.tok.pad_token_id,
        )
        self.stats = dict(
            pairs=0,
            unique_pairs=0,
            batches=0,
            real_tokens=0,
            padded_tokens=0,
            tokenize_s=0.0,
            gpu_s=0.0,
        )
        self._check_template()
        if self.compiled:
            import torch._dynamo  # ty: ignore[unresolved-import]

            for key in ("recompile_limit", "cache_size_limit"):
                if hasattr(torch._dynamo.config, key):
                    setattr(torch._dynamo.config, key, 64)
            self.forward = torch.compile(self.model, dynamic=True)  # default mode
        else:
            self.forward = self.model
        self.warmup_s = 0.0
        if warmup and self.compiled:
            t = time.perf_counter()
            self._warmup()
            self.warmup_s = time.perf_counter() - t

    # ------------------------------------------------------------------ public API
    def probs(self, pairs) -> np.ndarray:
        """[N, 3] float32 class probabilities (columns = self.labels) for (premise, hypothesis) pairs, input order."""
        index: dict[str, int] = {}
        n = len(pairs)
        prem = np.empty(n, dtype=np.int64)
        hyp = np.empty(n, dtype=np.int64)
        for k, (a, b) in enumerate(pairs):
            prem[k] = index.setdefault(a, len(index))
            hyp[k] = index.setdefault(b, len(index))
        return self.probs_indexed(list(index), prem, hyp)

    def entail(self, pairs) -> np.ndarray:
        """[N] float32 P(entailment), pair[0] = premise (the contract of NLIScorer.directional)."""
        return self.probs(pairs)[:, self.entail_index]

    def probs_indexed(self, texts, premise_idx, hypothesis_idx) -> np.ndarray:
        """Like probs(), pairs given as indices into ``texts``. Deduplicates (premise, hypothesis) index pairs."""
        prem = np.asarray(premise_idx, dtype=np.int64)
        hyp = np.asarray(hypothesis_idx, dtype=np.int64)
        if prem.size == 0:
            return np.zeros((0, len(self.labels)), dtype=np.float32)
        nt = len(texts)
        uniq, inv = np.unique(prem * nt + hyp, return_inverse=True)
        ua, ub = uniq // nt, uniq % nt
        self.stats["pairs"] += int(prem.size)
        self.stats["unique_pairs"] += int(uniq.size)
        out = self._score(texts, ua, ub)
        return out[inv.reshape(-1)]

    # ------------------------------------------------------------------ internals
    def _encode_texts(self, texts, used):
        """Token ids (no special tokens) of the used texts -> flat array + per-text offset/length (index = text id)."""
        t = time.perf_counter()
        ids = self.tok(
            [texts[i] for i in used],
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        lens = np.fromiter(map(len, ids), dtype=np.int64, count=len(ids))
        flat = np.fromiter(
            (x for row in ids for x in row), dtype=np.int64, count=int(lens.sum())
        )
        off = np.zeros(len(texts), dtype=np.int64)
        ln = np.zeros(len(texts), dtype=np.int64)
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        off[used], ln[used] = starts, lens
        # special slots appended after the real tokens: CLS, SEP, PAD
        ext = np.concatenate(
            [flat, np.array([self.cls, self.sep, self.pad], dtype=np.int64)]
        )
        self.stats["tokenize_s"] += time.perf_counter() - t
        return ext, off, ln, flat.size

    def _check_template(self):
        """Hand-assembled ids must equal the tokenizer's pair encoding (the reference path)."""
        texts = [t for p in _PROBE_PAIRS for t in p]
        ext, off, ln, base = self._encode_texts(texts, np.arange(len(texts)))
        a = np.arange(0, len(texts), 2)
        ids, mask, _ = self._assemble(ext, off, ln, base, a, a + 1)
        for r, (p, h) in enumerate(_PROBE_PAIRS):
            # batch call, as NLIScorer.directional does (a single call treats an empty hypothesis as "no pair")
            ref = self.tok([p], [h], truncation=True, max_length=self.max_len)[
                "input_ids"
            ][0]
            got = ids[r, : int(mask[r].sum())].tolist()
            if got != ref:
                raise RuntimeError(
                    f"CompiledNLI: pair template mismatch for {(p, h)!r}: {got} != {ref}"
                )

    @staticmethod
    def _assemble_static(ext, off, ln, base, a, b):
        la, lb = ln[a][:, None], ln[b][:, None]
        n = (la + lb + 3)[:, 0]
        L = int(n.max())
        pos = np.arange(L, dtype=np.int64)[None, :]
        cls_slot, sep_slot, pad_slot = base, base + 1, base + 2
        src = np.where(pos == 0, cls_slot, pad_slot)
        src = np.where((pos >= 1) & (pos <= la), off[a][:, None] + pos - 1, src)
        src = np.where((pos == la + 1) | (pos == la + lb + 2), sep_slot, src)
        src = np.where(
            (pos >= la + 2) & (pos <= la + lb + 1), off[b][:, None] + pos - la - 2, src
        )
        return ext[src], (pos < n[:, None]).astype(np.int64), n

    def _assemble(self, ext, off, ln, base, a, b):
        return self._assemble_static(ext, off, ln, base, a, b)

    def _batches(self, order, n):
        """Token-budget batches over pairs already sorted by length (same rule as the benchmark)."""
        out, cur, cur_l = [], [], 0
        for i in order:
            li = int(n[i])
            if cur and (len(cur) + 1) * max(cur_l, li) > self.budget:
                out.append(cur)
                cur, cur_l = [], 0
            cur.append(i)
            cur_l = max(cur_l, li)
        if cur:
            out.append(cur)
        return out

    def _run(self, ids, mask, nreal):
        torch = self.torch
        if (
            ids.shape[0] < 2
        ):  # one-row batch: pad with a copy (avoids a dynamic-shape recompile)
            ids, mask = np.repeat(ids, 2, axis=0), np.repeat(mask, 2, axis=0)
        if self.device.startswith("cuda"):
            ids_t = (
                torch.from_numpy(ids).pin_memory().to(self.device, non_blocking=True)
            )
            mask_t = (
                torch.from_numpy(mask).pin_memory().to(self.device, non_blocking=True)
            )
        else:
            ids_t, mask_t = torch.from_numpy(ids), torch.from_numpy(mask)
        with torch.inference_mode():
            logits = self.forward(input_ids=ids_t, attention_mask=mask_t).logits
            return torch.softmax(logits[:nreal].float(), dim=-1).clone()

    def _score(self, texts, ua, ub) -> np.ndarray:
        torch = self.torch
        used = np.unique(np.concatenate([ua, ub]))
        ext, off, ln, base = self._encode_texts(texts, used)
        n = ln[ua] + ln[ub] + 3
        long_ = np.flatnonzero(n > self.max_len)
        short = np.flatnonzero(n <= self.max_len)
        order = short[np.argsort(n[short], kind="stable")]
        t = time.perf_counter()
        outs, idx = [], []
        for batch in self._batches(order, n):
            bi = np.asarray(batch)
            ids, mask, nn = self._assemble(ext, off, ln, base, ua[bi], ub[bi])
            outs.append(self._run(ids, mask, len(bi)))
            idx.append(bi)
            self.stats["batches"] += 1
            self.stats["real_tokens"] += int(nn.sum())
            self.stats["padded_tokens"] += int(max(2, len(bi)) * ids.shape[1])
        if long_.size:  # longer than the model cap: let the tokenizer truncate exactly as the reference does
            for s in range(0, long_.size, 8):
                bi = long_[s : s + 8]
                enc = self.tok(
                    [texts[i] for i in ua[bi]],
                    [texts[i] for i in ub[bi]],
                    truncation=True,
                    max_length=self.max_len,
                    padding=True,
                    return_tensors="np",
                )
                outs.append(
                    self._run(
                        enc["input_ids"].astype(np.int64),
                        enc["attention_mask"].astype(np.int64),
                        len(bi),
                    )
                )
                idx.append(bi)
        res = np.empty((ua.size, len(self.labels)), dtype=np.float32)
        if outs:
            res[np.concatenate(idx)] = torch.cat(outs).cpu().numpy()
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        self.stats["gpu_s"] += time.perf_counter() - t
        return res

    def _warmup(self):
        """Compile once up front on synthetic pairs spanning short to long rows (both small and full batches)."""
        words = "media bias framing selection slant coverage agenda gatekeeping tone".split()
        texts, prem, hyp = [], [], []
        for k, L in enumerate((4, 12, 30, 60, 120, 250)):
            for r in range(3):
                texts.append(
                    " ".join(words[(k + r + i) % len(words)] for i in range(L))
                )
        for i in range(len(texts)):
            for j in range(len(texts)):
                if i != j:
                    prem.append(i)
                    hyp.append(j)
        stats = dict(self.stats)
        self.probs_indexed(
            texts, prem[:1], hyp[:1]
        )  # one-pair batch (padded to two rows)
        self.probs_indexed(texts, prem, hyp)
        self.stats = stats
