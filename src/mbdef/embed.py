"""The embedding cache: sentence vectors computed once on a GPU, found again by their text.

Shared by Step 5 and the later taxonomy step (docs/step-5-post-processing/embed.md). It reads and writes the stored cache's files byte for byte. `build` (`mbdef embed build`) fills the cache that Step 5 reads; it runs its model with vLLM, which supports Python 3.14 but not 3.15 or newer, so on a newer Python it stops with `VLLM_MISSING`.

Vectors are L2-normalised float16 rows in `<root>/<model>/vectors.npy`, with the sha1 of each row's text in `keys.txt` (one per line, no final newline). `manifest.json` pins everything that changes a vector: model revision, prompt, pooling, precision and quantisation, maximum length and engine. New texts are added only when the current settings equal the manifest; otherwise writing fails. The stored Qwen3-Embedding-8B cache is `embeddings/qwen3-emb-8b/` in the data root.
"""

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np

from mbdef.postprocess import groundtruth

TASK = "Represent the meaning of this scholarly concept and its definition"
# Prompt formats exactly as each model card writes them: Qwen3-Embedding and F2LLM (v1) use "Query:" directly followed by the text; F2LLM-v2's code sample uses "Query: " with a space.
INSTRUCT = f"Instruct: {TASK}\nQuery:"
INSTRUCT_V2 = f"Instruct: {TASK}\nQuery: "
MODELS = {
    # Decoder models: EOS (last-token) pooling, the instruction on every text (symmetric similarity), tokenised by the model's own tokenizer, which appends EOS after truncation.
    # Revisions pinned.
    "qwen3-emb-0.6b": dict(
        repo="Qwen/Qwen3-Embedding-0.6B",
        revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        pooling="last",
        prefix=INSTRUCT,
    ),
    "qwen3-emb-8b": dict(
        repo="Qwen/Qwen3-Embedding-8B",
        revision="1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
        pooling="last",
        prefix=INSTRUCT,
        engine="vllm",
    ),
    "f2llm-4b": dict(
        repo="codefuse-ai/F2LLM-4B",
        revision="6507d2322910a7a8449af8dfc39a165cc3a7e0bc",
        pooling="last",
        prefix=INSTRUCT,
        engine="vllm",
    ),
    # 28 GB in BF16 does not fit a 24 GB GPU: FP8 there; `bf16=True` keeps BF16 on 80 GB GPUs.
    "f2llm-v2-14b": dict(
        repo="codefuse-ai/F2LLM-v2-14B",
        revision="8be6125a8acb7c6e103ae8dcd754e098a6ddc90d",
        pooling="last",
        prefix=INSTRUCT_V2,
        engine="vllm",
        quantization="fp8",
    ),
    # Encoder model: CLS pooling, no instruction (as in its model card).
    "bge-m3": dict(
        repo="BAAI/bge-m3",
        revision="5617a9f61b028005a4858fdac845db406aefb181",
        pooling="cls",
        prefix="",
    ),
}
HF_MAX_LENGTH, VLLM_MAX_LENGTH = 512, 1024
VLLM_CHUNK = 20000  # texts per vLLM call: bounded memory, one model load
# The share of the GPU's memory vLLM may take (the value the stored cache was built with; lower it when another program holds GPU memory). It and BF16 for F2LLM-v2 are arguments with pinned defaults, never read from the environment.
GPU_SHARE = 0.88


def config(model, bf16=False):
    """Everything that determines a vector for ``model``; ``bf16`` keeps a model that is quantised by default (F2LLM-v2-14B) in BF16."""
    spec = MODELS[model]
    vllm = spec.get("engine") == "vllm"
    quant = None if bf16 else spec.get("quantization")
    return dict(
        model=model,
        repo=spec["repo"],
        revision=spec["revision"],
        pooling=spec["pooling"],
        prefix=spec["prefix"],
        engine="vllm" if vllm else "transformers",
        dtype="bfloat16" if vllm else "float16",
        quantization=quant,
        max_length=VLLM_MAX_LENGTH if vllm else HF_MAX_LENGTH,
        normalize="l2",
        storage="float16",
        text_views="definition_text=term+': '+text; term_text=term",
    )


def definition_text(term, text):
    """The one text view used for every definition vector (production and ground truth alike)."""
    return f"{term}: {text}"


def term_text(term):
    return term


def text_hash(text):
    return hashlib.sha1(text.encode()).hexdigest()


class Cache:
    def __init__(self, root, model):
        self.dir = Path(root) / model
        self.dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        kf, vf, mf = (
            self.dir / "keys.txt",
            self.dir / "vectors.npy",
            self.dir / "manifest.json",
        )
        self.keys = (
            kf.read_text().split("\n") if kf.exists() and kf.stat().st_size else []
        )
        self.vectors = np.load(vf) if vf.exists() else None
        self.index = {k: i for i, k in enumerate(self.keys)}
        self.manifest = json.loads(mf.read_text()) if mf.exists() else None

    def check_writable(self, cfg):
        """Refuse to mix vectors from different configurations in one cache."""
        if self.manifest is None:
            if self.keys:
                raise ValueError(
                    f"{self.dir}: vectors without manifest.json; adopt them explicitly first"
                )
            (self.dir / "manifest.json").write_text(json.dumps(cfg, indent=1))
            self.manifest = cfg
        elif {k: v for k, v in self.manifest.items() if k != "adopted"} != cfg:
            diff = {
                k: (self.manifest.get(k), cfg.get(k))
                for k in (set(cfg) | set(self.manifest)) - {"adopted"}
                if self.manifest.get(k) != cfg.get(k)
            }
            raise ValueError(f"{self.dir}: configuration differs from manifest: {diff}")

    def missing(self, texts):
        seen, out = set(self.index), []
        for t in texts:
            h = text_hash(t)
            if h not in seen:
                seen.add(h)
                out.append(t)
        return out

    def add(self, texts, vectors):
        if not texts:
            return
        start = len(self.keys)
        self.keys += [text_hash(t) for t in texts]
        self.index.update({k: start + i for i, k in enumerate(self.keys[start:])})
        vectors = vectors.astype(np.float16)
        self.vectors = (
            vectors if self.vectors is None else np.vstack([self.vectors, vectors])
        )
        np.save(self.dir / "vectors.npy", self.vectors)
        (self.dir / "keys.txt").write_text("\n".join(self.keys))

    def get(self, texts):
        return self.vectors[  # ty: ignore[not-subscriptable]
            [self.index[text_hash(t)] for t in texts]
        ].astype(np.float32)


VLLM_MISSING = (
    "This step runs its model with vLLM, which is not installed (vLLM supports Python 3.14 but not 3.15 or newer). "
    "Run it in a Python 3.14 environment: `uv sync --python 3.14`."
)


def encode_vllm(texts, cfg, gpu_share=GPU_SHARE):
    try:
        from vllm import LLM  # ty: ignore[unresolved-import]
        from vllm.config import PoolerConfig  # ty: ignore[unresolved-import]
        from vllm.inputs import TokensPrompt  # ty: ignore[unresolved-import]
    except ImportError:
        raise SystemExit(VLLM_MISSING) from None
    from transformers import AutoTokenizer  # ty: ignore[unresolved-import]

    max_length = cfg["max_length"]
    tok = AutoTokenizer.from_pretrained(cfg["repo"], revision=cfg["revision"])
    ids = tok(  # ty: ignore[call-non-callable]
        [cfg["prefix"] + t for t in texts], truncation=True, max_length=max_length
    )["input_ids"]
    llm = LLM(
        model=cfg["repo"],
        revision=cfg["revision"],
        tokenizer_revision=cfg["revision"],
        runner="pooling",
        convert="embed",
        dtype=cfg["dtype"],
        quantization=cfg["quantization"],
        max_model_len=max_length,
        gpu_memory_utilization=gpu_share,
        pooler_config=PoolerConfig(seq_pooling_type="LAST"),
        max_num_batched_tokens=32768,
        max_num_seqs=512,
        enable_prefix_caching=False,
        seed=0,
    )
    import gc

    import torch  # ty: ignore[unresolved-import]

    chunks = []
    for s in range(0, len(ids), VLLM_CHUNK):
        prompts = [TokensPrompt(prompt_token_ids=x) for x in ids[s : s + VLLM_CHUNK]]
        try:  # raw pooled tensors: avoids one Python float per dimension
            outs = llm.encode(prompts, pooling_task="embed", use_tqdm=True)
            v = torch.stack([o.outputs.data for o in outs]).float()
        except TypeError, AttributeError:
            outs = llm.embed(prompts, use_tqdm=True)
            v = torch.tensor([o.outputs.embedding for o in outs], dtype=torch.float32)
        v = torch.nn.functional.normalize(v, dim=-1)
        chunks.append(v.cpu().numpy().astype(np.float16))
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return np.vstack(chunks)


def encode(texts, model, batch_tokens=65536, bf16=False, gpu_share=GPU_SHARE):
    import torch  # ty: ignore[unresolved-import]
    from transformers import AutoModel, AutoTokenizer  # ty: ignore[unresolved-import]

    cfg = config(model, bf16)
    if cfg["engine"] == "vllm":
        return encode_vllm(texts, cfg, gpu_share)
    max_length = cfg["max_length"]
    tok = AutoTokenizer.from_pretrained(
        cfg["repo"],
        revision=cfg["revision"],
        padding_side="left" if cfg["pooling"] == "last" else "right",
    )
    net = (
        AutoModel.from_pretrained(
            cfg["repo"], revision=cfg["revision"], dtype=torch.float16
        )
        .cuda()
        .eval()
    )
    inputs = [cfg["prefix"] + t for t in texts]
    lengths = [
        len(tok(t, truncation=True, max_length=max_length)["input_ids"])  # ty: ignore[call-non-callable]
        for t in inputs
    ]
    order = np.argsort(lengths)[::-1]
    out = np.zeros((len(texts), net.config.hidden_size), dtype=np.float32)
    i = 0
    with torch.inference_mode():
        while i < len(order):
            n = max(1, min(512, batch_tokens // max(lengths[order[i]], 1)))
            idx = order[i : i + n]
            enc = tok(  # ty: ignore[call-non-callable]
                [inputs[j] for j in idx],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to("cuda")
            hidden = net(**enc).last_hidden_state
            if cfg["pooling"] == "last":
                vec = hidden[:, -1]
            else:
                vec = hidden[:, 0]
            vec = torch.nn.functional.normalize(vec.float(), dim=-1)
            out[idx] = vec.cpu().numpy()
            i += n
    del net
    torch.cuda.empty_cache()
    return out


def ensure(cache_root, model, texts, bf16=False, gpu_share=GPU_SHARE):
    cache = Cache(cache_root, model)
    todo = cache.missing(texts)
    if todo:
        cache.check_writable(config(model, bf16))
        cache.add(todo, encode(todo, model, bf16=bf16, gpu_share=gpu_share))
    return cache


# `mbdef embed build`: the cache that Step 5 reads, from the corpus store, the ground truth and the texts of the embedding comparison (docs/step-5-post-processing/embed.md).
MODEL = "qwen3-emb-8b"
# The stored cache was filled in two passes as the corpus grew: first the seed papers, then the seed and the closure papers, once the closure papers were in the corpus store. Its row order follows the passes.
PASSES = (("seeds",), ("seeds", "closure"))
# The 55,072 texts of the comparison of five embedding models, in the data root: 7,628 seed definitions with their term masked as "[X]" and 47,444 terms; the stored cache holds them too.
COMPARISON = "frozen/artifacts/experiments/taxonomy-curation-v2/bench-extra-texts.json"
SAMPLE_SEED = 20261023  # the seed of `build --sample`


def ground_truth_texts(paths):
    """'<term>: <definition>' of every ground-truth definition in file order (every record of every file, all splits; no label is read), and the terms of the records that have one."""
    definitions, terms = [], set()
    for path in paths:
        for paper in json.loads(Path(path).read_text(encoding="utf-8"))["papers"]:
            for record in paper["definition_records"]:
                texts = [d["text"] for d in record["definitions"] if d["text"].strip()]
                definitions += [definition_text(record["term"], t) for t in texts]
                if texts:
                    terms.add(record["term"])
    return definitions, terms


def pass_texts(db, corpora, ground_truth, comparison):
    """The texts of one pass over the store's papers of CORPORA, in the order they are embedded:
    '<term>: <definition>' of every definition in the store's row order, the ground truth's definitions, every term of these papers and of the ground truth sorted, the comparison texts."""
    marks = ",".join("?" * len(corpora))
    rows = db.execute(
        "SELECT m.display_term, o.text FROM occurrences o JOIN mentions m USING(mention_id)"
        " JOIN papers p ON p.paper_id = m.paper_id"
        f" WHERE o.field = 'definitions' AND p.corpus IN ({marks}) ORDER BY o.rowid",
        corpora,
    ).fetchall()
    terms = {
        term
        for (term,) in db.execute(
            "SELECT m.display_term FROM mentions m JOIN papers p USING(paper_id)"
            f" WHERE p.corpus IN ({marks})",
            corpora,
        )
    }
    gt_definitions, gt_terms = ground_truth
    return (
        [definition_text(term, text) for term, text in rows]
        + gt_definitions
        + [term_text(t) for t in sorted(terms | gt_terms)]
        + comparison
    )


def build(
    out,
    store,
    comparison,
    ground_truth=None,
    model=MODEL,
    replay=None,
    sample=None,
    gpu_share=GPU_SHARE,
):
    """Embed every text of the passes that the cache OUT/<model>/ lacks, pass by pass. With REPLAY, each vector comes from that recorded cache folder instead of the GPU (a check on a CPU); with SAMPLE, only that many texts of the passes are embedded, drawn with SAMPLE_SEED."""
    db = sqlite3.connect(
        Path(store).resolve().as_uri() + "?mode=ro&immutable=1", uri=True
    )
    try:
        gt = ground_truth_texts(ground_truth or [p for p, _ in groundtruth.paths()])
        extra = json.loads(Path(comparison).read_text(encoding="utf-8"))
        plan = [(", ".join(c), pass_texts(db, c, gt, extra)) for c in PASSES]
    finally:
        db.close()
    if sample:
        every = list(dict.fromkeys(t for _, texts in plan for t in texts))
        picked = np.random.default_rng(SAMPLE_SEED).choice(
            len(every), sample, replace=False
        )
        plan = [(f"a sample of {len(every):,}", [every[i] for i in sorted(picked)])]
    encoder = None
    if replay:
        from mbdef.postprocess.embeddings import EmbeddingCache

        encoder = EmbeddingCache(Path(replay).resolve()).vectors
    for number, (label, texts) in enumerate(plan, 1):
        cache = Cache(out, model)
        todo = cache.missing(texts)
        if todo:
            cache.check_writable(config(model))
            try:
                vectors = (
                    encoder(todo)
                    if encoder
                    else encode(todo, model, gpu_share=gpu_share)
                )
            except KeyError as missing:
                raise SystemExit(
                    f"{replay} has no recorded vector: {missing}"
                ) from None
            cache.add(todo, vectors)
        print(f"pass {number} ({label}): {len(texts):,} texts, {len(todo):,} new")
    write_receipt(
        out, model, store, comparison, ground_truth, replay, sample, gpu_share
    )
    return 0


def write_receipt(
    out, model, store, comparison, ground_truth, replay, sample, gpu_share
):
    from importlib import metadata

    from mbdef import files

    folder = Path(out) / model
    try:
        vllm = None if replay else metadata.version("vllm")
    except metadata.PackageNotFoundError:
        vllm = None
    settings = dict(
        model=model,
        manifest=config(model),
        passes=[list(c) for c in PASSES],
        replay=None if replay is None else str(replay),
        sample=sample,
        sample_seed=SAMPLE_SEED if sample else None,
        gpu_share=gpu_share,
        vllm=vllm,
    )
    inputs = {"store": Path(store), "comparison": Path(comparison)}
    inputs |= {f"ground_truth_{i}": Path(p) for i, p in enumerate(ground_truth, 1)}
    if replay:
        inputs["replay"] = Path(replay)
    names = ("keys.txt", "vectors.npy", "manifest.json")
    receipt = files.receipt(
        "mbdef embed build",
        settings,
        inputs,
        {name: folder / name for name in names if (folder / name).exists()},
    )
    files.write_json(Path(out) / files.RECEIPT, receipt)
    rows = len((folder / "keys.txt").read_text().split("\n"))
    print(
        f"cache {folder}: {rows:,} rows, keys.txt sha256 {receipt['outputs']['keys.txt']['sha256']}"
    )
