# The embedding cache: `mbdef embed`

| | |
|---|---|
| **Paper step** | Step 5 (post-processing) reads the definition vectors of this cache; the later taxonomy step reads the same cache |
| **Code** | [`src/mbdef/embed.py`](../../src/mbdef/embed.py) |
| **Command** | `mbdef embed build`: fill the cache from the corpus store and the ground truth (on a GPU; `--replay` checks it on a CPU). It runs its model with vLLM, which supports Python 3.14 but not 3.15 or newer: there the build stops with a message |
| **Data** | the stored cache `embeddings/qwen3-emb-8b/` in the data root ([docs/setup/data-root.md](../setup/data-root.md)): `mbdef data fetch embeddings/`; the build's inputs are [below](#inputs) |
| **Checked by** | [`tests/test_embed.py`](../../tests/test_embed.py): the build on a miniature corpus, round trip, byte equality with the fixture cache, Step 5's reader; three slow tests on the stored cache |

## What it is

Embedding a text with an 8-billion-parameter model takes a GPU, so each text is embedded once and its vector kept in a cache. Step 5 reads the definition vectors of the stored Qwen3-Embedding-8B cache, and the later taxonomy step will read the same vectors. The package writes the stored cache's format byte for byte: the fixture cache below was written by the original implementation of the cache, and the package's files equal it (tested).
`mbdef embed build` fills a cache the way the stored one was filled.

A cache is one folder per model, `<root>/<model>/`, with three files:

| File | What it holds |
|---|---|
| `vectors.npy` | one row per text: the L2-normalised vector, stored as float16 (NumPy's `.npy` format) |
| `keys.txt` | the sha1 of each row's text, one per line in row order, with no final newline |
| `manifest.json` | the settings that decide a vector: model, revision, prompt, pooling, precision, quantisation, maximum length, engine, text view; `json.dumps(…, indent=1)`, no final newline |

A text is found by its sha1, so the text itself is not stored. The text of a definition is `definition_text(term, text)` = `"<term>: <text>"`; the text of a bare term is the term. The model's instruction prefix is added only when encoding and is not part of the key.

## Worked example: the fixture cache

[`tests/fixtures/phase3/embed-cache/qwen3-emb-8b/`](../../tests/fixtures/phase3/embed-cache/qwen3-emb-8b) holds four texts and four-number rows:

| Row | Text | sha1 in `keys.txt` | Row in `vectors.npy` |
|---|---|---|---|
| 0 | `media bias: A slant in news coverage.` | `53d10a31…` | 0.1826 0.3652 0.548 0.7305 |
| 1 | `framing` (a bare term) | `a67b95a6…` | 0.3792 0.4548 0.531 0.6064 |
| 2 | `Medienverzerrung: Schräge Berichterstattung – „Bias“.` | `3c6d5540…` | 0.4263 0.4736 0.521 0.5684 |
| 3 | `echo chamber: A closed space where beliefs are amplified.` | `3a5abeb1…` | 0.447 0.4814 0.5156 0.5503 |

Row 0 is `[1, 2, 3, 4] / √30` = 0.18257, 0.36515, 0.54772, 0.73030, rounded to float16. `keys.txt` is 4 × 41 − 1 = 163 bytes (40 hex digits and a newline per row, none after the last), and `vectors.npy` is a 128-byte header plus 4 × 4 × 2 bytes = 160 bytes. `printf 'media bias: A slant in news coverage.' | sha1sum` gives `53d10a31e3e976843fb9c4f1229c44c735d513fb`, the first line of `keys.txt`.

## Build the cache: `embed build`

```bash
mbdef embed build --out /path/to/new/caches   # vLLM takes 0.88 of the GPU memory: 21 GiB of a 24 GiB GPU
```

The build reads the texts below, works out which of them the cache `OUT/qwen3-emb-8b/` lacks, embeds those with Qwen3-Embedding-8B (vLLM, the settings of `config("qwen3-emb-8b")`) and appends their rows. It goes in **two passes**, because the stored cache was filled that way as the corpus grew: first the seed papers, then the seed and the closure papers (after the closure papers were added to the corpus store). Each pass takes, in this order:

1. `"<term>: <definition>"` of every definition of the pass's papers in the corpus store, in the order the store holds them (its row order);
2. `"<term>: <definition>"` of every ground-truth definition: the relevant file, then the non-relevant file, papers and records in file order, blank definitions left out (all splits; no label is read);
3. every term of the pass's papers and every ground-truth term that has a definition, sorted, as bare text;
4. the comparison texts (below).

A text already in the cache (from an earlier pass, or earlier in the same pass) is skipped, so the rows are in the order the texts first appear. Rows already in `OUT` are never changed, and a cache there must have the same settings (its `manifest.json`), or the build stops. A build that stopped is continued by running it again; on a complete cache it adds nothing.

### Worked example: the miniature corpus

The tests build a cache from a store of two seed papers and one closure paper (synthetic text). The store's definitions, in its row order: `echo chamber: A closed space where beliefs are amplified.` (closure), `media bias: A slant in news coverage.` (seed), `agenda setting: Choosing which issues the news covers.` (seed) and `media bias: A slant in news coverage.` again (closure); the first seed paper also has the term `framing` without a definition. The ground truth has three definitions (one of them the same `media bias` text), and the comparison texts are `[X]: a slant in news coverage.` and `framing`.

| Rows | Pass | Texts | New |
|---|---|---|---|
| 0-1 | 1: seed papers | their 2 definitions | 2 |
| 2-3 | 1 | 3 ground-truth definitions | 2 (`media bias: …` is row 0) |
| 4-8 | 1 | 5 terms, sorted: `agenda setting`, `framing`, `hostile media effect`, `media bias`, `tokenization` | 5 |
| 9 | 1 | 2 comparison texts | 1 (`framing` is row 5) |
| 10 | 2: seed and closure papers | 4 definitions | 1, `echo chamber: …` |
| 11 | 2 | 3 ground-truth definitions, 6 terms, 2 comparison texts | 1, `echo chamber` |

So pass 1 reads 2 + 3 + 5 + 2 = 12 texts and adds 10 rows; pass 2 reads 4 + 3 + 6 + 2 = 15 texts and adds 2. One pass over all three papers would put `echo chamber: …` in row 0: the same rows in another order.

### Inputs

| Input | Default (flag) | What is read |
|---|---|---|
| The corpus store | `$DATA_ROOT/frozen/artifacts/experiments/taxonomy-curation-v2/library.sqlite3` (`--store`) | `papers` (each paper's corpus, `seeds` or `closure`), `mentions` (each paper's terms) and `occurrences` (the definition texts, field `definitions`), read only; Step 5 reads the same store ([docs/step-5-post-processing/postprocess.md](postprocess.md#inputs)) |
| The ground truth | MBDef-Gold's `$DATA_ROOT/MBDef-Gold/validated_gt.json`, then `$DATA_ROOT/MBDef-Gold/validated_gt_non_relevant.json` (`data fetch MBDef-Gold/`; `--ground-truth`, repeatable) | the term and definition texts of every record |
| The comparison texts | `$DATA_ROOT/frozen/artifacts/experiments/taxonomy-curation-v2/bench-extra-texts.json` (`--comparison`) | a JSON list of 55,072 texts that the comparison of five embedding models embedded: 7,628 seed definitions with their term masked as `[X]` (`[X]: the framing process in which …`) and 47,444 terms; the stored cache holds them, so the build embeds them too |
| The model | `qwen3-emb-8b` (`--model`) | Qwen/Qwen3-Embedding-8B at revision `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af` |

`mbdef data fetch frozen/artifacts/experiments/taxonomy-curation-v2/` brings the store and the comparison texts ([docs/setup/data-fetch.md](../setup/data-fetch.md)).

| Flag | What it does |
|---|---|
| `--out OUT` | the folder of caches; the cache is `OUT/<model>/` and the receipt `OUT/receipt.json` |
| `--replay FOLDER` | take every vector from a recorded cache folder (`keys.txt`, `vectors.npy`) instead of the GPU; a text it lacks stops the build. A check on a CPU |
| `--sample N` | embed only N of the texts, drawn with the pinned seed `SAMPLE_SEED` (20261023) and kept in cache order: a quick check on a GPU before the whole build |
| `--gpu-share F` | the share of GPU memory vLLM may take (default `GPU_SHARE`, 0.88); lower it when another program holds GPU memory |

### Outputs

`OUT/qwen3-emb-8b/` with the three files of a cache, `OUT/receipt.json` (the settings, the sha256 of every input and of the three files, the code commit), and one line per pass. With the pinned inputs:

```text
pass 1 (seeds): 226,725 texts, 178,937 new
pass 2 (seeds, closure): 429,518 texts, 201,945 new
cache OUT/qwen3-emb-8b: 380,882 rows, keys.txt sha256 65ebe17668f856bc1fe1ca0c50c63143b7a8eb613a0ede186fa65029cb8479dd
```

| Texts of each kind (new rows) | Pass 1: 3,869 seed papers | Pass 2: 8,425 seed and closure papers |
|---|---|---|
| Store definitions | 121,607 (121,299 new) | 272,619 (150,189 new) |
| Ground-truth definitions | 2,699 (2,634 new) | 2,699 (0 new) |
| Terms | 47,347 (47,347 new) | 99,128 (51,756 new) |
| Comparison texts | 55,072 (7,657 new) | 55,072 (0 new) |
| All | 226,725 (178,937 new) | 429,518 (201,945 new) |

121,299 + 2,634 + 47,347 + 7,657 = 178,937 and 150,189 + 51,756 = 201,945 new rows, 380,882 in all: the stored cache. A text counts once however often it occurs: of the 2,699 ground-truth definitions, 65 are texts the seed papers' definitions already gave. Encoding pass 1's 178,937 texts took 22 minutes on a 24 GB GPU (vLLM 0.28.0); how long pass 2 took was not recorded.

Step 5 reads the cache at `$DATA_ROOT/embeddings/qwen3-emb-8b/`; a cache built elsewhere is used by putting its three files there.

### The stored cache

`embeddings/qwen3-emb-8b/` in the data root is the cache built in the two passes above:

| File | Bytes | sha256 |
|---|---|---|
| `vectors.npy` | 3,120,185,472 = 128 + 380,882 rows × 4,096 numbers × 2 bytes | `37456fbb…` |
| `keys.txt` | 15,616,161 = 380,882 × 41 − 1 | `65ebe176…` |
| `manifest.json` | 433 | `43fe979b…` |

```json
{
 "model": "qwen3-emb-8b",
 "repo": "Qwen/Qwen3-Embedding-8B",
 "revision": "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
 "pooling": "last",
 "prefix": "Instruct: Represent the meaning of this scholarly concept and its definition\nQuery:",
 "engine": "vllm",
 "dtype": "bfloat16",
 "quantization": null,
 "max_length": 1024,
 "normalize": "l2",
 "storage": "float16",
 "text_views": "definition_text=term+': '+text; term_text=term"
}
```

`config("qwen3-emb-8b")` gives exactly this, and `Cache.check_writable` writes it byte for byte (sha256 `43fe979b…`, tested). Step 5's Label Groups read this cache. Its hashes are checked, so never add texts to it in place: a build with other inputs goes to another folder.

## Use it

```python
from mbdef import embed, files

files.load_env()  # DATA_ROOT from .env
cache = embed.Cache(files.data_root() / "embeddings", "qwen3-emb-8b")
rows = cache.get(
    [embed.definition_text("media bias", "…")]
)  # float32, one row per text
```

| Name | What it does |
|---|---|
| `MODELS`, `config(model)` | the five pinned models and the settings a cache's manifest must equal |
| `definition_text`, `term_text`, `text_hash` | the text views and the key (sha1) |
| `Cache(root, model)` | opens `<root>/<model>/`; `get(texts)`, `missing(texts)`, `add(texts, vectors)`; `check_writable(cfg)` writes the manifest of a new cache and refuses settings that differ from an existing one |
| `encode_vllm(texts, cfg, gpu_share)`, `encode(texts, model, bf16=False, gpu_share=0.88)` | embed on a GPU with vLLM (decoder models) or transformers (`bge-m3`, `qwen3-emb-0.6b`); torch, transformers and vLLM are imported only when called |
| `ensure(root, model, texts, bf16=False, gpu_share=0.88)` | embed the texts the cache lacks and add them |
| `build(out, store, comparison, …)` | `embed build`: the passes `PASSES` over `pass_texts(…)`, from `ground_truth_texts(GROUND_TRUTH)`, the store and `COMPARISON` |

Two settings are arguments with pinned defaults (method settings are never read from the environment): `bf16=True` keeps F2LLM-v2-14B in BF16 instead of FP8 (on GPUs with 80 GB; it changes that model's manifest, `config(model, bf16=True)`), and `gpu_share` sets vLLM's share of GPU memory (`GPU_SHARE`, 0.88).
The environment variables `TAXONOMY_BF16` and `TAXONOMY_EMBED_GPU_SHARE` change nothing (`tests/test_embed.py` checks it).

Step 5 reads the cache with its own reader (`postprocess/embeddings.py`: memory-mapped, read-only), with `definition_text` and `text_hash` from here, and finds the cache through the data root, not through a checkout's own folders ([docs/step-5-post-processing/postprocess.md](postprocess.md)). `embed build --replay` uses that reader for the recorded cache.

A manifest's `"adopted"` note (two older caches have one) is ignored when settings are compared.

## How it was checked

- **The build on a miniature corpus** (`test_build_fills_the_cache_in_two_passes_seed_papers_first`, in CI): the twelve rows of the worked example above, in that order, written down by hand from the rules; each row's vector is the one of its text in the recorded cache (recorded in another order); the manifest has the stored manifest's sha256; and the two printed counts.
- **A text the recorded cache lacks** stops the build after pass 1, and running it again with a complete recorded cache adds only pass 2's two rows; a third run adds nothing and leaves `keys.txt` as it was (`test_build_stops_…`, `test_build_on_a_complete_cache_…`).
- **`--sample 3`** gives 3 of the 12 rows, in cache order, the same 3 on every run; the receipt names the settings and the sha256 of the store and of `keys.txt` (`test_build_sample_…`, `test_build_writes_a_receipt_…`).
- **The stored cache, rebuilt on a CPU** (slow test `test_build_replaying_the_stored_cache_rebuilds_its_three_files_byte_for_byte`, about 40 s): `mbdef embed build --out <new folder> --replay $DATA_ROOT/embeddings/qwen3-emb-8b` with the pinned inputs prints the counts above, and the three files it writes have the stored sha256: `keys.txt` `65ebe176…`, `vectors.npy` `37456fbb…`, `manifest.json` `43fe979b…` ([`data/data-root.tsv`](../../data/data-root.tsv)). `keys.txt` proves the texts and their order; the vectors come from the stored cache, so `vectors.npy` proves only that the rows are put together in that order.
- **A sample on a GPU** (slow test `test_a_sample_embedded_on_a_gpu_matches_the_stored_vectors`, by hand, one GPU job at a time under `flock`): `embed build --sample 256` without `--replay` embeds 256 texts of the build again and compares each with its stored float16 row: the cosine, and how many float16 steps each number is from the stored one. Run on a 24 GB GPU (vLLM 0.32.0; the cache was built with vLLM 0.28.0), 2 min 26 s: of the 256 rows, 250 are identical to the stored ones; the other 6 differ in 126 of their 24,576 numbers, 121 of them by one float16 step and 5 by 7 or 8 steps (the largest difference 0.000244). Over all 1,048,576 numbers that is 99.988% identical and 99.9995% within one float16 step; the cosine with the stored row is at least 0.999918 (median 1.000003), above the test's bound 0.999. The few larger steps are the GPU's arithmetic in bfloat16 on a different batch of texts than the full build's, not a change of model, revision or text.
- **Round trip** (`test_a_written_cache_reads_back_its_float16_rows_by_text`): write the fixture texts in two appends, open the cache again, and get back the float16 rows by text, in any order; the keys are the four sha1 above.
- **Byte equality with the fixture cache** (`test_the_files_are_byte_identical_to_the_fixture_cache`): the three files written by the package equal the committed files that the original implementation of the cache wrote for the same texts and rows.
- **Step 5's reader** (`test_step5_reads_a_cache_written_here`): Step 5's `EmbeddingCache` (memory-mapped `vectors.npy`, `keys.txt` split on newlines, the sha1 of `"term: text"`) finds the rows the cache wrote.
- **Manifest**: the manifest the package writes for `qwen3-emb-8b` has the stored manifest's sha256.
- **Mixed settings refused**: a cache made with FP8 refuses BF16, and vectors without a manifest refuse new texts.
- **The stored cache's settings** (slow): it opens with 380,882 rows of 4,096 numbers and accepts `config("qwen3-emb-8b")`.
- Run the slow tests with `DATA_ROOT` set: `DATA_ROOT=… uv run pytest -m slow -s tests/test_embed.py` (on a GPU that others use too, wait your turn with `flock <lockfile> …`).

## How to adapt it

- **Another corpus.** Give `--store` a store with the same three tables; its papers' `corpus` must be `seeds` or `closure`, or change `PASSES`. One pass over all papers (`PASSES = (("seeds", "closure"),)`) gives the same rows in another order: Step 5 finds rows by their text, so its results do not change, but the files are then not byte for byte the stored ones. The comparison texts are only there because the stored cache holds them: `--comparison` with a file holding `[]` leaves them out.
- **Another model.** Add it to `MODELS` with a pinned revision and pass `--model`: the manifest records the settings, and a cache refuses vectors made with other settings, so two models never mix in one folder.
- **More texts.** Never add texts to the stored `qwen3-emb-8b` cache in place: its hashes are checked; build into a separate folder.
