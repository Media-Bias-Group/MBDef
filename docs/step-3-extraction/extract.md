# Extraction: `mbdef extract`

| | |
|---|---|
| **Paper step** | Step 3 (information extraction): definition records from the full text of papers |
| **Code** | [`src/mbdef/extract/`](../../src/mbdef/extract): `papers.py` (TEI text, windows), `records.py` (the rules after the model), `extractor.py` (the DSPy extractor and the programs), `run.py` (the command); `verifier.py` and `verify.py` (the candidate verifier); `relevance.py` and `filter.py` (the relevance filter); `plan.py` (a corpus run's shards and jobs); `derive.py` (a run's variants and their comparison); the language model through [`llm.py`](../../src/mbdef/llm.py) ([docs/setup/llm.md](../setup/llm.md)) |
| **Commands** | `mbdef extract run --out FOLDER`, then per part `extract verify --extraction FOLDER --out FOLDER` and `extract filter --run FOLDER --out FOLDER`; a corpus in shards: `extract plan --teis TEIS --out RUN`, the jobs it lists, then `extract derive --set RUN --out FOLDER`; the seed run's variants from its saved parts: `extract derive --out FOLDER` |
| **Checked by** | [`tests/test_extract.py`](../../tests/test_extract.py) and [`tests/test_extract_verify.py`](../../tests/test_extract_verify.py) (recorded responses on synthetic papers, no model), [`tests/test_extract_filter.py`](../../tests/test_extract_filter.py) (recorded BGE vectors, the real classifiers) and [`tests/test_extract_derive.py`](../../tests/test_extract_derive.py) (`extract plan` and `extract derive` on a miniature run); the filter's recheck on the released cohorts and on three shards of the seed run ([below](#how-the-verifier-and-the-filter-were-checked)) |

## Run it on your papers

With a model server and its address in `.env` (`DSPY_API_BASE`, `DSPY_API_KEY`; [below](#run-it) for the released model's `vllm serve` command), and your papers as GROBID TEIs named `<paper id>.tei.xml`, or as PDFs named `<paper id>.pdf` with a GROBID running ([docs/step-1-2-papers/collect.md](../step-1-2-papers/collect.md#run-it) starts one):

```bash
uv run mbdef extract run --teis my-papers/ --out out/run      # or --pdfs my-pdfs/
for part in w8k w16k+w32k; do
  uv run mbdef extract filter --run "out/run/$part" --records union --out "out/filtered/$part"
done
cat out/filtered/*/predictions.jsonl > out/predictions.jsonl
```

`out/run/predictions.jsonl` holds the extractor's records, one per line, before any filter; `out/predictions.jsonl` the records the relevance filter keeps, the paper's "P≥28 Gemma only" extractor. Score either against MBDef-Gold with `evaluate score --predictions` ([docs/step-3-extraction/evaluate.md](evaluate.md#score-your-extractor)). Another model is `--model` and the server's address ([How to adapt it](#how-to-adapt-it)); your own programs are `extract train` ([Training the programs](#training-the-programs-extract-train)).

## What this page covers

This page covers the three stages of Step 3: the extraction, the [candidate verifier](#the-candidate-verifier-extract-verify) and the [relevance filter](#the-relevance-filter-extract-filter), how a whole corpus is run in shards ([`extract plan`](#a-corpus-run-extract-plan-the-jobs-extract-derive)), and how the parts of a run are pooled into its four variants ([`extract derive`](#the-seed-runs-variants-extract-derive): for example P≥28 pools the filtered `w8k` and `w16k+w32k` records with `records.merge`); scoring is `mbdef evaluate score` ([docs/step-3-extraction/evaluate.md](evaluate.md#scoring-extractions-defscore-f1-and-matching)).

How the three programs were trained, and how to train new ones, is at the end of this page: [Training the programs](#training-the-programs-extract-train).

## What the step does

The released extractor is Gemma 4 12B (`google/gemma-4-12B-it-qat-w4a16-ct`, revision `1d2c2d7f…`) with three saved prompt programs, each trained for one window size (in `data/extractors/`, [its README](../../data/extractors/README.md)). A program is an instruction plus one worked example.

1. **Text.** Each paper's GROBID TEI gives its title and its text: the abstract, then every block of the body and back matter, without the bibliography.
2. **Windows.** The text is cut into overlapping windows of characters. A window ends at the last line break in its second half, so it rarely cuts a paragraph, and the next one starts `overlap` characters earlier. Nothing is cut off to fit the model.
3. **One model call per window.** The model gets the title and the window and answers with definition records: a term and seven lists of evidence (definitions, synonyms, description, related terms, subtypes, supertypes, examples), each copied from the window. Thinking is on, with at most 32,768 output tokens, seed 9 and no temperature, so the model samples with its own settings.
4. **Copied, or dropped.** A string is kept only if it occurs in the window (ignoring case); a record whose term is not in the window is dropped. Nothing is ever repaired.
5. **Union.** A paper's windows are merged per *part*: records with the same term (ignoring case) become one, and a term without any definition is dropped. Every kept definition becomes one candidate for the verifier, linked to the windows that hold it.

A **part** is one or more programs run on the same papers, each on its own window size. The released extractor has two parts, both in the chat format:

| Part | Programs and windows (characters) | The released TEST run's name |
|---|---|---|
| `w8k` | the `w8k` program on 8,000 / 1,000 | TR-bootstrap-8k |
| `w16k+w32k` | the `w16k` program on 16,000 / 2,000, and the `w32k` program on 32,000 / 4,000 | TR-bootstrap-16k32k |

**A worked example: "The science of fake news" (paper `73bfad11…`, part `w8k`).** Its text has 19,656 characters, so it gives three windows of 7,364, 7,508 and 6,782 characters. The model answered with 3, 6 and 5 records:

- In window 1 it named the term *Internet platform-based interventions*, which is not written in the window, so the record is dropped (its definition was in the window; the term was not). 13 records stay.
- *fact checking* is in windows 0 and 1; the union makes it one record.
- *priming*, *framing* and *agenda setting* (window 2) have a description and related terms but no definition, so the union drops them.

The union keeps 9 records with 9 definitions, for example *fake news*: "fabricated information that mimics the output of the news media in form, but not in organizational process or intent", with the supertype *misinformation*. These are exactly the records of the released run.

## Run it

With a server for the model and its address in `.env` (`DSPY_API_BASE`, `DSPY_API_KEY`; [docs/setup/data-root.md](../setup/data-root.md)). The released runs used vLLM 0.28 with the model's own reasoning parser, so the answer excludes the thinking, and its own sampling settings:

```bash
vllm serve google/gemma-4-12B-it-qat-w4a16-ct --revision 1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee \
    --reasoning-parser gemma4 --generation-config auto --max-model-len 65536
```

Then:

```bash
uv run mbdef extract run --out "$DATA_ROOT/extract/run" \
    --paper 4243aa9bacd51bc02d4240378e7528912962b53e --paper 73bfad11b96a69cb882028ead115751adb55252d
```

Without `--paper` it runs every TEI in `--teis` (default `data/paper_sources.zip`, the 150 ground-truth papers' TEIs, which hold paper text and are not part of this repository: give `--teis` your own GROBID TEIs). It prints the counts per part:

```json
{
  "w8k": {"papers": 2, "windows": 6, "failed_windows": 0, "records": 10, "definitions": 10, "candidates": 10, "model_calls": 6},
  "w16k+w32k": {"papers": 2, "windows": 5, "failed_windows": 0, "records": 16, "definitions": 25, "candidates": 25, "model_calls": 5}
}
```

**Without a model: replay.** `--replay FOLDER` (repeatable) answers every call from recorded request receipts and never contacts a server, so the run needs no `.env` keys. A call whose request was not recorded stops the run and names it:

```text
Stopped: a model call failed in part w8k: [openai/google/gemma-4-12B-it-qat-w4a16-ct] request_sha256 c5c8c9e1… was not recorded; its request is in OUT/w8k/requests-extract/c5c8c9e1….json. …
```

The released runs' receipts hold paper text and are not public; [`tests/test_extract.py`](../../tests/test_extract.py) replays a run recorded on synthetic papers instead.
The released TEST run used the ground truth's titles, which differ from the TEI's for 2 of its 29 papers; pass them with `--titles` (a JSON file `{paper id: title}`).

## Inputs

| Input | Where | What it is |
|---|---|---|
| TEIs | `--teis`, a folder or zip of `<paper id>.tei.xml` (default `data/paper_sources.zip`, which holds paper text and is not part of this repository: give your own) | GROBID output; the paper id is the file name, URL-decoded |
| Programs | `--programs` (default `data/extractors/gemma12b-bootstrap/`), one folder per program | `program.json` and `manifest.json`; a program whose sha256 differs from its manifest is refused |
| PDFs | `--pdfs`, instead of `--teis`: a folder of `<paper id>.pdf`; `--grobid-url` (default `http://localhost:8070`) | converted by a running GROBID into `tei/` in `--out` (no identity check: the file name is the paper id); a PDF that does not convert is reported and left out |
| Titles | `--titles`, optional | replace the TEI titles |
| Endpoint | `.env`: `DSPY_API_BASE`, `DSPY_API_KEY` | an OpenAI-compatible server; not read in a replay |
| Recorded receipts | `--replay`, optional | answers for every call |

## Outputs

| File in `--out` | What it is |
|---|---|
| `<part>/papers/<sha256 of the paper id>/extraction/<branch>-<window>.json` | One window: its text, the model's raw records, the kept records, the raw response and the `request_sha256` of its calls |
| `<part>/papers/<…>/extraction-failures/<branch>-<window>.json` | A window that gave no records: its answer did not finish, or could not be parsed |
| `<part>/papers/<…>/union.json` | The paper's merged records (`records`, `raw_records`) and the verifier's `candidates` |
| `<part>/requests-extract/<request_sha256>.json` | Every call's exact request and full response (or error) |
| `<part>/usage-extract.jsonl` | One line per call: tokens, finish reasons, settings ([docs/setup/llm.md](../setup/llm.md)) |
| `predictions.jsonl` | Every paper's records, the parts' unions pooled with `records.merge`, one record per line: `{"paper_id", "term", "definitions", ...}`, what `evaluate score --predictions` reads |
| `tei/<paper id>.tei.xml` | With `--pdfs`: GROBID's TEI of each PDF |
| `receipt.json` | The settings (model, revision, format, maximum tokens, seed, thinking, each program's sha256 and windows, GROBID's address and version with `--pdfs`), the hashes of the TEIs, programs and replay folders, each paper's TEI hash and title, the counts, the code commit and the time |

`branch` numbers the programs of a part from 0, so `1-00000.json` in `w16k+w32k` is the first 32k window. Raw outputs belong in the data root, not in git.

## How it was checked

- **The extraction tests** ([`tests/test_extract.py`](../../tests/test_extract.py)): the command on the two TEST papers of [`tests/synthetic_papers.py`](../../tests/synthetic_papers.py) (1,070 and 859 characters of made-up text), with three programs on windows of 400/80, 650/120 and 900/150 characters (parts `a` and `b+c`: 4 + 4 and 4 + 3 windows), replayed from a run recorded with a stand-in model (`tests/fixtures/synthetic-extract-replay.zip`): every window and union equals the recorded one, each union holds the paper's three defined terms and its one example once, and the receipt and the usage log are checked; so are `--shard`, the refusal of a changed program and of a changed TEI, the stop on an unrecorded call, and the replay of a call that failed when it was recorded. `predictions.jsonl` is checked against the unions; `--pdfs` against a fake GROBID on a local port (the same records as from the TEIs); `--thinking unset` changes every request.
- **A live run** (by hand, on a 24 GB laptop GPU): the released model served by vLLM 0.32 on two TEST papers ("Branding the nation" and "The science of fake news", from their TEIs), 11 windows in 3.5 minutes; then `extract filter`, `extract verify` with Gemma itself, and `evaluate score --predictions` (8 of the 10 reference definitions found). The same with another model (Qwen3-VL-2B-Instruct, `--thinking unset`, both formats). `--pdfs` was not run with a real GROBID.
- **All released Gemma bootstrap cohorts** (by hand, on the released run's recorded files, which hold paper text and are not public): TEST (29 papers) and DEV (21 papers), both parts, both formats, 8 runs, 2,005 windows. Every window and all 200 unions are identical to the saved run (the worked example above among them), the 3 windows that failed then fail again, and the number of calls equals the saved usage logs. Each run takes about 5 s.
- **The original implementation:** the TEI text and the windows of all 150 ground-truth papers, the formatted requests of the three programs, and the rules after the model were checked against its outputs.

**The released runs had four failed calls.** One call in each of four of the eight cohorts failed in the client with `KeyError: 'temperature'` before any answer. In the chat format DSPy then asked once more in JSON, and the window kept that answer; in the JSON format the window gave no records. A replay does the same.

## How to adapt it

- **Another model or server.** Set `DSPY_API_BASE` and `DSPY_API_KEY` in `.env` (any OpenAI-compatible endpoint: vLLM, OpenRouter, …), and pass `--model` (the name the server knows) and `--revision` (only recorded). `--thinking` is the thinking switch each request sends: `on` (the released runs), `off`, or `unset` (no switch, for an API that refuses vLLM's `chat_template_kwargs`); a model without thinking ignores `on`. `--max-tokens` and `--seed` are flags. The repository makes no paid call ([docs/setup/llm.md](../setup/llm.md)).
- **Another program.** Put a folder with `program.json` and `manifest.json` (with `program`, `program_sha256`, `window_chars` and `overlap_chars`) under `--programs`, and name it with `--part`; join programs with `+` to run them as one part. A program trained by `PaperExtractor` loads as is.
- **Another output format.** `--format json` asks for a JSON object instead of DSPy's field markers (the P≥20 variant also pools the two JSON parts). A small model may do better with it: Qwen3-VL-2B-Instruct left 6 of 11 windows without records in the chat format (it wrote until the token limit) and 1 of 11 in JSON.
- **Other papers.** `--teis` (or `--pdfs`) and `--paper`; `--titles` for titles other than the TEI's.
- **Speed.** `--workers` windows are sent at the same time (default 16); a vLLM server batches them.
- **The rules after the model** are `sanitize`, `merge` and `union` in `records.py`; the instruction and the record schema are in `extractor.py`. Changing either changes the requests, so a replay of the released run stops at the first call.
- **What to rerun after a change:** the verifier and filter stages, scoring on the ground truth, and everything after Step 3 (the corpus closure and the later steps).

## The candidate verifier: `extract verify`

A second model checks every definition the extractor kept. Each definition of a paper's union is a **candidate**. The verifier shows the model the window the candidate was found in, the paper's title and up to 16 candidates of that window at once (a **batch**), and asks for one verdict per candidate: is the quoted text a definition of the term (`yes`, `no` or `uncertain`), is the term relevant to media bias, and a short reason. A candidate found in two windows is asked about in both.

- **The rule** (`definitions-strict`, as released): a candidate is kept if one of its verdicts says `definition: yes`. Relevance is left to the filter. A candidate whose verdicts are not all there (its batch failed, or the answer left it out) is rejected; nothing is repaired. A record whose definitions are all rejected is dropped; the other lists of a kept record stay as they are.
- **The answer** is checked on the raw text the server sent, against the JSON schema of exactly the batch's ids, which the request also sends for constrained decoding. A valid answer that leaves ids out keeps the verdicts it has; any other bad answer fails the batch. A failed call (the server is unreachable, or a replayed call was not recorded) stops the run.
- **The batches** are cut so that the prompt, counted with the verifier model's own tokenizer, plus 4,096 output tokens fits 32,768 tokens; a window is never cut. The largest of the 8,090 batches of the released TEST and DEV cohorts had 18,418 tokens, so in practice a batch is up to 16 candidates of one window.
- **The released settings** (the defaults): Qwen 3.6 35B-A3B (`Qwen/Qwen3.6-35B-A3B`, revision `995ad96e…`), thinking off, temperature 0, seed 9, at most 4,096 output tokens. The P≥24 variant instead let Gemma verify its own records: `--model google/gemma-4-12B-it-qat-w4a16-ct --revision 1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee`, everything else the same.

**A worked example: "The science of fake news" (paper `73bfad11…`, part `w8k`, Qwen).** Its union has 9 candidates, one per record, in 3 windows; *fact checking* is in two of them, so the 3 batches ask for 5, 3 and 2 verdicts. All 10 came back. *Familiarity bias* got `definition: no` (the model's reason: the quote "describes the effects of familiarity bias … but does not explicitly define the term"), the other 8 got `yes`, so `verified.json` keeps 8 of the 9 records.

```bash
uv run mbdef extract verify --extraction "$DATA_ROOT/extract/run/w8k" \
    --out "$DATA_ROOT/extract/verified/w8k"
```

It reads `DSPY_API_BASE` and `DSPY_API_KEY` from `.env` (or `--replay FOLDER`, as in `extract run`), and the model's tokenizer from the Hugging Face cache (`--tokenizer FOLDER` to give another), and prints the counts:

```json
{"papers": 2, "candidates": 10, "batches": 4, "unbatchable_contexts": 0, "failed_batches": 0, "missing_verdicts": 0, "candidates_rejected_for_missing_verdict": 0, "kept_records": 9, "kept_definitions": 9, "model_calls": 4}
```

`--extraction` is one part's folder of `extract run` (or a released cohort's `run/`); run it once per part. Into `--out` it writes, per paper (`papers/<sha256 of the paper id>/`), the union it verified, one receipt per batch (`verification/<batch>.json`: ids, the raw response, the verdicts, the status), every verdict per candidate and the rejected ones (`decisions.json`), the kept records (`verified.json`) and the paper's counts (`paper-summary.json`), all in the released runs' JSON format; and, as `extract run` does, `requests-verifier/`, `usage-verifier.jsonl`, the kept records of every paper as `predictions.jsonl` and `receipt.json`.

## The relevance filter: `extract filter`

A record's **feature text** is its term and its definitions, one per line. The BGE encoder (`BAAI/bge-base-en-v1.5`, revision `a5beb1e3…`, float32, the first token's vector L2-normalised, 32 texts per batch) turns it into a vector, and a small classifier (logistic regression) gives the probability that the record is relevant to media bias. A record is kept, unchanged, if the probability is at least the filter's cutoff. Nothing is truncated: a record whose input has more than 512 tokens cannot be classified and is dropped (listed in `capacity_excluded_record_indices`).

The three frozen filters were trained on the 1,171 records of the ground truth's 80 training papers (592 relevant), each with the cutoff that lets 95 % of the relevant records pass in five-fold cross-validation. They are frozen inputs in `data/frozen/results/`; `extract filter` refuses a zip whose classifier, training or configuration file differs from the frozen sha256. How they were fitted, and how to fit one again: [`extract train-filter`](#training-the-relevance-filter-extract-train-filter).

| `--filter` | Row | Zip | Classifier input | Cutoff | Used by |
|---|---|---|---|---|---|
| `ordinary` (default) | R6e24b3d | `semantic-relevance-v1.zip` | the feature text | 0.3088 | the released variants P≥20, P≥28, P≥28 Gemma only |
| `title` | R0bea6bc | `semantic-title-relevance-v2.zip` | the title, a line break, the feature text | 0.2372 | the grid's `filtered-title` stages |
| `title-blocks` | Ra1da1ae | `semantic-title-blocks-v3.zip` | the feature text's vector next to the title's | 0.2489 | a row of the held-out TEST rescoring |

**A worked example: "A Comprehensive Survey on Graph Neural Networks" (paper `1901.00596v4`, TEST, part `w8k`).** Its union has 28 records. `ordinary` keeps 4: *directed graph* (0.334), *undirected graph* (0.361), *diffusion process* (0.332) and *attention mechanisms* (0.340); all others are below 0.3088, for example *message passing process* (0.279) and *GraphRNN* (0.151). `title` keeps 2 (*undirected graph* and *message passing process*). In "The science of fake news" every record passes, and the one record of "Branding the nation" (paper `4243aa9b…`), *nation branding*, has 0.4995.

```bash
uv run mbdef extract filter --run "$DATA_ROOT/extract/verified/w8k" \
    --out "$DATA_ROOT/extract/filtered/w8k"
```

`--run` is a folder with `papers/*/union.json`: a verifier run (the default `--records verified` filters each `verified.json`) or an extraction run (`--records union` filters the union's records, the released "no verifier" variants). The title comes from `union.json`. BGE comes from the Hugging Face cache (`mbdef cluster download --model bge --out RECEIPT_FOLDER`, or `--bge FOLDER`), and runs on `SCIDEF_NLI_DEVICE` from `.env` (`cpu`, `cuda` or `auto`), as the scoring models do.
Into `--out` it writes, per paper, `papers/<sha256 of the paper id>/filtered.json` (the kept records, every record's probability, the excluded ones, the filter and cutoff), a recording of the encoder (`encoder/encoded.json` with each text's sha256 and token count, `encoder/vectors.npy`), the kept records of every paper as `predictions.jsonl` (one record per line, as `extract run` writes it), and `receipt.json`. With `--replay FOLDER` (a recorded `encoder/`) no model is loaded, and a text that was not recorded stops the run.

`relevance.apply_filter(encoder, load_filter("ordinary"), title, records)` is the one function that applies a filter. The seed run applied the same filter on the cluster and saved the filtered records in its parts; [`extract derive`](#the-seed-runs-variants-extract-derive) reads those, it does not filter again.

## How the verifier and the filter were checked

- **Verifier tests** ([`tests/test_extract_verify.py`](../../tests/test_extract_verify.py)): the verifier on the synthetic papers' recorded run (`verify/` in `tests/fixtures/synthetic-extract-replay.zip`): the same files byte for byte, the three definitions of each paper kept and its example rejected, an answer that is not JSON rejecting its batch, and an unrecorded call stopping the run. CI has no Qwen tokenizer, so it counts tokens with a one-token-per-word stand-in (`tests/fixtures/word-tokenizer`), which gives the same batches.
- **The released verifier** (by hand, on the released TEST run's recorded requests, which hold paper text and are not public): the Qwen verifier and Gemma's self-verification on "Branding the nation" and "The science of fake news", both parts, asked exactly the recorded requests and wrote `verified.json`, `decisions.json` and `paper-summary.json` byte-identical to the saved ones; with the models' own tokenizers every verification receipt is byte-identical, token counts included.
- **Filter tests** ([`tests/test_extract_filter.py`](../../tests/test_extract_filter.py)): the real frozen classifiers on recorded BGE vectors (made by `extract filter` on a 24 GB GPU; the slow test remakes them and agrees within 1e-5): for both parts and the three record sets (union, Gemma's verification, Qwen's), and for the graph-network survey, the kept records equal the published stages and every probability is within 1e-5 of the released one; `--cutoff` and a changed zip are tested too. The filter reads records only, so its fixture (`tests/fixtures/filter-replay.zip`: the released records, filter files and BGE recordings, byte for byte) holds no paper text.
- **Recheck on the released cohorts and the seed run** (by hand, a 24 GB GPU): `extract filter` again on the saved records, compared with the published `filtered-ordinary`, `-noverify` and `-qv` stages. The 8 TEST and DEV cohorts of the released extractor: 24 stages, 600 paper results, identical kept records, 0 decision flips, max |Δp| 1.46e-6, in 3 to 4 minutes. Shards 000, 048 and 096 of the seed run's four parts and the Qwen verifier's passes: 30 stages, 1,110 paper results, identical, 0 flips, max |Δp| 2.57e-6, in 4 to 6 minutes.
- **The original implementation:** the verifier's requests and response schema for all 16 recorded batches, its verdicts and kept records, its partial answers and its batch cuts, and the filter's records and probabilities (≤ 1e-6, all three filters, on the GPU) were checked against its outputs.

## How to adapt the verifier and the filter

- **Another verifier model.** `--model` and `--revision` (its tokenizer is read from the Hugging Face cache at that revision, or `--tokenizer FOLDER`), `--max-tokens`, `--seed`; the endpoint in `.env`.
  Thinking and temperature are `THINKING` and `TEMPERATURE` in `verify.py`.
- **Another instruction or rule.** The instruction is `VerifyCandidates` in `verifier.py` (changing it changes every request, so a replay of the released runs stops at the first call); the rule is `keep`; the batch size `MAX_CANDIDATES`.
- **Another cutoff.** `--cutoff`; the receipt records it next to the frozen one.
- **Another filter.** `--filter title` or `title-blocks`; a retrained classifier (`extract train-filter`) needs its own zip and an entry in `FILTERS` in `relevance.py` with its files' sha256.
- **No verifier.** `extract filter --run <extraction part> --records union`.

## Training the relevance filter: `extract train-filter`

| | |
|---|---|
| **Paper step** | Step 3 (relevance filtering): how the three frozen filters were fitted |
| **Code** | [`src/mbdef/extract/train_filter.py`](../../src/mbdef/extract/train_filter.py); the classifier inputs and the encoder from [`relevance.py`](../../src/mbdef/extract/relevance.py) |
| **Command** | `mbdef extract train-filter --filter ordinary --out FOLDER` |
| **Checked by** | [`tests/test_extract_train_filter.py`](../../tests/test_extract_train_filter.py): the three filters refitted from their stored vectors (CI), a miniature ground truth with synthetic text and a recorded encoder (CI), and a refit with BGE itself (by hand) |

### What it does

The three filters of [`extract filter`](#the-relevance-filter-extract-filter) were fitted one study each, on the same records, folds and settings. `extract train-filter` fits one of them the same way:

1. **Records.** Every definition record of the ground truth's TRAIN papers: first the relevant ones from MBDef-Gold's `validated_gt.json` (label 1), then the not relevant ones from its `validated_gt_non_relevant.json` (label 0), paper by paper, record by record; both are in `$DATA_ROOT/MBDef-Gold/` (`data fetch MBDef-Gold/`). The TRAIN papers are the relevant file's papers with split `train`. That gives 1,171 records of 80 papers: 592 relevant and 579 not.
2. **Classifier input.** As the filter reads it when it is applied: for `ordinary` the record's feature text (the term and its definitions, one per line), for `title` the paper's title, a line break and the feature text, for `title-blocks` the feature text's BGE vector next to the title's (1,536 numbers). The title is the paper's `metadata.title`, which must be the same in both files.
3. **Folds.** The 80 papers go into 5 folds (scikit-learn's `StratifiedGroupKFold`, shuffled with seed 9; a paper's records always stay together), which hold out 15, 21, 13, 13 and 18 papers. Each record gets an *out-of-fold* probability from a classifier fitted on the other four folds. The classifier is a logistic regression (C = 1, the `liblinear` solver, at most 1,000 iterations, seed 9; `config.json` names it) on the vectors as they are; a fit that does not converge stops the run.
4. **Cutoff.** The relevant records' out-of-fold probabilities, sorted from high to low; the cutoff is the ⌈0.95 × 592⌉ = 563rd of them, the largest cutoff that still keeps 95 % of the relevant records.
5. **The filter** is one more classifier, fitted on all 1,171 records. It is saved as the frozen ones were (joblib with pickle protocol 4), so the two files can be compared byte for byte.

**A worked example: the `ordinary` filter.** The first record, *dual-process approaches* (paper `0209e602…`, relevant), is in fold 2 and gets 0.4802 from the classifier that did not see its paper. Sorted from high to low, the 562nd, 563rd and 564th relevant records have 0.3096, **0.3088** and 0.2992, so the cutoff is 0.3087965279332379: 563 of the 592 relevant records reach it (recall 95.1 %), and so do 283 of the 579 others (precision 66.5 %). The highest of those others is *Flak* (0.9082); the highest relevant record that the cutoff loses is *prose literacy* (0.2798). At 0.5 instead, 462 relevant and 115 other records would pass. The other two filters:

| `--filter` | Features | Cutoff | Out of fold at the cutoff: relevant / others kept | At 0.5 |
|---|---|---|---|---|
| `ordinary` | 768 | 0.3088 | 563 of 592 / 283 of 579 | 462 / 115 |
| `title` | 768 | 0.2372 | 563 / 406 | 444 / 155 |
| `title-blocks` | 1,536 | 0.2489 | 563 / 336 | 463 / 123 |

### Run it

The vectors come from BGE (as in `extract filter`: the Hugging Face cache or `--bge FOLDER`, on `SCIDEF_NLI_DEVICE` from `.env`; about 2 minutes on a 24-thread CPU for the 1,171 texts), from a recording of it (`--replay FOLDER`), or from the frozen filter's own zip, which stores the vector of every text its study encoded (`--stored-vectors`; no model, no GPU, a few seconds):

```bash
OPENBLAS_CORETYPE=Haswell uv run mbdef extract train-filter --filter ordinary --stored-vectors \
    --out "$DATA_ROOT/extract/filters/ordinary"
```

The logistic regression's solver and the probabilities run on OpenBLAS, which picks its kernels by CPU, and kernels sum in different orders. The frozen filters were fitted with OpenBLAS's `Haswell` kernels (also the ones it picks on AMD Zen CPUs); `OPENBLAS_CORETYPE=Haswell` forces them on any x86-64 CPU with AVX2, and the receipt records which kernels ran (`blas`). With other kernels the refit is the same filter to about 1e-14 but not the same file: with the older `Sandybridge` kernels, for example, the largest coefficient difference is 1.0e-14, the cutoff moves in its last digits, and the same 563 relevant and 283 other records pass out of fold.

It prints what it fitted and how that compares with the frozen filter of the same name:

```json
{"filter": "ordinary", "records": 1171, "papers": 80, "relevant_records": 592,
 "cutoff": 0.3087965279332379, "kept_out_of_fold": {"relevant": 563, "not_relevant": 283},
 "vectors": "stored in the frozen zip",
 "frozen_filter": {"zip": "semantic-relevance-v1.zip", "classifier_file": "identical",
                   "cutoff": "identical", "folds": "identical", "max_coefficient_difference": 0.0}}
```

### Inputs and outputs

| Input | Where | What it is |
|---|---|---|
| Relevant records | `--ground-truth` (default `$DATA_ROOT/MBDef-Gold/validated_gt.json`, MBDef-Gold) | its TRAIN papers' records are label 1; its split decides which papers are TRAIN |
| Records that are not relevant | `--non-relevant` (default `$DATA_ROOT/MBDef-Gold/validated_gt_non_relevant.json`) | the same papers' other records, label 0 |
| Filter | `--filter` (`ordinary`, `title` or `title-blocks`) | which classifier input, and which frozen zip to compare with |
| Frozen zips | `--filters` (default `data/frozen/results`) | compared with; the vectors of `--stored-vectors` |
| Encoder | `--bge FOLDER`, `--replay FOLDER` or `--stored-vectors`; `.env`: `SCIDEF_NLI_DEVICE` | where the vectors come from |

| File in `--out` | What it is |
|---|---|
| `model.joblib` | the classifier |
| `config.json` | the encoder settings that `extract filter` checks in a filter's zip, and the fit's settings |
| `training.json` | records, papers, relevant and other records, features, the cutoff, and the out-of-fold counts, precision, recall, F1 and MCC at the cutoff (`selected`) and at 0.5 (`default`) |
| `folds.json` | each fold's training and held-out papers |
| `oof.json` | every record (paper, file, index, term, definitions, feature text) with its label, out-of-fold probability and fold |
| `encoder/` | a recording of the encoder's vectors, which `--replay` reads (not with `--stored-vectors`) |
| `receipt.json` | the settings, the hashes of the inputs and outputs, the comparison with the frozen filter, the code commit |

### How it was checked

- **The three refits from the stored vectors** (CI, with `OPENBLAS_CORETYPE=Haswell`): for each filter, `model.joblib` is the frozen zip's file byte for byte (sha256 `c01a88ff…`, `392cd5b1…` and `e1d7f42a…`, the hashes `relevance.FILTERS` pins), so every coefficient is the same; `folds.json` is the frozen file byte for byte; the cutoff, and every record's out-of-fold probability and fold, equal the frozen `training.json` and `oof.json`; `training.json` equals the frozen one except the keys each study added for its own cross-checks against earlier studies (for example `identical_lexical_fold_assignment`). The records rebuilt from the ground truth are the ones the studies saved (`training-inputs.json` in the zips), in the same order. With the `Sandybridge` kernels the command reports a different classifier file, a largest coefficient difference of at most 1e-12 and the same records passing.
- **A miniature** (CI): ten TRAIN papers with synthetic text and a recorded encoder with synthetic vectors: only TRAIN records are read, every paper is held out in exactly one fold, the cutoff keeps ⌈0.95 × 20⌉ = 19 relevant records and is the largest that does; a text missing from the recording stops the run before anything is written, and titles that differ between the two files are refused.
- **With BGE itself** (by hand, `pytest -m slow tests/test_extract_train_filter.py`; on a 24-thread CPU, 2 min 13 s): the 1,171 vectors are within 3.8e-6 of the stored ones, the out-of-fold probabilities within 1.2e-6, the folds identical, the same 563 relevant and 283 other records pass, the cutoff is 0.3087962659 (frozen 0.3087965279) and no coefficient differs by more than 4.0e-6. The classifier file therefore differs in its last digits: a filter fitted on freshly computed vectors is a new filter.
- **Not done by this command:** the studies' evaluation on the 21 DEV papers (the verifier's records filtered at the cutoff and at 0.5, then scored), their plots, and one check of the labels: each study also matched every training record to its row in the annotation export and stopped unless that row confirmed the label. The export is not in the repository; it confirmed all 1,171 labels, which the two ground-truth files hold.

### How to adapt it

- **A new ground truth:** `--ground-truth` and `--non-relevant` in the same format; BGE computes the vectors.
  To apply the result with `extract filter`, zip the folder's `model.joblib`, `training.json` and `config.json` as `<study>.zip` and give it an entry in `FILTERS` in `relevance.py` with the study, the cutoff, the number of features and the three files' sha256. The entry's name decides the classifier input when the filter is applied (`title` and `title-blocks` read the title), so a refitted `title` filter replaces the `title` entry, and a new name reads the feature text as `ordinary` does.
- **Another fold count, seed, recall target or regularisation:** `--folds`, `--seed`, `--target-recall` and `--regularisation` (the logistic regression's C); `config.json` and the receipt record them. For example `--target-recall 0.9` sets the cutoff at the ⌈0.9 × 592⌉ = 533rd relevant record. Another kind of classifier: `classifier` in `train_filter.py`.
- **Another classifier input:** `features` in `train_filter.py` and `probabilities` in `relevance.py` together, so that training and applying read the same input.
- **What to rerun after a change:** `extract filter` with the new filter on the verified records, then everything after it (the seed run's variants, the closure, Step 5).

## Training the programs: `extract train`

| | |
|---|---|
| **Paper step** | Step 3 (information extraction): the prompt training behind the released programs |
| **Code** | [`src/mbdef/extract/train.py`](../../src/mbdef/extract/train.py) (training and the program checks); the language model through [`llm.py`](../../src/mbdef/llm.py), the metric from [`evaluate/metric.py`](../../src/mbdef/evaluate/metric.py) |
| **Commands** | `mbdef extract train --window w8k --out FOLDER`; `mbdef extract check-programs`; the dataset: `mbdef extract train-dataset --out FOLDER` |
| **Checked by** | [`tests/test_extract_train.py`](../../tests/test_extract_train.py) (a run on synthetic papers from recorded responses and recorded pair scores, no model; the released dataset, split and plan files); [`tests/test_extract_train_dataset.py`](../../tests/test_extract_train_dataset.py) |

### What it does

The three programs in `data/extractors/gemma12b-bootstrap/` were trained one run per window size, by DSPy's **BootstrapFewShot** with Gemma 4 12B (thinking on, the same model settings as extraction). Bootstrap keeps the extractor's instruction and adds one worked example, a *demo*.
`extract train` runs the same path:

1. **Papers.** The training dataset holds the ground truth's 80 TRAIN and 21 DEV papers with their titles, TEI hashes and reference records (`data/frozen/results/training-20260925/runner/train-dataset.json`, the released runner's input; `split.json` beside it adds the 32 TEST ids, which are refused). Each paper's TEI must have the recorded sha256, and its parsed text too. The papers are put in one fixed order, `random.Random(9)` over the dataset's order, as the released runs did. TRAIN papers are Bootstrap's training set; DEV papers are held out and never read (MIPRO and GEPA used them to choose).
2. **Bootstrap.** For each TRAIN paper in that order, the extractor without a demo reads every window of the paper (one model call each) and the paper's merged records, before any string check, are scored against its reference records with the full-record score (stored as `M3-full-v1`, matches at 0.5; 0 if an answer could not be parsed). The first paper that scores **above 0** gives the demo, and Bootstrap stops: one demo at most, no reference records as demos.
3. **The demo** is one of that paper's window calls, as it happened: the window, the title, and the model's reasoning and records. DSPy draws which window with a random number seeded by a hash of the pickled calls; that hash depends on Python's string hashing, which changes from process to process unless `PYTHONHASHSEED` is set. The receipt records it.
4. **Checks.** The program must load, keep the extractor's signature, have at most one demo, and every demo must be, word for word, a window of a TRAIN paper under that paper's title. The same checks run on any saved program with `extract check-programs`, which also counts how many of each demo's record strings occur in its window.

**A worked example: the released w8k run.** The first TRAIN paper of the seed-9 order is "Racism and Health: Evidence and Needed Research" (`f5c28232…`), 9 windows of 8,000 characters. Its 9 calls gave 23 records after merging by term, which met the paper's 6 reference records (*Race*, *Racism*, *Institutional racism*, *Cultural racism*, *stereotype threat*, *discrimination*). Every reference record found its record; for example the reference definition of *stereotype threat*, "the anxieties and expectations that can be activated in stigmatized groups when negative stereotypes about their group are made salient", met the model's "This term refers to the anxieties and expectations …" at a DefScore pair score of 1.0. The full-record score is the mean F1 over the 9 fields with a string on at least one side: terms 0.41, definitions 0.35, synonyms 0.33, subtypes 0.28, related terms 0.14, and 0 for description, supertypes, examples and citations, so 0.168. That is above 0, so Bootstrap stopped after this first paper and 9 calls (the w16k and w32k runs likewise, after 5 and 2 calls, at 0.237 and 0.226; the released runs did not record these scores, they were recomputed here from the recorded answers). The released w8k program's demo is window 3 (numbered from 0) of the 9.

### Run it

With a server for the model (as for `extract run`), its address in `.env` (`DSPY_API_BASE`, `DSPY_API_KEY`), and `SCIDEF_NLI_DEVICE` for the metric's two scoring models (BGE and ModernBERT, [docs/step-3-extraction/evaluate.md](evaluate.md#scoring-extractions-defscore-f1-and-matching)):

```bash
PYTHONHASHSEED=0 uv run mbdef extract train --window w8k \
    --out "$DATA_ROOT/extract/programs/w8k"
```

It trains on all 80 TRAIN papers until the first success and prints a summary, for example this one of a replay of the released w8k run on the first 4 TRAIN and 2 DEV papers:

```json
{"window": "w8k", "papers": {"train": 4, "dev": 2},
 "papers_tried": [{"paper_id": "f5c282326822c5c1cca05a49940024480948e09c", "score": 0.16843737112664178}],
 "demo": {"paper_id": "f5c282326822c5c1cca05a49940024480948e09c", "window": 6, "windows": 9},
 "model_calls": 9, "failed_calls": 0}
```

Run the three window sizes as three commands at the same time against one server; within a run, the windows of a paper are sent together (`--workers`, default 16), which gives the same program as sending them one by one. The output folder is then a program for extraction: `extract run --programs "$DATA_ROOT/extract/programs" --part w8k --part w16k+w32k`.

**Without a model: replay.** `--replay FOLDER` answers every call from recorded request receipts and `--replay-scores FILE` every pair score from a recording (`--record-scores FILE` writes one); then the run needs no `.env` keys and no GPU. A call or a pair score that was not recorded stops the run before any program is written. The released runs' calls hold paper text and are not public; [`tests/test_extract_train.py`](../../tests/test_extract_train.py) replays a run recorded on synthetic papers (`tests/fixtures/synthetic-training-replay.zip`, made by `tests/fixtures/make_synthetic_replays.py`).

### Inputs and outputs

| Input | Where | What it is |
|---|---|---|
| Training dataset | `--dataset` (default `data/frozen/results/training-20260925/runner/train-dataset.json`) | `papers`: each paper's `paper_id`, `split` (`train` or `dev`), `title`, `tei_sha256`, `text_sha256` and `gold_records`, in the ground truth's order |
| Split | `--split` (default `…/runner/split.json`) | `trainset`, `valset` and `forbidden_test_ids`; the dataset must agree with it |
| TEIs | `--teis` (default `data/paper_sources.zip`, which holds paper text and is not part of this repository) | the papers' GROBID TEIs |
| Papers | `--paper`, optional, repeatable | train on these papers of the dataset only; a TEST paper is refused |
| Endpoint, scoring device | `.env`: `DSPY_API_BASE`, `DSPY_API_KEY`, `SCIDEF_NLI_DEVICE` | not read in a replay |

| File in `--out` | What it is |
|---|---|
| `program.json` | the trained program (DSPy's saved state: the instruction, the field names and the demo) |
| `manifest.json` | `program`, `program_sha256`, `window_chars`, `overlap_chars`, the model: what `extract run --programs` reads |
| `requests-train/<request_sha256>.json`, `usage-train.jsonl` | every model call in full, and one usage line per call ([docs/setup/llm.md](../setup/llm.md)) |
| `receipt.json` | the settings (model, revision, window, maximum tokens, seed, thinking, the optimizer's settings, the metric and its threshold, the seed-9 order), the TRAIN and DEV papers, the hashes of the inputs, the papers tried with their scores, where the demo comes from, `PYTHONHASHSEED`, the code commit and the time |

The plan files that the released program manifests name (`plan.json`, `78e4cdc7…`, and `plan-amendment-1.json`, `5374c7ae…`) and the released runner's job plan (`runner/job-plan.json`, with the optimizer settings of all three optimizers) are in the same frozen folder ([../data/frozen/README.md](../../data/frozen/README.md)).

### The training dataset: `extract train-dataset`

`mbdef extract train-dataset --out FOLDER` builds the dataset and split above from the ground truth and the TEIs ([`train.py`](../../src/mbdef/extract/train.py), `build_dataset`; checked by [`tests/test_extract_train_dataset.py`](../../tests/test_extract_train_dataset.py)):

- **`train-dataset.json`**: the TRAIN and DEV papers of `--ground-truth` (default MBDef-Gold's `$DATA_ROOT/MBDef-Gold/validated_gt.json`; the file names it `data/validated_gt.json`, where the released training read it), in its order, each with its place in that order, its split, its title, its TEI's file name, sha256 and size, the sha256 and length of the text `extract run` reads from it, its number of windows at 8,000, 16,000 and 32,000 characters, and its reference records. The title is the paper's `metadata.title` in `--metadata` (default `data/ground_truth/v3.json`), or the TEI's own title for a paper missing there (none of the released 101). The TEIs come from `--teis` (default `data/paper_sources.zip`, not part of this repository), one per paper, named by its id; a missing TEI or a TEI without text stops the build.
- **`split.json`**: the TRAIN ids (`trainset`), the DEV ids (`valset`) and every TEST id (`forbidden_test_ids`: splits `original_test`, `new_test` and `test`) of the three files `--ground-truth`, `--non-relevant` and `--metadata`. TEST papers are read for their ids only. A paper whose split differs between the files, or a TRAIN or DEV paper that is TEST elsewhere, stops the build.
- Both record the sha256 of the training plan and its amendment (`--plan`, `--amendment`, default the frozen `plan.json` and `plan-amendment-1.json`; the amendment must name the plan's sha256). Their fixed text fields (the schema names, `title_source`, `order`, `note`) keep the released wording, and the input paths are written as given, so that run from the repository root with the defaults the two files are the released ones byte for byte.

**A worked example.** The first paper of the released dataset (position 0) is DEV paper `016f7a07…`, "Racial and Ethnic Disparities in Crime and Criminal Justice in the United States": its TEI has 240,295 bytes, its text 140,017 characters, which make 23 windows of 8,000 characters, 11 of 16,000 and 6 of 32,000.
The dataset has 80 TRAIN and 21 DEV papers, and the split refuses 32 TEST ids:

```bash
uv run mbdef extract train-dataset --out "$DATA_ROOT/extract/train-dataset"
```

```json
{"papers": {"train": 80, "dev": 21}, "forbidden_test_ids": 32,
 "sha256": {"train-dataset.json": "237803dd4de365fd4fe85f9ed3bdb2a35841eaf9701802e16262a7de04940bcd",
            "split.json": "26c76170a0ddf9acb72525802c08bdfa9eea67f6e0dc1edf465a8db9a65fdccc"}}
```

Those are the hashes of the frozen files and the ones the released runner's job plan pinned (CI checks the frozen files against the job plan; a rebuild needs the ground-truth TEIs). A miniature with synthetic papers checks every field, the TEI-title fallback, the TEST ids of all three files and the refusals. **Not rebuilt:** `runner/job-plan.json`, the cluster run's own plan (its servers, folders, deadline and the optimizers' settings, with the time it was written); `extract train` does not read it. The released builder also matched a TEI to a paper by DOI or title when its file name was not a paper id; every released TEI is named by its paper id, so this command reads them by id only.

### How it was checked

- **The training tests** ([`tests/test_extract_train.py`](../../tests/test_extract_train.py)): a w8k run on the six TRAIN and DEV papers of `tests/synthetic_papers.py` with their own dataset and split, replayed from a run recorded with a stand-in model and stand-in pair scores (`tests/fixtures/synthetic-training-replay.zip`). The first TRAIN paper of the seed-9 order (s1) has references that paraphrase its definitions and scores 0, the second (s2) scores 1, so Bootstrap stops after 2 calls and the program's demo is s2's one window; the program equals the recorded one, and the unrecorded call, the unrecorded pair score, the TEST paper and `check-programs` are tested on it. The frozen dataset and split are the ones the released runner's job plan pinned, the plan files the ones the program manifests name, and the released dataset and split refuse a TEST paper.
- **The released runs, replayed** (by hand, from their recorded calls and pair scores, which hold paper text and are not public): the three Bootstrap runs (w8k, w16k, w32k: 9, 5 and 2 calls) on the first 4 TRAIN and the first 2 DEV papers made exactly the released calls and no other, the program equalled the released one except for the demo, and under `PYTHONHASHSEED=0` the w32k program was the released file byte for byte; `check-programs` on the three released programs gives the table below.
- **A small live run** (by hand, on a 24 GB laptop GPU, not the full 80 papers): `--window w32k` on the first TRAIN paper (`--paper f5c28232…`), with a dataset rebuilt by `extract train-dataset`, and the released model served by vLLM: 2 calls in 1.6 minutes, a score of 0.186 (the released run: 0.226; the model samples), the demo from window 0 of 2, and `check-programs` passed. With Qwen3-VL-2B-Instruct (`--thinking unset`) the same run scored 0.068 and kept a demo with 247 of its 579 record strings not in the window: Bootstrap accepts any score above 0, so a weak model can teach the program its own mistakes.

**What the checks found in the released programs** (`extract check-programs`):

| Program | Demo | Its record strings in the window |
|---|---|---|
| `w8k` | window 3 of 9 of "Racism and Health" | 30 of 30 word for word |
| `w16k` | window 2 of 5 | 5 of 5 |
| `w32k` | window 0 of 2 | 22 of 26 word for word; 3 terms in another case (*Structural racism* is written "structural racism"); 1 definition not in the window: "Discrimination ... exists in two forms: (a) where individuals …", which the model shortened with "..." |

A demo is the model's answer before the string check: extraction drops strings that are not in the window, but the demo keeps them, and the w32k program shows this shortened quote to the model at every call.

**The demo window cannot be redrawn as released.** Replaying the released answers draws window 6 for w8k and window 0 for w16k under `PYTHONHASHSEED=0` (released: 3 and 2), and other windows in processes with other hash seeds; only the w32k draw matches. Everything else about the programs (the calls, the scores, the paper the demo comes from, the instruction) is reproduced.

**Differences from the released runner**, none of which changes what is trained: DSPy's response cache is off and LiteLLM does not retry a failed call (released: a private cache, 3 retries; the released runs had no failed call); there is no context cap on `max_tokens` (released: lowered when prompt plus 32,768 tokens would pass 65,536; never needed, because Bootstrap's calls carry no demo, and no call of the three runs was capped); the TEST guard is the dataset itself (it holds no TEST paper) plus the refusal of TEST ids, where the released runner also ended the process on any TEST id; and there is no deadline watchdog or resume, which MIPRO's and GEPA's hours-long runs needed.

### How to adapt it

- **Another model or server:** `--model`, `--revision` and `.env`, as for `extract run`. `--thinking` (`on`, `off` or `unset`, as for `extract run`), `--max-tokens`, `--seed`.
- **Other papers or a new ground truth:** build a dataset and a split with `extract train-dataset --ground-truth … --metadata … --teis …` ([above](#the-training-dataset-extract-train-dataset)) and pass them with `--dataset` and `--split`. `--paper` trains on a subset in the same order.
- **Another window size:** add it to `WINDOWS` in `train.py`.
- **More demos, or a stricter acceptance:** `MAX_BOOTSTRAPPED_DEMOS`, and `metric_threshold` of `dspy.BootstrapFewShot` in `train()` (released: none, so any score above 0 is accepted); `--threshold` is the metric's match cutoff.
- **Another optimizer:** replace `dspy.BootstrapFewShot(...)` in `train()`. MIPROv2 and GEPA take the DEV papers as `valset` (`paper_examples` of the DEV papers); their released settings are in `runner/job-plan.json` under `optimizers`. They were not selected. The released runs used DSPy 3.3.1; DSPy 3.4.1 changed GEPA's default proposer, so a GEPA run with the same seed differs from the released one.
- **What to rerun after a change:** `extract run` with the new programs (`--programs`), the verifier and filter stages, scoring on DEV (`evaluate score`), and everything after Step 3.

## A corpus run: `extract plan`, the jobs, `extract derive`

The seed papers (3,869) and the closure's papers (4,598 extracted) are too many for one command. The released runs cut them into **shards** of about 40 papers, ran each part's extraction, verifier and filter on every shard, and then pooled the parts into the variants. `extract plan` writes that cut and the list of commands (the **jobs**); the jobs are the commands above (`extract run`, `extract verify`, `extract filter`); and [`extract derive --set`](#the-seed-runs-variants-extract-derive) pools their per-paper outputs. `plan` and `derive` call no model.

1. **Rows.** One row per paper: its id, its title (`frozen_title`) and its TEI's sha256. They come from `--rows` (a JSON list; every key of a row is kept as given, and a row's `tei_sha256` must be its TEI's), or else from every TEI in `--teis`, with the TEI's own title. `--paper` keeps only the given papers.
2. **Windows.** Each paper's text is cut into windows at the three sizes of the released programs (8,000, 16,000 and 32,000 characters), as `extract run` cuts it, and the windows are counted.
3. **Shards.** There are ceil(papers / `--shard-papers`) shards (default 40 papers each). The papers, longest first (by their 16k plus 32k windows, then by paper id), go one at a time to the shard with the least work so far (then the fewest papers, then the lowest number). A paper's work is its 16k and 32k windows plus half its 8k windows, rounded down. Each shard lists its papers by id.
4. **Jobs.** For every part of every variant in the variants file (`--picks`, default `data/extractors/picks.json`, the four released variants) and every stage it needs, together with the stages these read first:

   | Stage | Command | Reads | Model server |
   |---|---|---|---|
   | `combined` | `extract run --shard … --part w8k --format chat` (the part's programs and format) | the shard's TEIs | the part's model (Gemma) |
   | `verified-self` | `extract verify --model <the part's model>` | `combined` | the part's model |
   | `verified-qwen` | `extract verify` (Qwen, the default) | `combined` | Qwen |
   | `filtered-ordinary`, `filtered-title` | `extract filter --filter ordinary` (or `title`) | `verified-self` | none (BGE) |
   | `filtered-…-noverify` | `extract filter --records union` | `combined` | none |
   | `filtered-…-qv` | `extract filter` | `verified-qwen` | none |

   The released variants need five stages for each of the two chat parts (`combined`, `verified-self`, `filtered-ordinary-noverify`, `verified-qwen`, `filtered-ordinary-qv`) and two for each of the two JSON parts (`combined`, `filtered-ordinary-noverify`): 14 jobs per shard. Each stage goes to its own folder, `runs/<model>/<arm>/<format>/<shard>/<stage>/`; `combined` is the part's folder inside `extract run`'s output (`…/<shard>/extract/w8k`).

**A worked example: the seed papers.** With the seed run's rows (`runner-rows.json`, in its `inputs/` folder in the data root; `mbdef data fetch` fetches it) and its TEIs (`seed-tei/` beside it, not public: they hold paper text; the GROBID TEIs of the seed papers you fetched take their place):

```bash
INPUTS=…   # the seed run's inputs/ folder in the data root
uv run mbdef extract plan --teis "$INPUTS/seed-tei" --rows "$INPUTS/runner-rows.json" \
    --out "$DATA_ROOT/corpus/seeds"
```

It prints `3869 papers in 97 shards of 29 to 42 papers; 14 jobs per shard in …/plans/jobs.tsv`, so 97 × 14 = 1,358 jobs over 29,333 windows of 8k, 14,815 of 16k and 7,969 of 32k. The longest paper (`pdf-f254647d…`: 135 windows of 8k, 64 of 16k and 31 of 32k, so its work is 64 + 31 + 135 // 2 = 162) goes first, to `shard-000`, which then only takes short papers: it ends with 29 papers and 304, 151 and 79 windows (151 + 79 = 230 long windows). Every shard ends with between 229 and 240 long windows. These are the 97 shard files of the released seed run, byte for byte. The jobs of `shard-000` for the `w8k` chat part, in `plans/jobs.tsv` (`RUN` is the `--out` folder, `…/shard-000` is `RUN/runs/gemma12b/TR-bootstrap-8k/chat-default-think/shard-000`):

| Stage | Model server | Command |
|---|---|---|
| `combined` | `gemma12b` | `mbdef extract run --teis …/seed-tei --shard RUN/shards/shard-000.json --part w8k --format chat --out …/shard-000/extract` |
| `verified-self` | `gemma12b` | `mbdef extract verify --extraction …/shard-000/extract/w8k --out …/shard-000/verified-self --model google/gemma-4-12B-it-qat-w4a16-ct --revision 1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee` |
| `filtered-ordinary-noverify` | `-` | `mbdef extract filter --run …/shard-000/extract/w8k --out …/shard-000/filtered-ordinary-noverify --filter ordinary --records union` |
| `verified-qwen` | `qwen` | `mbdef extract verify --extraction …/shard-000/extract/w8k --out …/shard-000/verified-qwen` |
| `filtered-ordinary-qv` | `-` | `mbdef extract filter --run …/shard-000/verified-qwen --out …/shard-000/filtered-ordinary-qv --filter ordinary` |

**Running the jobs.** Run them from the folder `extract plan` ran in (the commands name the folders as they were given). A shard's jobs are listed in an order that works; so is one model server at a time: first a shard's Gemma jobs (with `DSPY_API_BASE` in `.env` pointing at a Gemma server), then its Qwen jobs (a Qwen server), then its filters (BGE on `SCIDEF_NLI_DEVICE`):

```bash
RUN="$DATA_ROOT/corpus/seeds"
awk -F'\t' '$1 == "shard-000" && $4 == "gemma12b" {print $5}' "$RUN/plans/jobs.tsv" | uv run sh -e
awk -F'\t' '$1 == "shard-000" && $4 == "qwen" {print $5}' "$RUN/plans/jobs.tsv" | uv run sh -e
awk -F'\t' '$1 == "shard-000" && $4 == "-" {print $5}' "$RUN/plans/jobs.tsv" | uv run sh -e
uv run mbdef extract derive --set "$RUN" --out "$DATA_ROOT/derive/seeds-run"
```

On a cluster, one array task per shard and model server does the same ([docs/setup/cluster.md](../setup/cluster.md)). A job that stops leaves its folder half written; delete the folder and run the job again (each command wants a new or empty `--out`). `extract derive` reports a stage folder that is not there as `PROBLEM: missing folder runs/…` and counts its papers as missing for the variants that need it.

**The closure's papers.** Their TEIs are named by the acquisition, not by paper id, in the sub-folders of the closure's `grobid/` folder (here `$GROBID`), each beside the conversion receipt that names its paper; `--teis` reads that folder as it is (and so do the planned `extract run` jobs). Their DOIs come from the acquisition, so give the closure run's own DOIs with `--metadata` (the closure run's `rows-all.json`, `paper-metadata.json` and `grobid/` folder are in the data root, [docs/step-4-closure/closure.md](../step-4-closure/closure.md#inputs)):

```bash
uv run mbdef extract plan --teis "$GROBID" --rows rows-all.json \
    --metadata paper-metadata.json --prefix closure --out "$DATA_ROOT/corpus/closure"
```

This gives 4,598 papers in 115 shards of 30 to 42 papers, with 39,085, 19,693 and 10,498 windows: the totals the closure run recorded (checked, in about 8 s). (The closure run cut its papers into seven batches of shards instead; the shards do not change any derived file, because `extract derive` pools per paper.) Without `--metadata` the run's `paper-metadata.json` holds the first DOI each TEI names (the header's, if it has one; `parse_tei`): for the seed papers these are the seed manifest's DOIs, all 3,869 of them, but 1,818 closure papers have a DOI only from the acquisition.

**A worked example without a model: two ground-truth papers.** The released TEST run's recorded answers for "Branding the nation" (`4243aa9b…`) and "The science of fake news" (`73bfad11…`) cover extraction, Gemma's own verification, Qwen's verification and the filter's BGE vectors of both chat parts (they hold paper text and are not public). Planned with the released variants less P≥20 (whose JSON parts were not recorded), the two papers make one shard of 2 papers with 6, 3 and 2 windows, and 10 jobs. Every line of `plans/jobs.tsv`, each run with `--replay` (and the stand-in tokenizer of the verifier's tests), and then `extract derive --set` give these numbers: for "The science of fake news", P≥28 pools the `w8k` part's 8 filtered records and the `w16k+w32k` part's 8 into 8 records with 15 definitions, and P≥24 pools 9 and 8 records of Gemma's own verification into 9 records with 16 definitions. Each part's records are the ones the released run published for that stage, and P≥24 and P≥28 share 28 of their 29 definitions (Jaccard 0.9655).

### How the corpus run was checked

- **`extract plan`** ([`tests/test_extract_derive.py`](../../tests/test_extract_derive.py), in CI): five synthetic TEIs ([`tests/mini_derive.py`](../../tests/mini_derive.py) works out their windows and shards by hand), the shard files, sha256 and window counts, the DOIs, the jobs and their commands, rows kept as given, `--metadata`, and the refusal of a TEI whose sha256 is not its row's. `extract run --shard` gives the same windows and unions as `--paper` with the same papers ([`tests/test_extract.py`](../../tests/test_extract.py)), and refuses a TEI whose sha256 is not the shard's.
- **The full run on recorded answers** (by hand, on the released TEST run's recordings, which are not public): the worked example above, every job of `plans/jobs.tsv` and `extract derive`, against the released TEST run's published stages.
- **Per-paper outputs against tarballs** (in CI): the miniature run of `extract derive` ([below](#the-seed-runs-variants-extract-derive)) kept as per-paper folders instead of tarballs gives the same files, byte for byte, except the time and the output folder in `DONE.json` and `compare.json`, which also differ between two runs from the tarballs; and a missing stage folder is reported.
- **On the recorded seed run** (by hand, with its TEIs and part tarballs, which hold paper text and are not public; CPU, about 1 minute): `extract plan` with the run's rows and TEIs writes its 97 shard files byte for byte (the sha256 of the list of their sha256 is `1694132d…` for both) and each shard's papers and windows, and its DOIs are the seed manifest's. Three shards (000, 048 and 096, as in the filter's recheck; 111 papers) were converted from the part tarballs into per-paper folders: each part's `union.json` and `verified.json` (Gemma's and Qwen's) copied, and each filter's `filtered.json` written from the probabilities the run recorded, not from its published stage. `extract derive` on them writes the same files as on the tarballs, and every paper's file of the four variants is the released run's file, byte for byte (444 files; the sha256 of the list of their sha256 is `71e7486c…` for both).
- **The closure's rows:** `extract plan` accepted every TEI of the closure run's 4,598 rows (sha256), read from the closure's `grobid/` folder by their conversion receipts, and gave its recorded window totals (above). Its parts were not rerun or converted: they are not in the data root.
- **Not done here:** a live run of the jobs with a model (it needs model servers on a GPU).

### How to adapt the corpus run

- **Other variants:** `--picks` with another variants file; any stage of the table above.
- **Other parts:** a part's arm and format become `extract run`'s `--part` and `--format` through `RUN_PARTS` and `RUN_FORMATS` in `extract/plan.py`; its model through `MODELS`.
- **Other shards:** `--shard-papers` and `--prefix`; the balance rule is `cut_shards`.
- **Another layout or other commands:** edit the `jobs` of `plans/grid.json` (each names its folder and its command with `{shard}`); `extract derive` reads the folders the plan names.

## The seed run's variants: `extract derive`

The seed run ran the released extractor's parts on the 3,869 seed papers, each with its verifier and filter stages, and saved every part's stage records. A **variant** (i.e., selected extractor's configuration) is one stage of one or more parts, as `picks.json` lists them. `extract derive` turns the saved records into the four released variants, without any model:

| Variant | Parts | Stage | Definitions |
|---|---|---|---:|
| P≥20 | `w8k` and `w16k+w32k`, each in the chat and the JSON format (4 parts) | no verifier, then the filter (`filtered-ordinary-noverify`) | 106,339 |
| P≥24 | `w8k` and `w16k+w32k`, chat (2 parts) | Gemma's own verifier, no filter (`verified-self`) | 77,560 |
| P≥28 | the same 2 parts | the Qwen verifier, then the filter (`filtered-ordinary-qv`) | 62,433 |
| P≥28 Gemma only (`P>=28-single-model`, plan amendment 9) | the same 2 parts | no verifier, then the filter | 69,795 |

1. **Derive.** For each variant and paper, the parts' records are put one after the other in the listed order and merged with `records.merge` (same term ignoring case: one record; a term without a definition is dropped); a variant of one part keeps its records as they are. Each paper gets a file per variant, each variant a `predictions.json`, and the **union** pools all four variants per paper the same way. The union is the input of the corpus closure.
2. **Compare.** The variants are compared without gold: per paper, a definition is its term and text after NFKC, case folding and collapsing white space. For each pair: the definitions both have, those only one has, and the Jaccard index (shared divided by all). It also counts which part each variant's definitions come from.

**A worked example: "Emotional Reframing of Economic News using a Large Language Model" (`pdf-03bc50f4…`).** For P≥20 its four parts kept 3, 2, 4 and 3 records (`w8k` chat, `w8k` JSON, `w16k+w32k` chat, `w16k+w32k` JSON); merged, that is 6 records with 10 definitions, for example *News Framing or Media Framing*: "News Framing or Media Framing usually involves the use of language that shapes information delivery and the reader's opinion formation [4,11]." P≥24 keeps 4 records (5 definitions), P≥28 4 (6), P≥28 Gemma only 5 (7), and the union of all four is again 6 records with 10 definitions, because every definition of the smaller variants is also in P≥20. Over all papers, P≥20 and P≥28 share 62,391 definitions, 43,948 are only in P≥20 and 42 only in P≥28: Jaccard 62,391 / (62,391 + 43,948 + 42) = 0.5865.

### Run it

The seed run's plan and saved parts (`frozen/runs/production-20260925/corpus/seeds/{plans,parts}/`, 3.7 GB) are not public: the parts hold paper text. With them in the data root, the variants are derived with

```bash
uv run mbdef extract derive --out "$DATA_ROOT/derive/seeds"
```

Without them the command stops and says so; the variants of a run of your own come from `extract plan`, the jobs it lists and `extract derive --set RUN` ([above](#a-corpus-run-extract-plan-the-jobs-extract-derive)).
The derived variants themselves are in the data root (`frozen/runs/production-20260925/corpus/seeds/derived/` and `compare/`, `mbdef data fetch`).

It needs `zstd` (the parts are zstd tarballs) and takes about 30 s on a workstation (16 processes read the 582 tarballs it needs, once to derive and once to compare). It prints:

```text
P>=20: 3869 papers, 54877 records, 106339 definitions
P>=24: 3869 papers, 49190 records, 77560 definitions
P>=28: 3869 papers, 38659 records, 62433 definitions
P>=28-single-model: 3869 papers, 43142 records, 69795 definitions
union: 3869 papers, 65495 records
P>=20 vs P>=24: 61816 definitions shared, Jaccard 0.5063
P>=20 vs P>=28: 62391 definitions shared, Jaccard 0.5865
…
P>=28 vs P>=28-single-model: 62377 definitions shared, Jaccard 0.893
```

A missing tarball is printed as `PROBLEM: missing archive parts/…`; its papers count as missing for the variants that need it (`missing_part_outputs`), and the other parts' records are still pooled.

### Inputs

The run's folder, `--set` (default `$DATA_ROOT/frozen/runs/production-20260925/corpus/seeds`, [docs/setup/data-root.md](../setup/data-root.md); `data fetch` brings its `picks.json`, `shards/`, `derived/` and `compare/`, while its plan and parts are not public):

| Input | What it is |
|---|---|
| `picks.json` | The variants: for each, its parts, its stage and the selection's numbers; `--picks` replaces it |
| `plans/grid.json` | The 97 shards; each names its manifest of papers by its path on the machine that ran it, which is read from `shards/` by file name and checked against the plan's sha256 |
| `shards/shard-*.json` | The papers of each shard: paper id and title |
| `parts/<model>/<arm>/<format>/shard-*.tar.zst` | Each part's saved run per shard; the stage records are `cohorts/<shard>/…/stages/<stage>/<paper>.json` inside. A Gemma part has `shard-*.qv.tar.zst` too, with the Qwen verifier's stages |
| or `runs/…` | A run planned by [`extract plan`](#a-corpus-run-extract-plan-the-jobs-extract-derive) instead: its plan's `jobs` name each part's stage folder per shard, with `papers/<sha256 of the paper id>/` and the file of the stage's command: `union.json` (`records`) for `combined`, `verified.json` for `verified-…`, `filtered.json` for `filtered-…`; a paper without its file counts as missing. The run's `paper-metadata.json` gives the DOIs |
| `../../inputs/seed-manifest.jsonl` | The papers' DOIs (the TEI header's, else the other one it lists); a closure run's `paper-metadata.json` in the run's folder is read too |

### Outputs

| File in `--out` | What it is |
|---|---|
| `derived/<variant>/papers/<paper id, URL-quoted>.json` | One paper's records in one variant, with its parts' record counts, stage, title |
| `derived/<variant>/predictions.json` | All papers of a variant (with title and DOI), the variant's settings, the missing part outputs |
| `derived/union/predictions.json` | All four variants pooled per paper |
| `derived/DONE.json` | Per variant its counts and the path and sha256 of its `predictions.json`; the problems; the sha256 of `picks.json`; the time |
| `compare/compare.json`, `per-paper.json`, `compare.md` | The comparison: totals, pairs, part contributions; per paper; as a table |
| `receipt.json` | The settings, the hashes of the inputs (`parts/` is the 3.66 GB folder; a planned run's `runs/` instead), the outputs' hashes, the counts, the code commit and the time |

Every file has the released run's byte format (indent 1, sorted keys). The union's `note` keeps the released wording ("the three picks' records … (P>=20, P>=24, P>=28)"), although the run had four variants by then and the union pools all four, because the note is part of the released file; likewise `compare.json` gives `shared_by_all_three` only for exactly three variants (`null` for four).

### How it was checked

- **The saved run, derived again** (by hand, with its plan and parts, 32 s): all 4 × 3,869 paper files, the four `predictions.json`, the union, `per-paper.json` and `compare.md` are byte-identical to the saved run; `DONE.json` and `compare.json` are identical without their time and with each `predictions.json` path from `derived/` on. The counts: 3,869 papers; P≥20 106,339, P≥24 77,560, P≥28 62,433 and P≥28 Gemma only 69,795 definitions; Jaccard 0.5063, 0.5865, 0.7378, 0.893; 42 P≥28 definitions not in P≥20.
- **Tests** ([`tests/test_extract_derive.py`](../../tests/test_extract_derive.py), in CI): a miniature run ([`tests/mini_derive.py`](../../tests/mini_derive.py): 2 shards, 3 papers, 3 parts, the four variants) whose numbers are worked out by hand: the pooling and merge per variant, the `.qv` tarballs, the union, a missing tarball, the comparison's pairs and part contributions, a second run that differs only in the time and the output folder, and the receipt. The same run kept as per-paper folders (as `extract run`, `verify` and `filter` write them) gives the same files, and a missing stage folder is reported; three real shards were checked the same way ([above](#how-the-corpus-run-was-checked)).
- **The original implementation:** `derive` and `compare` were checked against its outputs on the same miniature run, with a missing tarball and a variant without a selected configuration: the same files byte for byte, except the time and the folder (and a missing tarball is named by its path in the run's folder, not its full path).
- The relevance filter inside the saved stages was rechecked on 3 of the 97 shards ([above](#how-the-verifier-and-the-filter-were-checked)).

Not rebuilt: `derived/reference/` and `compare/compare-with-reference.json` in the saved run, an earlier set of variants and its comparison, and the closure's copy of `derived/` ([docs/step-4-closure/closure.md](../step-4-closure/closure.md)).

### How to adapt it

- **Other variants of the same run.** Write a variants file (in the format of `picks.json`) with other parts or stages (any stage the tarballs hold: `combined`, `verified-self`, `filtered-title`, `filtered-ordinary`, their `-noverify` forms, and for Gemma `verified-qwen`, `filtered-title-qv`, `filtered-ordinary-qv`) and pass `--picks`.
- **Another run with this layout** (for example the closure run's set): `--set FOLDER`.
- **A new extraction run** (`extract run`, `verify`, `filter`): plan it with [`extract plan`](#a-corpus-run-extract-plan-the-jobs-extract-derive) and run its jobs; `--set` is then the plan's `--out`. `derive` and `compare` both read the parts through `read_stages`.
- The pooling rule is `records.merge`; the comparison's key is `definition_keys` in `derive.py`.

## Not in this step

- **Step 5's use of the filter.** Step 5 replays the same frozen filter on the ground truth with its own reader of the zip (`postprocess extractor-gate`, [docs/step-5-post-processing/postprocess.md](../step-5-post-processing/postprocess.md)).
- **Window-level training examples** and other chunking strategies were not on the released path: the released training used one example per paper (`paper_examples`).
