# Relevance: `mbdef relevance`

| | |
|---|---|
| **Paper step** | Steps 1 and 2: which papers of the starting corpus are about media bias. Two language models read each paper's title and abstract, a third decides where they disagree, and the papers marked relevant are the ones the seed list ([docs/step-1-2-papers/seeds.md](seeds.md)) was made from |
| **Code** | [`src/mbdef/relevance/`](../../src/mbdef/relevance/): [`classify.py`](../../src/mbdef/relevance/classify.py) (the requests, the answer parser, the replay), [`decide.py`](../../src/mbdef/relevance/decide.py) (agreement, disagreements, the list) and the prompt templates in [`prompts/`](../../src/mbdef/relevance/prompts/) |
| **Commands** | `mbdef relevance classify`, `… compare`, `… export` |
| **Checked by** | [`tests/test_relevance.py`](../../tests/test_relevance.py): a replay of eight made-up papers through the three commands, the replay's guard, the answer parser, a live run against a local test server; and (slow, with the data root) a replay of the released answers that gives the released list row for row |

## Run it on your papers

Write a papers file, a CSV with one row per paper: `corpus_id` (unique), `paperid` (any id; the same value if you have no other), `title`, `abstract` and optionally `doi`.
Each classifier needs an OpenAI-compatible server: `DSPY_API_BASE` and `DSPY_API_KEY` in `.env` ([docs/setup/data-root.md](../setup/data-root.md)), or `--api-base` for one command (the key always comes from `.env`).
For Qwen and Gemma a local vLLM server with the model served under the classifier's name (`vllm serve Qwen/Qwen3.6-35B-A3B --served-model-name qwen3_6_35b_a3b`; or pass `--model`); for GLM, OpenRouter:

```bash
uv run mbdef relevance classify --classifier qwen  --papers papers.csv --out out/qwen
uv run mbdef relevance classify --classifier gemma --papers papers.csv --out out/gemma
uv run mbdef relevance compare --qwen out/qwen --gemma out/gemma --out out/compare
uv run mbdef relevance classify --classifier glm --papers out/compare/disagreements.csv \
    --api-base https://openrouter.ai/api/v1 --out out/glm
uv run mbdef relevance export --compare out/compare --glm out/glm --out out/final
```

`out/final/relevant-papers.csv` lists the relevant papers.
Any model can take any of the three roles: `--model` (its name on the server) and `--api-base`; the role names (`--qwen`, `--gemma`, `--glm`) are those of the released run.
A long run keeps no partial results: split the papers file, run each part into its own folder, and join the `answers.csv` files (with one `metadata.json`) before `compare`.

## What the step does

1. **`classify`** asks one classifier about every paper with a title or an abstract: one chat request with a system prompt (three definitions of media bias and the rule) and a user prompt with the title and the abstract ([The prompts](#the-prompts)). The answer is reduced to one label (`relevant`, `not relevant`, or `can't determine` for "not sure" and anything else): the text after the model's reasoning, its explicit final answer if it gives one, else the last label it names. It runs for the two classifiers, **Qwen** (Qwen3.6-35B-A3B, served as `qwen3_6_35b_a3b`) and **Gemma** (Gemma 4 12B instruction-tuned, served as `gemma_4_12b_it`).
2. **`compare`**: a paper both answered `relevant` is **agreed**; a paper with two different labels is a **disagreement**; any other paper is not relevant.
3. **`classify` again**, with the arbiter **GLM** (GLM 5.2 through OpenRouter, `z-ai/glm-5.2`), on the disagreements only. Its prompt first shows eight labelled papers chosen to be diverse.
4. **`export`** writes the relevant papers: the agreed ones, then the disagreements the arbiter answered `relevant` (**tie-break**).

All requests send `max_tokens` 2048 and no temperature, so each model samples with its own settings.

**A worked example.** *Sharing Lies: A Socio-Technical Review on False Information Dissemination* (paper id `hash_ac28c0b97b92b880`). Gemma answered `relevant`; Qwen reasoned that false information is not media bias and answered `not relevant`. The labels differ, so GLM decided, after its eight examples: `relevant`. The paper is in the list with route `tie-break`, and it is a seed paper with the same route. *Media as Actors of Two-Stage Cultural Gatekeeping* got `relevant` from both (`agreed`); *Contrastive Perplexity for Controlled Generation* was a disagreement that GLM answered `not relevant`, so it is not in the list.

## Replicate the released list

Fetch the data root's `relevance/` folder (`uv run mbdef data fetch relevance`, [docs/setup/data-fetch.md](../setup/data-fetch.md)). A replay of the released answers needs no model and no key:

```bash
R="$DATA_ROOT/relevance"
uv run mbdef relevance classify --classifier qwen  --replay "$R/answers/qwen"  --out out/qwen
uv run mbdef relevance classify --classifier gemma --replay "$R/answers/gemma" --out out/gemma
uv run mbdef relevance compare --qwen out/qwen --gemma out/gemma --out out/compare
uv run mbdef relevance classify --classifier glm --papers out/compare/disagreements.csv \
    --replay "$R/answers/glm" --out out/glm
uv run mbdef relevance export --compare out/compare --glm out/glm --out out/final
```

It takes about a minute and prints:

```text
qwen: 14482 answers (11457 not relevant, 2998 relevant, 27 can't determine) in 8s; papers: 14862; without title or abstract: 300; not answered: 90; paired with another paper's text: 47
gemma: 14562 answers (9376 not relevant, 5074 relevant, 112 can't determine) in 7s; papers: 14862; without title or abstract: 300; not answered: 33; paired with another paper's text: 116
answered by both: 14482; both not relevant: 9305; agreed: 2963; disagreements: 2196; both can't determine: 18
glm: 2192 answers (1033 not relevant, 925 relevant, 234 can't determine) in 2s; papers: 2196; without title or abstract: 4; not answered: 1; paired with another paper's text: 7
agreed: 2963; tie-break: 925; relevant: 3888; disagreements: 2196; arbiter answers: 2192
```

| Stage | Count |
|---|---:|
| Papers of the starting corpus (the same TEI files as [`data/seeds/starting-corpus.tsv`](../../data/seeds/starting-corpus.tsv)) | 14,862 |
| without title and abstract, never sent | 300 |
| answered by both classifiers | 14,482 |
| agreed (both `relevant`) | 2,963 |
| disagreements, given to GLM | 2,196 |
| GLM's `relevant` answers (tie-break) | 925 |
| relevant papers: 2,963 + 925 | 3,888 |
| removed before extraction: 4 notices that are not papers, 15 without a title ([docs/step-1-2-papers/seeds.md](seeds.md)) | 19 |
| **Seed papers** | **3,869** |

What a replication should know:

- **2,196 disagreements.** The recorded run has 2,196; GLM answered 2,192 of them, and the other 4 had no title or abstract to send.
- **Answers are paired with papers by order.** The released answers name only the `paperid`, which several papers share (a placeholder DOI such as `10.1145/nnnnnnn.nnnnnnn`, a DOI cut short, or no metadata). So the k-th answer under an id belongs to the k-th paper under that id. The answers were recorded in the order they came back, so for shared ids an answer can belong to another paper's text: 47 of Qwen's, 116 of Gemma's and 7 of GLM's (each replay prints its number). The replay keeps this pairing, because it made the released list; matched by text, 2,968 papers would be agreed and 2,189 disagreements. A live run writes the `corpus_id` with each answer, so its answers cannot be mixed up.
- **Qwen's prompts carried no examples.** Its run was configured with similar labelled examples, but none existed for the corpus papers, so every prompt it was sent is word for word Gemma's zero-shot prompt ([The prompts](#the-prompts)).
- **"Not answered"** counts papers with text that got no answer: Qwen's run never answered 80 papers, all under 9 shared ids it answered for other papers; and the pairing by order gives a few answers to a paper without text, whose id-mate then has none.
- **Answers cut off:** 353 of Qwen's answers stopped at the 2,048-token limit inside the model's reasoning; their label is the last one the reasoning names (212 `not relevant`, 139 `relevant`, 2 `can't determine`).

## The prompts

Each request is two messages: the system prompt, the same for every classifier, and a user prompt rendered from one of four templates, the prompt styles of the paper. The templates are Jinja, in [`prompts/`](../../src/mbdef/relevance/prompts/); `classify` renders them with the paper's `title` and `abstract` (and the examples' `title`, `abstract` and `manual_label`) and strips the result.

The system prompt (`system_prompt.j2`):

```jinja
You are an expert in media bias research.

Media Bias can be defined in the following ways:

1. Media Bias describes slanted news coverage or other biased media content, which can be intentional, i.e., purposefully express a tendency towards a perspective, ideology, or result, or unintentional.
2. In addition to the fact that media bias always has a journalistic format, the most repeated characteristics in the definitions of media bias that we can find in the literature are: (1) that is slanted, that is, it has the intention of influencing the opinion of the receiver in a particular direction; (2) that the same bias is frequent and/or consistent for each media/journalist; and (3) that journalists may add bias in their attempt to create memorable stories.
3. Media bias is the tendency to, consciously or unconsciously, report a news story in a way that supports a pre-existing narrative instead of providing unprejudiced coverage of an issue.

Determine whether an academic article is relevant to media bias research based only on its title and abstract.
- Classify as "relevant" if the article studies media bias directly or investigates concepts that help explain, measure, detect, produce, perceive, or mitigate media bias.
- Classify as "not relevant" only if no meaningful connection to media, journalism, news, communication, information dissemination, or media bias can be inferred from the title and abstract.
- If you are unsure classify as "not sure".
```

**ZS**, zero-shot (`zero_shot.j2`): the paper alone.

```jinja
Article to classify:
Title: {{ title }}
Abstract: {{ abstract }}

Return exactly one label and no explanation:
relevant
not relevant
not sure

Answer:
```

**CoT**, chain of thought (`cot.j2`): ZS after one sentence that asks for reasoning first.

```jinja
Think step-by-step before providing the answer.

Article to classify:
Title: {{ title }}
Abstract: {{ abstract }}

Return exactly one label and no explanation:
relevant
not relevant
not sure

Answer:
```

**FS-similar**, few-shot with similar examples (`few_shot_similar.j2`): ZS after labelled papers that are like the paper being classified.

```jinja
{% if examples %}
Use these similarly labeled examples as guidance:
{% for example in examples %}
- Title: {{ example.title }}
  Abstract: {{ example.abstract }}
  Label: {{ example.manual_label }}
{% endfor %}
{% endif %}


Article to classify:
Title: {{ title }}
Abstract: {{ abstract }}

Return exactly one label and no explanation:
relevant
not relevant
not sure

Answer:
```

**FS-diverse**, few-shot with diverse examples (`few_shot_diverse.j2`): the same text, with one fixed set of labelled papers chosen to differ from each other.

```jinja
{% if examples %}
Use these diverse labeled examples as guidance:
{% for example in examples %}
- Title: {{ example.title }}
  Abstract: {{ example.abstract }}
  Label: {{ example.manual_label }}
{% endfor %}
{% endif %}


Article to classify:
Title: {{ title }}
Abstract: {{ abstract }}

Return exactly one label and no explanation:
relevant
not relevant
not sure

Answer:
```

**How the examples were chosen.** Both sets come from a hand-labelled sample of papers. Each labelled paper's title and abstract becomes a TF-IDF vector of words and word pairs, reduced to 128 dimensions. FS-similar: for each labelled paper, the three other labelled papers most like it by cosine similarity, at least one `relevant` and one `not relevant`. FS-diverse: k-means with eight clusters (seed 42); for each label, the labelled paper nearest the centre of one cluster after another, until there are four `relevant` and four `not relevant`: eight examples, the same for every paper.

**Which were used for the final classification.**

| Classifier | Model | Prompt style | `CLASSIFIERS` template | Examples |
|---|---|---|---|---|
| Qwen | Qwen3.6-35B-A3B | FS-similar, as configured; the text sent is ZS's | `zero_shot` | none |
| Gemma | Gemma 4 12B (instruction-tuned) | ZS | `zero_shot` | none |
| GLM, on the disagreements only | GLM 5.2 | FS-diverse | `few_shot_diverse` | the eight diverse examples: `$DATA_ROOT/relevance/examples.csv` (`--examples`); the miniature's two are [`tests/fixtures/relevance-mini/examples.csv`](../../tests/fixtures/relevance-mini/examples.csv) |

Without examples FS-similar renders to ZS's text exactly (the `if` block is empty and the blank lines before the paper are stripped), so `CLASSIFIERS` gives Qwen the `zero_shot` template, which builds the same requests; the replay checks each recorded prompt against them, and `test_the_similar_examples_prompt_without_examples_is_the_zero_shot_prompt` checks that the two templates give the same text. CoT is not used by any classifier. The model and prompt comparison is reported in the paper's appendix.

## How the list relates to the seed list

[`data/seeds/`](../../data/seeds/) is what Step 3 extracted from, and it was made from this list:

- The 3,888 relevant papers' PDFs were copied into one hand-over folder, by DOI or id. The folder held 3,888 PDFs, but not one for each row: the copy matched file names, and some ids are shared.
- Step 3 converted those PDFs again and kept the ones with an article body and a title: the **3,869 seed papers**. [docs/step-1-2-papers/seeds.md](seeds.md) follows the 3,888 down to the 3,869.
- Every seed's Step 2 id (`step2_id` in `seed-papers.tsv`) is a `paperid` of this list, and each of the 3,865 seeds with route `agreed` or `tie-break` has a row of this list with that id and that route. The other 4 seeds came along under an id they share with a listed paper.

## Inputs

| Input | Where | What it is |
|---|---|---|
| The papers | `$DATA_ROOT/relevance/papers.csv`, or `--papers` | One row per paper: `corpus_id` (unique), `paperid`, `title`, `abstract`, `doi` (optional). The released file has the 14,862 papers of the starting corpus in the order the released run read them (`corpus_id` is the TEI file name without `.grobid.tei.xml`) |
| The arbiter's examples | `$DATA_ROOT/relevance/examples.csv`, or `--examples` | The eight hand-labelled papers (title, abstract, label) GLM's prompt shows, in their order |
| The released answers | `$DATA_ROOT/relevance/answers/{qwen,gemma,glm}/` | Each classifier's recorded answers (`normalized_predictions.csv`: paper id, model, both prompts, the answer, token counts, its label) and the run's settings (`metadata.json`) |
| The server | `DSPY_API_BASE`, `DSPY_API_KEY` in `.env`, or `--api-base` | Live runs only |

[`data/data-root.tsv`](../../data/data-root.tsv) lists the folder (`relevance`, 8 files, 204,310,142 bytes) with its digest. The papers and the examples hold abstracts, and the answers hold them inside the prompts, so none of this is in git.

## Outputs

| File | Written by | What it is |
|---|---|---|
| `answers.csv` | `classify` | One row per answer: `corpus_id`, `paperid`, the template, the model, the system prompt, the prompt, the answer, its token counts and its label. Live, in the papers' order; replayed, in the recorded order. A folder with it and `metadata.json` can be replayed |
| `metadata.json` | `classify` | The classifier, model, template, `max_tokens` and temperature |
| `not-answered.csv` | `classify` | Each paper without an answer, with the reason: `no title or abstract`, `not answered`, or the error of a live request that failed after three retries |
| `agreed.csv` | `compare` | The agreed papers: `corpus_id`, `paperid`, `doi`, `title`, both labels |
| `disagreements.csv` | `compare` | The disagreements, with title and abstract: the arbiter's papers file |
| `relevant-papers.csv` | `export` | The relevant papers: `corpus_id`, `paperid`, `doi`, `title`, the three labels and the route (`agreed` or `tie-break`); the released run gives 3,888 rows |
| `receipt.json` | each command | The command, settings, inputs and outputs with their sha256, the code commit, and the counts it printed |

## How it was checked

[`tests/test_relevance.py`](../../tests/test_relevance.py):

- **The whole step on eight made-up papers** (fast): [`tests/fixtures/relevance-mini/`](../../tests/fixtures/relevance-mini/) holds their recorded answers in the released form, made by [`make_relevance_mini.py`](../../tests/fixtures/make_relevance_mini.py), which says which case each paper is (agreed, a tie-break the arbiter keeps, one it drops, both not relevant, no text, two papers under one id whose answers were recorded out of order, a paper one classifier never answered). The test runs the five commands and checks the list row by row.
- **The replay's guard** (fast): with one title changed, the recorded answer to the old title stops the replay and names its `request_sha256`; a replay with another `--max-tokens` than the recorded one stops too.
- **The answer parser** (fast): nine answers and their labels, from a bare label to a final answer after reasoning that names the other label.
- **A live run** (fast): against a local test server, one request per paper with text, with the model, the messages, `max_tokens` and the key as above; a replay of the live run's folder gives the same answers without a request.
- **The released run** (slow, needs the data root's `relevance/` folder): every recorded answer answers a prompt this code builds, every recorded label is the one the parser here gives, the counts are the ones above, and the list is the released one row for row (its sha256 is pinned).

## How to adapt it

- **Other settings:** `--max-tokens` and `--temperature` (sent only when given), `--workers` (32 for Qwen and Gemma, 8 for GLM, as in the released run).
- **Other prompts or examples:** the templates in [`prompts/`](../../src/mbdef/relevance/prompts/) and `--examples`. The classifiers' templates and whether they show examples are `CLASSIFIERS` in `classify.py` (FS-similar also needs examples chosen for each paper, which this step does not make). A changed prompt changes every request, so a replay of the released answers then stops at the first answer.
- **Another rule:** the agreement rule is in `compare_command` and the export in `export_command`, both in `decide.py`.

## Not in this step

- **The search** that found the starting corpus (Step 1) is a manual Scopus query: [docs/step-1-2-papers/seeds.md](seeds.md#the-search-query) gives it word for word. To collect and convert PDFs with this repository, see [`collect`](collect.md).
- **The hand-labelled sample** and the choice of the few-shot examples.
- **The copy of the relevant papers' PDFs into the hand-over folder**, and Step 3's conversion of them: the seed list ([docs/step-1-2-papers/seeds.md](seeds.md)) and [`extract`](../step-3-extraction/extract.md).
