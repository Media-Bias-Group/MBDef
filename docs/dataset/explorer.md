# MBDef Explorer: `mbdef explorer build` and `explorer data`

| | |
|---|---|
| **What it is** | A web page to browse MBDef, the Media Bias Definitions Dataset ([dataset.md](dataset.md)): every term mention the dataset releases, the concepts they are merged into, their relevance scores, the links between concepts, and the papers |
| **Code** | [`explorer/src/`](../../explorer/src/) (the page: one HTML shell, a CSS, HTML and JavaScript file per tab), [`src/mbdef/explorer.py`](../../src/mbdef/explorer.py) (the build), [`src/mbdef/explorer_data.py`](../../src/mbdef/explorer_data.py) (the page data: map, topics, nearest mentions) |
| **Command** | `mbdef explorer build`, `mbdef explorer data` |
| **Checked by** | `tests/test_explorer.py` and `tests/test_explorer_data.py` (miniature datasets); both commands check the page data against the dataset's tables |
| **Published by** | [`.github/workflows/pages.yml`](../../.github/workflows/pages.yml), run by hand |

It is a static site: one HTML page, `explorer/index.html`, that reads JSON files from `explorer/data/`, with no server code. Both are built, not kept in git.

## Words the page uses

- a **term mention** is one term as one paper defines it, with one or more definitions;
- a mention is **kept** when its relevance score (the ensemble of three scorers, [postprocess.md](../step-5-post-processing/postprocess.md)) reaches the cut, 0.4150, and **strict** when it also reaches the stricter cut, 0.5850; MBDef holds only kept mentions, and so does the page (the relevance filter left out 5,328 of the corpus's 67,394);
- a **label group** holds every mention whose label is the same after normalisation (case, plural, spelling, hyphens, word order, a bracketed acronym); on the page each label group is one **concept** of the dataset, named as the dataset names it.

## What each tab shows

Every tab opens with its headline numbers and one interactive centrepiece; every further chart, table and worked example is one click away, in a panel that shows its key number while closed. Hero areas sit on a dark band in both themes, as on the poster.

| Tab | Centrepiece | One click away |
|---|---|---|
| Overview | The landing page: the four headline numbers (34,916 terms, 103,476 definitions, 6,115 papers, 36 topics), a turning 3D map of every mention in its topic colours, "Find a term" with suggestions, and cards that open each tab | The dataset in numbers: mentions per term, definitions per mention, relevance scores, terms per paper, what else the papers say |
| Map | Each of the 62,066 term mentions as a dot in 3D or 2D, placed by the meaning of its definitions and coloured by topic, score, strict or not, seed or cited paper, or chosen concepts; a filterable topic list; click a dot to read the mention. Where topic names would overlap, the topic led by "media bias" keeps its name first, then the other topics led by a bias concept | How to read the map, a worked example ("framing"), a table of the 36 topics |
| Concepts | Search every concept and spelling, filter by strict, merged spellings, size and topic, and read a concept: its definitions by paper, what its papers say, its family, similar concepts, its spellings and how alike its mentions are | How merging works and its limits (the 17 word-order merges flagged as likely wrong), how alike mentions are, concept sizes, the full table |
| Relations | Where the papers of two linked concepts agree or disagree on how they relate, pair by pair | The six extracted fields, the strongest links, roles against roles, how a link is made, near-duplicate concepts, cosine and name likeness |
| Relevance | The three scorers' ranks of every mention in 3D, with the cut and the strict cut as planes (every mention lies on or beyond the cut), and a worked example of one mention's score | Two scorers against each other, where they disagree most, mentions by percentile, each scorer, the plane check |
| Papers | Search and filter the 6,115 papers, sort them, and open one to see its terms and topics | Terms against definitions per paper, seed against cited papers, terms no other paper defines |

**Worked example.** Open the concept "framing" in the Concepts tab and click its mention from the paper "Agenda-Setting, Priming, and Framing". The drawer shows its two definitions and its scores: classifier 0.906, basal 1.000, judge 1.000, ensemble 0.938. Because 0.938 is at least the cut 0.4150 the mention is **kept**, and because it is also at least 0.5850 it is **strict**. The concept "framing" has 421 mentions, all kept: they count as 421 term mentions but one concept. The mention most similar to this one, by the vectors of its definitions, is a mention of "frame" (cosine similarity 0.967).

The 3D views load three.js 0.160.0 from cdn.jsdelivr.net and the fonts come from Google Fonts, so the page needs an internet connection for them.

## Inputs

The MBDef dataset, in the folder where `mbdef data fetch` puts it (`$DATA_ROOT/MBDef/`, or the folder `--dataset` names). The build reads two parts of it:

| Part | What the build takes from it |
|---|---|
| `explorer/` | the page data: the JSON files the page reads, with their sha256 in `SHA256SUMS` |
| `data/{instances,definitions,concepts,papers}.parquet` | the tables the page data is checked against |

The page data holds exactly the dataset's 62,066 term mentions, and besides what the tables hold it gives every mention its map position, topic and 8 nearest mentions, says how alike the mentions of a concept are, and holds the scorers' scores on the 1,700 labelled train and dev records of MBDef-Gold (the Relevance tab ranks every mention among them). These come from the embedding vectors of the definitions (Qwen3-Embedding-8B, not part of the dataset) and from Step 5's decisions, so `explorer data` makes them once ([below](#make-the-page-data-explorer-data)) and they are published with the dataset rather than computed by the build.
MBDef on Hugging Face holds the page data in the form this build writes it (without the keys that record where and when it was made): `dataset build --explorer-data` put it there after the same checks ([dataset.md](dataset.md#outputs)), so a build from it writes the same files again.

| Files | What they hold |
|---|---|
| `mentions/00.json` … `41.json` | every term mention: label, paper, label group, the three scores and the ensemble score (×10,000), strict, map position, topic, the 8 nearest mentions |
| `defs/00.json` … `41.json` | the definition texts of each mention |
| `fields/00.json` … `41.json` | the other extracted fields of each mention (description, synonyms, related terms, subtypes, supertypes, examples) |
| `terms/00.json` … `17.json` | each concept's merged view of those fields, from the dataset's `terms` table |
| `groups.json` | the label groups, one per concept: the dataset's name, its mentions, how alike they are, the 8 most similar concepts |
| `merges.json` | how the spellings of a concept were merged (every concept with two or more spellings) |
| `papers.json` | the papers: id, title, seed or cited paper, how many mentions and definitions |
| `relations.json` | the links between concepts that the papers' name fields make |
| `meta.json` | the counts (with the number of mentions the relevance filter left out), the two cuts, the scorers' weights, the topics, the shard sizes |
| `ensemble-ref.json` | the scorers' reference scores and every mention's exact ranks, for the Relevance tab |

Shard `k` of `mentions/`, `defs/` and `fields/` holds the mentions `1500·k` to `1500·k + 1499` (`shard` in `meta.json`); a shard of `terms/` holds 2,000 label groups.

## Outputs

`explorer build` writes `explorer/index.html` from `explorer/src/` and `explorer/data/` from the page data, with a new `SHA256SUMS`. Before it writes anything it checks:

- the page: the joined script passes `node --check` (when it does not, the build names the scripts that fail);
- the page data: every file is the one its `SHA256SUMS` lists, and no other file is there;
- the page data against the tables: its mentions are exactly the rows of `instances`, each with the same term, paper, strict flag and four scores (to 4 decimals), so page data that still holds a dropped mention is refused; their definitions are the rows of `definitions`, in the same order (an unreadable character, U+FFFD in the table, is a "?" on the page); the mentions of each label group are one concept of `concepts`; its papers are the rows of `papers`, with the same titles and counts of mentions;
- the site fits GitHub Pages: every file under 100 MB, at most 1 GB in all.

It copies the files unchanged, except two: `meta.json` and `ensemble-ref.json` lose the keys that record where and when they were made (commits, run times, local paths), and `meta.json`'s notes on the method are written in plain words. On the real data it prints:

```
The page data agrees with the MBDef tables: 62,066 mentions, 103,476 definitions, 34,916 concepts, 6,115 papers.
Wrote explorer/index.html (476 KB) and explorer/data/ (151 files, 55.7 MB; the largest, groups.json, 5.4 MB). The site is 56.1 MB.
```

The page loads about 19 MB when it opens (`meta.json`, the 42 mention shards, `groups.json`, `papers.json`, `merges.json`) and the rest only when a tab or a drawer needs it: the definitions, fields and merged views shard by shard, `relations.json` and `ensemble-ref.json` when their tab opens. It reads `meta.json` first, so the Overview shows its headline numbers at once and draws its map as the mention shards arrive; a thin line under the header shows the progress, and the other tabs open when everything is in.

## Make the page data: `explorer data`

`mbdef explorer data --out FOLDER` writes the page data for exactly what MBDef releases, from an MBDef dataset folder (`--dataset`, by default where `data fetch` puts it) and the Qwen3-Embedding-8B vectors of the definitions (`--vectors`, by default `$DATA_ROOT/embeddings/qwen3-emb-8b`, [embed.md](../step-5-post-processing/embed.md)).
It keeps the term mentions of the `instances` table and leaves out any other; of the mentions the relevance filter dropped, only their number is kept, in `meta.json`.
Then it computes again, on these mentions alone, everything that depends on which mentions are there:

| What | How |
|---|---|
| a mention's vector | the L2-normalised mean of its definitions' vectors, each embedded as "term: definition" |
| the map | PCA to 50 dimensions, then UMAP (cosine distance, 30 neighbours, minimum distance 0.15) to 3D and to 2D, scaled so that the 99.5th percentile of a coordinate's distance from the median is 1,000 |
| the 36 topics | k-means (k = 36, 4 starts, seed 20261031) on the 50 PCA dimensions, numbered by size; a topic is named after its three most frequent concepts (a capital that only starts a concept's label is lowered) |
| the nearest mentions | each mention's 8 most similar others, by exact cosine on the full vectors |
| a concept's spread | the mean cosine of each pair of its mentions, and of each mention to their normalised centroid |
| the baselines | the mean cosine of 200,000 random pairs (0.28) and of each mention with its nearest one (0.80) |

A concept's name and how its spellings joined come from the dataset's `concepts` table; scores, fields, merged views and links come from the page data, renumbered.
The settings are those of the first map, which placed all 67,394 corpus mentions; the seeds are fixed (PCA and UMAP 20261005, k-means 20261031, the random pairs 20261022), and k-means runs in float64 on one thread, because in float32 its threads add up in any order and give other topics from run to run.
So a second run, on the page data it wrote, writes the same files byte for byte (checked on the real data: 151 files, 55.7 MB, in about 2 minutes on a laptop with a GPU for the nearest mentions).
Before it writes anything it checks that every stored neighbour similarity of two released mentions comes out of the vectors again (within 0.0015), so that other vectors are refused, and it checks the output the way `explorer build` does.

**What changed with the released mentions alone.** The first map had a topic named "tokenization · latent dirichlet allocation · attention mechanism · lemmatization" (2,270 mentions), of which the relevance filter had dropped 61%, the only topic where it dropped more than 20%. On the released mentions no such topic forms: its 877 released mentions go mostly to "metaphor · conceptual metaphors · presupposition" (228), "stance detection · stance · stance classification" (125), "fairness · individual fairness · group fairness" (110) and "social media · social bots · memes" (84).

**Why seed 20261031.** The topics are one reasonable division of a continuous space rather than natural groups, and k-means gives another, equally tight one for every seed: 30 seeds (20261006 to 20261035) gave topics where the most similar other mention is in the same topic for 71.1% to 72.5% of the mentions, and that agree with each other at an adjusted Rand index of about 0.5 to 0.7. The seed was chosen by reading the topic names: in 5 of the 30, a topic is led by the concept "media bias", and of those, 20261031 has three clean bias topics, "media bias · political bias · bias" (718 mentions: media bias, political bias, informational bias, the hostile media phenomenon, coverage, selection and statement bias), "bias · gender bias · implicit bias" (1,349, 89% of them a concept with "bias" in its name) and "confirmation bias · selective exposure · motivated reasoning" (874), and the closest fit of all 30 (72.2%; the first topics, read on the released mentions, 71.4%). With seed 20261005, the first map's, "media bias" was only the third concept of "bias · gender bias · media bias". The other 30 seeds agree with the chosen topics at 0.51 to 0.74, and the first topics at 0.56. The choice is a reading aid for the map and changes nothing else: map positions, nearest mentions and every number outside the topics are the same for every seed.

The command needs the embedding cache (about 3 GB, in the data root, not in the dataset) and `umap-learn` (in the `analysis` extra and the dev group).
`tests/test_explorer_data.py` runs it on a miniature in which a dropped mention comes first, so that every index of the released ones moves, with a stand-in for the map; it checks every file, that a second run writes the same bytes, that the output passes `explorer build`'s checks, and that other vectors are refused.

## Build it and look at it

```bash
uv run mbdef data fetch MBDef/        # the dataset, into $DATA_ROOT/MBDef/ (docs/setup/data-fetch.md)
uv run mbdef explorer build           # writes explorer/index.html and explorer/data/; needs node
python -m http.server 8000 --directory explorer   # then open http://localhost:8000/
```

Opening `index.html` as a file does not work: the browser does not let a page read `data/` from disk.

## Publish it

The workflow [`pages.yml`](../../.github/workflows/pages.yml) runs only by hand (Actions → pages → Run workflow). It installs the package, fetches the dataset, runs `explorer build`, checks `explorer/data/` against its `SHA256SUMS`, and deploys `index.html` and `data/`. Before the first run:

1. Pages must be enabled: Settings → Pages → Build and deployment → Source: GitHub Actions.
2. On GitHub's free plan, Pages works only for a public repository.
3. While the dataset on Hugging Face is private, the repository secret `HF_TOKEN` must hold a token that can read it.
4. The site is public, and it shows every definition text of the dataset word for word (and nothing the dataset does not hold).

## How to adapt it

Edit the files in `explorer/src/` and run `explorer build` again: `shell.html` holds three markers, which the build replaces with `base.css` and each tab's `.css`, each tab's `.html` section, and `core.js` followed by each tab's `.js` in its own block (so a tab's names stay its own). A new tab needs its three files and its name in `TABS` of [`explorer.py`](../../src/mbdef/explorer.py) and in the tab bar of `shell.html`. The page's outside links (`{{project_url}}`, `{{mbdef_url}}`, `{{gold_url}}`, `{{explorer_url}}` in `shell.html`) are filled in by `links()` from the addresses the dataset cards use (`dataset/__init__.py`, `files.py`). The shared look and parts are in `base.css` and `core.js`: colour tokens for both themes and the dark band, big-number tiles (`stat`), panels built on first open (`expander`), section heads, the topic colours (`mapFrame`, `topicPalette`) and the glossary. To show the page on another dataset, give `--dataset` a folder that holds the same tables in `data/` and its page data in `explorer/`.
