# Tutorials: exploring MBDef and MBDef-Gold

Six Jupyter notebooks show how to read MBDef and MBDef-Gold and reproduce analyses that support taxonomy development. The code they call is in [`src/mbdef/analysis/`](../src/mbdef/analysis/), so you can use it from your own scripts too.

| Notebook | What it shows | Models |
|---|---|---|
| [01_dataset_basics.ipynb](01_dataset_basics.ipynb) | The ten merged terms that the most papers define, and every definition of media bias with its paper | none |
| [02_definition_clustering.ipynb](02_definition_clustering.ipynb) | The definitions of media bias and its subtypes clustered by meaning (MiniLM, PCA, HDBSCAN), against a TF–IDF baseline and other settings, with the coverage of the ten largest clusters | MiniLM (CPU is enough) |
| [03_concept_connections.ipynb](03_concept_connections.ipynb) | The subtype and supertype links that the papers report between media bias, bias, misinformation, disinformation, fake news and propaganda | none |
| [04_definitional_divergence.ipynb](04_definitional_divergence.ipynb) | How similar MBDef-Gold's definitions of one term are across papers, by DefScore | BGE and ModernBERT (several GB; a GPU helps) |
| [05_research_disciplines.ipynb](05_research_disciplines.ipynb) | The fields the papers come from, by OpenAlex's primary topic, seed and closure papers apart, with the Social Sciences subfields | none (asks the OpenAlex API) |
| [06_taxonomy_coverage.ipynb](06_taxonomy_coverage.ipynb) | How many papers define each of the 11 bias types that the taxonomies of Spinde et al. and Rodrigo-Ginés et al. both cover, with a LaTeX-ready figure | none |

## Run them

From the repository root, with the [Install](../README.md#install) steps done and `DATA_ROOT` set in `.env` ([docs/setup/data-root.md](../docs/setup/data-root.md)):

```bash
uv sync --extra tutorials                                 # Jupyter, sentence-transformers, langdetect, wordfreq, matplotlib
uv run mbdef data fetch MBDef/ MBDef-Gold/    # the two datasets, each file checked by its sha256
uv run jupyter lab tutorials/
```

Run a notebook's cells from top to bottom; the notebooks do not depend on each other. What they write (tables, figures as PNG and SVG, and the full results behind them as JSON and CSV) goes to `tutorials/outputs/`, which git ignores. Models download into the Hugging Face cache on first use. Notebook 05 needs the network on its first run (an `OPENALEX_API_KEY` in `.env` is optional) and keeps OpenAlex's answers in `outputs/research-disciplines/openalex-cache.json`. The figures use Times Roman as the paper does, or a font with the same widths (Nimbus Roman, TeX Gyre Termes) when Times is not installed.

## The helpers

| Module | What it gives |
|---|---|
| `mbdef.analysis` | `load()`: MBDef's five tables as polars DataFrames, from `$DATA_ROOT/MBDef/data/`; `load_gold()`: MBDef-Gold's three splits in one DataFrame. Both take another folder as an argument. |
| `analysis.queries` | `find_concept` (a merged term by any spelling), `definitions_of` (its definitions with their papers), `hierarchy` (subtype-to-supertype edges, each with the papers' claims behind it) |
| `analysis.clustering` | `run` (select, screen for English, embed, cluster, compare) and `make_figure` |
| `analysis.connections` | `network_inputs` and `connections_figure` (the network at the fixed positions of [`network_layout.json`](../src/mbdef/analysis/network_layout.json)) |
| `analysis.divergence` | `definitions`, `build_pairs`, `run` (DefScore of every pair, resumable) and `figure` |
| `analysis.disciplines` | `run` (look the papers up in OpenAlex, resumable through its cache, then count and draw), `enrich`, `summarize` and `make_figure` |
| `analysis.taxonomy_coverage` | `run` (match, count, check, draw), `audit` and `make_figure`; the types and search terms are in [`assets/taxonomy-coverage/concepts.csv`](assets/taxonomy-coverage/concepts.csv) |

The rest is the pipeline's own code, not a copy: the tables are the released parquet files that `data fetch` checks, and `dataset build` has already checked them against each other ([docs/dataset/dataset.md](../docs/dataset/dataset.md)); merged terms are the release's Label Match groups, so no further spelling normalisation is applied; DefScore is `mbdef.evaluate.metric` with its pinned models and calibration ([docs/step-3-extraction/evaluate.md](../docs/step-3-extraction/evaluate.md)); OpenAlex is asked through the request code of `collect openalex` (`collect.openalex.get_json`).

## Methods

**Counts.** A merged term's paper count (`concepts.n_papers`) is the number of distinct papers that define it, not a word count in the full texts. Lookups return every definition with its source paper. All analyses of MBDef use every released row, not only the stricter subset (`instances.strict`).

**Merged terms.** A merged term joins the Extracted Concepts whose labels are the same up to case, punctuation, spacing, plural forms, British or American spelling and word order (Label Match, [GLOSSARY.md](../GLOSSARY.md)), and is shown under its most frequent spelling ("frames", "echo chambers"). Synonyms are not merged, so "frame" and "framing" stay apart.

**Clustering.** It follows the reported hierarchy two subtype steps down from media bias and screens for English without translating. MiniLM embeddings, PCA to 50 dimensions, L2 normalisation and HDBSCAN (minimum cluster size 20, minimum samples 5, leaf selection). The report keeps the exclusions, representative passages, unassigned passages, the TF–IDF comparison and the sensitivity checks. Results near cluster boundaries can differ across library versions and hardware; the report records the versions.

**Network.** The union of all shortest undirected paths between the six requested terms, then every directed relation the papers report among the kept terms. Arrows point to the supertype and are labelled by distinct supporting papers. The fixed layout reproduces the publication figure for MBDef 1.0.0; another release that brings in other terms needs their positions in `network_layout.json`.

**Divergence.** DefScore on the relevant records of all three paper-disjoint splits of MBDef-Gold, used descriptively; the calibration is applied as fitted, not refitted. The figure shows the cross-paper scores of the ten terms in the most papers (at least four); within-paper pairs and pairs of different terms are saved as reference distributions. MBDef-Gold's paper ids are its own and do not join with MBDef's.

**Disciplines.** One category per paper and level, from its OpenAlex primary topic, looked up by OpenAlex id and else by DOI; papers OpenAlex does not classify are counted as unclassified, never inferred from their titles. The report also counts all of a paper's topics. OpenAlex reclassifies works over time; the cache records when each answer was retrieved.

**Taxonomy coverage.** A definition counts for a type when a search term has the Label Match key of its term, or of a synonym its own paper gives for that term; Gatekeeping and Framing matches must name news, media, journalism, journalist(s) or the press. Distinct papers per type, seed and closure apart. The matches are automatic, not a semantic review; `manual_exclusions.csv` takes the rejected ones.

These analyses show how MBDef can support taxonomy development. Neither the clusters, the reported hierarchy links nor the similarity scores make a validated taxonomy. Read the dataset cards on Hugging Face and cite the datasets when you use them.
