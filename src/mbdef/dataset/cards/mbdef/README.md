
{{banner}}{{badges}}
# MBDef: media-bias definitions from the scientific literature

> **Version {{version}}.** {{links}}

Researchers write about media bias and the concepts around it (framing, gatekeeping, hostile media effect, fake news, stance and many more), and they define these terms differently. MBDef collects those definitions as the papers state them, so that the field's vocabulary can be compared and organised into a taxonomy.

It holds **{{concepts}} terms with {{definitions}} definitions from {{papers}} papers**. The [MBDef Explorer]({{explorer_url}}) shows it as a web page. The hand-labelled ground truth is a dataset of its own, [MBDef-Gold]({{gold_url}}).
{{explorer_files}}
## The tables

Start with `terms`: one row per term with everything the papers say about it. The other four tables hold the rows it is built from and share ids, so any merged item can be traced to the papers that state it.

| Table | One row is | Rows |
|---|---|---:|
| `terms` | a term with all its definitions, description, synonyms, related terms, subtypes, supertypes and examples, merged across papers | {{concepts}} |
| `concepts` | a term: every term-in-paper whose label is the same up to spelling | {{concepts}} |
| `instances` | a term as one paper defines it ("term-in-paper"), with its relevance scores | {{instances}} |
| `definitions` | one definition text, quoted from its paper | {{definitions}} |
| `papers` | a source paper: DOI, title, year, open-access status, licence | {{papers}} |

Every term-in-paper that the relevance filter keeps is included. {{strict}} of them also pass a stricter cut (the column `strict`; the paper says how the cuts were set).

```python
from datasets import load_dataset

terms = load_dataset("{{repo_id}}", "terms", split="corpus")  # start here
concepts = load_dataset("{{repo_id}}", "concepts", split="corpus")
instances = load_dataset("{{repo_id}}", "instances", split="corpus")
definitions = load_dataset("{{repo_id}}", "definitions", split="corpus")
papers = load_dataset("{{repo_id}}", "papers", split="corpus")
```

## Read more

Each page covers one thing; the [paper]({{paper_url}}) has the details.

| Page | What it answers |
|---|---|
| [What one row is]({{files_url}}/docs/tables.md) | What the five tables hold, with a worked example |
| [Columns]({{files_url}}/docs/columns.md) | What every column means |
| [How it was made]({{files_url}}/docs/how-it-was-made.md) | The five steps, from the literature search to the kept terms |
| [Which terms are kept]({{files_url}}/docs/relevance.md) | How a term-in-paper is scored and kept or dropped |
| [Term merging]({{files_url}}/docs/term-merging.md) | How spellings become one term, and how to trace a merge |
| [The terms view]({{files_url}}/docs/terms-view.md) | The merged items of `terms` and the other extracted fields |
| [Licence and rights]({{files_url}}/docs/licence.md) | What is licensed how |

## Limits

- **No person has checked the corpus terms yet.** The filter keeps {{kept_share}} of them.
- **Definitions are machine-extracted.** The filter judges the term, not each text.

The paper has the evaluation and the full limits.

## Licence

Our annotations are licensed {{licence_name}}. The definition texts are quotations: their copyright stays with the papers' authors and publishers. If you are a rights holder and want a quotation removed, open a discussion on this repository. Details: [Licence and rights]({{files_url}}/docs/licence.md).

## Citation

Citation details will follow.

## Version

This card describes version {{version}}. Each version is a commit of this repository, tagged with its number, so a version can be loaded by its revision. `provenance.json` lists every input file with its SHA-256, the code commit and the command that built this version.
