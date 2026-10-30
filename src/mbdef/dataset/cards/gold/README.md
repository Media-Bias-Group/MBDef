
{{banner}}{{badges}}
# MBDef-Gold: hand-labelled media-bias definitions

> {{links}}

MBDef-Gold is the hand-labelled ground truth behind [MBDef]({{mbdef_url}}), the dataset of media-bias definitions extracted from the scientific literature. It holds **{{relevant_papers}} papers with {{relevant}} records and {{relevant_definitions}} definitions**, and beside them {{non_relevant}} candidate records that the annotators judged not relevant ({{records}} records of {{papers}} papers in all).

A record is a term as one paper defines it, with its definitions and six more fields (description, synonyms, related terms, subtypes, supertypes, examples), each item with the references the paper cites for it. Every record is labelled relevant or not relevant to a media-bias taxonomy. It was used to train and choose MBDef's extractor and to fit and test MBDef's relevance filter.

## Splits

Splits are fixed per paper.

| Split (`split` / Hugging Face name) | Records | Relevant | Papers |
|---|---:|---:|---:|
{{split_table}}

```python
from datasets import load_dataset

gold = load_dataset("{{repo_id}}")  # train / validation / test
```

## Samples

Real records from the train split, drawn at random with a fixed seed: {{sample_relevant}} relevant and {{sample_not_relevant}} not relevant, each from a different paper. A record is a term as one paper defines it; the label says whether it belongs in a media-bias taxonomy.

{{samples_image}}<details>
<summary>All {{sample_count}} sample records as a table</summary>

*Votes* is how many annotators judged the record relevant / not relevant; the label is the majority. Not relevant means the term does not belong in a media-bias taxonomy, not that the definition is wrong. A definition is cut after {{sample_chars}} characters. The picture shows {{sample_poster}} of each label: the first ones, with a split vote among them when there is one.

| Term | Label | Votes | Definition | Paper |
|---|---|---|---|---|
{{sample_table}}

</details>

## Read more

| Page | What it answers |
|---|---|
| [What it holds]({{files_url}}/docs/what-it-holds.md) | Where the records come from and how they were labelled |
| [Files and columns]({{files_url}}/docs/files-and-columns.md) | What each file and column is |
| [Limits]({{files_url}}/docs/limits.md) | What to keep in mind before using it |
| [Licence and rights]({{files_url}}/docs/licence.md) | What is licensed how |

## Limits in short

- **The test split is not a clean benchmark any more.** It was read {{test_reads}} times while MBDef was built.
- **The labels are about relevance to a media-bias taxonomy**, not about whether a text is a good definition.

Please do not train language models on this ground truth. Canary string: `{{canary}}`

## Licence

Our annotations are licensed {{licence_name}}. The definition texts are quotations: their copyright stays with the papers' authors and publishers. If you are a rights holder and want a quotation removed, open a discussion on this repository. Details: [Licence and rights]({{files_url}}/docs/licence.md).

## Citation

Citation details will follow.
