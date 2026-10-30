# The terms view

Part of [MBDef]({{repo_url}}). `terms` gathers, for each concept, what all its papers say about it, field by field, and merges items that say the same thing. It builds on the [other extracted fields](#other-extracted-fields) of each row.

## How items merge

- **Names** (`synonyms`, `related_terms`, `subtypes`, `supertypes`) merge by the rule that merges the terms themselves ([Term merging]({{files_url}}/docs/term-merging.md), with the vocabulary of all corpus terms): {{tv_merge_example}}. A name whose spelling is one of the dataset's concepts links to it (`links_to`), so the supertypes and subtypes form links between concepts.
- **Sentences** (`definitions`, `description`, `examples`) merge only when they read the same after case, punctuation and spacing are set aside, as when several papers quote one source.
- **`papers`** counts the distinct papers that state an item: {{tv_agree}}. `variants` records every merge, and `sources` gives each paper's own wording and citations.
- **Fields are kept as each paper puts them.** {{tv_fields_note}}

{{tv_example}}

| Field | Items | After merging | Merged from several forms | Stated by 2+ papers | Name a concept |
|---|---:|---:|---:|---:|---:|
{{terms_table}}

{{terms_self}} name items name their own concept (for example a term listed as its own synonym); they are kept as extracted.

## Other extracted fields

Besides its definitions, the extraction lists what a paper says about a term in six more fields, the same fields the ground truth annotates. They come from the same extraction runs as the definitions and stay with their term: Step 5 judged the term, not each item.

- Every item is exactly what the extraction model wrote into that field for that paper.
- Only the definitions were checked by the second model.
- Nothing moves an item to another field or votes across papers: a name a paper lists under several fields (for example as both a synonym and a related term) stays under each, and when papers disagree, every paper's choice is kept.

| Field | Items | Rows with any |
|---|---:|---:|
{{extra_table}}

In all, {{extra_items}} items on {{extra_rows}} of the {{instances}} rows. The extraction's quality figures ([How it was made]({{files_url}}/docs/how-it-was-made.md), step 3) are measured on definitions; the other fields have no such measurement.
