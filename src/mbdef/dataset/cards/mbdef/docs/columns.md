# Columns

Part of [MBDef]({{repo_url}}). The columns of the five tables, one list per table. What a row is: [What one row is]({{files_url}}/docs/tables.md).

## `terms`

- `concept_id`, `term`: the concept and its name (its most frequent spelling).
- `spellings`: every spelling, with how many papers write it.
- `papers`: how many papers define the concept. `strict`: true when at least one of its rows passes the stricter cut.
- `definitions`, `description`, `synonyms`, `related_terms`, `subtypes`, `supertypes`, `examples`: the merged items, most often stated first ([how items merge]({{files_url}}/docs/terms-view.md)). Each item has:
  - `text`: its most frequent form;
  - `papers`: how many papers state it;
  - `links_to`: for names, the concept the item names, if it is one of the dataset's concepts;
  - `variants`: the forms that were merged, with their papers (empty when there was one form);
  - `sources`: every paper and row that states it (`paper_id`, `mention_id`, `def_id` for definitions, `as_written` when the paper's wording differs from `text`, and `citations`).

## `concepts`

- `concept_id`: a stable id ("g" plus 12 hex digits).
- `label`: the most frequent spelling. `labels`: all spellings, most frequent first.
- `match_key`: the normalised word set that the spellings share.
- `n_instances`, `n_papers`, `n_definitions`: the counts. `n_strict`: how many of the concept's rows pass the stricter cut.
- `merge_flags`: for concepts that join different spellings, the kinds of join someone may want to check: same words in a different order, spacing, a dropped bracket or acronym, a plural with two possible singulars, an irregular or "-ics" plural, British spelling. {{flagged_concepts}} concepts have at least one flag.
- `joined_by`: every key that two or more of the concept's spellings share, with those spellings ([tracing a merge]({{files_url}}/docs/term-merging.md)).
- `mention_ids`: the concept's rows in `instances`.

## `instances`

- `mention_id`: the row id. `concept_id`: the concept the row belongs to.
- `term`: the term as the paper writes it (its original label).
- `word_set_key`, `spacing_key`: the two normalised keys the merging compares.
- `normalisation`: each step that changed the term before matching, in plain words.
- `paper_id`: the source paper.
- `corpus`: `seeds` (found by the literature search) or `closure` (cited by a seed paper).
- `n_definitions`: how many definitions the row has.
- `ensemble_score`: the relevance score that decided the row, between 0 and 1. `strict`: true when the row also passes the stricter cut ({{strict_cut}}).
- `classifier_score`, `basal_score`, `judge_score`: the three member scores.
- `classifier_ref_rank`, `basal_ref_rank`, `judge_ref_rank`: each member score placed among that member's scores on the labelled training records, 0 to 1 ([how scores are combined]({{files_url}}/docs/relevance.md)).
- `description`, `synonyms`, `related_terms`, `subtypes`, `supertypes`, `examples`: everything else the extraction lists for the term in this paper, each a list of `{text, citations}` (empty when the paper gives none; [what they are]({{files_url}}/docs/terms-view.md)).

## `definitions`

- `def_id`: the row id.
- `mention_id`, `concept_id`, `paper_id`, `term`: the row's term-in-paper, concept, paper and term.
- `text`: the definition as quoted.
- `citations`: the references the paper gives for it, as written. {{with_citations}} definitions have at least one.

## `papers`

- `paper_id`: the paper's id in this dataset. `corpus`: `seeds` or `closure`.
- `doi`: known for {{doi_papers}} papers. `title`.
- `year`: known for {{year_papers}} papers ({{year_openalex}} from OpenAlex).
- `openalex_id`. `oa_status`: OpenAlex's open-access status.
- `open_licence`: the most open licence that OpenAlex records for any version of the paper. `licence_url`: where that version is.
- `n_instances`, `n_definitions`: the counts.
