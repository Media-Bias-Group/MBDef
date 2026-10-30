# What one row is

Part of [MBDef]({{repo_url}}). The dataset has five tables; **start with `terms`**. The other four hold the rows it is built from and share ids. What every column means: [Columns]({{files_url}}/docs/columns.md).

| Table | One row is | Rows |
|---|---|---:|
| `terms` | a concept with all its definitions, description, synonyms, related terms, subtypes, supertypes and examples, each item merged across papers, with how many papers state it | {{concepts}} |
| `concepts` | a concept: every term-in-paper whose label is the same up to spelling | {{concepts}} |
| `instances` | a term as one paper defines it ("term-in-paper"), with the relevance scores and everything else extracted for it (description, synonyms, related terms, subtypes, supertypes, examples) | {{instances}} |
| `definitions` | one definition text of a term-in-paper, quoted from the paper | {{definitions}} |
| `papers` | a source paper: DOI, title, year, open-access status, licence | {{papers}} |

## Worked example

The concept “{{ex_label}}” (`concept_id` {{ex_concept_id}}) has {{ex_n_spellings}} spellings, among them {{ex_spellings}}. It appears as a defined term in {{ex_papers}} papers ({{ex_instances}} term-in-paper rows) with {{ex_definitions}} definitions.

One of its rows in `instances` is `{{ex_mention}}`: the term “{{ex_term}}” in *{{ex_title}}* ({{ex_year}}, DOI {{ex_doi}}, licence {{ex_licence}}). Its relevance score is {{ex_score}}, combined from three scores: embedding classifier {{ex_clf}}, LLM decision model {{ex_basal}} and retrieval judge {{ex_judge}} ([how they are combined]({{files_url}}/docs/relevance.md)).

One of its rows in `definitions` (`{{ex_def_id}}`) reads:

> {{ex_def}}

{{ex_citation_sentence}}

{{ex_fields_sentence}}
