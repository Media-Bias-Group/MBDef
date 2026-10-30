# Term merging

Part of [MBDef]({{repo_url}}). Kept terms are merged when their labels are the same up to spelling (the Label Match rule). Merged terms are the rows of `concepts` and `terms`.

## The rule

The rule:
- ignores case, hyphens and spacing ("click-bait" = "clickbait"), quotation marks, a leading article, and stop words between content words;
- treats an acronym given in brackets as the same label ("fake news (FN)" = "fake news");
- treats word order as irrelevant ("bias in the media" = "media bias");
- maps British to American spelling ("judgement" = "judgment");
- reduces plurals to the singular. When a word has two possible singulars that both occur ("refugees": "refugee" or "refuge"), it takes the one that more labels end with. It also covers "-ses" plurals ("crises"), "-ics" plurals ("infodemics"), and irregular plurals ("phenomena", "criteria"). Plurals that name something else stay apart ("news media" and "new media", "ethics" and "ethic").

It does not merge synonyms with different words ("news bias" and "media bias" stay separate). It does not split one spelling that is used with different meanings.

## What it did here

The spelling groups are formed once over all {{all_terms}} term-in-paper rows of the corpus, then restricted to the kept rows. When a label is kept in some papers and dropped in others, its concept contains only the papers where it was kept. Before Step 5, the {{all_terms}} rows formed {{all_groups}} groups:
- {{groups_all_kept}} were kept whole;
- {{groups_all_dropped}} were dropped whole;
- {{groups_split}} were split, kept in some papers and dropped in others.

The {{instances}} kept rows use {{spellings}} distinct spellings and form {{concepts}} concepts: {{single_concepts}} have a single term-in-paper row and {{multi_concepts}} have more than one. The largest is “{{top_label}}”, with {{top_instances}} term-in-paper rows.

## Tracing a merge

Every row records how its term was normalised, and every concept records why its spellings were joined:

- **The row.** In `instances`, row `{{tr_mention}}` has the term “{{tr_term}}”. Its `normalisation` reads {{tr_steps}}, giving `word_set_key` “{{tr_ws}}” and `spacing_key` “{{tr_sp}}”.
- **The concept.** That puts it in the concept “{{tr_concept}}” (`{{tr_cid}}`, {{tr_n}}). The concept's spellings are {{tr_spellings}}.
- **Why they were joined.** Its `joined_by` says: {{tr_joins}}.
- **Flags.** Its `merge_flags`: {{tr_flags}}.

The columns are explained in [Columns]({{files_url}}/docs/columns.md). Merging is by spelling only; the [paper]({{paper_url}}) lists the merges that may join different meanings.
