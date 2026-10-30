# Files and columns

Part of [MBDef-Gold]({{repo_url}}).

## Files

- `data/train.parquet`, `data/validation.parquet`, `data/test.parquet`: one row per record (columns below).
- `validated_gt.json` (the relevant records) and `validated_gt_non_relevant.json` (the non-relevant ones): the same records as the curation wrote them, by paper, with each paper's split and metadata.
- The repository's other files describe how the ground truth was made: the guide, the statistics, the manifest and the report.
- `provenance.json`: every input file with its SHA-256, the code commit and the command that built the split files.

## Columns

- `record_id` (`<paper_id>:<1 relevant | 0 not>:<index>`), `paper_id`, `split`, `term`.
- `relevant`: true or false.
- `votes_relevant`, `votes_not_relevant`: the annotators' votes.
- `definitions`, `description`, `synonyms`, `related_terms`, `subtypes`, `supertypes`, `examples`: each a list of `{text, citations}`.
- `paper_title`, `paper_doi`, `paper_year`.
