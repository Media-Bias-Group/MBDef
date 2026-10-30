# What it holds

Part of [MBDef-Gold]({{repo_url}}). The records come from two sets of the same {{papers}} papers:

- **Relevant records:** {{relevant}} records, found in {{relevant_papers}} of the papers. The papers are SciDef's DefExtra ({{defextra_papers}} papers) and {{seed_papers}} papers of MBDef's seed corpus. Candidate definitions were proposed by a language model, checked by human annotators (majority vote on disagreements) and cleaned by a curator.
- **Non-relevant candidates:** {{non_relevant}} records from the same papers that the annotators judged not relevant.

Each record carries its votes. The label is the majority vote in every record.

The splits (train, validation, test) are fixed per paper; their sizes are in the table on the [README]({{repo_url}}).

It was used to train and choose the extractor of [MBDef]({{mbdef_url}}) and to fit and test MBDef's relevance filter. Its paper ids are a separate id scheme from MBDef's.
