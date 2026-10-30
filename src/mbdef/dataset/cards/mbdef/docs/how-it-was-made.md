# How it was made

Part of [MBDef]({{repo_url}}). The pipeline has five steps (see the published paper for more details).

1. **Literature search.** A keyword search in Scopus found {{search_documents}} documents. {{open_access_documents}} are openly accessible, and {{parsed_documents}} of them were parsed into structured text with GROBID.
2. **Paper relevance.** Large language models judged whether each paper discusses media bias. {{view_seed_papers}} papers were judged relevant; these are the **seed papers**.
3. **Definition extraction.** An extraction pipeline based on SciDef reads each paper and lists the terms it defines. For each term it copies, word for word, the definitions and six more fields (description, synonyms, related terms, subtypes, supertypes, examples), with the references cited for each item.
   - The model is Gemma 4 12B with reasoning. It reads each paper in two passes, in overlapping windows of 8,000, 16,000 and 32,000 characters; an item is kept only if its text is found word for word in the window.
   - A second model, Qwen3.6-35B-A3B, checks every definition against the text. A first relevance filter (a classifier on the term and its definitions) keeps or drops whole terms.
   - The other fields are not checked: they stay as the extraction model wrote them.
   - The pipeline was trained and chosen on the hand-labelled ground truth, [MBDef-Gold]({{gold_url}}). On that ground truth's test papers it finds {{extractor_recall}} of the annotated definitions, at a precision of {{extractor_precision}}.
4. **Corpus closure.** The seed papers cite works that define the same concepts but were missed by the search. Works cited in what was extracted (a definition, or a neighbouring field such as an example) were identified, downloaded as open-access PDFs, checked against the citation, and run through the same extraction. Only one level is followed: the cited works' own citations are not.
   - The corpus then holds {{view_papers}} papers: {{view_seed_papers}} seed and {{view_closure_papers}} closure.
   - It holds {{definition_items}} extracted definition items. {{repeated_items}} of them repeat an identical text for the same term in the same paper, which leaves {{all_definitions}} distinct definitions of {{all_terms}} term-in-paper rows in {{all_papers}} papers.
5. **Relevance and term merging.** Extraction also picks up terms unrelated to media bias, such as "deep learning". Step 5 decides for every term-in-paper whether it belongs to a media-bias taxonomy, then merges the kept terms by spelling.
   - Only the kept rows are in this dataset: {{instances}} of {{all_terms}} ({{kept_share}}). {{seed_instances}} come from seed papers and {{closure_instances}} from closure papers.
   - How a row is kept or dropped: [Which terms are kept]({{files_url}}/docs/relevance.md). How spellings merge: [Term merging]({{files_url}}/docs/term-merging.md).
