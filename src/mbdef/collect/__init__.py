"""Paper step 1, collect: find, download and convert the PDFs of the papers to collect (docs/step-1-2-papers/collect.md).

The actions run over the files the earlier ones wrote in the step's folder (`$DATA_ROOT/collect` unless `--folder` names another; docs/step-1-2-papers/collect.md gives the order):

- `crossref.py` and `semantic_scholar.py`: find the DOI of a queued paper that has none (and, by DOI, more open copies);
- `openalex.py`: look each queued paper up with OpenAlex (by DOI, or by title, year and author);
- `open_access.py`: look up what OpenAlex left without a PDF with Unpaywall;
- `download.py`: download the PDFs found into zip batches;
- `grobid.py`: convert each PDF to TEI with a running GROBID service, and keep it only if it is the paper that was asked for.

`identity.py` holds the identity rules (DOIs, compared text, paper ids, TEI headers, conversion receipts), which the closure step uses too, and `ledger.py` the CSV ledgers and run receipts.
"""
