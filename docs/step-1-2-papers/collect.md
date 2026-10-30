# Collect: `mbdef collect`

| | |
|---|---|
| **Paper step** | Step 1 (collecting the full texts of the papers), as far as this repository covers it. The seed papers' Step 1 ran outside this repository; this code collected the 4,556 cited papers that the corpus closure (Step 4) added |
| **Code** | [`src/mbdef/collect/`](../../src/mbdef/collect/): [`crossref.py`](../../src/mbdef/collect/crossref.py), [`semantic_scholar.py`](../../src/mbdef/collect/semantic_scholar.py), [`open_access.py`](../../src/mbdef/collect/open_access.py), [`openalex.py`](../../src/mbdef/collect/openalex.py), [`download.py`](../../src/mbdef/collect/download.py), [`grobid.py`](../../src/mbdef/collect/grobid.py), with the identity rules in [`identity.py`](../../src/mbdef/collect/identity.py) and the ledgers in [`ledger.py`](../../src/mbdef/collect/ledger.py) |
| **Commands** | `mbdef collect crossref`, `… semantic-scholar`, `… openalex`, `… open-access`, `… download`, `… semantic-scholar-doi`, `… grobid` |
| **Checked by** | the end-to-end test [`tests/test_collect.py`](../../tests/test_collect.py), one test file per command, and a replay of the released closure's lookup ledgers ([below](#how-it-was-checked)). It is not rerun against live services: their answers change |

## Run it on your papers

**PDFs you already have:** start a GROBID ([Run it](#run-it)) and give the folder to the extractor, which converts each `<paper id>.pdf` itself: `mbdef extract run --pdfs my-pdfs/ --out out/run` ([docs/step-3-extraction/extract.md](../step-3-extraction/extract.md#run-it-on-your-papers)).
**A list of papers to find and download:** write it as a collection queue (below) and run the commands of [Run it](#run-it); they look up DOIs and open-access copies, download the PDFs, and keep a TEI only if its title page shows the paper that was asked for. `extract run --teis "$DATA_ROOT/collect/grobid"` reads the kept TEIs as they are.

## What the step does

The input is a **collection queue**: a CSV with one row per paper to collect (title, authors, year, DOI, the web address the reference gave, and a status). Only rows with status `missing` are collected; the others (for example `review`, a reference that could be two different works) are skipped and counted.
The closure step's corpus check (`closure run`, [docs/step-4-closure/closure.md](../step-4-closure/closure.md)) writes this queue for the works the seed papers cite.

The commands run in this order, all in the step's folder. The first two find DOIs for queued papers that have none, the next two find PDF addresses, `download` fetches the PDFs of each list of addresses, `semantic-scholar-doi` finds more addresses for the papers still without a PDF, and `grobid` converts:

1. **`crossref`** searches [Crossref](https://www.crossref.org/documentation/retrieve-metadata/rest-api/) once for each row without a DOI, by its title and up to three author surnames (20 records, ranked by relevance). A record counts only if its title (or "title: subtitle") is the same after case and punctuation are ignored, one of its years (issued, published, print, online or posted) is one the row names, and one author surname is shared. The row takes a DOI only if exactly one DOI passes: two or more are `ambiguous`, none is `no_match`, and a row whose title is under 12 characters, or that names no year or no author, is `insufficient_evidence` (a short title is no evidence of which work it is).
2. **`semantic-scholar`** asks [Semantic Scholar](https://api.semanticscholar.org/api-docs/graph)'s title match about each row Crossref settled as `no_match` (or failed on three times), and judges its one best match by the same rule. Semantic Scholar sometimes names a DOI that is not the work's own (a book review's, say), so that DOI is kept only if Crossref's record of it passes the rule too, or if it is the arXiv DOI of the matched arXiv id; otherwise it is dropped and the match's PDF and arXiv copy stay.
3. **`openalex`** asks [OpenAlex](https://docs.openalex.org/) about each row: by its DOI (the queue's, or one Crossref or Semantic Scholar found), else by a title search. The work OpenAlex returns for a DOI must carry that DOI. A DOI OpenAlex does not know leaves the row `not_found` (`doi_not_in_openalex`): the DOI is kept for Unpaywall, and the title is not searched. The title search (`--title-rule 2026-09-29`, the default) filters OpenAlex's records on the title (`title.search:`, commas dropped) and on the years the row names and the years either side, 200 records a page. A record whose first author is not one of the cited authors is dropped (a book review lists its reviewer first, the book's authors after). A record counts if its title is the same after case and punctuation are ignored, an author surname is shared, and its year is one the row names (`title_filter`); only if no record has such a year does a record a year off count instead (`title_filter_year_off_by_one`: a preprint's year, say). One such record is the match; two, or more records than the page shows, make the row `ambiguous`, and no PDF is chosen. A row that names no year is searched but cannot match (`not_found`).
   A row without a DOI whose Semantic Scholar title match has a PDF address is not searched while that PDF may still come: only once its download failed for good (a 404 or 403, or a page that is not a PDF; not a 429, a server error or a timeout, which a later `download` run retries). The run counts these rows as `waiting_for_a_download`.
   `--title-rule 2026-09-28` selects the rule the released closure's first title searches used (`matched_by` `title`): a full-text search filtered to the row's years, with the same rule for a record but no first-author check and no year off by one. Under the default rule, a row the earlier rule left `ambiguous` or `not_found` is searched again, and a row it matched keeps its match, as in the released closure. OpenAlex lists every copy of a work; repository copies come first, and the others are kept as alternates.
4. **`open-access`** asks [Unpaywall](https://unpaywall.org/products/api) about what OpenAlex left without a PDF, and keeps Unpaywall's best open-access PDF address. It asks about the DOI of each row OpenAlex answered `not_open_access`, `open_access_but_no_pdf` or `not_found`: the queue's, a found one, or the DOI of OpenAlex's title match. A row OpenAlex has not answered yet is asked about its own or a found DOI. A row OpenAlex found a PDF for, or left `ambiguous`, is not asked. The web address the reference gave is not used, as in the released closure; `--reference-urls` makes it a fallback when Unpaywall has no PDF, and the only source of a row without a DOI.
5. **`download`** tries each candidate's addresses in order, with a browser's TLS fingerprint (many publishers refuse plain scripts), until one answers with a PDF: the bytes `%PDF-` within the first kilobyte, and at most 80 MiB. PDFs are stored uncompressed in zip batches of 5,000. It reads one list of candidates per run (OpenAlex's, Unpaywall's, Semantic Scholar's); a paper already downloaded is skipped.
   When no address gives a PDF, it tries up to four more (the recovery): [Europe PMC](https://europepmc.org/)'s rendering (`https://europepmc.org/articles/PMC…?pdf=render`) of each PubMed Central paper the addresses name, and, when an address answered with an HTML page, the PDF that the page of the first or second address declares for indexers (`<meta name="citation_pdf_url">`, or a `<link>` of type PDF). If these fail too, the error names them after ` | recovery: `. Europe PMC (`europepmc.org`, `www.ebi.ac.uk`) is asked at most once every 3 seconds, and not for 60 seconds after a 429 or 503 (or as long as its `Retry-After` says, at most 600 seconds).
6. **`semantic-scholar-doi`** asks Semantic Scholar about the DOI of every paper OpenAlex settled that still has no downloaded PDF, 500 DOIs per request, for the open copies it knows: its open-access PDF, the arXiv copy, and for a paper in PubMed Central, [Europe PMC](https://europepmc.org/)'s rendering of it as a PDF (`https://europepmc.org/articles/PMC…?pdf=render`). Europe PMC is not searched: the PubMed Central id comes from Semantic Scholar. Hosts that copy papers without the rights holder's permission (Sci-Hub, LibGen and the like) are never used. `download` then reads these candidates too.
7. **`grobid`** sends each downloaded PDF to a running [GROBID](https://grobid.readthedocs.io/) service and keeps the TEI it returns only if it has body text and its title page shows the paper that was asked for (the identity rule, below). Each kept TEI gets a receipt beside it that binds the queue row, the PDF's and the TEI's sha256 and the rule that matched. The closure step reads these receipts: its tables count the closure papers by the rule that matched.

**A DOI found for a row without one** is looked up by `openalex` and `open-access` as if the queue had it: Crossref's, else the one Semantic Scholar's match kept. OpenAlex's row is recorded under the original queue row (its `query_doi` stays empty) with `matched_by` naming who found the DOI (`crossref_doi`, `s2_doi`); if OpenAlex lacks that DOI the row is `not_found` (`crossref_doi_not_in_openalex`) and keeps the DOI for Unpaywall, without a title search, because a search already found the DOI.

Each command appends one row per attempt to its **status ledger** and rewrites a small snapshot of the current candidates or failures. The latest row of a paper is its state. A run that stops (a crash, a spent rate limit, a timeout) is resumed by running the same command again: settled papers are not asked again, and errors are retried. Every run also appends a receipt to `receipts.jsonl`.

**The identity rule** (`identity_match` in `grobid.py`), in order:

| Rule | When it holds |
|---|---|
| `doi` | the title page shows the requested DOI |
| *refused* | the title page shows a different DOI that is real evidence (for example a chapter `…-11` of a requested book) |
| `title_year_author` | no usable DOI on the title page; same title, a shared author surname, and a year the row names |
| `title_author_year_missing` | as above, but the title page shows no year |
| `title_author_year_off_by_one` | as above, but the year is one off (a preprint year, say) |
| `title_in_text` | GROBID read no title; the requested title and an author surname open the text |
| `llm_verified` | none of these, but the closure step's PDF identity check accepted these exact PDF bytes (`--identity-verdicts`) |

A DOI on the title page that says nothing about which work the PDF is does not count either way: a template DOI such as `10.1145/nnnnnnn.nnnnnnn`, the requested DOI with text glued on, the same DOI with other punctuation, a dataset DOI (Zenodo, figshare, …), a preprint DOI, or one cut short or garbled. The title, author and year decide then.

**A worked example: one cited paper through the step (the released closure).** A seed paper cites *Trolling for engagement: Australian legacy news outlets seeking audience interaction metrics on Facebook through deliberately divisive content* (Hurcombe, E., 2019, DOI `10.5210/spir.v2019i0.10985`).

- Its paper id is `doi_10.5210_spir.v2019i0.10985__454fb465…__06f3208d186c8269`: `doi_` and the DOI with each run of characters that is unsafe in a file name replaced by one `_` (at most 120 characters); then the sha1 of the DOI (`454fb465b2c16b30779920683ef5f038f8a84a6d`), so two DOIs that look alike as file names stay apart; then 16 hex digits of the sha256 of the row's fields (base id, queue status, title, authors, year, DOI and web address), so the id stays the same when other rows leave the queue.
- `download` stored its 82,991-byte PDF as `do/doi_10.5210_spir.v2019i0.10985__….pdf` in a zip batch.
- GROBID read the title in capital letters, no DOI and no year from the title page. The title is the same once case is ignored and Hurcombe is a shared author, so the TEI was kept under the rule `title_author_year_missing`, with the PDF's sha256 `b366ebdb…` and the TEI's sha256 `44db1f4d…` in its receipt.

**Worked examples of the lookups (the released closure).**

- *Crossref.* A seed paper cites *Deciding what's true: The rise of political fact-checking in American journalism* (Graves, L., 2016) without a DOI. The search (`Deciding what's true: … journalism Graves`) returned 20 records: 3 with the same title, 1 of them with a year the citation names, and that one with the author Graves. One DOI passed, so the row is `matched` with `10.7312/grav17506` (an `edited-book`, years 2016 and 2017; its first title is "Deciding What's True", the rest of the cited title is its subtitle or another title of the record). OpenAlex then looked that DOI up under the original row (`crossref_doi`): not open access.
- *Semantic Scholar.* *Detecting and tracking political abuse in social media* (Conover, M. D., and five others, 2011): Crossref's one record with that title has no year the citation names (`no_match`, "title but no matching year"). Semantic Scholar's best title match is the same title, 2011, with an author surname of the citation, so it is `matched`. It names the DOI `10.1609/icwsm.v5i1.14127`, but Crossref's record of that DOI gives no year the citation names, so the DOI is dropped (`dropped 10.1609/icwsm.v5i1.14127: Crossref record fails the rule (title but no matching year)`) and only the match's PDF address is kept as a candidate.
- *Semantic Scholar by DOI, Europe PMC.* OpenAlex and Unpaywall knew `10.1002/acp.4002` as open access but had no PDF address. Semantic Scholar names a copy at PubMed Central (`PMC9537901`); the candidate lists that address, then Europe PMC's rendering `https://europepmc.org/articles/PMC9537901?pdf=render`, and the PDF was downloaded from the second.
- *OpenAlex's title search, a year off.* *Unsupervised domain adaptation by backpropagation* (Ganin, Y., and Lempitsky, V., 2015): neither Crossref nor Semantic Scholar gave it a DOI. The search (`title.search:Unsupervised domain adaptation by backpropagation,publication_year:2014|2015|2016`) found no record with the title, a cited author and the year 2015, and one with the year 2014: the arXiv preprint (OpenAlex `W1882958252`, DOI `10.48550/arxiv.1409.7495`). So the row is `has_pdf` under `title_filter_year_off_by_one`, with the PDF at `https://arxiv.org/pdf/1409.7495`.
- *Unpaywall after OpenAlex.* *History, collective memories, or national memories? …* (Carretero, M., and two others, 2017): OpenAlex's title search found one record a year off (2018), the chapter `10.1093/oso/9780190230814.003.0013`, not open access. Unpaywall, asked about that DOI, named a PDF in a university repository (`repositorio.uam.es`): the one Unpaywall PDF of the released closure for a DOI that OpenAlex's title search found.
- *The download recovery and Europe PMC's pace.* *Sharing of misinformation is habitual, not just lazy or biased* (Anderson, I., Ceylan, G., and Wood, W., 2023): PubMed Central's PDF address (`https://pmc.ncbi.nlm.nih.gov/articles/PMC9942822/pdf/pnas.202216614.pdf`) answered with an HTML page, so the recovery tried Europe PMC's rendering of `PMC9942822`, which answered 429. That day Europe PMC answered 769 requests with 429 in a two-hour burst, which is why it is paced. Three minutes later the rendering gave the 749,908-byte PDF.

What the lookups gave in the released closure (the ledgers in the data root, [docs/step-4-closure/closure.md](../step-4-closure/closure.md#inputs)):

| Lookup | Rows | Outcome |
|---|---:|---|
| Crossref, rows without a DOI | 13,937 | 5,341 `matched`, 8,074 `no_match`, 198 `ambiguous`, 324 `insufficient_evidence` |
| Semantic Scholar title match, Crossref's `no_match` rows | 8,074 | 1,951 `matched` (95 with a DOI kept: 91 arXiv DOIs, 4 that Crossref confirmed; 616 DOIs dropped; 497 with a PDF or arXiv address), 6,123 `no_match` |
| OpenAlex by the queue's DOI | 9,119 | 3,174 with a PDF, 1,035 open access without a PDF, 4,602 not open access, 308 not in OpenAlex |
| OpenAlex by a found DOI | 5,407 | 5,312 by Crossref's DOI (1,541 with a PDF), 95 by Semantic Scholar's (93 with a PDF) |
| OpenAlex title search, rows without any DOI | 8,178 | rule `2026-09-29`: 7,817 (1,494 matched, 381 of them a year off, 281 with a PDF; 376 `ambiguous`, 5,947 `not_found`); the earlier rule `2026-09-28`: 361 (356 matched, 56 with a PDF; 4 `ambiguous`, 1 `not_found`) |
| Unpaywall, what OpenAlex left without a PDF | 10,522 | 326 with a PDF (311 of them for papers OpenAlex found a PDF of later), 1,499 open access without a PDF, 8,243 not open access, 454 not known |
| Semantic Scholar by DOI | 11,416 | 10,335 known, 3,476 with an open copy (354 in PubMed Central, 306 on arXiv) |

Downloads that came from these copies: 420 from a title match's address, 804 from a copy that Semantic Scholar named for a DOI (279 of all downloads were Europe PMC's rendering).

How the 4,556 closure papers of the released closure were matched (`pdf_identity_check` in the release's `closure/papers.csv`):

| Rule | Papers |
|---|---:|
| `doi` | 2,199 |
| `title_author_year_missing` | 1,096 |
| `title_year_author` | 544 |
| `llm_verified` | 426 |
| `title_author_year_off_by_one` | 202 |
| `existing_corpus` (taken from the project's earlier corpus by the closure step, not converted here) | 64 |
| `title_in_text` | 25 |

## Run it

GROBID runs as a separate service; the command never starts one. With Docker:

```bash
docker run --rm --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.9.1-crf          # CPU only
docker run --rm --gpus all --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.9.1-full  # with a GPU
```

Put the queue at `$DATA_ROOT/collect/citation-papers-to-collect.csv` (or pass `--collection`), set `UNPAYWALL_EMAIL` in `.env` (and, optionally, `CROSSREF_MAILTO`, `OPENALEX_API_KEY` and `SEMANTIC_SCHOLAR_API_KEY`), then:

```bash
uv run mbdef collect crossref                     # DOIs for rows without one
uv run mbdef collect semantic-scholar             # for the rows Crossref left without one
uv run mbdef collect download --candidates "$DATA_ROOT/collect/semantic_scholar_candidates.csv"
uv run mbdef collect openalex                     # by DOI, else by title
uv run mbdef collect open-access                  # what OpenAlex left without a PDF
uv run mbdef collect download --candidates "$DATA_ROOT/collect/openalex_candidates.csv"
uv run mbdef collect download                     # the Unpaywall candidates
uv run mbdef collect semantic-scholar-doi         # open copies of what is still missing
uv run mbdef collect download --candidates "$DATA_ROOT/collect/semantic_scholar_doi_candidates.csv"
uv run mbdef collect grobid
```

`crossref` and `semantic-scholar` run first: `openalex` and `open-access` read the DOIs they found.
The title matches' PDFs are downloaded before `openalex`, which does not search the title of a row whose title-match PDF may still come (rerun `openalex` after a later `download` to search the rows whose PDF failed for good). `openalex` runs before `open-access`, which asks Unpaywall about what OpenAlex left without a PDF.

Each command prints its counts and the files it wrote, for example `collect open-access`:

```text
{"eligible": 1, "errors": 0, "resolved": 1, "skipped_review": 1, "unresolved": 0}
status: …/collect/open_access_status.csv
candidates: …/collect/open_access_candidates.csv
```

Here the queue had three rows (the end-to-end test, below): one `review` row was skipped, OpenAlex had found a PDF of the paper without a DOI, so Unpaywall was asked only about the paper OpenAlex knows without a PDF, and it had one. `collect openalex` exits 1 when OpenAlex's daily budget is spent (progress is kept; rerun later), and `collect grobid` exits 1 when a PDF was not converted (`failed_grobid.csv` says why; rerun to retry).

## Inputs

| Input | Where | What it is |
|---|---|---|
| The collection queue | `$DATA_ROOT/collect/citation-papers-to-collect.csv`, or `--collection` | One row per paper to collect; columns `queue_status`, `title`, `authors` (`Surname, I.; …`), `year`, `doi` (several joined by `; `; a published DOI is preferred over an arXiv one), `url` and seven columns the closure step uses |
| `UNPAYWALL_EMAIL` | `.env` | The contact address Unpaywall asks for; `open-access` needs it |
| `OPENALEX_API_KEY` | `.env`, optional | A larger daily budget for `openalex` (sent as a bearer token, never written to a receipt) |
| `CROSSREF_MAILTO` | `.env`, optional | The contact address for Crossref's polite pool (`crossref`, and the DOI check of `semantic-scholar`); `UNPAYWALL_EMAIL` if unset. Receipts say only whether it was set |
| `SEMANTIC_SCHOLAR_API_KEY` | `.env`, optional | A Semantic Scholar API key (one request a second of one's own); without it, `semantic-scholar` and `semantic-scholar-doi` share the public rate limit |
| The DOIs found | `crossref_status.csv`, `semantic_scholar_status.csv` in the folder | Read by `semantic-scholar` (Crossref's `no_match` rows), `openalex`, `open-access` and `semantic-scholar-doi` |
| The downloads so far | `pdf_download_status.csv` in the folder | Read by `openalex`: a row without a DOI waits while its title match's PDF may still come |
| OpenAlex's answers | `openalex_status.csv` in the folder | Read by `open-access`, which asks Unpaywall about what OpenAlex left without a PDF |
| A GROBID service | `--grobid-url` (default `http://localhost:8070`) | GROBID 0.9.1, the version the released closure used |
| PDF identity verdicts | `--identity-verdicts`, optional | `accepted.json` of the closure step's PDF identity check |

## Outputs

All in the step's folder, `$DATA_ROOT/collect/` unless `--folder` names another:

| File | Written by | What it is |
|---|---|---|
| `crossref_status.csv` | `crossref` | Every search, one row per attempt: the query, the verdict and its reason, the DOI, the matched record's title, years, first author and type, how many records passed each part of the rule, and the DOIs that passed all of it |
| `semantic_scholar_status.csv` | `semantic-scholar` | Every title match: the verdict, the match's id, title, year, score, arXiv id and PDF address, the DOI kept, and what the Crossref check said (`crossref_record_passes`, `arxiv_doi`, `dropped …`) |
| `semantic_scholar_candidates.csv`, `semantic_scholar_doi_candidates.csv` | `semantic-scholar`, `semantic-scholar-doi` | The papers with an address now, in the columns of `openalex_candidates.csv`; `download --candidates` reads them |
| `semantic_scholar_doi_status.csv` | `semantic-scholar-doi` | One row per paper asked about: Semantic Scholar's id, open-access flag, status, licence, PDF address, arXiv and PubMed Central ids (`status_code` 404 when it does not know the DOI) |
| `open_access_status.csv`, `openalex_status.csv` | `open-access`, `openalex` | Every lookup, one row per attempt (append-only) |
| `open_access_candidates.csv`, `openalex_candidates.csv` | `open-access`, `openalex` | The papers with a PDF address now; `download` reads one of them |
| `pdf_batches/pdf_batch_<n>.zip` | `download` | The PDFs, as `<xx>/<paper id>.pdf`; each member's comment holds its paper id and DOI |
| `pdf_download_status.csv`, `failed_pdf_downloads.csv` | `download` | Every download attempt (a failed one names each address tried and what it answered, the recovery's after ` \| recovery: `), and the papers whose last attempt failed |
| `grobid/<xx>/work_<sha256>.grobid.tei.xml` | `grobid` | The kept TEI files, named by the hash of the paper id and DOI |
| `grobid/<xx>/work_<sha256>.grobid.tei.closure.json` | `grobid` | The receipt of each TEI: the request, the rule that matched, the PDF's and the TEI's sha256 |
| `grobid_status.csv`, `failed_grobid.csv` | `grobid` | Every conversion attempt, and the PDFs whose last attempt failed |
| `receipts.jsonl` | every command | One receipt per run: the command, its settings, the hashes of its inputs and outputs, the code commit, the time and the counts |

## How it was checked

- **The end-to-end test** (`tests/test_collect.py`) runs `openalex`, `open-access`, `download` and `grobid` on a three-row queue. Unpaywall, OpenAlex, two publishers and GROBID are one local replay server that answers only recorded requests; every other address goes to a closed proxy port, so nothing reaches the network. It checks the counts, the two PDFs in the zip, the two TEI files with receipts the closure step accepts (one `doi`, one `title_year_author`), and every run receipt: the command, the pinned defaults, the input hashes and the commit.
- **One test file per command** (`tests/test_collect_open_access.py`, `…_openalex.py`, `…_download.py`, `…_grobid.py`, `…_crossref.py`, `…_semantic_scholar.py`) covers resuming, retries, rate limits, alternates, the size cap, every outcome of the Crossref and title-match rules and of the Crossref check of a DOI, the open copies (arXiv, Europe PMC, hosts never used), and every case of the identity rule (`tests/identity_cases.py`, 29 cases, 14 of them title-page DOIs).
  `tests/test_collect_found_dois.py` runs `crossref`, `semantic-scholar`, `openalex` and `open-access` in turn and checks that a found DOI is looked up under the original row. `…_openalex.py` covers both title rules, the first-author check, the year off by one and the second search of the earlier rule's misses; `…_download.py` the recovery (a landing page's PDF, Europe PMC's rendering, the error of a failed one) and Europe PMC's pace; `tests/test_collect_order.py` the order across commands (a title match's PDF before OpenAlex's title search, Unpaywall after OpenAlex).
- **The released closure's lookups and download recovery, replayed** (on a CPU, no live request, about 4 minutes). The ledgers keep what each answer contributed, not the answer itself: for Crossref the counts and the matched record's DOI, first title, years, first author and type; for Semantic Scholar the match's id, title, year, score, DOI, arXiv id and PDF address, and what the Crossref check said; for OpenAlex the work's id, DOI, title, year and PDF addresses; for Unpaywall the PDF address and its host type; for a failed download each address tried and what it answered. So from each ledger row the smallest answer carrying the same evidence was rebuilt and served on a local port, and the original collection code and these commands were run on the released queue (23,005 `missing` rows). Both gave the recorded rows, field for field apart from the time:
  - 13,907 Crossref searches and 8,052 title matches (620 with a Crossref check of their DOI);
  - 14,458 OpenAlex lookups by DOI (5,402 by a found DOI, 306 of a queue DOI that OpenAlex lacks);
  - 8,164 OpenAlex title searches, each under the rule its ledger row names: 7,803 under rule `2026-09-29` and 361 under the earlier one;
  - 10,478 Unpaywall lookups (441 by the DOI of an OpenAlex title match, 306 of a queue DOI OpenAlex lacks);
  - 10,359 papers asked about by DOI;
  - 278 failed downloads with their recovery (159 papers; 48 tried a Europe PMC rendering), run in-process.
    For 38 of them, recorded in the recovery's first form, the error kept the recovery's answer without its address, so for those only the original and this downloader were compared.
    A successful recovery is not told apart from other downloads in the ledger (its row keeps only the address that worked); the tests cover it.

  That the original code gave the recorded rows shows that the rebuilt answers carry what the rows recorded. Where two rows send the same request but their answers, rebuilt from each row's own ledger row, differ (211 Crossref, 228 title-match and 10 title-search rows), those rows ran again in rounds in which each request has one answer. The commands choose whom to ask from the final ledgers, so 8 title searches and 329 Unpaywall lookups made before a later answer (a Semantic Scholar PDF that downloaded later, a PDF that OpenAlex found later) ran with the ledgers as they stood at the recorded request. Left out: 40 rows without a DOI that Crossref was never asked about (OpenAlex's title search had settled them before the Crossref stage ran; they are among the title searches), 1 paper asked about by DOI without a recorded answer, and 58 rows of papers no longer in the queue (14 title searches, 44 Unpaywall lookups).
  From the recorded ledgers, the candidate files give the same addresses in the same order as the original code's list of downloads, for each of the 3,745 works it would fetch from a Semantic Scholar address (391 title matches, 2,013 copies found by DOI, and 1,341 OpenAlex or Unpaywall candidates with Semantic Scholar's copies after their own; 396 of the addresses are Europe PMC renderings).
- **Against the original implementation**: run on the same recorded answers, the original collection functions and these commands gave the same ledgers, zips, TEI files and receipts (apart from times and folder paths), and the same identity helpers.
- **The released TEI folder** (read only): the receipt reader accepts all 4,534 receipts that GROBID conversions wrote, and the identity rule, given each receipt's request, gives the rule the receipt recorded, for all 4,534. (The folder's other 65 receipts are `existing_corpus` ones that the closure step wrote; one of them has no body text and is refused, which leaves the 64 papers of the table above.)

The step is not rerun against the released data: rerunning it would call Unpaywall, OpenAlex, the publishers and GROBID again, and their answers change. The closure step reads the released TEI files and receipts as frozen inputs.

## How to adapt it

- **Another API endpoint** (a mirror, a proxy): `--api-url` of `open-access` and `openalex`.
- **Keys:** `UNPAYWALL_EMAIL` and `OPENALEX_API_KEY` in `.env`, and, both optional, `CROSSREF_MAILTO` (Crossref's polite pool; else `UNPAYWALL_EMAIL`) and `SEMANTIC_SCHOLAR_API_KEY` ([data-root.md](../setup/data-root.md)).
- **Another GROBID:** `--grobid-url`; for a GROBID on a cluster node, start it there and point the URL at it. `--concurrency` (default 10, as the run that wrote the released TEI folder; the released closure's other GROBID run used 8) should not exceed the service's own pool, which is 10 in GROBID's default configuration.
  `--consolidate-header` lets GROBID look the header up online (off in the released closure).
- **Speed and politeness:** `--sleep-seconds` (Unpaywall 1.0, OpenAlex 0.1, Crossref 0.2, Semantic Scholar's title match 1.111, its DOI batch 1.2 after each batch), `--check-sleep-seconds` (the Crossref check of a Semantic Scholar DOI, 2.5), `--max-retries` (`openalex`), `--max-attempts` and `--backoff-seconds` (`crossref`: 9 attempts, waits doubling from 1 s; `semantic-scholar`: 6, waits growing by 5 s), `--timeout-seconds`, and `--concurrency` of `download` (default 4). Crossref's own advertised rate, if lower, slows `crossref` down. An HTTP 429 doubles `crossref`'s pause for the rest of the run (at most 5 s); it doubles `semantic-scholar`'s pause too (at most 10 s), which every answered request then shortens by a tenth, back to the starting pace.
- **Size and batching:** `--max-pdf-mb` and `--pdf-batch-size` of `download`.
- **A smaller run:** `--limit` on any command; rerun to continue.
- **Crossref's search and its rule's threshold:** `--rows` (records per search, 20) and `--min-title-chars` (a shorter title is too little evidence, 12); the receipt records both. For example `--min-title-chars 80` leaves a row titled "Deciding what's true: The rise of political fact-checking" (56 characters as compared) without a DOI.
- **The lookup rules** are `match` in `crossref.py`, `match` and `check_doi` in `semantic_scholar.py`, `open_copies` there (with `NOT_AUTHORISED_HOST`), OpenAlex's title search `_title_filter` and `first_author_cited` in `openalex.py` (`--title-rule` picks the earlier one, `_full_text_search`), and what Unpaywall is asked, `unpaywall_queries` there (`--reference-urls` of `open-access` adds the reference's own address). A changed rule changes the ledgers: run the lookups on a fresh folder.
- **The download recovery and Europe PMC's pace:** `recovery_urls` in `download.py` (at most `RECOVERY_URLS` more addresses), `--europe-pmc-url` of `download` for another Europe PMC, and the paced hosts with their gaps in `PACED_HOSTS` (cool-down `PACE_COOLDOWN_SECONDS`). Which failures count as final, so that `openalex` stops waiting for a title match's PDF, is `failure_class` and `RETRY_LATER` there.
- **The identity rule** is `identity_match` in `grobid.py`, with the DOIs that are no evidence in `doi_evidence`, and the search-result rule is `matches_metadata` in `openalex.py`. A change there changes which TEI files are kept, so rerun `collect grobid` on a fresh folder, then the closure step's extraction (Step 4) and everything after it.

## Not in this step

- **The scheduling of the released closure's collection**, which does not change what a paper's lookups and downloads give, is not part of this repository. The lookups ran in parallel threads and in passes, under daily OpenAlex quotas spread over several keys and a daily allowance of title searches; a work cited by several queue rows was asked about and downloaded once (here each row is asked, and gets the same answer); downloads ran with at most a few at a time per host and one at a time for Europe PMC (here Europe PMC's requests are spaced 3 seconds apart, and other hosts are not limited); a failed download was tried again only in some cases (a 429, server error or timeout up to three more times, a paced host's 429 or server error hourly up to ten attempts, an address not tried yet, and once more when the recovery came in), and Unpaywall stopped asking about a DOI after three errors. Here `download` tries every paper without a PDF again on each run and `open-access` retries its errors on each run: with the same answers that gives the same rows, and against live services it can only add PDFs.
- **Which title rule a row of the released closure got** depended on when it was searched: its first searches used the earlier rule. The ledger names it (`matched_by`); a fresh run uses rule `2026-09-29` for every row, and `--title-rule 2026-09-28` the earlier one.
- **The PDF identity check** that writes `--identity-verdicts` (a language model compares a PDF that the rule could not verify with the reference) belongs to the closure step (`closure identity`, [docs/step-4-closure/closure.md](../step-4-closure/closure.md#the-pdf-identity-check)).
