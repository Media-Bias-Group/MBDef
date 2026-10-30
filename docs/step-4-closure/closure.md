# Closure: `mbdef closure`

| | |
|---|---|
| **Paper step** | Step 4 (corpus closure): the works the seed papers cite in their extracted items (definitions, synonyms, descriptions, related terms, subtypes, supertypes and examples), one level deep, and the release folder of the corpus |
| **Code** | [`src/mbdef/closure/`](../../src/mbdef/closure/): `run.py`, `extend_gt.py`, `citations.py`, `bibliography.py`, `manual_citations.py`, `corpus_check.py`, `tei_sources.py` (citations to a queue of works to collect); `pdf_identity.py` (the PDF identity check); `inputs.py` (the bundle's four lists), `exclusions.py`, `existing.py`, `definition_citations.py`, `release_folder.py`, `bundle.py` (the release bundle). The identity rules (DOIs, compared text, TEI headers, conversion receipts) are [`collect/identity.py`](../../src/mbdef/collect/identity.py): the closure step imports them from the collect step |
| **Commands** | `closure run`, `closure resolve`, `closure manual`, `closure check`; `closure identity evidence`, `… judge`, `… decide`, `… merge`; `closure inputs`, `closure bundle` |
| **Checked by** | the end-to-end test [`tests/test_closure.py`](../../tests/test_closure.py) (a two-paper closure through `collect`), [`tests/test_closure_identity.py`](../../tests/test_closure_identity.py) (five synthetic identity checks replayed with no network), [`tests/test_closure_bundle.py`](../../tests/test_closure_bundle.py) (a miniature release), [`tests/test_closure_inputs.py`](../../tests/test_closure_inputs.py) (the bundle's four lists on the same miniature), the tests of each rule; the released closure, rebuilt from the data root, equals the released folder ([below](#how-it-was-checked)) |

## What the step does

The seed papers' extracted items cite other works. The closure collects the cited works that are open access, extracts their definitions too, and publishes both sets with the table that says which seed definition cites which collected paper. It goes one level deep: a collected paper's own citations are never followed.

```text
seed definitions + seed TEI --closure run--> queue of works not in the corpus ---+
                                                                                 v
   collect (crossref, semantic-scholar, open-access, openalex, download, grobid) <-- docs/step-1-2-papers/collect.md
                    |  TEI files and receipts          PDFs the rule refused --closure identity--> accepted.json
                    v                                                                      |  (--identity-verdicts)
   closure run --collect-folder (what is left)                                             v
                    |                                        collect grobid keeps them with the rule `llm_verified`
                    v
   extract (the closure papers' definitions, docs/step-3-extraction/extract.md) --> rows-all.json + derived/
                    |
                    v
   closure bundle: left out, existing corpus, links, identity decisions, release folder + MANIFEST
```

The steps, in order:

1. **`closure run`** resolves every citation string of the seed items to a reference-list entry of its paper (`[1]`, `(Hurcombe, 2019)`), groups the entries into cited works (by DOI; an entry without a DOI joins the one existing group with the same title whose years and authors do not conflict), and decides for each work whether a paper with that identity is already in the corpus roots: `present`, `missing`, or `review` (it could be two different works, or the link was uncertain). The works still `missing` or `review` are the **collection queue** (`before-queue.csv`) that [`collect`](../step-1-2-papers/collect.md) takes. After the collect step, `closure run --collect-folder` checks again with the TEI files it kept.
2. **`closure identity`** is the check for a PDF that the identity rule of `collect grobid` refused as unverified: a language model is shown the cited reference and the PDF's first two pages and says whether it is the cited work; the answer counts only if a program confirms it from the PDF text ([below](#the-pdf-identity-check)).
3. **`closure inputs`** writes four small lists the bundle reads: the collected papers, what was known of each when the duplicate check ran, the collected papers no seed reference cites, and the existing-corpus works not to add ([below](#the-bundles-four-lists-closure-inputs)).
4. **`closure bundle`** is the offline end: it leaves out collected papers whose text is another paper's, takes in the cited works found only in the project's earlier corpus, builds the table of links, takes the identity check's decisions and writes the release folder with a README and a manifest.

**A worked example: one collected paper in the released closure.** A seed paper (`pdf-48e4450c…`) defines *shareworthiness* and cites "Hurcombe, 2019" for it. `closure run` linked that citation string to the reference-list entry (method `ref_target`, confidence high), whose DOI is `10.5210/spir.v2019i0.10985`; no TEI with that DOI was in the corpus, so the work went into the queue. [`collect`](../step-1-2-papers/collect.md#what-the-step-does) found its PDF, GROBID read the title in capital letters and no year, and the PDF was kept under the rule `title_author_year_missing`. The extraction read it (11 definitions at P≥20, 6 at P≥24, 6 at P≥28, 8 at P≥28 with one model), and the release says in `closure/papers.csv`:

```text
paper_id            doi_10.5210_spir.v2019i0.10985__454fb465…__06f3208d186c8269
doi                 10.5210/spir.v2019i0.10985
pdf_identity_check  title_author_year_missing
n_citing_seed_papers 1      n_citing_seed_definitions 1
definitions_P20 11   definitions_P24 6   definitions_P28 6   definitions_P28-single-model 8
```

and in `links/definition-links.csv` one row: seed paper `pdf-48e4450c…`, term `shareworthiness`, field `definitions`, citation `Hurcombe, 2019`, work `doi:10.5210/spir.v2019i0.10985`, status `extracted`, closure paper as above, `in_P20` 1 and the other three variants 0 (the definition is in the P≥20 variant only).

**What became of the 25,316 cited works of the released closure** (`links/cited-works.csv`, by `work_status`):

| Status | Works | Meaning |
|---|---:|---|
| `resolved_no_oa_pdf` | 9,961 | identified, but no open-access PDF |
| `not_found` | 6,252 | no provider knew it (books, reports, garbled references) |
| `extracted` | 4,574 | collected, confirmed and extracted: the 4,556 closure papers (a few works that the grouping kept apart are the same PDF) |
| `review_not_auto_acquired` | 2,112 | a queue row that was not `missing`: an uncertain or ambiguous link, deliberately not fetched |
| `oa_pdf_found_download_failed` | 1,292 | an open-access link was known, the download failed (mostly bot protection) |
| `present_in_corpus` | 431 | already a seed paper (or the text of a left-out copy of one) |
| `ambiguous` | 392 | two works fit equally well |
| `pdf_rejected_unconfirmed` | 230 | a PDF came, but nothing showed it is the cited work |
| `pdf_rejected_other_work` | 51 | a PDF came, carrying another work's DOI |
| `pdf_unusable` | 21 | a PDF came without usable text |

## Run it

A new closure, with the files of the seed papers (`seeds.json`: their extracted definitions; `seed-tei/`: their TEI files):

```bash
uv run mbdef closure run --dataset seeds.json --out "$DATA_ROOT/closure/new/run" \
    --corpus seed-tei --tei-dir seed-tei
# the queue is $DATA_ROOT/closure/new/run/before-queue.csv; collect it (docs/step-1-2-papers/collect.md)
uv run mbdef collect crossref --collection "$DATA_ROOT/closure/new/run/before-queue.csv"
uv run mbdef collect semantic-scholar --collection "$DATA_ROOT/closure/new/run/before-queue.csv"
uv run mbdef collect open-access --collection "$DATA_ROOT/closure/new/run/before-queue.csv"
# ... openalex, download, semantic-scholar-doi, download, grobid (docs/step-1-2-papers/collect.md, "Run it")
uv run mbdef closure run --dataset seeds.json --out "$DATA_ROOT/closure/new/run" \
    --corpus seed-tei --tei-dir seed-tei --collect-folder "$DATA_ROOT/collect"
```

`closure run` prints a summary; for the two-paper example of the end-to-end test:

```text
{"before": {"present": 1, "missing": 1, "review": 0, "unique_cited_papers": 2, ...},
 "after": {"present": 2, "missing": 0, ...}, "markers": 3, "linked_markers": 3, ...}
queue for `collect`: …/before-queue.csv
```

The other three first-stage commands are the ones for the ground truth's own citations (Step 4 for the benchmark papers, with a ledger of manual links, YAML): `closure resolve` (to `citations.json`, `report.json`, `review.csv` and the ledger), `closure manual` (applies the completed ledger rows, all or none), `closure check` (the corpus check on a resolved dataset). `closure run` runs the first and the last in one go for a seed set.

The PDF identity check:

```bash
uv run mbdef closure identity evidence --out ids/r1            # needs pdftotext (poppler)
uv run mbdef closure identity judge --evidence ids/r1/evidence.json --out ids/r1/judged \
    --base http://127.0.0.1:8123/v1                                       # a vLLM server with the model
uv run mbdef closure identity decide --evidence ids/r1/evidence.json \
    --verdicts ids/r1/judged/verdicts.json --out ids/r1/decided
uv run mbdef collect grobid --identity-verdicts ids/r1/decided/accepted.json
```

A later round passes the earlier evidence (`evidence --earlier ids/r1/evidence.json`) and joins the accepted lists with `closure identity merge --earlier … --new … --round 2 --out …`. The released closure used three rounds (284, 155 and 4 papers accepted; 443 in all, of which 426 are released closure papers). When a collect folder holds no other verified paper to take a negative control's PDF from, `evidence` skips that control.

The release bundle, from the released closure's files in the data root (about 1.7 GB written):

```bash
uv run mbdef data fetch frozen/                # the inputs, below; the first time only
uv run mbdef closure bundle --out "$DATA_ROOT/closure/bundle"
```

It prints each step's counts and ends with `release: …/release (42158 files verified)`. It reads four small lists (`--audit-inputs`, by default the released ones); `closure inputs --out …` writes them from the run's files, and [below](#the-bundles-four-lists-closure-inputs) is how the released ones are rebuilt. Every folder is a flag (`--ledgers`, `--closure-set`, `--closure-tei`, `--seeds`, …, `closure bundle --help`) and the release's name is `--version`.

## Inputs

The release is rebuilt from files that earlier steps wrote. All of them are in the data root (the paths are below `$DATA_ROOT`; `data/data-root.tsv` lists each with its sha256 and where to get it), except the two in git:

| Folder or file | What it is | From |
|---|---|---|
| `frozen/runs/production-20260925/corpus/seeds/{derived,compare}` | the seed run's four variants and their comparison | [extract derive](../step-3-extraction/extract.md#the-seed-runs-variants-extract-derive) |
| `frozen/runs/production-20260925/inputs/{runner-rows.json,seed-tei/}` | the seed papers' extraction rows and 3,884 TEI files | the seed run's copy |
| `frozen/runs/production-20260925/inputs/existing-corpus-tei.{zip,sha256}` | the project's earlier corpus: 14,862 TEI files in a zip, and their hashes | the seed run's copy |
| `frozen/closure/closure-v11/extraction/unpacked/` | the closure papers' extraction: `rows-all.json`, `derived/`, `compare/`, `batches.json`, `set.json`, `picks.json`, `paper-metadata.json` | the closure extraction run's copy |
| `frozen/closure/closure-v11/links-inputs/` | the closure run's ledgers: `before.json`, `before-queue.csv`, `provider.csv`, `unpaywall.csv`, `download.csv`, `download-dedup.json`, `grobid-laptop-snapshot.csv`, `union-dataset.json`, `cited-works.json` (the audit's grouping, which the grouping must equal) | the closure run's working copy |
| `frozen/closure/taxonomy-closure-20260928/prod/closure/run/grobid/` | the 4,599 collected closure papers' TEI files with one receipt each (9,198 files, 538 MB) | the closure run's working copy |
| `…/grobid-left-out-v11/` | the TEI files of 6 papers moved out after the duplicate check | the closure run's working copy |
| `frozen/closure/closure-v11/{papers-info,all-candidates,uncited,existing-exclude}.json` | the bundle's four lists as released: per-paper facts for the duplicate rule, the 4,605 papers compared, 9 papers no seed cites, 2 papers not to add; `closure inputs` rebuilds all four, two of them in another order ([below](#the-bundles-four-lists-closure-inputs)) | the closure run's working copy |
| `frozen/closure/closure-v11/links-inputs-v1/` | the closure run's earlier ledgers (those of release v1, extraction batches 1-6), the same ten files as `links-inputs/`; `closure inputs` counts `papers-info.json`'s citing seed papers from them | the public deposit (`data/data-root.tsv`, a zstd tar like every folder) |
| `frozen/closure/taxonomy-closure-20260928/identity-llm/` | the three identity rounds' `evidence.json` and `verdicts.json` (`r2/`, `r3/`), and the released decisions, which `--identity-reference` compares with | the closure run's working copy |
| `frozen/closure/definition-corpus-v1.1/` | the released folder, 42,159 files, 1.7 GB: what a rebuild is compared with | the release itself (a zstd tar with its `MANIFEST.sha256`) |
| `data/frozen/docs/closure-loss-audit.md` | the loss audit as the release recorded it, copied into the release | git |
| `data/frozen/results/closure-audit-20261001/` | the released duplicate-text record, exclusions and existing-corpus list, which `--audit-reference` compares with | git |

`data fetch` checks every file it fetches against the sha256 that `data/data-root.tsv` lists (for example `…/run/grobid: fetched, sha256 matches`).

## What `closure bundle` does, step by step

1. **Left out.** Every closure paper's TEI is reduced to its words (the tags are stripped, so the header and the reference list count too), and 8-word sequences (one every 4th word) are hashed. Two papers are the same document when at least 80% of the smaller paper's sequences occur in the other. A closure paper with a seed paper's text is left out (the work is a seed paper whose TEI header GROBID garbled); of two closure papers with the same text one stays: the one already extracted, then the one with a published DOI (not an arXiv DOI), then the one more seed papers cite, then the smaller paper id. Papers no seed reference cites (collected through a DOI GROBID had put on the wrong reference entry) are left out too: 48 papers in the released closure (15 seed copies, 24 closure copies, 9 not cited).
   *Example:* `doi_10.1007_s42001-018-0015-z…` has the same text as seed paper `pdf-3966deca…` (share 1.0).
2. **Existing corpus.** A cited work that is `present` only in the earlier corpus (the zip) was never fetched and never extracted, but its TEI exists: it becomes a closure paper with a receipt (`existing_corpus`, the zip member, the corpus check's method). 64 of the released closure papers are such works. Their files must equal the ones in the collected TEI folder; the bundle stops if not.
3. **Links.** One row per citation: the seed definition, the citation string, the reference entry, the cited work after the merges M1 (same DOI), M2 (same title, year and first author) and M3 (the same OpenAlex work), its status, and the closure paper it became. The grouping must equal the closure audit's own (`cited-works.json`); the run stops otherwise. The helpers that group (`nfkc_words`, `strong_doi`, `surname`, `UnionFind`) are in `closure/definition_citations.py`.
4. **Identity decisions.** `decide` on each round's saved evidence and verdicts, then the merges.
5. **The release folder.** Variant files are checked against the SHA-256 in their `DONE.json`, TEI files against the extraction rows, the README's numbers are counted from the copied files, `MANIFEST.sha256` lists every file, and `verify` checks every manifest line again.

## The bundle's four lists: `closure inputs`

Steps 1 and 2 of the bundle read four small lists besides the run's ledgers and folders. `closure inputs` writes them from the run's own files, with the rules the released lists were made with, into `--out`, with a `receipt.json` (settings, the sha256 of every input, counts). It looks for same-text pairs among the existing-corpus works alone, not together with the collected papers; on the released files the pairs are the same two.

| List | What it holds | Released closure |
|---|---|---|
| `all-candidates.json` | every collected closure paper, `{"paper_id", "tei"}`: one per conversion receipt (`*.grobid.tei.closure.json`) in the collected TEI folder (`--closure-tei`), in the order of their paths in that folder, each TEI path below `--tei-prefix` (by default the folder as `--closure-tei` gives it, or `$DATA_ROOT/frozen/closure/…/grobid` when that flag is left out, so the list names no folder of the machine) | 4,605 papers |
| `papers-info.json` | for the same papers, in the same order, what was known when the duplicate check ran: the DOI the paper was collected under (its receipt's `requested_doi`), whether it was extracted already (in the closure set's first `--extracted-batches` batches), and how many seed papers cited it (the links of that time: step 3's rules on `--earlier-ledgers` over the papers extracted then, every cited work that became the paper counted) | 4,605 papers, 4,503 extracted (batches 1-6: 2,841 + 218 + 282 + 3 + 1,110 + 49) |
| `uncited.json` | the papers of the closure set that no link row reaches (step 3's rules on `--ledgers`, with the existing-corpus works added and the copies the duplicate check finds left out), apart from those copies; sorted | 9 papers |
| `existing-exclude.json` | the cited works found only in the existing corpus whose TEI has the same text as another such work's (the 80% rule of step 1): of each pair, the one fewer seed papers cite, on a tie the one with the smaller paper id | 2 of 67 works |

*Examples from the released closure.* The first entry of `papers-info.json` is `doi_10.1609_icwsm.v4i1.14009…`: `{"doi": "10.1609/icwsm.v4i1.14009", "extracted": true, "n_citing_seed_papers": 1}`. The citing counts are those of the earlier ledgers (`links-inputs-v1/`), as release v1 listed them: `hash_13fbee78f4471164…` has 12; on the final ledgers (`links-inputs/`), where reference DOIs on the wrong entry are ignored, it has 16, and 14 of the 4,503 extracted papers would get another count. The count is only the duplicate rule's third tie-break.
`doi_10.1017_pan.2024.29…` is one of the 9 uncited papers: it was collected through such a DOI (on the earlier ledgers 2 seed papers seemed to cite it); with the DOI ignored, no link row reaches it. Of the 67 works found only in the existing corpus, two pairs have the same text (share 1.0): `doi_10.18653_v1_2021.acl-long.416…`, which 7 seed papers cite, and `hash_35193c447714ecd8…`, which 1 cites, so the second is not added; `hash_04d1fa4e1abc4bb2…` and `hash_a40ddc378578c7bd…`, 1 each, so the smaller id, `hash_04d1…`, is not added. The bundle adds the other 65; 64 are in the release (the extraction refused one: batch 7's `excluded` entry in `batches.json`).

In outline:

```text
papers            = the receipts in --closure-tei, in the (bytewise) order of their TEI paths
all-candidates    = [{paper_id, tei: --tei-prefix/<subfolder>/<TEI file>}]
extracted         = rows-all.json, the papers of the first --extracted-batches batches
papers-info       = {paper_id: {doi: requested_doi, extracted, n_citing_seed_papers: links(--earlier-ledgers, extracted)}}
works             = the cited works found only in the existing corpus (before.json of --ledgers)
existing-exclude  = for each pair of works with one text: the one fewer seed papers cite (tie: smaller paper id)
copies            = the duplicate rule of step 1 on all-candidates and papers-info, nobody uncited yet
uncited           = sorted(papers of rows-all.json without a link row in links(--ledgers, works kept, copies left out)) - copies
```

**The released lists, rebuilt.** Three things of the released run are not the data root's defaults. (1) The lists were made before six of the papers left out were moved aside into `grobid-left-out-v11/`, so they list the folder as it was; a copy of both folders made of hard links (no extra space) puts it back (the `cp -al` line below). (2) The released lists have the order in which the file system of their machine listed the folder: ext4 lists a folder by a hash of each file name, seeded per file system, so that order comes back nowhere else.
`closure inputs` lists the papers in the order of their TEI paths instead, the same on every machine, so `all-candidates.json` and `papers-info.json` equal the released files once those are put in that order (the comparison below); the other two lists do not depend on the order. (3) The `tei` paths name the folder of the machine the list was made on; the bundle reads only the file names, and `--tei-prefix` writes that folder, read here from the released list. The citing counts need the earlier ledgers in `frozen/closure/closure-v11/links-inputs-v1/` ([Inputs](#inputs)).

```bash
R="$DATA_ROOT/frozen/closure/taxonomy-closure-20260928/prod/closure/run"
V="$DATA_ROOT/frozen/closure/closure-v11"
W="$DATA_ROOT/closure-inputs"
mkdir -p "$W" && cp -al "$R/grobid" "$W/grobid-as-checked" && cp -al "$R/grobid-left-out-v11/." "$W/grobid-as-checked/"
P=$(python3 -c 'import json, os, sys; print(os.path.dirname(os.path.dirname(json.load(open(sys.argv[1]))[0]["tei"])))' "$V/all-candidates.json")
uv run mbdef closure inputs --out "$W/rebuilt" --closure-tei "$W/grobid-as-checked" --tei-prefix "$P"
for f in uncited existing-exclude; do cmp "$V/$f.json" "$W/rebuilt/$f.json"; done
python3 - "$V" "$W/rebuilt" <<'PY'   # the released order of the other two lists, set aside
import json, sys
from pathlib import Path
v, w = (Path(a) for a in sys.argv[1:])
listed = sorted(json.loads((v / "all-candidates.json").read_text()), key=lambda c: c["tei"].encode())
info = json.loads((v / "papers-info.json").read_text())
assert json.dumps(listed) == (w / "all-candidates.json").read_text()
assert json.dumps({c["paper_id"]: info[c["paper_id"]] for c in listed}) == (w / "papers-info.json").read_text()
print("all-candidates.json and papers-info.json: the released lists in the order of their TEI paths")
PY
```

It prints `"candidates": 4605, "extracted": 4503, "copies": 39, "uncited": 9, "existing_only_works": 67, "existing_not_added": 2` (the 39 copies are the 15 seed copies and 24 closure copies of step 1; with the 9 uncited papers they are its 48) and takes about 1 min 40 s on one CPU core. `cmp` finds no difference in `uncited.json` and `existing-exclude.json` (the released sha256 `197b7315…` and `3cd696c6…`), and the other two are the released files (`e7b133b3…`, `cb4a12ef…`) in the order of their TEI paths (`a7ed0fcb…`, `6b77f21e…`).

## Outputs

`closure bundle --out OUT` writes:

| Folder | What it is |
|---|---|
| `release/` | the release: `seeds/` and `closure/` (`papers.csv`, `tei/`, `derived/<variant>/predictions.json`, `…/papers/`, `…/union/`, `DONE.json`), `links/` (`definition-links.csv`, `cited-works.csv`, `definition-links-summary.json`), `provenance/` (batches, comparisons, `exclusions.json`, `release-facts.json`, the loss audit), `README.md`, `MANIFEST.sha256` |
| `links/` | the same three link files |
| `audit/` | `duplicate-texts.json`, `exclusions.json`, `existing-corpus-added.json` |
| `existing/grobid/` | the TEI files and receipts written for the existing-corpus works |
| `identity/` | per round `accepted.json` and `identity-decisions.json`, and `accepted-all.json`, `accepted-all-r3.json` |
| `receipt.json` | the settings, the sha256 of every input, and the counts of every step |

The release README says in plain words what each folder holds and how to follow a definition to its source.
Two builds differ only in their build time (`Built …` in the README and `built_utc` in the facts; the manifest lists their hashes).

## The PDF identity check

`collect grobid` keeps a converted PDF only if its title page shows the cited work (the rule accepts a missing year, a year one off and a missing title, [docs/step-1-2-papers/collect.md](../step-1-2-papers/collect.md#what-the-step-does)). Correct PDFs still fail that rule, for example when GROBID read a garbled title or a year more than one off. For those, a verdict of the model counts only if all of these hold (`checks` in `pdf_identity.py`):

| Check | Holds when |
|---|---|
| `same_work` | the model answered `same_work` (not `different_work` or `cannot_tell`) |
| `author_in_text` | a cited author surname (3 letters or more) is in the PDF text |
| `title_in_text` | the cited title is in the PDF text (case and punctuation ignored) |
| `title_close` and `pdf_title_in_text` | alternatively: the title the model read from the PDF is in the text and is at least 85% like the cited title (or, for a cited title of at least 20 characters, one contains the other) |

The check is run on controls as well, so its error rate is known: positive controls (PDFs verified by DOI or by title, year and author, which the model must accept) and negative controls (the cited reference paired with a different paper's PDF, which it must refuse). `decide` prints how many of each the program accepted. The model, revision, prompt, temperature (0), token limit (500), schema and "thinking off" are pinned defaults (`closure identity judge --help`); every request is written as a receipt, `requests/<request_sha256>.json`, and `--replay FOLDER` answers from such receipts and sends nothing. A request that is not recorded stops the run: it is the guard that makes a changed prompt, setting or evidence fail loudly. A live run of the check needs a GPU server; CI runs only its replay.

## How it was checked

- **The end-to-end test** (`tests/test_closure.py`): two seed papers cite one work between them and one cites the other. `closure run` finds the shared work missing and the other `present`, and queues the missing one; `collect grobid` (a fake GROBID on a local port) converts its PDF; the second `closure run` finds nothing left. The receipt names the command, the pinned settings, the commit and the dataset's hash.
- **The identity check, with no network** (`tests/test_closure_identity.py`): five synthetic checks (three candidates, a positive and a negative control; `tests/fixtures/synthetic-identity-replay.zip`, made by `tests/fixtures/make_synthetic_replays.py` with a stand-in model) are judged from the recorded answers, which gives the recorded verdicts, and `decide` gives the recorded decisions. They have the outcomes of the released checks: one candidate accepted, one the model accepts and the program checks refuse, one different work, a positive control accepted and a hard negative refused (the released checks hold the PDFs' first pages, so they are not public). A changed evidence text is an unrecorded request and stops the run. A live run against a local server sends the pinned request (temperature 0, 500 tokens, the schema, strict) and records receipts that replay it. The miniature closure of the bundle tests uses the same checks.
- **A miniature closure** (`tests/test_closure_bundle.py`, built by `tests/mini_bundle.py`, worked out by hand): two seed papers, five cited works, one of them a copy of a seed paper and one only in the existing corpus. The tests check the exclusions, the six link rows and their statuses, the README's numbers, every manifest line, the identity decisions, the receipt, and that two builds differ only in the build time.
- **Against the original implementation**: on the same inputs, the original closure code and this package gave the same bibliography records, the same resolved dataset and report, and the same files from a whole run.
- **The rules' own tests**: the bibliography, citation resolution, the corpus check, the manual links, the adversarial identities and whole-run invariants, the DOI cases of the identity fixes and the independent identity oracle at a small scale (`tests/test_closure_scale.py`: 100 seed papers, 10,000 references).
- **The bundle's four lists** ([`tests/test_closure_inputs.py`](../../tests/test_closure_inputs.py)): on the miniature, every collected paper is a candidate (W1, W5 and W4), `papers-info.json` has each paper's DOI, whether batch 1 extracted it and its citing seed papers (2, 1 and 0), a collected paper no seed cites is uncited while the seed copy W5 is not, of two existing-corpus works with one text the one fewer seed papers cite (or the smaller id) is not added, and the bundle built on the rebuilt lists is the release built on the carried ones. On the released closure's files the four lists are rebuilt, two byte for byte and two in the order of their TEI paths ([above](#the-bundles-four-lists-closure-inputs)).
- **No closure module loads code by its file path or reads git history**, and the loss audit the bundle copies into the release is the frozen file with the released hash (`tests/test_closure_self_contained.py`).
- **The released closure, rebuilt.** `closure bundle` on the released closure's files in the data root, with `--audit-reference data/frozen/results/closure-audit-20261001 --identity-reference "$DATA_ROOT/frozen/closure/taxonomy-closure-20260928/identity-llm"`, checks that its duplicate-text record, exclusions, existing-corpus list and identity decisions equal the released ones. Its release folder equals the released one (`frozen/closure/definition-corpus-v1.1/`) byte for byte, apart from the build time (in `README.md` and `provenance/release-facts.json`) and one time stamp in `closure/derived/DONE.json` (`filtered.union_recounted_utc`, written into the released file by hand), and the manifest lines of those three files. So the rebuild has the released 42,158 manifest lines, 4,556 closure papers, 69,611 closure definitions at P≥28, the counts of every identity rule, and 443 accepted verdicts.

## How to adapt it

- **Another seed set:** `closure run --dataset … --corpus … --tei-dir …`; the collect step and the extraction in between are the same commands as for the seed papers.
- **Another model or server for the identity check:** `--base`, `--model`, `--revision` of `closure identity judge` (the receipts record them); the prompt and schema are `SYSTEM`, `user_prompt` and `SCHEMA` in `pdf_identity.py`.
  A different decision rule is `checks`; rerun `decide` on the saved verdicts to see its effect.
- **Another duplicate rule:** `SHARE`, `SHINGLE_WORDS` and `SHINGLE_STEP` in `exclusions.py`, and `rank` for which copy stays.
- **Another release:** the folders and `--version` of `closure bundle`. The README's text is `readme(facts)` in `release_folder.py`; its numbers come from the files.
- **Byte formats:** the audit files, the identity files and the release keep the formats they were released with (`write_duplicates`, `write_exclusions`, `pdf_identity.write`, `release.finish`).
- **A changed resolver:** reports name the resolver (`taxonomy_pipeline.closure.extend_gt/5`) and the sha256 of the closure code, so `closure manual` refuses a report made by other code; rerun `closure resolve` after a code change.

## Not in this step

- **Acquisition** is the [collect](../step-1-2-papers/collect.md) step: the Crossref, Semantic Scholar and Europe PMC lookups that found the released closure papers, and the chaining of a found DOI to OpenAlex and Unpaywall, are its commands (replayed on the released ledgers, [docs/step-1-2-papers/collect.md](../step-1-2-papers/collect.md#how-it-was-checked)). The scheduling of the released run (parallel passes, quotas, one download per work) and the few rules collect does not have ([docs/step-1-2-papers/collect.md](../step-1-2-papers/collect.md#not-in-this-step)) are not part of this repository. `closure run` does not start the collect stages itself.
- **The extraction of the closure papers** is the [extract](../step-3-extraction/extract.md) step run on the closure papers' TEI files (the set `closure`); the rows it reads (`rows-all.json`) come from the collected TEI files and their receipts. Its saved outputs are inputs here.
- **The language model run** of the identity check (a GPU server): its saved verdicts are inputs; only `decide` is rerun.
- **The pages and plots** built from the release are not part of this step.
