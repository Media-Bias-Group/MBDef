# Seeds: the seed list (paper Steps 1 and 2)

| | |
|---|---|
| **Paper step** | Steps 1 and 2: a database search with the PDFs converted to text (the starting corpus, 14,862 papers), and a relevance labelling by language models that picked the papers to extract from (the seed papers, 3,869). The search is a manual database query, given [below](#the-search-query); downloading and converting papers is [`collect`](collect.md); the relevance labelling is the [`relevance`](relevance.md) step, which replays it. The later steps start at the seed list |
| **Code** | none for the search (a manual query, [below](#the-search-query)); the relevance labelling is [`src/mbdef/relevance/`](../../src/mbdef/relevance/) ([relevance.md](relevance.md)). The list was made by [`tests/make_seed_list.py`](../../tests/make_seed_list.py) from the recorded inputs of the seed run and the two models' output files |
| **Commands** | none: the step is two data files, [`data/seeds/seed-papers.tsv`](../../data/seeds/seed-papers.tsv) and [`data/seeds/starting-corpus.tsv`](../../data/seeds/starting-corpus.tsv), and this page |
| **Checked by** | [`tests/test_seeds.py`](../../tests/test_seeds.py): the counts, columns and identifiers of both files, and (with the data root) the same papers, titles, DOIs and hashes as the seed run's recorded rows and seed manifest |

## What the steps did

**Step 1: the search and the starting corpus.** A keyword search in a bibliographic database (Scopus) for papers on media bias and the concepts around it, limited to English-language documents that are openly accessible and outside most medical and natural-science subject areas; [The search query](#the-search-query) gives the search string and reads it. The PDFs that could be fetched were converted with [GROBID](https://grobid.readthedocs.io/) to **TEI**, GROBID's structured text of a paper (title, abstract, body, references). 14,862 papers were converted: the **starting corpus**. Each has an identifier made from the DOI of its database record, for example `doi_10.1002_9781444314861`, and its TEI file is named after it.

**Step 2: the relevance labelling.** Two language models read the title and abstract of every paper in the starting corpus, after a short definition of media bias, and each answered with one label: relevant, not relevant or not sure. The files record the answer of **Qwen3.6-35B-A3B** and of **Gemma 4 12B**, both prompted without examples (Qwen's run was configured with labelled examples of similar papers, but none existed for these papers, so it got the same prompt as Gemma: [relevance.md](relevance.md#the-prompts)). The recorded name of "not sure" is `can't determine`; an answer that is none of the three is recorded under it too. Where the two models gave different answers, a third model (GLM 5.2) decided. The PDFs of the papers that ended up relevant were copied into one hand-over folder, and Step 3 started from that folder.

**From the folder to the seed papers.** Step 3 converted the folder's PDFs with GROBID where no saved conversion matched, took each paper's title from its TEI (or, where GROBID found none, from a lookup that the PDF's first page confirmed), and extracted definitions from every paper that had an article body and a title.

| Stage | Count | A worked example | Checked by |
|---|---:|---|---|
| Papers in the starting corpus | 14,862 | `doi_10.1002_9781444314861`, the first row of `starting-corpus.tsv` | the test, and the recorded TEI list |
| Answers of Qwen and of Gemma | 14,482 and 14,562 | Qwen answered `relevant` for 2,998 of its rows, Gemma for 5,074 | the recorded answers, replayed by [`relevance`](relevance.md) |
| Rows both models answered `relevant` | 2,963 | the first seed in the list below; counted by paper id and title | the recorded answers, replayed by [`relevance`](relevance.md) |
| Rows the models disagreed on, given to the third model | 2,196 | *Sharing Lies: A Socio-Technical Review on False Information Dissemination*: Qwen `not relevant`, Gemma `relevant` | the recorded answers, replayed by [`relevance`](relevance.md) |
| Relevant rows in total | 3,888 | 2,963 agreed + 925 kept by the third model | `relevance export` ([relevance.md](relevance.md)) |
| PDFs in the hand-over folder | 3,888 | each named after its Step 2 id; as many as the relevant rows, but not the same papers one for one (below) | the seed manifest, by the test (slow) |
| of these, with an article body in the TEI | 3,884 | 4 PDFs are notices (an erratum, a withdrawal, a table of contents) | the seed manifest, by the test (slow) |
| of these, with a title confirmed | **3,869** | 15 TEIs had no title and none could be confirmed; these are the **seed papers** | the runner rows, by the test (slow) |

## The search query

The full search of Step 1, word for word as it was entered in Scopus's advanced search:

```text
TITLE-ABS-KEY ( ( ( bias OR biases OR biased OR framing OR frame* OR slant ) W/5 ( media OR news* OR journalis* OR reporting OR article* OR headline* OR outlet* OR source* OR LANGUAGE OR editor* ) ) OR ( ( media OR news* OR journalis* OR reporting OR article* OR headline* OR outlet* OR source* OR LANGUAGE OR editor* ) W/5 ( bias OR biases OR biased OR framing OR frame* OR slant ) ) OR ( ( inequality OR unequal OR distort* OR imbalance* OR skew* OR selective ) W/5 ( attention OR discurs* OR exposure OR source* OR prominence OR content OR representation OR report* ) ) OR ( ( attention OR discurs* OR exposure OR source* OR prominence OR content OR representation OR report* ) W/5 ( inequality OR unequal OR distort* OR imbalance* OR skew* OR selective ) ) OR "coverage bias" OR "coverage imbalance" OR "coverage skew" OR "unequal coverage" OR "selective coverage" OR ( ( "agenda setting" OR both-sidesism OR "cherry picking" OR clickbait OR delegitimization OR demonization OR "differential prominence" OR disinformation OR "echo chamber" OR emotionalism OR "fake news" OR "filter bubble" OR "flawed logic" OR gatekeeping OR "hate speech" OR "hedge words" OR misinformation OR mudslinging OR normalwashing OR overrepresentation OR propaganda OR rumor OR scandalization OR scapegoating OR sensationalism OR spin OR stance OR stereotyping OR stigmatization OR underrepresentation OR "unsubstantiated claims" OR vagueness OR whataboutism ) W/7 ( media OR news* OR journalis* OR reporting OR article* OR headline* OR outlet* OR source* OR LANGUAGE OR editor* ) ) ) AND ( EXCLUDE ( SUBJAREA , "MEDI" ) OR EXCLUDE ( SUBJAREA , "MATH" ) OR EXCLUDE ( SUBJAREA , "PHYS" ) OR EXCLUDE ( SUBJAREA , "MATE" ) OR EXCLUDE ( SUBJAREA , "BIOC" ) OR EXCLUDE ( SUBJAREA , "ENVI" ) OR EXCLUDE ( SUBJAREA , "CHEM" ) OR EXCLUDE ( SUBJAREA , "EART" ) OR EXCLUDE ( SUBJAREA , "ENER" ) OR EXCLUDE ( SUBJAREA , "AGRI" ) OR EXCLUDE ( SUBJAREA , "CENG" ) OR EXCLUDE ( SUBJAREA , "PHAR" ) OR EXCLUDE ( SUBJAREA , "NURS" ) OR EXCLUDE ( SUBJAREA , "HEAL" ) OR EXCLUDE ( SUBJAREA , "IMMU" ) OR EXCLUDE ( SUBJAREA , "DENT" ) OR EXCLUDE ( SUBJAREA , "VETE" ) ) AND ( LIMIT-TO ( LANGUAGE , "English" ) ) AND ( LIMIT-TO ( OA , "all" ) )
```

**How to read it.** `TITLE-ABS-KEY` searches a document's title, abstract and keywords. A `*` matches any ending (`journalis*` finds *journalism*, *journalist*, *journalistic*), and `W/n` means that the two sides stand within *n* words of each other, in either order; the query also spells out each `W/5` pair in both orders. A document is found if it matches any one of these parts:

1. **A bias word near a media word.** A bias word (`bias`, `biases`, `biased`, `framing`, `frame*`, `slant`) within five words of a media word (`media`, `news*`, `journalis*`, `reporting`, `article*`, `headline*`, `outlet*`, `source*`, `LANGUAGE`, `editor*`). For example, *biased news* or *framing of the headlines*.
2. **An inequality or distortion word near an attention or representation word.** An inequality or distortion word (`inequality`, `unequal`, `distort*`, `imbalance*`, `skew*`, `selective`) within five words of an attention or representation word (`attention`, `discurs*`, `exposure`, `source*`, `prominence`, `content`, `representation`, `report*`). For example, *selective exposure* or *distorted representation*.
3. **A fixed coverage phrase:** "coverage bias", "coverage imbalance", "coverage skew", "unequal coverage" or "selective coverage".
4. **A bias-type term near a media word.** One of 33 terms for kinds of media bias and related phenomena ("agenda setting", both-sidesism, "cherry picking", clickbait, delegitimization, demonization, "differential prominence", disinformation, "echo chamber", emotionalism, "fake news", "filter bubble", "flawed logic", gatekeeping, "hate speech", "hedge words", misinformation, mudslinging, normalwashing, overrepresentation, propaganda, rumor, scandalization, scapegoating, sensationalism, spin, stance, stereotyping, stigmatization, underrepresentation, "unsubstantiated claims", vagueness, whataboutism) within seven words of a media word (the list of part 1). For example, *propaganda in the news*.

Three limits then apply to every result:

- **Subject areas.** A document that Scopus files under any of 17 subject areas is left out: Medicine (`MEDI`), Mathematics (`MATH`), Physics and Astronomy (`PHYS`), Materials Science (`MATE`), Biochemistry, Genetics and Molecular Biology (`BIOC`), Environmental Science (`ENVI`), Chemistry (`CHEM`), Earth and Planetary Sciences (`EART`), Energy (`ENER`), Agricultural and Biological Sciences (`AGRI`), Chemical Engineering (`CENG`), Pharmacology, Toxicology and Pharmaceutics (`PHAR`), Nursing (`NURS`), Health Professions (`HEAL`), Immunology and Microbiology (`IMMU`), Dentistry (`DENT`) and Veterinary (`VETE`).
- **Language.** English only (`LIMIT-TO ( LANGUAGE , "English" )`).
- **Access.** Open access only, of any kind (`LIMIT-TO ( OA , "all" )`).

The results were exported from Scopus in three RIS files, by year of publication: 1910–2020, 2021–2024 and 2025–2026. They are not in this repository ([What is not included](#what-is-not-included)).

## The seed list

[`data/seeds/seed-papers.tsv`](../../data/seeds/seed-papers.tsv): one row for each of the 3,869 seed papers, sorted by `paper_id`. It is tab-separated with the usual quoting (93 titles hold a double quote), so read it with a CSV reader set to the tab delimiter, for example `csv.DictReader(f, delimiter="\t")` or `pandas.read_csv(path, sep="\t")`. It holds identifiers, titles and labels, no text of a paper.

| Column | What it is |
|---|---|
| `paper_id` | The id the whole pipeline uses for the paper (`pdf-` and a hash), and the name of its TEI file (`<paper_id>.tei.xml`) in the seed run |
| `step2_id` | The id Step 2 gave the paper; the models' output files name their rows with it. It is not always unique (below) |
| `doi` | The DOI GROBID read in the TEI header, or empty. It is what GROBID read, not a checked DOI (below) |
| `title` | The title the extractor was given |
| `qwen_label`, `gemma_label` | What each model answered for this paper: `relevant`, `not relevant` or `can't determine`; empty if the paper's rows could not be told apart |
| `step2_route` | How Step 2 selected the paper: `agreed`, `tie-break` or `not-selected` |
| `tei_sha256`, `pdf_sha256` | The sha256 of the paper's TEI (the one the seed run read) and of its PDF |

**A worked example, the first row.** `pdf-00081f56…`, Step 2 id `doi_10.11114_smc.v13i4.8126`, DOI `10.11114/smc.v13i4.8126`, title *Media as Actors of Two-Stage Cultural Gatekeeping: Theoretical Framework and Interdisciplinary Context*. Both models answered `relevant`, so its route is `agreed` and nobody else had to decide.

**The route follows from the two labels.** Both `relevant` is `agreed`. Two different labels is `tie-break`: the third model kept the paper (it is in the hand-over folder). The list was made before the third model's answers were at hand, so for these papers it says only that the first two disagreed; `relevance export` ([relevance.md](relevance.md)) gives the third label. Anything else is `not-selected`: Step 2 did not select this paper itself, it came along under an id it shares with a paper that was selected (see below). For the 3,869 seeds:

| `qwen_label` | `gemma_label` | `step2_route` | Seeds | Example |
|---|---|---|---:|---|
| relevant | relevant | agreed | 2,947 | the first row, above |
| not relevant | relevant | tie-break | 901 | *Sharing Lies: A Socio-Technical Review on False Information Dissemination* |
| relevant | can't determine | tie-break | 7 | |
| can't determine | relevant | tie-break | 5 | |
| relevant | not relevant | tie-break | 4 | |
| not relevant | can't determine | tie-break | 1 | |
| not relevant | not relevant | not-selected | 1 | *The Reliance on Lifestyle Audits for Public Officials to Curb Corruption and Tax Evasion in Nigeria* |
| empty | relevant | not-selected | 1 | *Check-It: A plugin for Detecting and Reducing the Spread of Fake News and Misinformation on the Web* |
| empty | not relevant | not-selected | 1 | *Securing Manufacturing Using Blockchain* |
| empty | empty | not-selected | 1 | *In the frame: Signalling structure in academic articles and blogs* |
| | | **Total** | **3,869** | 2,947 agreed, 918 tie-break, 4 not selected |

**Why the ids are not always unique.** As far as the files show, Step 2 built a paper's id from the DOI that GROBID read in the PDF's header, or, where it read none, from a hash of the (empty) metadata. A DOI that was cut short (`10.17159/1727-`), a template placeholder (`10.1145/nnnnnnn.nnnnnnn`) or the DOI of a cited paper then gives different papers the same id, and every paper without a title or DOI got the one id `hash_da39a3ee5e6b4b0d` (17 Qwen rows and 78 Gemma rows). The hand-over folder is named by these ids. So a seed's two labels are found as follows (the rule is in the header of [`tests/make_seed_list.py`](../../tests/make_seed_list.py), and the test runs it on a miniature):

1. The PDF's file name starts with its Step 2 id. Exactly one id in the models' files has that spelling.
2. If neither model has more than one row under that id, those rows are the seed's.
3. If the id is shared, the rows whose title is the seed's title are the seed's.

A worked example of step 3: *Media Accessibility of Taboo in Arabic: The Analysis of a Political Film* has the id `doi_10.47476_jat.v8i1.2025`, which Step 2 also gave to *A Netnographic Study of Iranian Users' Reactions to Dubbing Censorship in Instagram Posts*. Each model has two rows under it, and the one with the seed's title says Qwen `not relevant`, Gemma `relevant`. For 3,607 of the 3,854 seeds whose id is not shared, the title the models saw is the seed's title; for the other 247 the models saw an abstract but no title (GROBID had found none), and the title was confirmed later. If no row of a shared id has the seed's title, that model's label is empty: Qwen has no row of its own for *Check-It* or *Securing Manufacturing Using Blockchain*, and neither model has one for *In the frame*, whose id is the no-title id that dozens of rows share.

**DOIs.** 3,385 seeds have a DOI and 484 have none (GROBID found none in the header). 3,350 of the 3,385 are plain DOIs; the other 35 have a full stop, bracket or semicolon stuck to them, a web address in front, or text glued on (`10.1080/13504630.2020.1787822,accessed14july2024.`). 224 seeds share one of 73 DOIs with another seed, so at most one seed of each group has that DOI as its own and the others carry the DOI of a cited paper (or are the same paper twice). Use `paper_id`, not the DOI, as the key.

**Titles.** 3,621 titles are the TEI header's. For another 248 papers GROBID found no title: 224 titles came from a DOI lookup in OpenAlex, 17 from the PDF's first page and 7 from the PDF's metadata, each accepted only when the title also stood on the first page of the PDF.

## The starting corpus

[`data/seeds/starting-corpus.tsv`](../../data/seeds/starting-corpus.tsv): one row for each of the 14,862 papers of Step 1, sorted by `corpus_id`: the paper's identifier (its TEI file name without `.grobid.tei.xml`) and the sha256 of that TEI file.

**A worked example.** The first row is `doi_10.1002_9781444314861` with sha256 `7af99dfb…`. A reader who converts the same PDF gets a file that is the same only if the GROBID version and settings are too, and the list tells you which file the seed run read. 2,458 of the 3,869 seeds read exactly a file of this list: their `tei_sha256` is one of this list's. The other 1,411 were converted for Step 3, with the GROBID 0.9.1 full image, so their TEIs are not in this list.

## Inputs

| Input | What it is |
|---|---|
| `runner-rows.json`, `seed-manifest.jsonl`, `existing-corpus-tei.sha256` | The seed run's recorded inputs: its 3,869 rows, the manifest of all 3,888 PDFs of the hand-over folder, and the hash of each starting-corpus TEI. They are in the data root: [`data/data-root.tsv`](../../data/data-root.tsv) lists them under `inputs/`, and [docs/setup/data-fetch.md](../setup/data-fetch.md) says how to fetch them |
| The two models' output files | One `normalized_predictions.csv` per model, with one row per classified paper. They hold the papers' abstracts, so they are not in git; they are in the data root's `relevance/answers/` ([relevance.md](relevance.md#inputs)) |

## Outputs

| File | What it is |
|---|---|
| [`data/seeds/seed-papers.tsv`](../../data/seeds/seed-papers.tsv) | The 3,869 seed papers (above); 1.4 MB |
| [`data/seeds/starting-corpus.tsv`](../../data/seeds/starting-corpus.tsv) | The 14,862 starting-corpus papers (above); 1.4 MB |

## What is not included

- **The search's results.** The records the database returned (titles, abstracts and metadata of the roughly fifty thousand documents) are covered by the database's licence. The search string is [above](#the-search-query), so a reader with access to the database can repeat the search.
- **The PDFs and the TEI texts.** They are the publishers' and authors' work. The lists carry the sha256 of each TEI (and of each seed's PDF) so that a copy you fetch and convert yourself can be compared with ours.
- **Code for the search:** there is none, the search is a manual query ([above](#the-search-query)). The starting corpus's PDFs were downloaded and converted with GROBID before Step 2; in this repository, [`collect`](collect.md) downloads and converts papers. The code that called the three models is the [`relevance`](relevance.md) step, and its prompts are in [`src/mbdef/relevance/prompts/`](../../src/mbdef/relevance/prompts/) ([relevance.md](relevance.md#the-prompts)).
- **The third model's label** of a `tie-break` seed: the list does not show it; `relevance export` does.
- **The models' answers** in git, for the reason above (they are in the data root).

## Known problems of the list

The list is what Step 3 extracted from, so it keeps the faults of the hand-over folder. None was repaired, because the released extraction ran on these papers.

- **3,888 rows and 3,888 PDFs are not the same papers one for one.** The folder is named by Step 2's ids, and some ids are shared by several papers, so its files are not the relevant rows one for one.
- **Four seeds were not selected by Step 2** (`not-selected`): they are PDFs that stood in for another paper with the same id. One PDF is also in the list twice: it was handed over under two file names, and the two seeds have the same title, `step2_id` (`doi_10.46687_rqth4999`) and `pdf_sha256` but different `paper_id` and TEI.
- **Eight papers on which both models agreed are not seeds,** and up to six tie-break papers may be missing too: they share an id with a seed, and the folder holds the other paper's PDF. The tie-break papers cannot be told apart without the third model's answers.
- **Fifteen papers have no seed row for want of a title.** Ten of them are papers on which both models agreed, five are tie-break papers.

## How it was checked

[`tests/test_seeds.py`](../../tests/test_seeds.py):

- **The files, as a reader sees them** (fast, run by CI): the columns; 3,869 seed rows with unique ids, a title and two 64-digit hashes each; the labels are the three recorded names or empty; each `step2_route` follows from its two labels, and the routes count 2,947, 918 and 4; the DOI counts above; 14,862 unique corpus identifiers with unique hashes; 2,458 seeds whose TEI is in the corpus list; no field is long enough to hold paper text.
- **Against the recorded inputs** (slow, needs the data root; run `uv run pytest -q tests/test_seeds.py` with `DATA_ROOT` set): the manifest has 3,888 rows, 4 notices and 15 papers without a title, and the 3,869 rows are exactly the rest; every seed's title, TEI hash, PDF hash and DOI is the recorded one; the Step 2 id is the one its PDF was named after; the corpus list is, row for row, the recorded TEI list; the seeds that share a TEI with the corpus are the ones the manifest says reused a saved conversion.
- **The maker on a miniature** (fast): a paper with its own id, a tie-break, a shared id resolved by title, a paper no row describes, and an id that matches no row (the maker stops).

The Step 2 numbers in the stage table (14,482, 14,562, 2,998, 5,074, 2,963, 2,196) and the counts of the known problems are counted from the two models' recorded answers (in the data root) and the tie-break manifest, which is not in this repository.
Counting the papers both models answered `relevant` by id and title gives 2,963; by id and the whole text of the title and abstract it gives 2,968. The seed list does not depend on either count.

## How to adapt it

To start the pipeline from your own papers, you need a TEI file for each, named `<paper id>.tei.xml`, and a title where GROBID found none: [`extract run`](../step-3-extraction/extract.md#inputs) reads them with `--teis` and `--titles`. The [`collect`](collect.md) commands download and convert PDFs the way the closure step does. Nothing in the pipeline reads `seed-papers.tsv`: the later steps read the seed run's rows and TEI files. The list is the record of which papers those were and why Step 2 picked them, and a table like it for your own papers (one row per paper: its id, DOI, title and the labels of your own relevance step) keeps that record. [`tests/make_seed_list.py`](../../tests/make_seed_list.py) is the join to start from: swap the two models' output files for your classifier's, and make sure the id in its rows is unique, which Step 2's was not.

## Not in this step

The extraction from the seed papers is [`extract`](../step-3-extraction/extract.md) (and [the seed run's variants](../step-3-extraction/extract.md#the-seed-runs-variants-extract-derive)); the works the seed papers cite are [`closure`](../step-4-closure/closure.md).
