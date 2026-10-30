# Closure loss audit: which missing papers are genuine, which were our fault (29 Sep 2026)

*Question from the owner: make sure we are not missing PDFs because of a technical error, but only because a citation is broken, a paper is not open access, or something else genuine. Every claim below comes from a measurement on real run data, a random sample (fixed seed) or a live re-query; the scripts are in `scripts/closure_ops/audit/`. How the closure works in general: [citation-closure-walkthrough.md](citation-closure-walkthrough.md).*

## 1. Verdict per loss

| Where papers are lost | Size | Technical or genuine? | Evidence | Action |
|---|---:|---|---|---|
| OpenAlex title search answered **"ambiguous"** | 1,453 rows (about 45% of all title searches) | **Technical (our rule).** The search matched title words anywhere in title, abstract or full text and the rule gave up whenever more than one page (200) of hits existed. 39 of 40 sampled rows were only a long result page; with a title-only filter exactly one record passed in 12 of 40 | re-query of 40 + 40 rows three ways | **Fixed** (title rule v2), all old answers re-asked |
| OpenAlex title search **"not found"** | 1,391 rows | Mostly genuine; about 8% technical (cited year one off the record) | same probe: 3 of 40 found with year ±1; closest titles show books, reports and garbled citations | Fixed within v2 (year ±1 only when no exact-year record) |
| Title matches that are **book reviews** | a few per hundred | Technical risk (wrong paper) | OpenAlex W2023116690 (a one-page 1967 review of *The Social Construction of Reality*) and W2016243413 (review of *Building Peace*) pass title + author + year ±1 | **Fixed**: the record's first author must be a cited author (costs 0 of 17 confirmed matches) |
| **PDF link returned a web page** | 533 works | **Partly technical**: 5 of 25 sampled pages declare the PDF in `citation_pdf_url`; most others are "checking your browser" pages of bot-protected hosts (MDPI, Springer, PMC, ZORA) | download recovery probe, 150 failures | **Fixed** (landing-page PDF link followed); bot protection is not worked around |
| **PubMed Central** PDFs blocked | 61 works | **Technical**: Europe PMC serves the same article through its official PDF rendering | 7 of 8 recovered | **Fixed** |
| **Transient errors** (5xx, timeouts, 202, network) | 167 works | **Technical**: 7 of 51 sampled succeed on a later try; each was tried at most twice | same probe | **Fixed**: up to three tries |
| **Blocked by the publisher** (403 / 405 / 418) | ~800 works | Genuine for us: open access, but the host refuses automated downloads (ACM, Sage, Wiley, Elsevier, OUP, T&F, …). Unpaywall knew **no** other copy for any of 126 sampled | probe T3/T4 | Not worked around. The Semantic Scholar stage below finds other copies for some |
| **Dead links / broken hosts** (404, TLS) | 73 works | Genuine | no alternative found in the sample | – |
| **Semantic Scholar never asked about works with a DOI** | 10,771 DOI works without a PDF | **Technical gap in our design**: Semantic Scholar was only used for works *without* a DOI. A batch check found open PDF links for 3,296; 2,165 of those URLs had never been tried (1,976 works OpenAlex and Unpaywall call closed or PDF-less; 389 failed downloads with a new URL) | batch lookup of all 10,771 DOIs; sample of 80: 5 downloaded, all genuine copies (repository, arXiv, journal site) | **Fixed**: new stage `s2doi.csv`; links from sites that re-host papers without permission (6 on `communicationcache.com`) are excluded |
| OpenAlex **server errors (503)** on title searches | 92 rows | Technical, self-healing (errors are never final; retried each cycle) | ledger | re-asked by the v2 pass |
| **Crossref found no DOI** | 7,892 DOI-less works | Mostly genuine: 5,925 have no record with the exact title; 1,967 have the title only in another year or without a cited author (sample: books cited by edition, e.g. *Nudge: The Final Edition*, Benjamin's *Illuminations*, Fiske & Taylor's *Social Cognition*) | reasons in `crossref.csv`, sample of 10 | Not relaxed: a wrong Crossref DOI would then pass the PDF identity check by that DOI. The papers among them are covered by the OpenAlex v2 rule |
| **Not open access** | 10,203 works | Genuine (paywalled) except where Semantic Scholar knows a green copy | batch lookup: 543 + 96 of them have a Semantic Scholar link | handled by the new stage |
| **Markers not linked** to a reference entry | 16,983 markers | Mostly genuine, see §3 | tests of two possible rescues, both rejected | not changed |
| **Wrongly linked markers** (short pointer text such as `2019)` or `[1,` counted as contained in `Figure 1` or `M.K. personal interview, 2019`) | 137 of 74,501 links (0.18%) | Technical, small | automatic author/year consistency check of every link | listed for a later fix; the fetched work is still one the paper cites |

## 2. Is linking correct? Automatic check of all 74,501 links

For every linked marker the marker's surnames and year were compared with the entry it was linked to:

| Linking method | Links | Author and year consistent |
|---|---:|---:|
| explicit GROBID pointer (`ref_target`) | 60,203 | 99.4% |
| author + year in the reference list | 5,883 | 100.0% |
| partial pointer text | 4,338 | 96.9% |
| several equally good entries (all kept, low confidence) | 1,898 | 100.0% |
| surname + year in the entry text | 1,609 | 89.7% |
| surname only | 316 | 100.0% |

Most "inconsistent" cases are GROBID misreading the *entry's* year, not a wrong link: `Carvalho and Burgess, 2005` → entry dated 1985 by GROBID; `Fanon, 2008` → the 1952 original of the 2008 edition. Those links are already marked low confidence and go to review, not to automatic fetching.

### Random sample of correct links (seed 20260929)

| Citing text (term) | Marker | Linked entry |
|---|---|---|
| agenda-setting theory of media | `McCombs and Shaw in 1972` | McCombs & Shaw (1972). *The agenda-setting function of mass media* |
| conspiracy theory | `DiFonzo, 2019` | DiFonzo & Uscinski (2019). *Conspiracy rumor psychology* |
| health sovereignty | `Piller et al., 2020` | Piller, Zhang, Li (2020). *Linguistic diversity in a time of crisis* |
| post-functionalist approach to European integration | `Hooghe and Marks 2009` | Hooghe & Marks (2009). *Postfunctionalist Theory of European Integration* |
| silence | `Galtung, 1969` | Galtung (1969). *Violence, peace, and peace research* |
| advocate frames | `Tewksbury et al., 2000` | Tewksbury, Jones, Peske (2000). *The Interaction of News and Advocate Frames* |
| fact-checking | `Rapeli, 2019` | Nieminen & Rapeli (2019). *Fighting misperceptions and doubting journalists' objectivity* |

### Wrong links found

| Marker | Linked to | Why |
|---|---|---|
| `Figure 1` | reference `[1]` (Sherief & Kulkarni 2018) | pointer text `[1,` is contained in "figure 1" |
| `M.K. personal interview, June 7, 2019` | a 2019 entry (Papadopoulou) | pointer text `2019)` is contained in the marker |
| `foot_7` | reference `[7]` | a footnote number, not a reference |

## 3. Markers that stay unlinked: genuine or technical? (random sample)

| Reason | Markers | Example | Assessment |
|---|---:|---|---|
| a number, but no pointer or entry matches it | 6,223 | `136` in a paper with 122 parsed entries | Mostly genuine (footnotes, entries GROBID dropped). **Tested rescue "number N = N-th entry"**: holds for 82% of linked numeric markers, but in the 57 papers where it held without exception it still linked a definition of machine learning to a Spanish patient-safety study (`7`) and a definition of deepfake to an unrelated paper (`40`). Rejected |
| the reference exists only as raw, unparsed text | 3,287 | `Entman 2007` → "EntmanRobert Journal of Communication 57 2007" | Mostly non-scholarly sources (CAAD 2024, *Cambridge Dictionary*, United Nations, Kofi Annan Foundation, WHO, European Council). **Tested rescue with Crossref free-text matching**: right work for 3 of 25 (Entman 1993, Frankfurt 2006, Mehrabi 2021), one false match (*Knight Foundation 2009* → a neurology paper by Knight). Rejected for automation; a candidate for manual review |
| surname and year not in the reference list | 3,227 | `Barthes, 1996: 258-259`, `Charaudeau, 2006, pp. 36-38` | genuine: the paper cites an edition or work it does not list |
| a bare year with several entries of that year | 2,181 | `1983` (third-person effect) | genuine ambiguity |
| no year and no surname | 1,178 | `Editor 3, C13`, `Herzog; Sohn, "The Co-mingling…" 7` | genuine |
| a page number only | 607 | `p.114` | not a citation |
| the paper has no reference list in its TEI | 270 | `33` | GROBID found no bibliography (a PDF quality issue) |

## 4. Samples of where cited works ended up

| Outcome | Example | Why |
|---|---|---|
| collected, identity by DOI | *Addressing disparities in academic medicine: What of the minority tax?* (2015) | DOI in the reference list → OpenAlex → publisher PDF |
| collected via Crossref | *Populists in government: Normal or exceptional?* (Askim et al. 2022) | no DOI in the reference; Crossref found 10.1017/gov.2021.30 |
| collected via the LLM check | *News coverage and social protest…* (McLeod 2007) | garbled PDF header, verified by the checked LLM verdict |
| now found by the v2 title rule | *A Survey of Fake News: Fundamental Theories, Detection Methods, and Opportunities* (2018, arXiv, has PDF); *Accounting for the force of the appeal to authority* (2011, has PDF) | v1 called them "ambiguous" (288 and 12,206 full-text hits) |
| not open access | *Framing: Towards clarification of a fractured paradigm* (Entman 1993), cited by 131 seed papers | paywalled at the publisher; no green copy known |
| open access but blocked | *A 1980 Letter on the Risk of Opioid Addiction* (NEJM) | HTTP 403 from the publisher |
| downloaded, correctly rejected | book DOI 10.1007/978-3-030-45046-5 requested, PDF is its chapter `…_9` | DOI conflict |
| downloaded, not confirmed | *Virtual violence: cyberspace, misogyny and online abuse* | right title, different author (Todd vs. cited Noble, Owen, Speed) |
| no record found | *Pierre Bourdieu: A critical introduction to media and communication theory*; *A Bringing Model for Media Research* (garbled "Bridging"); *Communication Studies* (a journal name given as title) | book / garbled citation / wrong title |

## 5. What the fixes changed (measured 29 Sep, 09:00–10:00 UTC)

- **Title rule v2**, on the first 1,758 rows re-asked: 286 identified (52 with a PDF, 204 not open access, 28 open access without a PDF link); "ambiguous" down from about 45% to 5%. About 1,081 re-asks and 5,900 first searches remain, 5 keys × 1,000 searches a day, so they finish about 1 Oct 00:00 UTC.
- **Download recovery** (landing pages, retries): 72 more PDFs at once (3,627 → 3,699).
- **Semantic Scholar by DOI**: 1,976 works newly with candidates plus 389 retries; running.
- New PDFs go through the same GROBID conversion, identity check (including the LLM check) and extraction as before, as a new batch.

## 6. Still open (known, small or genuine)

- The short-pointer linking defect (137 links); fix in `src/closure/citations.py` partial rung (require a real author/year surface string on both sides).
- Raw-only references (3,287 markers) could be reviewed by hand with the Crossref free-text candidate shown.
- About 800 open-access works stay behind publisher bot protection; a person can download them in a browser if they matter (the list is `download.csv` rows with `http_403` / challenge pages).

## 7. Second audit, 30 Sep 2026: seven more technical losses, all fixed

The owner asked again to make sure nothing is missing for a technical reason. Every failure class with more than a
handful of works was re-examined on the live ledgers; seven turned out to be our fault (the Semantic Scholar deadlock is
listed in the same row as the Crossref one it mirrors). Each fix has a test
(`tests/test_production_acquire_keys.py`, `tests/test_grobid_llm_identity.py`).

| Where papers were lost | Size | What was wrong | Evidence | Fix |
|---|---:|---|---|---|
| **Europe PMC answered 429 ("too many requests")** | 268 works | The download-recovery stage (29 Sep) sent Europe PMC's PDF rendering 769 requests in a two-hour burst; every retry of those works fell inside the same burst, so each work used up its three tries while rate-limited | `download.csv`: all 769 refusals between 09:00 and 11:00 UTC on 29 Sep; 79 Europe PMC downloads in the same hours succeeded | Europe PMC now gets one request at a time, at least 3 s apart, with a one-minute pause after a refusal; its 429 / 5xx failures are retried at least an hour apart, up to 10 attempts. First spaced pass: 84 of 191 downloaded |
| **Title-only works never searched again** | 155 works | The v2 title rule (29 Sep) re-asks old "ambiguous" / "not found" answers, but only after Crossref has looked at the work; the Crossref stage skipped every work that already had an OpenAlex answer. Neither stage ever took them (all 155 have titles, e.g. *23 ways to nudge*, 2019) | driver state: `awaiting_crossref` stuck at 155; every one had a stale v1 answer and no Crossref row | Crossref now takes such works; on restart it matched 44 of them (DOI found). The Semantic Scholar stage, next in line, skipped the other 107 for the same reason (found three hours later: `awaiting_s2` stuck at 107); fixed the same way |
| **Semantic Scholar link failed, no fallback** | 63 works | Works without a DOI that Semantic Scholar matched with a PDF link were only ever downloaded from that link; when it failed (34 web pages instead of a PDF, 11 publisher blocks, …) the OpenAlex title search was never tried. One match was even a different paper (*Algorithms for inverse reinforcement learning* linked to a veterinary journal) | queue rows with an S2 match, a failed download and no OpenAlex answer | a failed Semantic Scholar PDF now sends the work to the title search |
| **Correct PDFs rejected for a "DOI conflict"** | 79 of 138 rejected PDFs | The PDF identity check rejects a PDF whose DOI differs from the requested one. GROBID often reads a DOI that says nothing about the work: publisher template DOIs left in the PDF (`10.1145/nnnnnnn.nnnnnnn`, ACM's sample DOI `10.1145/1122445.1122456` on 8 PDFs, `10.1109/access.2017.doi`), the right DOI with journal text glued on (`10.1177/1464884915610988jou.sagepub.com`), punctuation variants, the DOI of the paper's dataset (Zenodo, figshare, Dataverse) or of its preprint | all 138 conflicts converted again and compared with the citation (`results/closure-audit-20260930/doi-conflict-probe.json`): 79 carry the cited title and a cited author | such DOIs are no evidence either way: the title, year and author checks (and then the language-model check) decide. A chapter of a requested book, a later correction, a co-published statement or another article's DOI still rejects the PDF (8 of the 79 stay rejected for that reason) |
| **Cited works counted too often** | 1,338 works | The closure audit merges references to the same work (same DOI; same title, year and first author; same OpenAlex record), but its merge step stopped after the first join, so groups of three or more were only partly merged (Entman 1993, *Framing: Toward clarification of a fractured paradigm*: 339 reference groups, counted as 338 works) | an independent re-implementation of the audit's grouping reproduced its 26,617 works exactly with the same early stop, and 25,279 without it | fixed; there are **25,279** distinct cited works. Nothing was downloaded twice or lost (downloads and batches deduplicate separately), but per-work counts such as "cited by N seed papers" were too low |
| **A work skipped by its own representative** | 7 works | Rows of one cited work are asked once: the representative row, the others wait. Two reference groups can produce the same queue row, so a work listed its representative twice, and the representative skipped itself; nobody was asked (e.g. de Vreese 2005, *News framing: Theory and typology*; Gamson & Modigliani 1989) | the last 7 works marked "unattempted": all had duplicated queue rows | fixed and tested; the 4 with a DOI are not open access, the other 3 wait for a title search after the daily budget resets |

Checked and found genuine (no change):

| Failure | Works | Why it is genuine |
|---|---:|---|
| GROBID returned no usable text | 21 PDFs | scans without a text layer (*Dilemmas in a General Theory of Planning*: 32 pages, under 300 characters of text) or one- and two-page flyers / abstracts; would need OCR |
| HTTP 202 from figshare, De Gruyter, some repositories | 161 works | these hosts answer automated requests with a browser challenge; not worked around |
| network errors, timeouts, server errors at other hosts | about 150 works | each tried four times over three days; the hosts are down or unreachable |

## 8. Final state at the release freeze (30 Sep 2026)

The acquisition was stopped on 30 Sep at 11:32 UTC; the release (laptop `~/definition-corpus-v1`, ZIB
`~/taxonomy/definition-corpus-v1`) and the group-cloud `Closure` folder hold this state. The definitions of the seed
papers point to **25,268 distinct cited works** (after the corrected merging of section 7); each ended as:

| Outcome | Works | What it means |
|---|---:|---|
| collected, confirmed, extracted | 4,518 | the **4,503 closure papers** (15 works that the grouping kept apart turned out to be the same PDF) |
| already a seed paper | 499 | nothing to fetch |
| identified, no open-access PDF | 9,931 | paywalled, or open access without a PDF link anywhere we can reach |
| not identified with certainty | 6,591 | 6,200 not found, 391 ambiguous: books, reports, web pages, garbled or incomplete citations |
| open-access link known, download failed | 1,294 | publisher bot protection (403 / challenge pages), dead links, and about 100 works whose Europe PMC rendering still answered 429 or 5xx at the freeze |
| PDF downloaded but rejected | 304 | 232 not shown to be the cited work (the language-model check included), 51 another work's PDF (it carries that work's DOI; e.g. a Spanish article citing Gamson & Modigliani 1989 offered as its open copy), 21 without usable text |
| linked only with low confidence, not fetched | 2,131 | ambiguous or contradictory links (section 2) |

How the 4,503 PDFs were confirmed to be the cited work: DOI 2,189; same title and a cited author with the year
missing 1,093; same title, year and author 547; checked language-model verdict 439 (two rounds on 29 and 30 Sep);
year one off 211; title and author opening the text 24.

Left out of this release on purpose: 3 title searches that the day's budget did not cover, and the Europe PMC works
above (their retries would have continued past the freeze).

## 9. Link-table fix (1 Oct 2026, after the release)

**Bug.** The exporter of `links/definition-links.csv` (and the closure audit's per-work counts) looked up each citation's
string in the item's citation list by `citation_index`. That index numbers the item's *resolved markers*: one string can
hold several markers ("Jamieson y Cappella, 2008;Sunstein, 2009"). The string a marker came from is stored next to it as
`raw`. The wrong lookup had three effects:
- **Lost rows:** 2,761 of the 78,242 citation occurrences had no row, so 39 closure papers had no citing row at all. The
  README's "0 without" counted something else: closure papers without a cited work.
- **Wrong strings:** other rows carried a sibling string of the same item, so their `in_P20` … `in_P28_single_model`
  flags could be wrong.
- **Low counts:** per-work counts of citing definitions, items and variants were too low.

**Not affected:** which works were looked up and fetched, their statuses, the 4,503 papers, and every definition and TEI.

**Fix.**
- The string is taken from `raw` and must occur in the item's citation list. An occurrence that cannot be placed is now
  counted in `occurrences_not_located` instead of being skipped silently.
- The README checks citing rows directly.
- The grouping check against the audit now compares work keys, merged groups and citing seed papers (identical), and
  reports the corrected counts separately.

**After the fix:**
- 78,242 link rows for 78,242 occurrences; 0 occurrences not placed;
- 0 closure papers without a citing row;
- corrected counts for 869 works (citing definitions), 1,105 (citing items) and 423 (citing variants);
- closure papers cited inside a definition sentence: 3,589 (was 3,552).

**One paper, sixteen works.** The closure paper table took each paper's citing counts from one of its cited works.
That is right for 4,502 papers. The other one is Mikolov et al. 2013, *Efficient estimation of word representations in
vector space*. It has no DOI, so its 16 citations stayed 16 separate works, and the counts came from a work with no
citing definition: 1 citing seed paper, 0 definitions. The table now adds up all the works of a paper. For papers with
one work, the build asserts that nothing changes. Mikolov's row now lists all 16 work keys and the seed papers and
definitions that cite any of them. As a result, the README count went from 3,588 to 3,589.

**Release.** Rebuilt with only the link files, the closure paper table, the README and the provenance changed. The ZIB
copy was updated and verified against the new manifest. The group-cloud `Closure` folder now also holds the link tables
(`Links/`) and the paper tables (`Papers/seed-papers.csv`, `Papers/closure-papers.csv`), all checksum-verified.

## 10. Version 1.1 (1–2 Oct 2026): DOIs on the wrong reference entries, works from the earlier corpus, copies

Found while checking the final numbers of the methods document; every item was measured on all data and has a test
(`tests/test_closure_misplaced_dois.py`, `tests/test_release_exclusions.py`, `tests/test_definition_links.py`).

| What | Size | What was wrong | Fix |
|---|---:|---|---|
| **DOIs on the wrong reference entry** | 1,146 of 191,819 entries in 298 seed papers | GROBID copies the citing paper's own DOI (header or page footer) into reference entries, or one entry's DOI into the next. The corpus check groups references by DOI first, so such a DOI merged unrelated works: 53 groups mixed different works, 393 citations sat under another work, and 117 works were never looked up (e.g. Wardle & Derakhshan 2017, *Information disorder*, 181 citations, hidden in a group keyed by a citing paper's own DOI) | `src/closure/bibliography.py` ignores a DOI that several clearly different entries of one reference list share (1,047) and the paper's own DOI on an entry that is not the paper itself (99). The linking and corpus check were rebuilt with it; the rebuild without the fix reproduces the original outputs exactly. Mixed groups fell to 31 and misgrouped citations to 118, mostly chapters under their book's DOI, spelling variants and garbled titles. 176 new lookups, 91 lookups no longer needed |
| **Works found only in the earlier corpus** | 67 works, cited by 105 seed papers | The corpus check counts a work as present if it is in the seed corpus *or* in the group's earlier corpus (`existing-corpus-tei.zip`); works present only there were never fetched and, not being seed papers, never extracted | their TEIs (checked against the corpus's SHA-256 list) became closure papers with receipt type `existing_corpus`: 65 after removing 2 pairs that were one TEI, 64 extracted (one TEI had no body text) |
| **Copies in the closure set** | 39 papers | text comparison of every collected paper with every seed paper and every other collected paper (8-word sequences, 80% shared): 15 collected papers are seed papers whose own headers were too garbled for the corpus check; 24 are a second copy of another collected paper (arXiv and published version, two rows of one work) | left out of the release; their citations count for the seed paper or for the copy kept (always one already extracted) |
| **Collected papers nobody cites** | 9 papers | collected in v1 through a DOI that GROBID had put on the wrong reference entry (e.g. a book chapter on the Gulf of Maine); once that DOI is ignored, no seed reference cites them | left out of the release |
| **Papers that lost their citing row** | 2 papers | the regrouping gave their work a new, title-based queue row | relinked by title (≥ 95% alike, one candidate only) |

Checked and left as limits: 118 citations still grouped under another title (see above); 224 seed papers whose metadata
DOI (from the seed collection, not used by the closure) is shared with another seed paper.
