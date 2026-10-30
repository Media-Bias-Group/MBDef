# Ground truth: `mbdef groundtruth`

| | |
|---|---|
| **Paper step** | Step 3 (information extraction): its ground truth, MBDef-Gold |
| **Code** | [`src/mbdef/groundtruth.py`](../../src/mbdef/groundtruth.py), with the report template [`groundtruth_report.tex`](../../src/mbdef/groundtruth_report.tex); the publishable copies of its inputs: [`pseudonymise.py`](../../src/mbdef/pseudonymise.py). The release's fixed texts (guide, Markdown report, figures, notes on the earlier ground truths) are files of MBDef-Gold, not of the code |
| **Commands** | `mbdef groundtruth build --out FOLDER`; `mbdef groundtruth pseudonymise --out FOLDER` ([publishable copies](#publishable-copies-groundtruth-pseudonymise)) |
| **Checked by** | the tests [`tests/test_groundtruth.py`](../../tests/test_groundtruth.py), with one test run by hand on the real data root (`-m slow`) |

## What the step does

Two annotators judged each of the 2,488 records of the synthetic V3 dataset in Label Studio: is it a definition, and is it relevant to media bias? A third annotator settled their disagreements, and a curator reviewed and edited the records. The curator's tool exported its final state as one zip, `validated-v3.zip`. This step turns that export into the release:

1. It reads the raw Label Studio exports again (`AnnotationExports`) and checks that the export agrees with them and with V3: the same 2,488 records, every decision reviewed, every record well formed, every kept record placed where the labels say.
2. It writes the **relevant set** (both answers *Yes*) and the **non-relevant set** (definition *Yes*, relevant *No*). The relevant set leaves out the 20 papers that kept no record.
3. It counts what changed between V3 and the final records, writes the statistics, fills the LaTeX report template and compiles it with Tectonic, then writes a hash manifest, the release zip and a receipt.

**A worked example: one record's path.** Record `3457607:2` is the term *Behavioral Bias* in paper
3457607. Both annotators answered *Yes* to both questions, so it goes into the relevant set. The curator completed its definition: V3 had "different user behavior across platforms, contexts, or different datasets"; the release has "Behavioral bias arises from different user behavior across platforms, contexts, or different datasets". It is one of the 174 edited records, and the report shows it as its first example of an approved edit.

Where the 2,488 V3 records went in MBDef-Gold:

| Result | Records | Definitions |
|---|---:|---:|
| Relevant set, `validated_gt.json` (130 papers) | 1,003 | 1,192 |
| Non-relevant set, `validated_gt_non_relevant.json` (150 paper entries, 135 with records) | 1,325 | 1,507 |
| Not a definition | 158 | |
| Definition judgement incomplete, both annotators said not relevant | 2 | |

## Run it

The two zips and the text snapshot of the annotation's method notes are not in git, because they may name the annotators. Pseudonymised copies of the two zips, which name no one and build the same data ([Publishable copies](#publishable-copies-groundtruth-pseudonymise)), are in the data root ([docs/setup/data-root.md](../setup/data-root.md)), and so are MBDef-Gold's fixed texts that `build` copies into the release. With `DATA_ROOT` set in `.env` and Tectonic installed:

```bash
uv run mbdef data fetch groundtruth/ MBDef-Gold/   # once: download the copies and MBDef-Gold, and check them
uv run mbdef groundtruth build --out /tmp/gt-release
```

The originals are not public, so a data root that holds only the pseudonymised copies is the usual case: `build` then reads the copies by default and writes MBDef-Gold's data files, the report and the README byte for byte. Three files name the copies instead of the originals: `validated_gt_stats.json` (the copy's sha256 `db6bc11b…` where the released one names `27ac4f29…`), `validated_gt_manifest.json` (the hashes of the export, of the statistics, of the code and of the six `provenance/` members, which are the copies, and no `history/IE-wiki-snapshot.txt`, which is not public) and so the release zip `validated_gt_release.zip` ([Publishable copies](#publishable-copies-groundtruth-pseudonymise)).

It takes about 1.5 s and prints a summary:

```json
{
  "primary": 1003,
  "non_relevant": 1325,
  "primary_definition_statements": 1192,
  "changed_primary_records": 174,
  ...
}
```

`build` writes only into `--out`, which must be new or empty; it never touches `data/` or the data root.

## Inputs

| Input | Where | sha256 | What it is |
|---|---|---|---|
| `validated-v3.zip` | `$DATA_ROOT/groundtruth/` | `27ac4f29…` | The curator's export: 12 members (not public). The build stops if its hash is neither this one nor its pseudonymised copy's. |
| `Annotations.zip` | `$DATA_ROOT/groundtruth/newAnnotations/` | `09192c66…` | The raw Label Studio exports, read again to check the export (not public) |
| `validated-v3.zip`, `newAnnotations/Annotations.zip` | `$DATA_ROOT/groundtruth/pseudonymised/` | `db6bc11b…`, `764c356e…` | Their [pseudonymised copies](#publishable-copies-groundtruth-pseudonymise), read by default when the two originals are not in the data root (or passed with `--export` and `--annotations`) |
| `IE-wiki-snapshot.txt` | `$DATA_ROOT/groundtruth/newAnnotations/raw/` | `a7d07031…` | A text snapshot of the method notes of the information-extraction annotation, only copied into the zip (skipped if absent; not public) |
| `ground_truth/v3.json`, `v2.json`, `scidef.json` | `data/` | | V3 (the records annotated) and the two earlier versions, for the statistics and the zip |
| `frozen/labelStudioXML.md` | `data/` | | The labelling interface; must equal the export's copy |
| `validated_gt_README.md`, `validated_gt_report.md`, `figures/`, `history/data-README.md` | `$DATA_ROOT/MBDef-Gold/` | | MBDef-Gold's fixed texts: the guide, the hand-written report and its figures, and the notes on the earlier ground truths; copied into the release as they are (`data fetch MBDef-Gold/` downloads them, [`data/data-root.tsv`](../../data/data-root.tsv) pins their hashes) |
| Tectonic 0.17 | system | | Compiles the report (it downloads its TeX bundle on first use) |

The flags `--export`, `--annotations`, `--wiki` and `--data` point at other copies of these inputs.

## Outputs

The published outputs are the dataset MBDef-Gold.
It is on Hugging Face (`mediabiasgroup/MBDef-Gold`, the repository named in `HF_DATASETS` in [`files.py`](../../src/mbdef/files.py)), not in git: `uv run mbdef data fetch MBDef-Gold/` downloads its records, statistics, manifest, guide, report, figures and notes on the earlier ground truths into `$DATA_ROOT/MBDef-Gold/` and checks each file's sha256 ([docs/setup/data-fetch.md](../setup/data-fetch.md#the-two-datasets)).
Every command that reads the ground truth (`extract train-dataset`, `extract train-filter`, `embed build`, `evaluate agreement`, Step 5's `postprocess` commands, `dataset build` and `dataset build-gold`) reads it there, unless a flag names another file. MBDef does not hold the ground truth; `dataset build-gold` writes MBDef-Gold's train, validation and test files and its dataset card ([docs/dataset/dataset.md](../dataset/dataset.md#mbdef-gold-dataset-build-gold)).
The release's fixed texts are MBDef-Gold files too: `build` reads them from `$DATA_ROOT/MBDef-Gold/` and copies them into `--out` as they are.

| File in `--out` | What it is |
|---|---|
| `validated_gt.json` | The relevant set: 130 papers, 1,003 records |
| `validated_gt_non_relevant.json` | The non-relevant set, byte for byte as exported |
| `validated_gt_stats.json` | Every count of the report, and the annotation statistics |
| `validated_gt_report.tex`, `.pdf` | The release report; the PDF carries the pinned date, so it is the same on every run |
| `validated_gt_README.md` | The release guide, copied from MBDef-Gold |
| `validated_gt_report.md`, `figures/` | Copies of the hand-written report and its figures |
| `newAnnotations/raw/final-export-20260908/` | The export, unpacked (12 files) |
| `validated_gt_manifest.json` | sha256 and size of every member of the zip except itself |
| `validated_gt_release.zip` | Everything above plus the earlier versions, the notes on them (`history/data-README.md`) and the code that built it (31 members), with `SHA256SUMS` |
| `receipt.json` | The settings, the hashes of every input and output, the code commit and the time |

## How it was checked

A rebuild from the original exports wrote the two datasets, the statistics, the guide, the report (LaTeX and PDF), the copied report and figures and the unpacked export byte for byte as MBDef-Gold holds them, with the same counts (130 papers, 1,003 relevant records, 1,192 definitions, 1,325 non-relevant records, …).
Its manifest differs from MBDef-Gold's only in the entry of the zip member `code/groundtruth.py`: that member is the builder itself, so its hash changes whenever the code does, and MBDef-Gold's manifest records the builder as it was when MBDef-Gold was built. Every data member matches.
[docs/setup/pinned.md](../setup/pinned.md) explains why a rebuild can differ.

For version 1.0.0 on Hugging Face the release was built again with its texts written for MBDef-Gold: the guide, the Markdown report, the figure notes and `history/data-README.md` name the dataset MBDef-Gold and no internal paths or dates, the figures are `figures/MBDef-Gold-*.png` (the same images), the report template names MBDef-Gold, and the release id is `MBDef-Gold-1.0.0`. The two record files came out byte for byte as before; the statistics differ only in the release id.

`test_the_pinned_copies_rebuild_mbdef_gold` (marked `slow`) repeats the check on the real data root from the pseudonymised copies ([below](#how-the-copies-were-checked)).
The fast tests build a two-paper export made up for the test (`tests/mini_groundtruth.py`, with made-up annotator ids) and check the datasets, the statistics, the zip, its manifest and the receipt. They need no network, no GPU and no Tectonic: a stand-in writes the PDF.

## How to adapt it

- **A new export of the same kind** (another annotation round, or definitions of another concept with the same 12-member export): pass `--export` and its hash with `--export-sha256`. The pinned hash is a guard, so that nobody releases an export by accident.
- **The release's words and fixed choices** are written for MBDef-Gold. For a new release, change them: the fixed texts in the data root's `MBDef-Gold/` folder (the guide `validated_gt_README.md`, the Markdown report, its figures and `history/data-README.md`), and in the code the report template (`groundtruth_report.tex`, which states counts in prose), the two records the report shows as examples (`EDIT_EXAMPLE`, `CITATION_EXAMPLE`), the release id (`RELEASE_ID`), and the PDF date and zip member time (`REPORT_DATE_EPOCH`, `ZIP_MEMBER_TIME`).
- **The rules that select records** (both answers *Yes* for the relevant set; *Yes* and *No* for the non-relevant set) are the two predicates in `analyze`.
- **After a new release**, upload the files you publish from `--out` to MBDef-Gold's Hugging Face repository, with the split files and dataset card that `dataset build-gold` writes from the new record files ([docs/dataset/dataset.md](../dataset/dataset.md#mbdef-gold-dataset-build-gold)), then put their new sha256 and sizes into the `MBDef-Gold/` rows of `data/data-root.tsv` and pin the new revision in `files.HF_DATASETS` ([docs/setup/data-fetch.md](../setup/data-fetch.md#how-to-adapt-it)).
  Then the steps that read the ground truth must be rerun: extraction scoring and evaluation (Step 3) and everything after it.

## Publishable copies: `groundtruth pseudonymise`

The two zips cannot be published as they are: they may name the annotators, they hold local paths, and the curator's export holds paragraphs of the papers. `groundtruth pseudonymise` writes copies of both that name no one and hold no local path and no paper text, and keeps everything `build` and [`evaluate agreement`](evaluate.md) read. `build` accepts the copy of the export without `--export-sha256`, because its hash is pinned beside the original's (`PSEUDONYMISED_EXPORT_SHA256` in `groundtruth.py`). The command needs the two originals, which are not public: for anyone else it documents how the public copies were made, and the copies themselves come with `data fetch`.

```bash
uv run mbdef groundtruth pseudonymise --out "$DATA_ROOT/groundtruth/pseudonymised"   # about 9 s
uv run mbdef groundtruth build --out /tmp/gt-from-copies \
  --export "$DATA_ROOT/groundtruth/pseudonymised/validated-v3.zip" \
  --annotations "$DATA_ROOT/groundtruth/pseudonymised/newAnnotations/Annotations.zip"
```

(`$DATA_ROOT` here is the shell's; set it to the folder `.env` names.) By default `pseudonymise` reads the two originals from the data root; `--export` and `--annotations` point at others.

### What it changes

Four rules, in [`pseudonymise.py`](../../src/mbdef/pseudonymise.py). On MBDef-Gold's exports the command printed how often each applied:

| Rule | Where | On MBDef-Gold's exports |
|---|---|---|
| 1. The user name and e-mail of an unsubmitted draft become the pseudonym of its Label Studio account, `account-<number>` | the raw exports, and their copy in the curator's export | 1 draft (task 512 of the weekly project 15): `user` becomes `account-<n>` and `created_username` becomes `account-<n>, <n>` |
| 2. The local path of each paper's PDF becomes `papers/<file name>` | `pdf-sources.json` (150 papers) and `final-touches.json` (37 touched records) | 187 paths; paper 3457607's becomes `papers/3457607.pdf` |
| 3. The paper passages behind the assistant's proposals are removed; their block numbers stay | `curation-suggestions.json` | 552 passages of 372 records (about 621,000 characters of the papers' text) |
| 4. The hashes that name what changed are recomputed | the fingerprint of the raw exports, and the final audit's seals | 3 fingerprints (`archive_sha256` in `curation-provenance.json`, `curation-suggestions.json`, `final-touches.json`) and 77 seals: the touch hash of 37 decisions and of 38 entries of their history, the audit's own hash and the provenance's copy of it |

The export's `README.txt` gets one paragraph that says this. `curation-suggestions.json` keeps its `sha256`, which names the file the curator worked with (the curation app hashed that file, so the copy cannot match it). Nothing else changes: every record, label, vote, annotator note and decision, the papers' metadata and SciDef's definition texts stay as they are. A changed JSON file is written back in its own format (Label Studio's compact JSON, the curation app's indented JSON), and the zips keep their member order, times and compression, so the copies differ from the originals only in the values above: 21 of the 22 entries of the raw exports' zip (19 files and 3 folders) and 6 of the 12 members of the curator's export are the originals byte for byte.

**What stays: the account numbers.** Label Studio names each annotator by an account number (`completed_by`, `updated_by`). The copies keep these numbers: they are not names, the released statistics (`validated_gt_stats.json`, `newAnnotations/annotation-audit.json`) and the per-account numbers of the agreement step already use them, and the organisers' share in `evaluate agreement` is counted from accounts 2 and 3. After rule 1, nothing in the copies links a number to a person.

**The check before writing.** Every member of both copies is scanned for an e-mail address, a local path (a home or user folder, a path from `~`, a Windows drive and the like) and a name field (`user`, `username`, `created_username`, `email`, `first_name`, `last_name`) that does not hold a pseudonym. If anything is left, nothing is written and the command names where it is, for example:

```text
Not written: the copies would still name a person or a local folder (an e-mail address in validated-v3.zip: README.txt). Add a rule for it (docs/step-3-extraction/groundtruth.md).
```

The snapshot of the method notes is not copied: `build` skips it when it is absent, and the release zip then has 30 members instead of 31 (no `history/IE-wiki-snapshot.txt`).

### Its outputs

| File in `--out` | sha256 (MBDef-Gold's exports) | Bytes | What it is |
|---|---|---:|---|
| `validated-v3.zip` | `db6bc11b909a273d…` | 3,315,131 | The curator's export, pseudonymised; it holds the copy below as `annotations-original.zip` |
| `newAnnotations/Annotations.zip` | `764c356ea79d19dc…` | 7,440,186 | The raw Label Studio exports, pseudonymised |
| `receipt.json` | | | The rules, the counts above, the hashes of the inputs and outputs, the code commit and the time |

[`data/data-root.tsv`](../../data/data-root.tsv) lists the two copies under `groundtruth/pseudonymised/` as files of the public deposit (`https:`), which `data fetch groundtruth/` downloads ([docs/setup/data-fetch.md](../setup/data-fetch.md)).

### The curation's files in `data/newAnnotations/`

`data/newAnnotations/` holds four files of the curation. `annotation-audit.json` is the machine-readable audit of the annotation round, which `evaluate agreement` reads for a cross-check ([docs/step-3-extraction/evaluate.md](evaluate.md)). The other three are working files of the curation app, published as copies with the rules above; the originals are not public, because they quote the papers or name local paths. Each copy changes only these values and keeps the app's own JSON format (indented, with a final newline):

| File | Bytes | sha256 | What it is |
|---|---:|---|---|
| `curation-suggestions-without-passages.json` | 556,426 | `49fd4bb2…` | The assistant's proposals with rule 3. The app's file (1,190,418 bytes, `de9b994c…`, the hash the export's `curation-suggestions.json` names) quotes 552 passages of 372 records, 620,880 characters in all; in the copy each source keeps only its block number. For example, record `016f7a07…:5` (*ecological fallacy*) has one source, block 62 with a passage of 3,693 characters; in the copy that source is `{"block": 62}`. |
| `final-touches-without-local-paths.json` | 226,092 | `b70736b8…` | The final audit's touches with rule 2: the PDF of each of the 37 touched records is named `papers/<file name>` (record `3457607:0`: `papers/3457607.pdf`) instead of its path on the machine the curation ran on |
| `pdf-sources-without-local-paths.json` | 76,337 | `b08ceeb7…` | The 150 papers' PDFs (`path` is already `papers/<file name>`); where each PDF was copied from (`source_path`) keeps only its file name (paper `73bfad11…`: `paper_73bfad11….pdf`) |

No command reads these three files: `build` and `evaluate agreement` read the curator's export. A rebuild with `--data` set to a copy of `data/` that holds the originals instead of the three copies wrote the same files, apart from the receipts, the SVG figures' time and random ids, and the data folder's own path that `agreement.json` lists among its inputs.

### How the copies were checked

On a CPU:

- Two runs of `pseudonymise` wrote the same bytes (`db6bc11b…` and `764c356e…` both times).
- `build` from the copies, without the snapshot of the method notes and without `--export-sha256`, wrote `validated_gt.json` (`afce3c8b…`), `validated_gt_non_relevant.json` (`02502de2…`), the guide, the report (`.tex` and `.pdf`, `9d942972…`), the copied report and the figures byte for byte as released, and as `build` writes them from the originals. `validated_gt_stats.json` is equal except `export_sha256`, which names the copy (`db6bc11b…` instead of `27ac4f29…`). Of the release zip's 31 members, 20 are the same as from the originals; 10 differ (the 6 changed members of the export, the export itself, the statistics, the manifest and `SHA256SUMS`), and the snapshot is missing.
- `evaluate agreement` from the copies gave the same Table 3 and the same numbers around it ([docs/step-3-extraction/evaluate.md](evaluate.md#how-it-was-checked)).

The fast tests in [`tests/test_groundtruth.py`](../../tests/test_groundtruth.py) run the command on the miniature export, which holds a made-up draft with a made-up name and e-mail, made-up local paths and a made-up passage: nothing of them is left, the account numbers stay, every unchanged member is the same, the seals hold, two runs agree, a trace no rule covers stops the command, and `build` writes the same data from the copies. `test_the_pinned_copies_rebuild_mbdef_gold` (marked `slow`) does the above on the real data root, against MBDef-Gold in `$DATA_ROOT/MBDef-Gold/`.

### How to adapt the copies

- **Another trace of a person** (a new Label Studio export with comments, or an export with annotator details): the check names it; add a rule beside the four in `pseudonymise.py` and a made-up case to the miniature export.
- **New copies** (other rules, another export): pin the new export copy's hash as `PSEUDONYMISED_EXPORT_SHA256` in `groundtruth.py` and update the two rows of `data/data-root.tsv`.

## Not in this step

- **How V2 and V3 were made.** This step starts from V3 (`data/ground_truth/v3.json`, sha256 `b421ae4c…`: 2,488 records of 150 papers, each record with synthetic fields beside its definitions). V3 came from SciDef's DefExtra definitions (`scidef.json`, `276b0877…`: 268 definitions of 75 paper ids) through V2 (`v2.json`, `69187bef…`: 272 records of the same papers with synthetic enriched fields); it merges SciDef's two pairs of duplicate publications into 73 papers and adds 77 papers from the seed corpus (MBDef-Gold's `history/data-README.md` lists the files and their hashes). The code that made V2 and V3 is not available, so this step is not rerun from SciDef: the three files are kept as they were, and the build reads V3 and checks the export against it.
- **The curation app**, the web tool the curator used, is not part of this repository: `build` needs only the export it wrote, and this module holds the reader of the raw exports and the checks.
- **`dataset_views`** (the three dataset views the experiments read: SciDef's original definitions grouped by V3 paper, MBDef-Gold's relevant set, and the relevant plus non-relevant records) and the text form `ascii_words` it joins titles by are in this module (`groundtruth.dataset_views(data)`). No command uses them: training reads the frozen training dataset of the released runs instead ([docs/step-3-extraction/extract.md](extract.md#training-the-programs-extract-train)).
- **MBDef-Gold's guide** (`validated_gt_README.md`) names the script that first built the release, not the command above. It is kept word for word, because the guide is part of the release and its hash is pinned; `mbdef groundtruth build` is the command that builds it now.
