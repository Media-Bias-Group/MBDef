# Fetching the data: `mbdef data fetch`

| | |
|---|---|
| **Paper step** | None of its own: it puts the data that is not in git into `DATA_ROOT`, where every step's command reads it: the two datasets MBDef and MBDef-Gold from Hugging Face, and the larger saved inputs and outputs of the steps from the public deposit |
| **Code** | [`src/mbdef/files.py`](../../src/mbdef/files.py) (`fetch`, `HF_DATASETS`, `read_list`, `download`, `unpack`) and the list [`data/data-root.tsv`](../../data/data-root.tsv) |
| **Command** | `mbdef data fetch [PREFIX ...]` |
| **Checked by** | [`tests/test_data_fetch.py`](../../tests/test_data_fetch.py), with stand-in web servers for Hugging Face and for the deposit: no test touches the network |

## What it does

The data root ([docs/setup/data-root.md](data-root.md)) is one folder per machine.
`data/data-root.tsv` lists every file that belongs in it with its sha256, its size and its source.
For each listed file under the prefixes you give (all of them, without a prefix):

1. **The file is there.** It is only checked: its sha256 (a folder: its folder digest) must be the listed one.
   If it differs, the command reports `CHANGED` and does not touch it.
   Nothing is ever overwritten.
2. **It is missing and its source is `hf:<dataset>/<path>`.** It is a file of one of the two datasets on Hugging Face: MBDef, the definitions extracted from the corpus ([docs/dataset/dataset.md](../dataset/dataset.md)), or MBDef-Gold, the ground truth ([docs/step-3-extraction/groundtruth.md](../step-3-extraction/groundtruth.md)).
   The file is downloaded from the dataset's repository at its pinned revision and kept only if its sha256 matches.
3. **It is missing and its source is `https:<path>`.** It is a file of the public deposit.
   The deposit's file names are flat (a Zenodo record allows no `/` in a name), so `<path>` is stored with each `/` replaced by `__`: the file is downloaded from `<deposit address>/<flat name>` and kept only if its sha256 matches.
   A folder is one zstd tar archive in the deposit (`https:<path>.tar.zst#<folder name>`): the archive is downloaded once per run, the member is unpacked with `zstd -d --long=31` and `tar`, and checked the same way.
4. **Its source is `derive:<command>`.** The command says `MISSING` and prints the command that makes the file.

The command exits 1 if any listed file is missing or differs, so it also serves as a check of a data root.

### The two datasets

The two Hugging Face repositories are named in one place, `HF_DATASETS` in [`files.py`](../../src/mbdef/files.py), each with the revision that `data fetch` downloads:

| Dataset | Repository | Folder in the data root | What `data/data-root.tsv` lists |
|---|---|---|---|
| MBDef | [`mediabiasgroup/MBDef`](https://huggingface.co/datasets/mediabiasgroup/MBDef) | `MBDef/` | its five tables (`data/*.parquet`) and the MBDef Explorer's page data (`explorer/`, 151 files with its `SHA256SUMS`) |
| MBDef-Gold | [`mediabiasgroup/MBDef-Gold`](https://huggingface.co/datasets/mediabiasgroup/MBDef-Gold) | `MBDef-Gold/` | the relevant and the non-relevant records (`validated_gt.json`, `validated_gt_non_relevant.json`), the same records as split files (`data/{train,validation,test}.parquet`), the statistics, the manifest, the guide, the report (`.tex`, `.pdf`, `.md`) with its figures, and the notes on the earlier ground truths (`history/data-README.md`) |

A file keeps its path in the repository below the dataset's folder: the row `MBDef-Gold/validated_gt.json … hf:MBDef-Gold/validated_gt.json` is the file `validated_gt.json` of `mediabiasgroup/MBDef-Gold`, downloaded from `https://huggingface.co/datasets/mediabiasgroup/MBDef-Gold/resolve/<revision>/validated_gt.json` to `$DATA_ROOT/MBDef-Gold/validated_gt.json`.
The dataset cards, `provenance.json` and the cards' banner (`assets/banner.png`) are not listed: no command reads them.

Both repositories hold version 1.0.0 as the commit tagged `v1.0.0`; `HF_DATASETS` pins that commit (MBDef `fa97dbb3…`, MBDef-Gold `e2efedfb…`), so a later upload changes nothing that `data fetch` gets until the pin and the rows change too.
Both repositories are private for now: a member of the `mediabiasgroup` organisation reads them with a Hugging Face key in `HF_TOKEN` (`.env`).
Without a key that can read them, or when a repository lacks a file, the download fails and the line names the dataset and its repository, then the address and the error Hugging Face answered with:

```text
$ uv run mbdef data fetch MBDef-Gold/validated_gt.json
MBDef-Gold/validated_gt.json: FAILED, the dataset MBDef-Gold (mediabiasgroup/MBDef-Gold on Hugging Face, revision e182076c602e9546874212bfab7693f193e7555e) has no file validated_gt.json, or is not uploaded or not public yet: https://huggingface.co/datasets/mediabiasgroup/MBDef-Gold/resolve/e182076c602e9546874212bfab7693f193e7555e/validated_gt.json: <the error>
1 of 1 files are missing or differ from data/data-root.tsv.
```

`HF_ENDPOINT`, as for the Hugging Face tools, replaces `https://huggingface.co` (the tests point it at a stand-in server).

### The deposit

The first line of `data/data-root.tsv` gives the address of the public deposit.
While it is the placeholder (`https://deposit.invalid/…`, until the deposit is uploaded), an `https:` file is reported as not uploaded yet.
With a Zenodo record as the deposit, the line holds `https://zenodo.org/records/<id>/files/` and the file `https:embeddings/qwen3-emb-8b/keys.txt` is the record's file `embeddings__qwen3-emb-8b__keys.txt`, at `https://zenodo.org/records/<id>/files/embeddings__qwen3-emb-8b__keys.txt`.

Files that have no public copy (the papers' full text, saved runs that hold windows of it, PDF first pages and paragraphs, and the original annotation exports) are not in `data/data-root.tsv` at all.
The commands that read one cannot run with the public data alone; each step's page says which inputs they are.

## Inputs

| Input | Where | What it is |
|---|---|---|
| The list | `data/data-root.tsv`, read from the folder the command runs in | A `#deposit` line with the deposit's address, a header line (`path`, `sha256`, `bytes`, `source`), then one line per file or folder |
| The two datasets' repositories | `HF_DATASETS` in `src/mbdef/files.py` | Each dataset's repository id and the revision to download |
| `DATA_ROOT` | `.env` | The folder to fill ([docs/setup/data-root.md](data-root.md)) |
| `HF_TOKEN`, `HF_ENDPOINT` | `.env` or the environment, both optional | A Hugging Face key for a private repository; another address for Hugging Face |
| `zstd` and `tar` | system | Only for folders (archive members) |
| `PREFIX ...` | the command line | Only the listed paths that start with one of these, for example `MBDef-Gold/` or `selection/` |

## Run it

```bash
uv run mbdef data fetch MBDef-Gold/                   # MBDef-Gold, the ground truth (2.8 MB)
uv run mbdef data fetch MBDef/                        # MBDef, the corpus dataset's tables (28 MB)
uv run mbdef data fetch frozen/ selection/            # several prefixes
uv run mbdef data fetch                               # everything listed (many gigabytes)
```

Each step's page says which prefix it needs ([docs/step-3-extraction/groundtruth.md](../step-3-extraction/groundtruth.md), [docs/step-3-extraction/evaluate.md](../step-3-extraction/evaluate.md), [docs/step-3-extraction/selection.md](../step-3-extraction/selection.md), [docs/step-3-extraction/extract.md](../step-3-extraction/extract.md), [docs/step-4-closure/closure.md](../step-4-closure/closure.md), [docs/step-5-post-processing/postprocess.md](../step-5-post-processing/postprocess.md)).

## Outputs

The files and folders of the list, at their listed paths under `DATA_ROOT`.
The command prints one line per listed file, in the list's order:

| Line | Meaning |
|---|---|
| `<path>: already there, sha256 matches` | Nothing to do |
| `<path>: fetched, sha256 matches` | Downloaded and checked |
| `<path>: CHANGED, its sha256 is <found>, not <listed>; not overwritten` | A different file is there: look at it yourself |
| `<path>: FAILED, <reason>` | The download failed (the address and the error; for a dataset file also the dataset, its repository and revision) or its sha256 differs; nothing was kept |
| `<path>: MISSING; make it with: <command>` | A `derive:` file |
| `<path>: MISSING; the public deposit is not uploaded yet (…)` | An `https:` file while the address is the placeholder |

The last line says how many files were chosen and how many are missing or differ.
A download is written to `<path>.part` and renamed only when its sha256 matches; an archive is unpacked in a scratch folder (`.fetch-…`) on the data root's file system, so that a folder can be moved in, and the folder is removed at the end.

## How it was checked

[`tests/test_data_fetch.py`](../../tests/test_data_fetch.py) runs the command against two stand-in web servers on the test machine, one for Hugging Face and one for the deposit, which log every path they are asked for.
It checks that:

- a dataset's files come from its repository at the pinned revision, are checked, and are kept at their path below the dataset's folder; a key in `HF_TOKEN` is sent, and none without it;
- a dataset that is not on Hugging Face yet fails with a line that names the dataset and its repository, and a dataset file with another sha256 is not kept; a row that names neither dataset fails without a download;
- files and a folder (a member of a zstd tar archive, downloaded once) come from the deposit and are checked; a deposit file with another sha256 is not kept, and a file the deposit lacks fails with its address;
- a `derive:` file prints its command, and an `https:` file says that the deposit is not uploaded while its address is the placeholder;
- a file that is already there is not downloaded again, and one that differs is reported and never overwritten;
- the repository's list gives every row a dataset, deposit or derived source and names no private store, commit or machine; each dataset row lies in its dataset's folder; and git holds neither dataset.

## How to adapt it

- **Other repositories for the two datasets.** Change `MBDEF_REPO` and `MBDEF_GOLD_REPO` in [`files.py`](../../src/mbdef/files.py), and pin each revision in `HF_DATASETS` to the commit of the upload (the repository's history on Hugging Face shows it); the rows of the list stay.
- **A new version of a dataset.** Upload it, then put each changed file's new sha256 and size into its row of `data/data-root.tsv` and pin the revision to the new commit.
- **Add a dataset file.** Add one line to `data/data-root.tsv`: its path below the dataset's folder, sha256, bytes, and `hf:<dataset>/<path in the repository>`.
- **Add a deposit file.** Put the file into the deposit under its flat name (each `/` of `<path>` replaced by `__`; a folder as `<path>.tar.zst`, made with `tar -C <parent> -c <folder> | zstd --long=31`), add one line to `data/data-root.tsv` (path, sha256 or folder digest, bytes, `https:<path>`), and read it in the step's code as `files.data_root() / "<path>"`, with the path as a named constant in the module that reads it.
  No listed path may contain `__`, so each flat name maps back to one path, and a Zenodo record holds at most 100 files by default (the tests check both).
- **Another deposit.** Change the address on the `#deposit` line; the paths below it stay.
- **A file that a command makes.** List it as `derive:<command>`.
