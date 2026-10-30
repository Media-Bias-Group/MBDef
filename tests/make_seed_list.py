"""Make `data/seeds/` (docs/step-1-2-papers/seeds.md): the seed list and the starting corpus's identifiers.

Run from the repository root, with the seed run's recorded inputs fetched into the data root (the three files
of `inputs/` that `data/data-root.tsv` lists: `runner-rows.json`, `seed-manifest.jsonl` and
`existing-corpus-tei.sha256`) and the two Step 2 classifiers' output files (`normalized_predictions.csv`, one
per model) copied somewhere. Those two files are not public (they hold abstracts; docs/step-1-2-papers/seeds.md, "What is not
included"), so outside the group this script documents how `data/seeds/` was made:

    uv run python tests/make_seed_list.py \\
        --runner-rows <inputs>/runner-rows.json --seed-manifest <inputs>/seed-manifest.jsonl \\
        --corpus-sha256 <inputs>/existing-corpus-tei.sha256 \\
        --qwen <qwen folder>/normalized_predictions.csv --gemma <gemma folder>/normalized_predictions.csv \\
        --out data/seeds

`seed-papers.tsv` has one row per seed paper: the paper id of the seed run, the id Step 2 used for it, the
DOI GROBID read in its header, its title, the labels the two models gave it, how Step 2 selected it, and the
sha256 of its TEI and of its PDF. `starting-corpus.tsv` has one row per paper of the starting corpus: its
identifier and the sha256 of its TEI. Nothing of a paper's text is written.

How a seed finds its two labels (the classifiers' files have one row per classified article, named by a Step 2
id that is not always unique):

1. The seed's PDF was handed over under a file name that starts with the Step 2 id, spelled with every run of
   characters other than letters and digits as one `_`. Exactly one id in the classifiers' files has that
   spelling, or the maker stops.
2. If neither model has more than one row under that id, those rows are the seed's. A model with no row gives
   no label.
3. If the id is shared (a model has several rows under it), the rows whose article title is the seed's title
   (letters and digits, case ignored) are the seed's. No such row, or two rows that disagree: no label.

The route follows from the two labels: both `relevant` is `agreed`; two different labels (the third model then
decided, and its answer is not recorded) is `tie-break`; anything else is `not-selected`.
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

SEED_COLUMNS = [
    "paper_id",
    "step2_id",
    "doi",
    "title",
    "qwen_label",
    "gemma_label",
    "step2_route",
    "tei_sha256",
    "pdf_sha256",
]
CORPUS_COLUMNS = ["corpus_id", "tei_sha256"]
SEEDS_FILE = "seed-papers.tsv"
CORPUS_FILE = "starting-corpus.tsv"
TEI_SUFFIX = ".grobid.tei.xml"
RELEVANT = "relevant"

ARTICLE = re.compile(r"Article to classify:\s*\n(.*?)\n\nReturn exactly", re.DOTALL)
TITLE = re.compile(r"Title:[ \t]*(.*?)[ \t]*\nAbstract:", re.DOTALL)


def sanitised(text: str) -> str:
    """The spelling of an id in a file name: each run of other characters than letters and digits is one `_`."""
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def normal(title: str) -> str:
    return re.sub(r"\W+", " ", title.casefold()).strip()


def classified_title(prompt: str) -> str:
    """The title of the article a classifier was asked about ('' if the paper had none)."""
    articles = ARTICLE.findall(prompt)
    found = TITLE.match(articles[-1]) if articles else None
    return found.group(1).strip() if found else ""


def route(qwen: str, gemma: str) -> str:
    if qwen == gemma == RELEVANT:
        return "agreed"
    if qwen and gemma and qwen != gemma:
        return "tie-break"
    return "not-selected"


def rows_by_id(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        grouped[r["paperid"]].append(r)
    return grouped


def label_of(rows: list[dict[str, str]]) -> str:
    labels = {r["clean_label"] for r in rows}
    return labels.pop() if len(labels) == 1 else ""


def seed_rows(
    runner_rows: list[dict],
    manifest_rows: list[dict],
    qwen_rows: list[dict[str, str]],
    gemma_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    manifest = {m["paper_id"]: m for m in manifest_rows}
    qwen, gemma = rows_by_id(qwen_rows), rows_by_id(gemma_rows)
    ids_by_spelling: dict[str, set[str]] = defaultdict(set)
    for step2_id in qwen.keys() | gemma.keys():
        ids_by_spelling[sanitised(step2_id)].add(step2_id)
    out = []
    for run in runner_rows:
        record = manifest[run["paper_id"]]
        spelling = record["source_filename"].split("__", 1)[0]
        ids = ids_by_spelling.get(spelling, set())
        if len(ids) != 1:
            raise SystemExit(
                f"{run['paper_id']}: {len(ids)} Step 2 ids are spelled {spelling} "
                f"(its file is {record['source_filename']}); expected exactly one."
            )
        (step2_id,) = ids
        by_qwen, by_gemma = qwen.get(step2_id, []), gemma.get(step2_id, [])
        # a shared id: the title says which rows are the seed's
        if len(by_qwen) > 1 or len(by_gemma) > 1:
            same = normal(run["frozen_title"])
            by_qwen = [
                r for r in by_qwen if normal(classified_title(r["prompt"])) == same
            ]
            by_gemma = [
                r for r in by_gemma if normal(classified_title(r["prompt"])) == same
            ]
        q, g = label_of(by_qwen), label_of(by_gemma)
        out.append(
            {
                "paper_id": run["paper_id"],
                "step2_id": step2_id,
                "doi": record.get("tei_header_doi") or "",
                "title": run["frozen_title"],
                "qwen_label": q,
                "gemma_label": g,
                "step2_route": route(q, g),
                "tei_sha256": run["tei_sha256"],
                "pdf_sha256": record["source_pdf_sha256"],
            }
        )
    return sorted(out, key=lambda r: r["paper_id"])


def corpus_rows(lines: list[str]) -> list[dict[str, str]]:
    """One row per line of `sha256sum` output over the starting corpus's TEI files."""
    rows = []
    for line in lines:
        sha, name = line.split(None, 1)
        rows.append(
            {"corpus_id": name.strip().removesuffix(TEI_SUFFIX), "tei_sha256": sha}
        )
    return sorted(rows, key=lambda r: (r["corpus_id"], r["tei_sha256"]))


def write_tsv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str) -> list[dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    for flag in (
        "runner-rows",
        "seed-manifest",
        "corpus-sha256",
        "qwen",
        "gemma",
        "out",
    ):
        parser.add_argument(f"--{flag}", required=True)
    args = parser.parse_args(argv)
    runner = json.loads(Path(args.runner_rows).read_text(encoding="utf-8"))
    manifest = [
        json.loads(line)
        for line in Path(args.seed_manifest).read_text(encoding="utf-8").splitlines()
    ]
    seeds = seed_rows(runner, manifest, read_csv(args.qwen), read_csv(args.gemma))
    corpus = corpus_rows(
        Path(args.corpus_sha256).read_text(encoding="utf-8").splitlines()
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_tsv(out / SEEDS_FILE, SEED_COLUMNS, seeds)
    write_tsv(out / CORPUS_FILE, CORPUS_COLUMNS, corpus)
    print(
        f"wrote {len(seeds)} seed papers and {len(corpus)} starting-corpus papers to {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
