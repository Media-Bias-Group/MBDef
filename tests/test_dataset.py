"""`mbdef dataset build`, `dataset export` and `dataset build-gold` (docs/dataset/dataset.md): MBDef's tables, card and JSON copy, and MBDef-Gold's.

tests/fixtures/dataset-mini/ holds miniature inputs with synthetic text and, in `expected/`, the parquet and JSON files that these commands wrote from them, which the tests compare with byte for byte. MBDef's corpus tables there are byte-identical to the ones the original implementation wrote from the same inputs, and MBDef-Gold's three split files to the ground-truth tables it wrote; the JSON copy changed only where it left the ground truth out.
"""

import hashlib
import json
import re
import sqlite3
from pathlib import Path

import polars as pl
import pytest
import yaml
from cli_helpers import run_cli

from mbdef import dataset as ds
from mbdef import files
from mbdef.dataset import gold as gold_module
from mbdef.dataset.card import BADGES, files_url, gold_badges, hub_url

MINI = Path(__file__).parent / "fixtures/dataset-mini"
INPUTS = MINI / "inputs"
EXPECTED = MINI / "expected"
REPO = Path(__file__).resolve().parents[1]
CONTEXT_MARKER = "SYNTHETIC-PARAGRAPH-CONTEXT"  # every paragraph context of the miniature store holds it
PARQUET = sorted(
    p.relative_to(EXPECTED) for p in (EXPECTED / "taxonomy-dataset").rglob("*.parquet")
)
JSON = sorted(
    p.relative_to(EXPECTED)
    for p in (EXPECTED / "taxonomy-dataset-json").rglob("*.json")
)
GOLD_PARQUET = sorted(
    p.relative_to(EXPECTED) for p in (EXPECTED / "MBDef-Gold").rglob("*.parquet")
)
GOLD_URL = f"https://huggingface.co/datasets/{files.MBDEF_GOLD_REPO}"
# the pages each card links, one topic each (src/mbdef/dataset/cards/)
MBDEF_PAGES = [
    f"{name}.md"
    for name in (
        "columns",
        "how-it-was-made",
        "licence",
        "relevance",
        "tables",
        "term-merging",
        "terms-view",
    )
]
GOLD_PAGES = [
    f"{name}.md" for name in ("files-and-columns", "licence", "limits", "what-it-holds")
]
# what no shipped text may hold: question and decision ids, the rule's working name and version id, dates, the ground truth's old name, the machine
PRIVATE = re.compile(
    r"\b(?:DR|S5|RC|SP|CG|RB|IS)-Q[0-9]+|\bD1[0-9]{2}\b|\brule C\b|\bC-20[0-9]{6}\b|\bNR[0-9]|\"platform\"|"
    r"20[0-9]{2}-[0-9]{2}-[0-9]{2}|[0-9]{1,2} (?:Sep|Oct)[a-z]* 20[0-9]{2}|BiasDef130|"
    r"research.notes|coordinator|development history"
)


def mini_store(folder: Path, contexts: bool = True) -> Path:
    """The miniature corpus store, loaded from its SQL text; without its paragraph contexts (the `contexts` table, which a rebuilt store does not have) when `contexts` is false."""
    store = folder / (
        "library.sqlite3" if contexts else "library-without-contexts.sqlite3"
    )
    if store.exists():
        return store
    db = sqlite3.connect(store)
    db.executescript((INPUTS / "store.sql").read_text())
    if not contexts:
        db.execute("DROP TABLE contexts")
        db.commit()
    db.close()
    return store


def ground_truth_flags() -> list[str]:
    return [
        *("--ground-truth", str(INPUTS / "ground-truth/validated_gt.json")),
        *(
            "--non-relevant",
            str(INPUTS / "ground-truth/validated_gt_non_relevant.json"),
        ),
    ]


def mini_flags(folder: Path, contexts: bool = True) -> list[str]:
    return [
        *("--decisions", str(INPUTS / "decisions/ensemble.csv")),
        *("--label-groups", str(INPUTS / "label-groups")),
        *("--term-merging", str(INPUTS / "term-merging")),
        *("--pick", str(INPUTS / "pick.json")),
        *("--store", str(mini_store(folder, contexts))),
        *("--corpus", str(INPUTS / "corpus")),
        *("--recorded", str(INPUTS / "recorded")),
        *ground_truth_flags(),
    ]


def gold_flags() -> list[str]:
    return [
        *ground_truth_flags(),
        *("--votes", str(INPUTS / "ground-truth/records.csv")),
        *("--recorded", str(INPUTS / "recorded")),
    ]


def dataset(*args: str):
    return run_cli(REPO, "dataset", *args)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """One build and export of the miniature: (the output folder, the build's and the export's completed runs)."""
    work = tmp_path_factory.mktemp("dataset")
    out = work / "out"
    build = dataset("build", *mini_flags(work), "--out", str(out))
    export = dataset("export", "--out", str(out))
    return out, build, export


@pytest.fixture(scope="module")
def gold_built(tmp_path_factory):
    """One `build-gold` of the miniature: (the output folder, the completed run)."""
    out = tmp_path_factory.mktemp("gold") / "out"
    return out, dataset("build-gold", *gold_flags(), "--out", str(out))


def shipped_texts(folder: Path) -> dict[str, str]:
    """Every text file below `folder`: what an upload would publish."""
    return {
        str(p.relative_to(folder)): p.read_text(encoding="utf-8")
        for p in sorted(folder.rglob("*"))
        if p.suffix in (".md", ".json") or p.name == "SHA256SUMS"
    }


def test_build_and_export_succeed(built):
    out, build, export = built
    assert build.returncode == 0, build.stdout + build.stderr
    assert export.returncode == 0, export.stdout + export.stderr
    report = json.loads((out / "validation.json").read_text())
    assert report["verdict"] == "PASSED"
    assert not [c for c in report["checks"] if c["status"] == "FAILED"]


@pytest.mark.parametrize("name", PARQUET, ids=str)
def test_the_tables_equal_the_expected_files_byte_for_byte(built, name):
    out, _, _ = built
    assert (out / name).read_bytes() == (EXPECTED / name).read_bytes()


@pytest.mark.parametrize(
    "name", [*JSON, Path("taxonomy-dataset-json/SHA256SUMS")], ids=str
)
def test_the_json_copy_equals_the_expected_files_byte_for_byte(built, name):
    out, _, _ = built
    assert (out / name).read_bytes() == (EXPECTED / name).read_bytes()


def test_mbdef_holds_no_ground_truth_and_links_mbdef_gold(built):
    """The ground truth is its own dataset, MBDef-Gold: MBDef has no table of it, and its card links it."""
    out, _, _ = built
    assert sorted(p.name for p in (out / "taxonomy-dataset/data").iterdir()) == [
        "concepts.parquet",
        "definitions.parquet",
        "instances.parquet",
        "papers.parquet",
        "terms.parquet",
    ]
    assert not (out / "taxonomy-dataset-json/ground_truth").exists()
    card = (out / "taxonomy-dataset/README.md").read_text()
    assert "ground_truth" not in card
    assert GOLD_URL in card
    provenance = json.loads((out / "taxonomy-dataset/provenance.json").read_text())
    assert "ground_truth" not in provenance["outputs"]


def test_no_shipped_text_holds_a_private_reference(built):
    """The card, provenance.json, SCHEMA.md and the card's copy: no question or decision id, no working name of the rule, no date, no old name."""
    out, _, _ = built
    texts = shipped_texts(out / "taxonomy-dataset") | shipped_texts(
        out / "taxonomy-dataset-json"
    )
    assert {"README.md", "provenance.json", "SCHEMA.md"} <= set(texts)
    for name, text in texts.items():
        assert not PRIVATE.findall(text), name
    provenance = json.loads(texts["provenance.json"])
    decisions = provenance["decisions"]
    assert decisions["text"] == "every definition text word for word"
    assert decisions["rows"] == "only the kept term-in-paper rows"
    assert set(provenance["code"]) == {
        "commit",
        "dirty",
        "python",
        "polars",
        "pyarrow",
    }
    assert "label group" not in texts["provenance.json"]


def test_no_output_holds_a_paragraph_context(built):
    out, _, _ = built
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        if path.suffix == ".png":  # the card's banner
            continue
        if path.suffix == ".parquet":
            text = pl.read_parquet(path).write_ndjson()
        else:
            text = path.read_text(encoding="utf-8")
        assert CONTEXT_MARKER not in text, path
    # the store does hold them, so the check above can fail
    assert CONTEXT_MARKER in (INPUTS / "store.sql").read_text()


def test_the_build_needs_no_paragraph_contexts(tmp_path):
    out = tmp_path / "out"
    done = dataset("build", *mini_flags(tmp_path, contexts=False), "--out", str(out))
    assert done.returncode == 0, done.stdout + done.stderr
    for name in PARQUET:
        assert (out / name).read_bytes() == (EXPECTED / name).read_bytes(), name


def test_the_card_names_every_table_and_fills_every_number(built):
    out, _, _ = built
    card = (out / "taxonomy-dataset/README.md").read_text()
    head = yaml.safe_load(card.split("---\n")[1])
    paths = sorted(f["path"] for cfg in head["configs"] for f in cfg["data_files"])
    assert paths == sorted(str(p.relative_to("taxonomy-dataset")) for p in PARQUET)
    assert not re.findall(r"\{\{\w+\}\}", card)  # every placeholder filled
    assert head["license"] == "cc-by-4.0"
    assert head["pretty_name"].startswith("MBDef")
    assert "Citation details will follow." in card
    # the JSON copy carries the same card
    assert (out / "taxonomy-dataset-json/DATASET-CARD.md").read_text() == card
    # the card is an overview; what it links is one page per topic, every placeholder filled
    pages = sorted((out / "taxonomy-dataset/docs").glob("*.md"))
    assert [p.name for p in pages] == sorted(MBDEF_PAGES)
    for page in pages:
        assert not re.findall(r"\{\{\w+\}\}", page.read_text()), page.name


def check_overview(folder: Path, repo_id: str, limit: int) -> None:
    """README.md's body stays within LIMIT lines, every link in a card or page to another page of its dataset names a file the build wrote, and every page is linked from the README."""
    base = files_url(repo_id) + "/"
    pages = {p: p.read_text() for p in sorted(folder.rglob("*.md"))}
    readme = pages[folder / "README.md"]
    body = re.sub(
        r"<details>.*?</details>\n", "", readme.split("---\n", 2)[2], flags=re.S
    )
    assert (
        len(body.splitlines()) <= limit
    )  # a collapsed table does not make the card longer
    linked = {
        link.removeprefix(base)
        for text in pages.values()
        for link in re.findall(r"\]\((" + re.escape(base) + r"[^)#]+)", text)
    }
    assert linked, folder
    assert all((folder / name).is_file() for name in linked), linked
    for page in pages:
        name = page.relative_to(folder).as_posix()
        assert name == "README.md" or base + name in readme, name


def test_the_card_is_an_overview_whose_links_reach_pages_that_exist(built):
    check_overview(built[0] / "taxonomy-dataset", files.MBDEF_REPO, 85)


def test_the_gold_card_is_an_overview_whose_links_reach_pages_that_exist(gold_built):
    check_overview(gold_built[0] / "MBDef-Gold", files.MBDEF_GOLD_REPO, 65)


def test_the_provenance_lists_every_input_by_hash_without_a_machine_path(built):
    out, _, _ = built
    provenance = json.loads((out / "taxonomy-dataset/provenance.json").read_text())
    store = next(
        i for i in provenance["inputs"] if i["path"].endswith("library.sqlite3")
    )
    assert (
        store["sha256"]
        == hashlib.sha256((out.parent / "library.sqlite3").read_bytes()).hexdigest()
    )
    assert not any(i["path"].startswith("/") for i in provenance["inputs"])
    assert {k: v["rows"] for k, v in provenance["outputs"].items()}["concepts"] == 25


def recorded_listing(
    path: Path, changed: Path | None = None, extra: str | None = None
) -> Path:
    """A sha256sum listing of the expected files; the line of `changed` names another hash, and `extra` adds a line for a file no build writes."""
    lines = []
    for p in [*PARQUET, *JSON]:
        digest = (
            "0" * 64
            if p == changed
            else hashlib.sha256((EXPECTED / p).read_bytes()).hexdigest()
        )
        lines.append(f"{digest}  {p}\n")
    if extra:
        lines.append(f"{'1' * 64}  {extra}\n")
    path.write_text("".join(lines))
    return path


def test_same_data_as_compares_the_build_with_a_recorded_hash_list(tmp_path):
    out = tmp_path / "out"
    listing = recorded_listing(tmp_path / "recorded.sha256")
    done = dataset(
        "build",
        *mini_flags(tmp_path),
        "--out",
        str(out),
        "--same-data-as",
        str(listing),
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "byte-identical" in done.stdout
    changed = recorded_listing(tmp_path / "changed.sha256", changed=PARQUET[0])
    done = dataset(
        "build",
        *mini_flags(tmp_path),
        "--out",
        str(out),
        "--same-data-as",
        str(changed),
    )
    assert done.returncode == 1
    assert str(PARQUET[0].relative_to("taxonomy-dataset")) in done.stdout
    assert (out / "taxonomy-dataset-FAILED").is_dir() and not (
        out / "taxonomy-dataset"
    ).exists()


def test_same_data_as_names_a_recorded_file_the_build_does_not_write_without_failing(
    tmp_path,
):
    """A listing of an older version that still holds the ground-truth tables (as 1.2.1's does): the files this build writes are compared, the others are named."""
    out = tmp_path / "out"
    extra = "taxonomy-dataset/data/ground_truth/train.parquet"
    listing = recorded_listing(tmp_path / "older.sha256", extra=extra)
    done = dataset(
        "build",
        *mini_flags(tmp_path),
        "--out",
        str(out),
        "--same-data-as",
        str(listing),
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "byte-identical" in done.stdout
    assert "data/ground_truth/train.parquet" in done.stdout


def test_same_data_as_compares_the_json_copy_with_a_recorded_hash_list(built, tmp_path):
    out, _, _ = built
    done = dataset(
        "export",
        "--out",
        str(out),
        "--same-data-as",
        str(recorded_listing(tmp_path / "a.sha256")),
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "byte-identical" in done.stdout
    changed = recorded_listing(tmp_path / "b.sha256", changed=JSON[0])
    done = dataset("export", "--out", str(out), "--same-data-as", str(changed))
    assert done.returncode == 1
    assert str(JSON[0]) in done.stdout


def test_an_output_folder_inside_a_frozen_folder_is_refused(tmp_path):
    """`--out` inside the data root's frozen folder (here a throwaway data root) is refused before anything is written; the repository's `data/frozen/` is refused by the same rule, checked without running a build there."""
    from mbdef import postprocess

    root = tmp_path / "data-root"
    (root / "frozen").mkdir(parents=True)
    inside = root / "frozen/dataset/x"

    for action, flags in (
        ("build", mini_flags(tmp_path)),
        ("build-gold", gold_flags()),
    ):
        done = run_cli(
            REPO,
            *("dataset", action, *flags, "--out", str(inside)),
            env={"DATA_ROOT": str(root)},
        )
        assert done.returncode != 0, action
        assert "Refused" in done.stderr, action
        assert not inside.exists(), action
    with pytest.raises(SystemExit, match="Refused"):
        postprocess.refuse_frozen(REPO / "data/frozen/dataset/x")


def test_schema_md_shows_fake_news_or_else_the_first_entry_with_a_supertype():
    """A corpus without "fake news" (an adapted one) still gets SCHEMA.md's example: the first entry that names a supertype; one without any supertype stops with a sentence."""
    from mbdef.dataset import export

    def entry(term, supertypes):
        return {"term": term, "supertypes": [{"text": s} for s in supertypes]}

    terms = [
        entry("bias", []),
        entry("framing", ["bias"]),
        entry("Fake News", ["news"]),
    ]
    assert (export.example(terms) or {}).get("term") == "Fake News"
    assert (export.example(terms[:2]) or {}).get("term") == "framing"
    assert export.example(terms[:1]) is None
    with pytest.raises(SystemExit, match="no example"):
        export.schema_md({}, terms[:1], "9.9.9")


# --- MBDef-Gold ---------------------------------------------------------------------------------------------------------


def test_build_gold_succeeds(gold_built):
    out, done = gold_built
    assert done.returncode == 0, done.stdout + done.stderr
    report = json.loads((out / "gold-validation.json").read_text())
    assert report["verdict"] == "PASSED"


@pytest.mark.parametrize("name", GOLD_PARQUET, ids=str)
def test_the_gold_splits_equal_the_expected_files_byte_for_byte(gold_built, name):
    out, _ = gold_built
    assert (out / name).read_bytes() == (EXPECTED / name).read_bytes()


def test_the_gold_folder_holds_the_record_files_unchanged(gold_built):
    out, _ = gold_built
    for name in ("validated_gt.json", "validated_gt_non_relevant.json"):
        assert (out / "MBDef-Gold" / name).read_bytes() == (
            INPUTS / "ground-truth" / name
        ).read_bytes()


def test_the_gold_card_counts_the_ground_truth_it_was_built_with(gold_built):
    """The split table and the counts come from the built split files (here the miniature's, not the released ones); the card names its three splits and links MBDef."""
    out, _ = gold_built
    folder = out / "MBDef-Gold"
    card = (folder / "README.md").read_text()
    head = yaml.safe_load(card.split("---\n")[1])
    splits = {
        f["split"]: f["path"] for cfg in head["configs"] for f in cfg["data_files"]
    }
    assert splits == {s: f"data/{s}.parquet" for s in ("train", "validation", "test")}
    assert not re.findall(r"\{\{\w+\}\}", card)
    everything = []
    for split, name in (("train", "train"), ("dev", "validation"), ("test", "test")):
        table = pl.read_parquet(folder / f"data/{name}.parquet")
        everything.append(table)
        relevant = table.filter(pl.col("relevant"))
        assert (
            f"| {split} / `{name}` | {table.height:,} | {relevant.height:,} | "
            f"{table['paper_id'].n_unique():,} |"
        ) in card
    g = pl.concat(everything)
    relevant = g.filter(pl.col("relevant"))
    holds = (folder / "docs/what-it-holds.md").read_text()
    assert f"**Relevant records:** {relevant.height:,} records" in holds
    assert (
        f"**Non-relevant candidates:** {g.height - relevant.height:,} records" in holds
    )
    assert sorted(p.name for p in (folder / "docs").glob("*.md")) == sorted(GOLD_PAGES)
    for page in (folder / "docs").glob("*.md"):
        assert not re.findall(r"\{\{\w+\}\}", page.read_text()), page.name
    assert "1,420" not in card
    assert f"https://huggingface.co/datasets/{files.MBDEF_REPO}" in card
    assert "Citation details will follow." in card


def test_no_gold_text_holds_a_private_reference(gold_built):
    out, _ = gold_built
    texts = shipped_texts(out / "MBDef-Gold")
    for name in ("README.md", "provenance.json"):
        assert not PRIVATE.findall(texts[name]), name


def test_both_cards_open_with_their_banner_and_link_the_paper_the_code_and_each_other(
    built, gold_built
):
    """Each banner is copied into its dataset folder and shown by a relative address, which works on a private repository too. MBDef's row of buttons leads to the Explorer, MBDef-Gold's (which is not about the Explorer) to MBDef."""
    mbdef = built[0] / "taxonomy-dataset"
    gold = gold_built[0] / "MBDef-Gold"
    mbdef_url = hub_url(files.MBDEF_REPO)
    for folder, banner, alt, badges, buttons, others in (
        (
            mbdef,
            ds.BANNER,
            "MBDef, the Media Bias Definitions Dataset",
            BADGES,
            [ds.EXPLORER_URL, ds.PROJECT_URL],
            [GOLD_URL, ds.EXPLORER_URL, ds.PROJECT_URL],
        ),
        (
            gold,
            ds.GOLD_BANNER,
            "MBDef-Gold, the hand-labelled ground truth of MBDef",
            gold_badges(mbdef_url),
            [mbdef_url, ds.PROJECT_URL],
            [mbdef_url, ds.PROJECT_URL],
        ),
    ):
        card = (folder / "README.md").read_text()
        body = card.split("---\n", 2)[2]
        assert body.lstrip().startswith(f"![{alt}]({ds.BANNER_PATH})\n\n" + badges)
        for url in buttons:
            assert f'<a href="{url}">' in body.split("\n# ", 1)[0], url
        assert (folder / ds.BANNER_PATH).read_bytes() == banner.read_bytes()
        for url in (ds.PAPER_URL, ds.CODE_URL, *others):
            assert f"]({url})" in card, url
    assert ds.EXPLORER_URL not in (gold / "README.md").read_text()
    assert ds.BANNER.read_bytes() != ds.GOLD_BANNER.read_bytes()


def test_no_banner_leaves_the_card_without_one(tmp_path):
    out = tmp_path / "out"
    done = dataset(
        "build-gold",
        *gold_flags(),
        "--out",
        str(out),
        "--no-banner",
        "--no-samples-image",
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert not (out / "MBDef-Gold/assets").exists()
    text = (out / "MBDef-Gold/README.md").read_text()
    assert "](assets/" not in text
    assert (
        text.split("---\n", 2)[2]
        .lstrip()
        .startswith(gold_badges(hub_url(files.MBDEF_REPO)))
    )


def test_the_gold_card_shows_a_seeded_sample_of_train_records_as_a_table_and_a_picture(
    gold_built,
):
    """The table lists the records `samples` draws (train only, one per paper, relevant ones first), folded away in a <details>; the picture is copied beside the banner; no test-split text is on the card."""
    folder = gold_built[0] / "MBDef-Gold"
    card = (folder / "README.md").read_text()
    g = pl.concat(
        [
            pl.read_parquet(folder / f"data/{n}.parquet")
            for n in ("train", "validation", "test")
        ]
    )
    chosen = gold_module.samples(g)
    assert chosen and chosen == gold_module.samples(g)  # the same draw every time
    assert {r["split"] for r in chosen} == {"train"}
    assert len({r["paper_id"] for r in chosen}) == len(chosen)
    assert [r["relevant"] for r in chosen] == sorted(
        (r["relevant"] for r in chosen), reverse=True
    )
    details = card.split("<details>", 1)[1].split("</details>", 1)[0]
    assert f"All {len(chosen)} sample records" in details
    table = [x for x in details.splitlines() if x.startswith("| ") and "---" not in x][
        1:
    ]
    assert len(table) == len(chosen)
    assert (folder / ds.SAMPLES_PATH).read_bytes() == ds.GOLD_SAMPLES.read_bytes()
    assert f"]({ds.SAMPLES_PATH})" in card.split("<details>", 1)[0]
    for r in g.filter(pl.col("split") == "test").iter_rows(named=True):
        assert r["definitions"][0]["text"] not in card


def test_a_table_cell_cannot_be_read_as_markdown():
    assert gold_module.markdown("a|b *c* _d_ [e](f) <g>") == (
        "a\\|b \\*c\\* \\_d\\_ \\[e\\](f) &lt;g&gt;"
    )
    assert gold_module.cut("one two three", 9) == "one two…"
    assert gold_module.cut("one  two\nthree", 50) == "one two three"


def test_no_samples_image_leaves_the_table_without_the_picture(tmp_path):
    out = tmp_path / "out"
    done = dataset("build-gold", *gold_flags(), "--out", str(out), "--no-samples-image")
    assert done.returncode == 0, done.stdout + done.stderr
    assert not (out / "MBDef-Gold" / ds.SAMPLES_PATH).exists()
    text = (out / "MBDef-Gold/README.md").read_text()
    assert ds.SAMPLES_PATH not in text and "<details>" in text


def test_the_picture_shows_the_first_records_of_each_label_with_a_close_call_among_them():
    """`poster_samples`: the first three of each label, but a split vote (not unanimous) replaces the last one when none of the three has one."""

    def row(label: str, relevant: bool, yes: int, no: int) -> dict:
        return dict(
            term=label, relevant=relevant, votes_relevant=yes, votes_not_relevant=no
        )

    rows = [row(f"r{i}", True, 2, 0) for i in range(4)] + [row("r-close", True, 2, 1)]
    rows += [row(f"n{i}", False, 0, 2) for i in range(4)] + [
        row("n-close", False, 1, 2)
    ]
    shown = [r["term"] for r in gold_module.poster_samples(rows)]
    assert shown == ["r0", "r1", "r-close", "n0", "n1", "n-close"]
    # a split vote already among the first three stays as it is; fewer than three records are all shown
    rows[1] = row("r1", True, 2, 1)
    assert [r["term"] for r in gold_module.poster_samples(rows)][:3] == [
        "r0",
        "r1",
        "r2",
    ]
    assert [r["term"] for r in gold_module.poster_samples(rows[:2])] == ["r0", "r1"]
