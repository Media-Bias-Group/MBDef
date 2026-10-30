"""`mbdef explorer build`: the MBDef Explorer, a static web page in `explorer/` (docs/dataset/explorer.md).

The page is made of the sources in `explorer/src/`: the CSS files, the tabs' HTML sections and the scripts (each tab's in its own block, so its names stay local) replace the three markers of `shell.html`, and the addresses of `links()` its `{{name}}` placeholders.
The joined script must pass `node --check`; when it does not, the build names the scripts that fail on their own.

The page's data comes with the MBDef dataset, in its folder `explorer/`: the JSON files the page reads, listed with their sha256 in `SHA256SUMS`.
They hold exactly the dataset's term mentions, and besides what its tables hold, what only the page shows: every mention's map position, topic and nearest mentions, how alike the mentions of a concept are, and the scorers' reference scores (`explorer data`, explorer_data.py, makes them).
The build checks those files against their `SHA256SUMS` and against the dataset's tables (every mention, score, definition, concept and paper).
It then writes them to `explorer/data/`, without the keys that record where and when they were made, with a new `SHA256SUMS`.
It refuses a site that GitHub Pages would not take.

The command runs from the repository root. Nothing is written when a check fails.
"""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import polars as pl

from mbdef import files

FOLDER = Path("explorer")
SRC = FOLDER / "src"
PAGE = FOLDER / "index.html"
DATA = FOLDER / "data"
SUMS = "SHA256SUMS"
# where `data fetch` puts the MBDef dataset, below DATA_ROOT (files.py)
MBDEF = Path(files.MBDEF)
BUNDLE = "explorer"  # the page's data, inside the dataset's folder
TABLES = "data"  # the dataset's tables, inside its folder
TABS = ("overview", "map", "concepts", "relations", "relevance", "papers")
CSS_MARKER, SECTIONS_MARKER, JS_MARKER = "/*CSS*/", "<!--SECTIONS-->", "//JS"
# The last line of the page's script: it starts the page, and says so on the page when the data fails to load.
BOOT = (
    "boot().catch((e) => { $('#loading').textContent = 'The dataset could not be loaded: ' "
    "+ e.message; console.error(e); });"
)
NODE_CHECK = ("node", "--check", "--input-type=module")
# U+FFFD, the mark of an unreadable character: the page shows "?" instead
REPLACEMENT_CHARACTER = "\ufffd"
SCORES = {
    "s": "ensemble_score",
    "c": "classifier_score",
    "b": "basal_score",
    "j": "judge_score",
}
# Keys of the page data that record where and when it was made (commits, run times, local paths): not published.
RUN_HISTORY = {
    "meta.json": ("provenance",),
    "ensemble-ref.json": ("sources", "provenance"),
}
# How the page's numbers were made, in place of the dataset's own notes, which name run folders and dates.
METHOD = {
    "vector": 'the L2-normalised mean of the Qwen3-Embedding-8B vectors of a mention\'s definitions, each embedded as "term: definition"',
    "umap": "a UMAP of the vectors of the released mentions (PCA to 50 dimensions, then UMAP with cosine distance, 30 neighbours and minimum distance 0.15, to 3D and 2D); 36 topics by k-means (k = 36) on the 50 PCA dimensions, named by their most frequent concepts",
    "spread": "the mean cosine of a label group's mentions to their normalised centroid; the mean pairwise cosine",
    "neighbours": "the 8 most similar other mentions, by exact cosine on the full vectors",
    "rule": {
        "name": "the label-match rule",
        "source": "src/mbdef/postprocess/label_match.py",
    },
    "concept_neighbours": "the 8 most similar kept concepts, by the cosine of the normalised mean of their kept mentions' vectors",
}
FIELDS_NOTE = (
    "items as the dataset reads them; the merged view is the dataset's terms table: names merged by the label-match "
    "rule and linked to the concept they name, sentences merged when they read the same once case, punctuation and "
    "spacing are set aside"
)
PAGES_SITE_LIMIT = 2**30  # a GitHub Pages site holds at most 1 GB
PAGES_FILE_LIMIT = 100 * 2**20  # and, like a repository, files under 100 MB


def dataset_folder() -> Path:
    """The MBDef dataset as `mbdef data fetch` places it: the explorer's one link to where it is."""
    return files.data_root() / MBDEF


def source_files() -> list[str]:
    """The names of the files in `explorer/src/` that the page is made of, in the order they are read."""
    return [
        "shell.html",
        "base.css",
        "core.js",
        *(f"{tab}.{kind}" for tab in TABS for kind in ("css", "html", "js")),
    ]


def node_error(script: str) -> str | None:
    """None if node parses `script` as a JavaScript module, else node's message."""
    try:
        done = subprocess.run(
            NODE_CHECK, input=script, capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        raise SystemExit(
            "No page: node is needed for the syntax check of the scripts (nodejs.org)."
        ) from None
    if done.returncode == 0:
        return None
    return done.stderr.strip() or "syntax error"


def read(name: str) -> str:
    path = SRC / name
    if not path.is_file():
        raise SystemExit(
            f"No page: {path} is missing; run this command from the repository root."
        )
    return path.read_text(encoding="utf-8")


def links() -> dict[str, str]:
    """The addresses the page links to, filled into its `{{name}}` placeholders: the one place each is set."""
    from mbdef.dataset import EXPLORER_URL, PROJECT_URL

    return {
        "explorer_url": EXPLORER_URL,
        "project_url": PROJECT_URL,
        "mbdef_url": f"https://huggingface.co/datasets/{files.MBDEF_REPO}",
        "gold_url": f"https://huggingface.co/datasets/{files.MBDEF_GOLD_REPO}",
    }


def page() -> str:
    """The page that the sources in `explorer/src/` make. SystemExit if a script fails node --check."""
    text = {name: read(name) for name in source_files()}
    shell = text["shell.html"]
    for name, url in links().items():
        shell = shell.replace("{{" + name + "}}", url)
    for marker in (CSS_MARKER, SECTIONS_MARKER, JS_MARKER):
        if shell.count(marker) != 1:
            raise SystemExit(
                f"No page: {SRC / 'shell.html'} must hold the marker {marker} once."
            )
    css = "\n".join(text[f"{name}.css"] for name in ("base", *TABS))
    sections = "\n".join(text[f"{tab}.html"] for tab in TABS)
    blocks = [f"{{ // ---- {tab}.js\n{text[f'{tab}.js']}\n}}" for tab in TABS]
    js = "\n".join([text["core.js"], *blocks, BOOT])
    built = (
        shell.replace(CSS_MARKER, css)
        .replace(SECTIONS_MARKER, sections)
        .replace(JS_MARKER, js)
    )
    if REPLACEMENT_CHARACTER in built:
        raise SystemExit("No page: the sources hold U+FFFD, an unreadable character.")
    message = node_error(js)
    if message:  # name the scripts that fail on their own, else the joined script
        scripts = ["core.js", *(f"{tab}.js" for tab in TABS)]
        failed = [(name, node_error(text[name])) for name in scripts]
        lines = [f"{name}: {error}" for name, error in failed if error]
        raise SystemExit(
            "No page: node --check failed on\n" + "\n".join(lines)
            if lines
            else f"No page: node --check failed on the joined script\n{message}"
        )
    return built


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checked_bundle(bundle: Path) -> dict[str, bytes]:
    """Every file that the bundle's SHA256SUMS lists, by its path there. SystemExit if one is missing or differs."""
    sums = bundle / SUMS
    if not sums.is_file():
        raise SystemExit(
            f"No page data: {bundle} is missing (or has no {SUMS}). The MBDef dataset holds it; get the dataset "
            f"with `mbdef data fetch` or name its folder with --dataset."
        )
    listed = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, _, name = line.partition("  ")
        listed[name] = digest
    found, bad = {}, []
    for name, digest in listed.items():
        path = bundle / name
        data = path.read_bytes() if path.is_file() else None
        if data is None or sha256(data) != digest:
            bad.append(f"{name} {'differs' if data is not None else 'is missing'}")
        else:
            found[name] = data
    unlisted = sorted(
        p.relative_to(bundle).as_posix()
        for p in bundle.rglob("*")
        if p.is_file() and p.name != SUMS
    )
    bad += [f"{name} is not listed" for name in unlisted if name not in listed]
    if bad:
        raise SystemExit(
            f"No page data: {bundle} is not what its {SUMS} lists: {', '.join(bad)}."
        )
    return found


def mentions(bundle: dict[str, bytes], meta: dict) -> dict[str, list]:
    """The page's table of term mentions: the columns of its shards `mentions/NN.json`, joined."""
    shards = -(-meta["counts"]["terms"] // meta["shard"])
    table: dict[str, list] = {}
    for number in range(shards):
        shard = json.loads(bundle[f"mentions/{number:02d}.json"])
        for key, column in shard.items():
            if key != "n":
                table.setdefault(key, []).extend(column)
    return table


def definitions(bundle: dict[str, bytes], meta: dict) -> list[list[str]]:
    """The definitions of every mention, in mention order, from the shards `defs/NN.json`."""
    shards = -(-meta["counts"]["terms"] // meta["shard"])
    return [
        texts
        for number in range(shards)
        for texts in json.loads(bundle[f"defs/{number:02d}.json"])
    ]


def plural(n: int, word: str) -> str:
    return f"{n:,} {word}{'' if n == 1 else 's'}"


def disagreements(bundle: dict[str, bytes], tables: Path) -> tuple[list[str], str]:
    """Where the page data and the dataset's tables disagree on what the dataset holds, and a summary of what was compared.

    Compared: the mentions, their scores and definitions, the concepts and the papers; the page data must hold exactly the rows of the tables.
    """
    meta = json.loads(bundle["meta.json"])
    m = mentions(bundle, meta)
    papers = json.loads(bundle["papers.json"])
    instances = pl.read_parquet(tables / "instances.parquet")
    table_defs = pl.read_parquet(tables / "definitions.parquet")
    concepts = pl.read_parquet(tables / "concepts.parquet")
    table_papers = pl.read_parquet(tables / "papers.parquet")
    problems = []

    kept = range(len(m["mention_id"]))
    on_page = set(m["mention_id"])
    rows = {row["mention_id"]: row for row in instances.iter_rows(named=True)}
    if on_page != set(rows):
        only_page, only_table = sorted(on_page - set(rows)), sorted(set(rows) - on_page)
        parts = (
            [f"{len(only_page):,} only on the page, e.g. {only_page[0]}"]
            if only_page
            else []
        )
        parts += (
            [f"{len(only_table):,} only in instances, e.g. {only_table[0]}"]
            if only_table
            else []
        )
        problems.append(
            f"mentions: {len(on_page):,} on the page, {len(rows):,} in instances ({'; '.join(parts)})"
        )
    differ = []
    for i in kept:
        row = rows.get(m["mention_id"][i])
        if row is None:
            continue
        same = (
            all(
                round(row[column] * 10_000) == m[key][i]
                for key, column in SCORES.items()
            )
            and bool(row["strict"]) == bool(m["strict"][i])
            and row["term"] == m["label"][i]
            and row["paper_id"] == papers["id"][m["paper"][i]]
        )
        if not same:
            differ.append(m["mention_id"][i])
    if differ:
        problems.append(
            f"the scores, strict flag, term or paper of {plural(len(differ), 'mention')} differ from "
            f"instances (e.g. {differ[0]})"
        )

    texts: dict[str, list[str]] = {}
    for row in table_defs.select("mention_id", "text").iter_rows():
        texts.setdefault(row[0], []).append(row[1])
    page_defs = definitions(bundle, meta)
    differ = []
    for i in kept:
        expected = texts.get(m["mention_id"][i], [])
        if [text.replace(REPLACEMENT_CHARACTER, "?") for text in expected] != page_defs[
            i
        ]:
            differ.append(m["mention_id"][i])
    if differ:
        problems.append(
            f"the definitions of {plural(len(differ), 'mention')} differ from definitions (e.g. {differ[0]})"
        )

    members: dict[int, set[str]] = {}
    for i in kept:
        members.setdefault(m["group"][i], set()).add(m["mention_id"][i])
    concept_sets = {frozenset(ids) for ids in concepts["mention_ids"].to_list()}
    no_concept = [g for g, ids in members.items() if frozenset(ids) not in concept_sets]
    if no_concept or len(members) != concepts.height:
        problems.append(
            f"the mentions of {plural(len(no_concept), 'label group')} are no concept of concepts"
            if no_concept
            else f"the page has {len(members):,} label groups, but concepts holds {concepts.height:,} concepts"
        )

    page_papers = {
        papers["id"][p]: (papers["title"][p], papers["n"][p])
        for p in range(len(papers["id"]))
    }
    table = {
        row["paper_id"]: (row["title"], row["n_instances"])
        for row in table_papers.iter_rows(named=True)
    }
    differ = sorted(
        p for p in set(page_papers) | set(table) if page_papers.get(p) != table.get(p)
    )
    if differ:
        problems.append(
            f"{plural(len(differ), 'paper')} "
            f"{'differs' if len(differ) == 1 else 'differ'} from papers (id, title or count of mentions; e.g. {differ[0]})"
        )
    summary = (
        f"{plural(len(kept), 'mention')}, {plural(table_defs.height, 'definition')}, "
        f"{plural(concepts.height, 'concept')}, {plural(table_papers.height, 'paper')}"
    )
    return problems, summary


def published(name: str, data: bytes) -> bytes:
    """A page data file as the site publishes it: without the keys that record where and when it was made."""
    if name not in RUN_HISTORY:
        return data
    content = json.loads(data)
    for key in RUN_HISTORY[name]:
        content.pop(key, None)
    if name == "meta.json":
        content["method"] = METHOD
        if "fields" in content:
            content["fields"]["note"] = FIELDS_NOTE
    return json.dumps(content, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


def pages_limits(sizes: dict[str, int]) -> list[str]:
    """What keeps a site of these files (path -> bytes) off GitHub Pages: a file of 100 MB or more, over 1 GB in all."""
    mb = 2**20
    problems = [
        f"{name} is {size / mb:.1f} MB; GitHub Pages takes files under 100 MB"
        for name, size in sizes.items()
        if size >= PAGES_FILE_LIMIT
    ]
    total = sum(sizes.values())
    if total > PAGES_SITE_LIMIT:
        problems.append(
            f"the site is {total / mb:.1f} MB; a GitHub Pages site holds at most 1 GB"
        )
    return problems


def build(dataset: Path | None = None) -> int:
    """Write `explorer/index.html` from `explorer/src/` and `explorer/data/` from the MBDef dataset in `dataset`.

    `dataset` defaults to where `data fetch` puts it.
    Returns 0 when written; SystemExit, with nothing written, when a check fails.
    """
    dataset = dataset_folder() if dataset is None else dataset
    built = page().encode("utf-8")
    bundle = checked_bundle(dataset / BUNDLE)
    problems, summary = disagreements(bundle, dataset / TABLES)
    if problems:
        raise SystemExit(
            f"No page data: {dataset / BUNDLE} disagrees with the MBDef tables in {dataset / TABLES}:\n"
            + "\n".join(f"- {problem}" for problem in problems)
        )
    data = {name: published(name, content) for name, content in sorted(bundle.items())}
    data[SUMS] = "".join(
        f"{sha256(content)}  {name}\n" for name, content in data.items()
    ).encode()
    sizes = {PAGE.relative_to(FOLDER).as_posix(): len(built)}
    sizes |= {
        f"{DATA.relative_to(FOLDER).as_posix()}/{name}": len(content)
        for name, content in data.items()
    }
    too_big = pages_limits(sizes)
    if too_big:
        raise SystemExit("No site: " + "; ".join(too_big) + ".")
    print(f"The page data agrees with the MBDef tables: {summary}.")
    staging = DATA.with_name(DATA.name + ".part")
    if staging.exists():
        shutil.rmtree(staging)
    for name, content in data.items():
        path = staging / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    if DATA.exists():
        shutil.rmtree(DATA)
    staging.replace(DATA)
    PAGE.write_bytes(built)
    mb = 2**20
    data_size = sum(len(content) for content in data.values())
    largest = max(data, key=lambda name: len(data[name]))
    print(
        f"Wrote {PAGE} ({len(built) / 1024:.0f} KB) and {DATA}/ ({len(data):,} files, {data_size / mb:.1f} MB; "
        f"the largest, {largest}, {len(data[largest]) / mb:.1f} MB). The site is {sum(sizes.values()) / mb:.1f} MB."
    )
    return 0
