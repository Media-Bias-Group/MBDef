"""`mbdef explorer build` on a miniature MBDef dataset: the page, its data, and the checks between them.

The miniature dataset has two term mentions of two papers, one concept ("framing").
Its page data (the folder `explorer/` of the dataset) holds exactly them, in shards of two mentions; page data that also holds a mention the dataset does not (one the relevance filter dropped) is refused.
These tests need `node`, which the page build calls for its syntax check.
"""

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from cli_helpers import run_cli

from mbdef import explorer
from mbdef.dataset import build as dataset_build
from mbdef.dataset.schema import frame

TABS = ("overview", "map", "concepts", "relations", "relevance", "papers")
SHELL = "<style>\n/*CSS*/\n</style>\n<main>\n<!--SECTIONS-->\n</main>\n<script>\n//JS\n</script>\n"


def write_sources(work: Path) -> Path:
    """An explorer folder whose sources hold one line per file, each naming its file."""
    src = work / "explorer/src"
    src.mkdir(parents=True)
    (src / "shell.html").write_text(SHELL, encoding="utf-8")
    (src / "base.css").write_text("body { margin: 0; }", encoding="utf-8")
    (src / "core.js").write_text(
        "const TABS = {};\nasync function boot() {}", encoding="utf-8"
    )
    for tab in TABS:
        (src / f"{tab}.css").write_text(f".{tab} {{}}", encoding="utf-8")
        (src / f"{tab}.html").write_text(
            f'<section id="v-{tab}"></section>', encoding="utf-8"
        )
        (src / f"{tab}.js").write_text(f"TABS.{tab} = {{}};", encoding="utf-8")
    return src


# The page that write_sources() makes, written out by hand.
# The CSS files joined with newlines, base.css first; the sections in tab order; core.js, then each tab's script in its own block, then the line that starts the page.
EXPECTED_PAGE = """\
<style>
body { margin: 0; }
.overview {}
.map {}
.concepts {}
.relations {}
.relevance {}
.papers {}
</style>
<main>
<section id="v-overview"></section>
<section id="v-map"></section>
<section id="v-concepts"></section>
<section id="v-relations"></section>
<section id="v-relevance"></section>
<section id="v-papers"></section>
</main>
<script>
const TABS = {};
async function boot() {}
{ // ---- overview.js
TABS.overview = {};
}
{ // ---- map.js
TABS.map = {};
}
{ // ---- concepts.js
TABS.concepts = {};
}
{ // ---- relations.js
TABS.relations = {};
}
{ // ---- relevance.js
TABS.relevance = {};
}
{ // ---- papers.js
TABS.papers = {};
}
boot().catch((e) => { $('#loading').textContent = 'The dataset could not be loaded: ' + e.message; console.error(e); });
</script>
"""

# The miniature page data, file by file.
# Mentions 0 and 1: "framing" in paper A, "Framing" in paper B. Scores are stored x 10,000.
MENTIONS: dict[str, Any] = {
    "mentions/00.json": {
        "n": 2,
        "label": ["framing", "Framing"],
        "paper": [0, 1],
        "group": [0, 0],
        "c": [9000, 8000],
        "b": [9500, 7000],
        "j": [10000, 6000],
        "s": [9381, 6950],
        "strict": [1, 1],
        "mention_id": ["m1", "m2"],
    },
    "defs/00.json": [
        ["Framing selects some aspects of a perceived reality."],
        ["Framing makes some aspects more salient.", "A frame is a schema."],
    ],
    "groups.json": {"name": ["framing"], "n": [2]},
    "papers.json": {
        "id": ["paper-a", "paper-b"],
        "title": ["Framing: Toward Clarification", "Media frames"],
        "n": [1, 1],
    },
    "relations.json": {"fields": ["sp"], "edges": []},
    "meta.json": {
        "cut": 0.415,
        "shard": 2,
        "counts": {
            "terms": 2,
            "dropped": 1,
            "definitions": 3,
            "concepts": 1,
            "papers": 2,
        },
        "concepts": {"concepts": 1},
        "method": {"umap": "UMAP of 5 Oct 2026 (results/p28-view-20261005/)"},
        "provenance": {"commit": "bcee3cc", "run_at": "2026-10-22T08:42:49+00:00"},
        "replaced_unreadable_characters": 0,
    },
    "ensemble-ref.json": {
        "n_ref": 2,
        "ref": {"c": [0.1, 0.9]},
        "sources": {"decisions": "/home/someone/results/ensemble.json"},
        "provenance": {"commit": "d80c672"},
    },
}

# The same mentions as the MBDef tables hold them.
INSTANCES = [
    {
        "mention_id": "m1",
        "concept_id": "g1",
        "term": "framing",
        "paper_id": "paper-a",
        "ensemble_score": 0.93814,
        "strict": True,
        "classifier_score": 0.9,
        "basal_score": 0.95,
        "judge_score": 1.0,
    },
    {
        "mention_id": "m2",
        "concept_id": "g1",
        "term": "Framing",
        "paper_id": "paper-b",
        "ensemble_score": 0.69501,
        "strict": True,
        "classifier_score": 0.8,
        "basal_score": 0.7,
        "judge_score": 0.6,
    },
]
DEFINITIONS = [
    {
        "def_id": "o1",
        "mention_id": "m1",
        "text": "Framing selects some aspects of a perceived reality.",
    },
    {
        "def_id": "o2",
        "mention_id": "m2",
        "text": "Framing makes some aspects more salient.",
    },
    {"def_id": "o3", "mention_id": "m2", "text": "A frame is a schema."},
]
CONCEPTS = [{"concept_id": "g1", "label": "framing", "mention_ids": ["m1", "m2"]}]
PAPERS = [
    {"paper_id": "paper-a", "title": "Framing: Toward Clarification", "n_instances": 1},
    {"paper_id": "paper-b", "title": "Media frames", "n_instances": 1},
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_dataset(
    folder: Path, page_data: dict | None = None, **tables: list[dict]
) -> Path:
    """A miniature MBDef dataset in `folder`: its tables in data/ and its page data in explorer/ with SHA256SUMS."""
    data = folder / "data"
    data.mkdir(parents=True)
    rows = {
        "instances": INSTANCES,
        "definitions": DEFINITIONS,
        "concepts": CONCEPTS,
        "papers": PAPERS,
    } | tables
    for name, table in rows.items():
        frame(table, name).write_parquet(data / f"{name}.parquet")
    bundle = folder / "explorer"
    sums = []
    for name, content in sorted((page_data or MENTIONS).items()):
        path = bundle / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")
        sums.append(f"{sha256(path.read_bytes())}  {name}\n")
    (bundle / "SHA256SUMS").write_text("".join(sums), encoding="utf-8")
    return folder


def build_here(work: Path, dataset: Path, monkeypatch, capsys) -> SimpleNamespace:
    """`explorer.build(dataset)` run in `work`, its outcome as the command line reports it.

    The command line's own start-up takes seconds, so most tests call the function.
    """
    monkeypatch.chdir(work)
    try:
        code, error = explorer.build(dataset), ""
    except SystemExit as stop:
        code, error = 1, f"{stop}\n"
    return SimpleNamespace(
        returncode=code, stdout=capsys.readouterr().out, stderr=error
    )


def built_files(work: Path) -> dict[str, bytes]:
    data = work / "explorer/data"
    return {
        p.relative_to(data).as_posix(): p.read_bytes()
        for p in sorted(data.rglob("*"))
        if p.is_file()
    }


def test_the_page_links_the_addresses_set_for_the_dataset_cards(tmp_path, monkeypatch):
    """`{{name}}` placeholders in shell.html take the addresses the dataset cards use, so each is set in one place."""
    from mbdef.dataset import EXPLORER_URL, PROJECT_URL

    src = write_sources(tmp_path)
    shell = (src / "shell.html").read_text(encoding="utf-8")
    (src / "shell.html").write_text(
        '<a href="{{project_url}}"></a><a href="{{mbdef_url}}"></a><a href="{{gold_url}}"></a>'
        '<meta content="{{explorer_url}}">' + shell,
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    built = explorer.page()

    assert built.startswith(
        f'<a href="{PROJECT_URL}"></a><a href="https://huggingface.co/datasets/mediabiasgroup/MBDef"></a>'
        f'<a href="https://huggingface.co/datasets/mediabiasgroup/MBDef-Gold"></a><meta content="{EXPLORER_URL}">'
    )
    assert "{{" not in built


def test_build_writes_the_page_and_the_data_the_page_reads(tmp_path):
    write_sources(tmp_path)
    dataset = write_dataset(tmp_path / "MBDef")

    done = run_cli(tmp_path, "explorer", "build", "--dataset", str(dataset))

    assert done.returncode == 0, done.stdout + done.stderr
    assert (tmp_path / "explorer/index.html").read_text(
        encoding="utf-8"
    ) == EXPECTED_PAGE
    built = built_files(tmp_path)
    assert sorted(built) == sorted([*MENTIONS, "SHA256SUMS"])
    for name in ("mentions/00.json", "defs/00.json", "groups.json", "relations.json"):
        assert built[name] == (dataset / "explorer" / name).read_bytes(), name
    assert done.stdout.startswith(
        "The page data agrees with the MBDef tables: 2 mentions, 3 definitions, 1 concept, 2 papers.\n"
        "Wrote explorer/index.html (1 KB) and explorer/data/ (8 files, "
    )


def test_the_published_data_holds_no_run_history(tmp_path, monkeypatch, capsys):
    """The dataset's page data records where and when it was made (commits, run times, local paths).

    The site publishes the data without those keys, and describes the map in plain words.
    """
    write_sources(tmp_path)
    dataset = write_dataset(tmp_path / "MBDef")

    done = build_here(tmp_path, dataset, monkeypatch, capsys)

    assert done.returncode == 0, done.stdout + done.stderr
    built = built_files(tmp_path)
    meta = json.loads(built["meta.json"])
    ref = json.loads(built["ensemble-ref.json"])
    assert "provenance" not in meta
    assert meta["method"] == explorer.METHOD
    assert meta["counts"] == MENTIONS["meta.json"]["counts"]
    assert ref == {"n_ref": 2, "ref": {"c": [0.1, 0.9]}}
    for text in (built["meta.json"], built["ensemble-ref.json"]):
        assert b"2026" not in text and b"results/" not in text and b"/home/" not in text


def test_sha256sums_lists_every_published_file(tmp_path, monkeypatch, capsys):
    write_sources(tmp_path)
    dataset = write_dataset(tmp_path / "MBDef")

    build_here(tmp_path, dataset, monkeypatch, capsys)

    built = built_files(tmp_path)
    listed = built.pop("SHA256SUMS").decode().splitlines()
    assert listed == [f"{sha256(built[name])}  {name}" for name in sorted(built)]


def test_the_dataset_is_read_from_the_data_root_by_default(tmp_path):
    write_sources(tmp_path)
    write_dataset(tmp_path / "root" / explorer.MBDEF)

    done = run_cli(
        tmp_path, "explorer", "build", env={"DATA_ROOT": str(tmp_path / "root")}
    )

    assert done.returncode == 0, done.stdout + done.stderr
    assert (tmp_path / "explorer/data/meta.json").exists()


def test_a_page_data_file_that_differs_from_its_sha256sums_stops_the_build(
    tmp_path, monkeypatch, capsys
):
    write_sources(tmp_path)
    dataset = write_dataset(tmp_path / "MBDef")
    (dataset / "explorer/groups.json").write_text("{}", encoding="utf-8")

    done = build_here(tmp_path, dataset, monkeypatch, capsys)

    assert done.returncode == 1
    assert done.stderr == (
        f"No page data: {dataset / 'explorer'} is not what its SHA256SUMS lists: groups.json differs.\n"
    )
    assert not (tmp_path / "explorer/data").exists()
    assert not (tmp_path / "explorer/index.html").exists()


@pytest.mark.parametrize(
    "tables, problem",
    [
        (
            {"instances": INSTANCES[:1]},
            "mentions: 2 on the page, 1 in instances (1 only on the page, e.g. m2)",
        ),
        (
            {"instances": [INSTANCES[0], INSTANCES[1] | {"judge_score": 0.61}]},
            "the scores, strict flag, term or paper of 1 mention differ from instances (e.g. m2)",
        ),
        (
            {"definitions": DEFINITIONS[:2]},
            "the definitions of 1 mention differ from definitions (e.g. m2)",
        ),
        (
            {"concepts": [CONCEPTS[0] | {"mention_ids": ["m1"]}]},
            "the mentions of 1 label group are no concept of concepts",
        ),
        (
            {"papers": [PAPERS[0], PAPERS[1] | {"title": "Media Frames"}]},
            "1 paper differs from papers (id, title or count of mentions; e.g. paper-b)",
        ),
    ],
)
def test_page_data_that_disagrees_with_the_tables_stops_the_build(
    tmp_path, tables, problem, monkeypatch, capsys
):
    write_sources(tmp_path)
    dataset = write_dataset(tmp_path / "MBDef", **tables)

    done = build_here(tmp_path, dataset, monkeypatch, capsys)

    assert done.returncode == 1
    assert done.stderr == (
        f"No page data: {dataset / 'explorer'} disagrees with the MBDef tables in {dataset / 'data'}:\n"
        f"- {problem}\n"
    )
    assert not (tmp_path / "explorer/data").exists()


def test_page_data_with_a_mention_the_dataset_does_not_hold_stops_the_build(
    tmp_path, monkeypatch, capsys
):
    """Page data that still holds a mention the relevance filter dropped (earlier page data did) is refused."""
    write_sources(tmp_path)
    page = json.loads(json.dumps(MENTIONS))
    shard = page["mentions/00.json"]
    page["mentions/01.json"] = {key: [] for key in shard} | {"n": 1}
    for key, value in dict(
        label="tokenization",
        paper=1,
        group=1,
        c=1000,
        b=500,
        j=0,
        s=500,
        strict=0,
        mention_id="m3",
    ).items():
        page["mentions/01.json"][key] = [value]
    page["defs/01.json"] = [["Tokenization splits text into tokens."]]
    page["meta.json"]["counts"]["terms"] = 3
    page["groups.json"] = {"name": ["framing", "tokenization"], "n": [2, 1]}
    page["papers.json"]["n"] = [1, 2]
    dataset = write_dataset(tmp_path / "MBDef", page)

    done = build_here(tmp_path, dataset, monkeypatch, capsys)

    assert done.returncode == 1
    assert (
        "- mentions: 3 on the page, 2 in instances (1 only on the page, e.g. m3)\n"
        in done.stderr
    )
    assert not (tmp_path / "explorer/data").exists()


def test_an_unreadable_character_counts_as_the_question_mark_the_page_shows(
    tmp_path, monkeypatch, capsys
):
    """The page data shows "?" where the dataset has U+FFFD, the mark of an unreadable character."""
    write_sources(tmp_path)
    definitions = [
        DEFINITIONS[0]
        | {"text": "Framing selects some aspects of a perceived reality�"},
        *DEFINITIONS[1:],
    ]
    page = json.loads(json.dumps(MENTIONS))
    page["defs/00.json"][0] = ["Framing selects some aspects of a perceived reality?"]
    dataset = write_dataset(tmp_path / "MBDef", page, definitions=definitions)

    done = build_here(tmp_path, dataset, monkeypatch, capsys)

    assert done.returncode == 0, done.stdout + done.stderr


def test_a_missing_dataset_is_named_with_how_to_get_it(tmp_path, monkeypatch, capsys):
    write_sources(tmp_path)

    done = build_here(tmp_path, tmp_path / "nowhere", monkeypatch, capsys)

    assert done.returncode == 1
    assert done.stderr.startswith(
        f"No page data: {tmp_path / 'nowhere' / 'explorer'} is missing"
    )
    assert "mbdef data fetch" in done.stderr


def test_a_script_that_node_cannot_parse_is_named_and_nothing_is_written(tmp_path):
    src = write_sources(tmp_path)
    (src / "papers.js").write_text("TABS.papers = {;", encoding="utf-8")
    dataset = write_dataset(tmp_path / "MBDef")

    done = run_cli(tmp_path, "explorer", "build", "--dataset", str(dataset))

    assert done.returncode == 1
    assert done.stderr.startswith("No page: node --check failed on\npapers.js: ")
    assert "core.js" not in done.stderr
    assert not (tmp_path / "explorer/index.html").exists()
    assert not (tmp_path / "explorer/data").exists()


@pytest.mark.parametrize(
    "sizes, problems",
    [
        ({"index.html": 500_000, "data/groups.json": 6_000_000}, []),
        (
            {"index.html": 500_000, "data/big.json": 100 * 2**20},
            ["data/big.json is 100.0 MB; GitHub Pages takes files under 100 MB"],
        ),
        (
            {f"data/{k}.json": 90 * 2**20 for k in range(12)},
            ["the site is 1080.0 MB; a GitHub Pages site holds at most 1 GB"],
        ),
    ],
)
def test_the_site_must_fit_github_pages(sizes, problems):
    assert explorer.pages_limits(sizes) == problems


def test_the_dataset_build_takes_the_published_page_data_only(
    tmp_path, monkeypatch, capsys
):
    """`dataset build --explorer-data` copies the page data that `explorer build` wrote into the dataset as explorer/, after the same checks against the tables; the dataset's own page data, which records its run history, is refused."""
    write_sources(tmp_path)
    dataset = write_dataset(tmp_path / "MBDef")
    assert build_here(tmp_path, dataset, monkeypatch, capsys).returncode == 0

    def run(bundle: Path) -> tuple[str, dict, Path]:
        folder = tmp_path / f"release-{bundle.parent.name}"
        shutil.copytree(dataset / "data", folder / "data")
        checks: dict[str, bool] = {}
        sentence = dataset_build.explorer_data(
            folder, bundle, lambda name, ok, detail="": checks.update({name: ok})
        )
        return sentence, checks, folder

    sentence, checks, folder = run(tmp_path / "explorer/data")
    assert all(checks.values()) and len(checks) == 3, checks
    assert built_files(tmp_path) == {
        p.relative_to(folder / "explorer").as_posix(): p.read_bytes()
        for p in sorted((folder / "explorer").rglob("*"))
        if p.is_file()
    }
    assert "It holds exactly the rows of these tables" in sentence

    _, checks, _ = run(dataset / "explorer")
    assert checks == {
        "explorer data: every file as its SHA256SUMS lists": True,
        "explorer data agrees with the tables (exactly their rows, scores, definitions, concepts, papers)": True,
        "explorer data holds no run history (commits, run times, local paths)": False,
    }
