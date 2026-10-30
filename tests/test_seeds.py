"""The seed list (`data/seeds/`): the 3,869 seed papers with their Step 2 labels, and the starting corpus's identifiers (docs/step-1-2-papers/seeds.md).

Three kinds of test, each at one seam:

- the maker (`make_seed_list.py`) on a miniature: how a seed paper gets its two relevance labels;
- the shipped files as a reader sees them: counts, columns, identifiers, hashes, and labels that agree with the route;
- the shipped files against the recorded inputs of the seed run (`runner-rows.json`, `seed-manifest.jsonl` and `existing-corpus-tei.sha256` in the data root): the same papers, titles, DOIs and hashes. These need the data root and are marked slow, so CI skips them.

Every title and text below is synthetic.
"""

import csv
import json
import os
import re
from collections import Counter
from pathlib import Path

import pytest
from make_seed_list import (
    CORPUS_COLUMNS,
    SEED_COLUMNS,
    corpus_rows,
    seed_rows,
)

REPO = Path(__file__).resolve().parent.parent
SEEDS_TSV = REPO / "data/seeds/seed-papers.tsv"
CORPUS_TSV = REPO / "data/seeds/starting-corpus.tsv"
DATA_ROOT_LIST = REPO / "data/data-root.tsv"

# The numbers the page docs/step-1-2-papers/seeds.md reports; the recorded inputs and the files decide whether they hold.
SEED_PAPERS = 3869
STARTING_CORPUS = 14862
ROUTES = {"agreed": 2947, "tie-break": 918, "not-selected": 4}
FOLDER_PDFS = 3888  # rows of the seed manifest: the PDFs Step 2 handed over
WITHOUT_BODY = 4  # of those, PDFs GROBID found no article body in (notices)
WITHOUT_TITLE = 15  # of the rest, papers whose title could not be confirmed
# seeds whose TEI is, byte for byte, one of the starting corpus's
SAME_TEI_AS_CORPUS = 2458
WITHOUT_DOI = 484  # seeds whose TEI header has no DOI
WITH_CLEAN_DOI = 3350  # of the other 3,385, those that are just a DOI
SEEDS_SHARING_A_DOI = 224  # seeds whose DOI is also the DOI of another seed (73 DOIs)

# ---------------------------------------------------------------------------------------------
# A miniature: synthetic classifier outputs, runner rows and seed manifest.
# ---------------------------------------------------------------------------------------------


def prompt(title: str) -> str:
    """The tail of a classifier prompt: the instructions, then the article, then the answer format."""
    return (
        "Some instructions.\n\nArticle to classify:\nTitle: "
        f"{title}\nAbstract: Synthetic abstract of {title or 'nothing'}.\n\n"
        "Return exactly one label and no explanation:\nrelevant\nnot relevant\nnot sure\n\nAnswer:"
    )


def row(step2_id: str, title: str, label: str) -> dict[str, str]:
    return {"paperid": step2_id, "prompt": prompt(title), "clean_label": label}


def sanitised(step2_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", step2_id).strip("_")


def seed(n: int, title: str, step2_id: str, doi: str = "") -> tuple[dict, dict]:
    """One runner row and its manifest row, as the seed run recorded them."""
    pid = f"pdf-{n:064x}"
    runner = {
        "paper_id": pid,
        "frozen_title": title,
        "tei_sha256": f"{n + 1000:064x}",
        "tei_size_bytes": 10,
        "tei_path": "/elsewhere/x.tei.xml",
    }
    manifest = {
        "paper_id": pid,
        "source_filename": f"{sanitised(step2_id)}__{step2_id}.pdf",
        "tei_header_doi": doi,
        "source_pdf_sha256": f"{n + 2000:064x}",
        "tei_sha256": runner["tei_sha256"],
    }
    return runner, manifest


QWEN = [
    row("doi_10.1_a", "Alpha study", "relevant"),
    row("doi_10.1_b", "Beta study", "not relevant"),
    row("doi_10.1_c", "", "relevant"),
    row("doi_10.1_shared", "Gamma study", "relevant"),
    row("doi_10.1_shared", "Delta study", "not relevant"),
    row("doi_10.1_clash", "Other study", "relevant"),
    row("hash_0", "", "not relevant"),
    row("hash_0", "", "relevant"),
]
GEMMA = [
    row("doi_10.1_a", "Alpha study", "relevant"),
    row("doi_10.1_b", "Beta study", "relevant"),
    row("doi_10.1_c", "", "can't determine"),
    row("doi_10.1_shared", "Gamma study", "relevant"),
    row("doi_10.1_shared", "Delta study", "relevant"),
    row("doi_10.1_clash", "Other study", "relevant"),
    row("doi_10.1_clash", "Intruder study", "relevant"),
    row("hash_0", "", "relevant"),
    row("hash_0", "", "not relevant"),
]
MINI_SEEDS = [
    # one row in each file; both relevant
    seed(1, "Alpha study", "doi_10.1_a", "10.1/a"),
    # one row in each file; they differ
    seed(2, "Beta study", "doi_10.1_b"),
    # the classifiers saw no title; their labels still count
    seed(3, "Found by lookup", "doi_10.1_c"),
    # the id is shared: the title picks the row
    seed(4, "Gamma study", "doi_10.1_shared"),
    seed(5, "Delta study", "doi_10.1_shared"),
    # the id is shared and Qwen has no such paper
    seed(6, "Intruder study", "doi_10.1_clash"),
    # shared, and no row has a title to pick by
    seed(7, "Some paper without a title", "hash_0"),
]


def mini() -> dict[str, dict[str, str]]:
    runner = [r for r, _ in MINI_SEEDS]
    manifest = [m for _, m in MINI_SEEDS]
    return {r["title"]: r for r in seed_rows(runner, manifest, QWEN, GEMMA)}


def test_a_paper_with_its_own_id_gets_the_label_of_each_model():
    alpha = mini()["Alpha study"]
    assert (alpha["qwen_label"], alpha["gemma_label"]) == ("relevant", "relevant")
    assert alpha["step2_route"] == "agreed"
    assert alpha["step2_id"] == "doi_10.1_a"
    assert alpha["doi"] == "10.1/a"


def test_two_different_labels_are_a_tie_break_and_the_recorded_label_names_are_kept():
    rows = mini()
    beta = rows["Beta study"]
    assert (beta["qwen_label"], beta["gemma_label"]) == ("not relevant", "relevant")
    assert beta["step2_route"] == "tie-break"
    # the models saw no title for this paper, but its id is its own, so the labels are still its
    found = rows["Found by lookup"]
    assert (found["qwen_label"], found["gemma_label"]) == (
        "relevant",
        "can't determine",
    )
    assert found["step2_route"] == "tie-break"
    assert found["doi"] == ""


def test_a_shared_id_is_resolved_by_the_title_the_models_saw():
    rows = mini()
    gamma, delta = rows["Gamma study"], rows["Delta study"]
    assert (gamma["qwen_label"], gamma["gemma_label"], gamma["step2_route"]) == (
        "relevant",
        "relevant",
        "agreed",
    )
    assert (delta["qwen_label"], delta["gemma_label"], delta["step2_route"]) == (
        "not relevant",
        "relevant",
        "tie-break",
    )


def test_a_paper_that_no_row_describes_has_no_label_and_was_not_selected():
    rows = mini()
    # Gemma has a row for the intruder, Qwen has none: one label is not enough to be selected
    intruder = rows["Intruder study"]
    assert (intruder["qwen_label"], intruder["gemma_label"]) == ("", "relevant")
    assert intruder["step2_route"] == "not-selected"
    # an id whose rows carry no title cannot be resolved at all
    blank = rows["Some paper without a title"]
    assert (blank["qwen_label"], blank["gemma_label"]) == ("", "")
    assert blank["step2_route"] == "not-selected"


def test_the_rows_are_sorted_by_paper_id_and_carry_exactly_the_columns():
    rows = list(mini().values())
    assert [r["paper_id"] for r in rows] == sorted(r["paper_id"] for r in rows)
    assert all(list(r) == SEED_COLUMNS for r in rows)
    assert rows[0]["tei_sha256"] == f"{1001:064x}"
    assert rows[0]["pdf_sha256"] == f"{2001:064x}"


def test_a_paper_whose_id_matches_no_classifier_row_stops_the_maker():
    runner, manifest = seed(9, "Lost study", "doi_10.1_missing")
    with pytest.raises(SystemExit, match="doi_10_1_missing"):
        seed_rows([runner], [manifest], QWEN, GEMMA)


def test_the_starting_corpus_is_the_file_stems_and_hashes_of_the_tei_list():
    lines = [
        f"{'ab' * 32}  doi_10.1002_x.grobid.tei.xml",
        f"{'cd' * 32}  hash_00ff.grobid.tei.xml",
    ]
    rows = corpus_rows(lines)
    assert rows == [
        {"corpus_id": "doi_10.1002_x", "tei_sha256": "ab" * 32},
        {"corpus_id": "hash_00ff", "tei_sha256": "cd" * 32},
    ]
    assert all(list(r) == CORPUS_COLUMNS for r in rows)


# ---------------------------------------------------------------------------------------------
# The shipped files, as a reader sees them.
# ---------------------------------------------------------------------------------------------


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


@pytest.fixture(scope="module")
def seeds():
    return read_tsv(SEEDS_TSV)


@pytest.fixture(scope="module")
def corpus():
    return read_tsv(CORPUS_TSV)


HEX = re.compile(r"[0-9a-f]{64}")


def test_the_seed_list_has_the_columns_and_the_3869_papers(seeds):
    assert list(seeds[0]) == SEED_COLUMNS
    ids = [r["paper_id"] for r in seeds]
    assert len(ids) == SEED_PAPERS
    assert len(set(ids)) == SEED_PAPERS
    assert ids == sorted(ids)
    assert all(re.fullmatch(r"pdf-[0-9a-f]{64}", i) for i in ids)


def test_every_seed_has_a_title_and_both_hashes_and_a_doi_only_if_one_was_read(seeds):
    assert all(r["title"] == r["title"].strip() and r["title"] for r in seeds)
    assert all(
        HEX.fullmatch(r["tei_sha256"]) and HEX.fullmatch(r["pdf_sha256"]) for r in seeds
    )
    # the DOI is what GROBID read in the TEI header: empty, a DOI, or a DOI with text stuck to it
    assert all(r["doi"] == "" or "10." in r["doi"] for r in seeds)
    assert sum(1 for r in seeds if r["doi"] == "") == WITHOUT_DOI
    # plain: starts 10.<digits>/ and does not end in a full stop, comma, semicolon or bracket
    clean = [
        r["doi"] for r in seeds if re.fullmatch(r"10\.\d{4,9}/\S*[^\s.,;>)]", r["doi"])
    ]
    assert len(clean) == WITH_CLEAN_DOI
    shared = Counter(r["doi"] for r in seeds if r["doi"])
    assert sum(n for n in shared.values() if n > 1) == SEEDS_SHARING_A_DOI


def test_the_labels_are_the_recorded_ones_and_the_route_follows_from_them(seeds):
    allowed = {"relevant", "not relevant", "can't determine", ""}
    for r in seeds:
        q, g = r["qwen_label"], r["gemma_label"]
        assert q in allowed and g in allowed
        if q == g == "relevant":
            expected = "agreed"
        elif q and g and q != g:
            expected = "tie-break"
        else:
            expected = "not-selected"
        assert r["step2_route"] == expected, r["paper_id"]
    assert dict(Counter(r["step2_route"] for r in seeds)) == ROUTES


def test_a_worked_example_of_each_route(seeds):
    rows = {r["paper_id"]: r for r in seeds}
    # agreed: the first seed of the list (also the worked example of docs/step-1-2-papers/seeds.md)
    first = rows["pdf-00081f56d928d71df9cd19a54beea769be820c518212c39e49218810348123ef"]
    assert first["title"].startswith(
        "Media as Actors of Two-Stage Cultural Gatekeeping"
    )
    assert (first["qwen_label"], first["gemma_label"], first["step2_route"]) == (
        "relevant",
        "relevant",
        "agreed",
    )
    assert first["doi"] == "10.11114/smc.v13i4.8126"
    # the four seeds Step 2 did not select: they came in through an id they share with a selected paper
    not_selected = sorted(
        r["title"][:24] for r in seeds if r["step2_route"] == "not-selected"
    )
    assert not_selected == [
        "Check-It: A plugin for D",
        "In the frame: Signalling",
        "Securing Manufacturing U",
        "The Reliance on Lifestyl",
    ]


def test_93_titles_hold_a_double_quote_so_the_page_says_to_read_the_quoting(seeds):
    assert sum(1 for r in seeds if '"' in r["title"]) == 93


def test_the_seed_list_holds_identifiers_titles_and_labels_only(seeds):
    # no column can carry paper text: every field is short
    assert max(len(v) for r in seeds for v in r.values()) < 600


def test_the_starting_corpus_has_the_14862_identifiers_and_hashes(corpus):
    assert list(corpus[0]) == CORPUS_COLUMNS
    ids = [r["corpus_id"] for r in corpus]
    assert len(ids) == STARTING_CORPUS
    assert len(set(ids)) == STARTING_CORPUS
    assert ids == sorted(ids)
    assert all(HEX.fullmatch(r["tei_sha256"]) for r in corpus)
    assert len({r["tei_sha256"] for r in corpus}) == STARTING_CORPUS


def test_the_seeds_that_share_a_tei_with_the_starting_corpus_are_the_2458_the_page_reports(
    seeds, corpus
):
    shared = {r["tei_sha256"] for r in corpus}
    assert sum(1 for r in seeds if r["tei_sha256"] in shared) == SAME_TEI_AS_CORPUS


# ---------------------------------------------------------------------------------------------
# The shipped files against the recorded inputs of the seed run (data root).
# ---------------------------------------------------------------------------------------------


def data_root() -> Path | None:
    env = REPO / ".env"
    for line in env.read_text().splitlines() if env.exists() else []:
        if line.startswith("DATA_ROOT="):
            return Path(line.split("=", 1)[1]).expanduser()
    value = os.environ.get("DATA_ROOT")
    return Path(value).expanduser() if value else None


def recorded_input(name: str) -> Path | None:
    """Where the data root keeps one of the seed run's recorded inputs: the row of data/data-root.tsv whose path ends in `inputs/<name>`."""
    if ROOT is None:
        return None
    rows = DATA_ROOT_LIST.read_text(encoding="utf-8").splitlines()[1:]
    paths = [
        r.split("\t")[0] for r in rows if r.split("\t")[0].endswith(f"/inputs/{name}")
    ]
    return ROOT / paths[0] if len(paths) == 1 else None


ROOT = data_root()
SEED_MANIFEST = recorded_input("seed-manifest.jsonl")
needs_the_recorded_inputs = pytest.mark.skipif(
    SEED_MANIFEST is None or not SEED_MANIFEST.is_file(),
    reason="needs the data root with the seed run's recorded inputs",
)


@pytest.fixture(scope="module")
def recorded():
    def read(name: str) -> str:
        path = recorded_input(name)
        assert path is not None, f"data/data-root.tsv has no unique row for {name}"
        return path.read_text(encoding="utf-8")

    runner = json.loads(read("runner-rows.json"))
    manifest = [json.loads(line) for line in read("seed-manifest.jsonl").splitlines()]
    tei_list = read("existing-corpus-tei.sha256").splitlines()
    return runner, manifest, tei_list


@pytest.mark.slow
@needs_the_recorded_inputs
def test_the_recorded_rows_and_manifest_have_the_funnel_the_page_reports(recorded):
    runner, manifest, _ = recorded
    assert len(manifest) == FOLDER_PDFS
    without_body = [m for m in manifest if m.get("disposition")]
    assert len(without_body) == WITHOUT_BODY
    without_title = [
        m for m in manifest if m.get("title_source") == "empty_tei_header_title"
    ]
    assert len(without_title) == WITHOUT_TITLE
    assert len(runner) == SEED_PAPERS == FOLDER_PDFS - WITHOUT_BODY - WITHOUT_TITLE
    left_out = {m["paper_id"] for m in without_body + without_title}
    assert {r["paper_id"] for r in runner} == {
        m["paper_id"] for m in manifest
    } - left_out


@pytest.mark.slow
@needs_the_recorded_inputs
def test_the_seed_list_is_the_recorded_seeds_with_their_titles_dois_and_hashes(
    recorded, seeds
):
    runner, manifest, _ = recorded
    rows = {r["paper_id"]: r for r in runner}
    records = {m["paper_id"]: m for m in manifest}
    assert {r["paper_id"] for r in seeds} == set(rows)
    for r in seeds:
        run, rec = rows[r["paper_id"]], records[r["paper_id"]]
        assert r["title"] == run["frozen_title"]
        assert r["tei_sha256"] == run["tei_sha256"] == rec["tei_sha256"]
        assert r["pdf_sha256"] == rec["source_pdf_sha256"]
        assert r["doi"] == (rec.get("tei_header_doi") or "")
        # the Step 2 id is the one the hand-over file was named after
        assert sanitised(r["step2_id"]) == rec["source_filename"].split("__", 1)[0]


@pytest.mark.slow
@needs_the_recorded_inputs
def test_the_titles_come_from_the_tei_header_or_a_lookup_as_the_page_reports(recorded):
    runner, manifest, _ = recorded
    seed_ids = {r["paper_id"] for r in runner}
    sources = Counter(m["title_source"] for m in manifest if m["paper_id"] in seed_ids)
    assert dict(sources) == {
        "tei_header_title_frozen_at_staging": 3621,
        "seed_titles:openalex:doi": 224,
        "seed_titles:first-page:search_hit": 17,
        "seed_titles:pdf-meta:pdf_sha256": 7,
    }


def test_one_pdf_is_in_the_list_twice(seeds):
    pdfs = Counter(r["pdf_sha256"] for r in seeds)
    twice = [sha for sha, n in pdfs.items() if n > 1]
    assert len(twice) == 1 and pdfs[twice[0]] == 2
    pair = [r for r in seeds if r["pdf_sha256"] == twice[0]]
    assert len({r["title"] for r in pair}) == len({r["step2_id"] for r in pair}) == 1
    assert (
        len({r["paper_id"] for r in pair}) == len({r["tei_sha256"] for r in pair}) == 2
    )


@pytest.mark.slow
@needs_the_recorded_inputs
def test_the_starting_corpus_is_the_recorded_tei_list(recorded, corpus):
    _, _, tei_list = recorded
    recorded_rows = [line.split(None, 1) for line in tei_list]
    assert len(recorded_rows) == STARTING_CORPUS
    assert [(r["corpus_id"], r["tei_sha256"]) for r in corpus] == sorted(
        (name.removesuffix(".grobid.tei.xml"), sha) for sha, name in recorded_rows
    )


@pytest.mark.slow
@needs_the_recorded_inputs
def test_the_seeds_that_share_a_tei_with_the_starting_corpus_were_converted_before(
    recorded,
):
    runner, manifest, _ = recorded
    saved = {
        m["paper_id"]
        for m in manifest
        if m.get("tei_provenance") == "saved_grobid_exact_identity"
    }
    assert sum(1 for r in runner if r["paper_id"] in saved) == SAME_TEI_AS_CORPUS
