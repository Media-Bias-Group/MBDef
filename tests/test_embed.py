"""The embedding cache writes and reads the stored cache's files, and `mbdef embed build` fills it.

A cache is three files in one folder per model: `vectors.npy` (float16 rows), `keys.txt` (the sha1 of each row's text, one per line) and `manifest.json` (the settings). docs/step-5-post-processing/embed.md shows the fixture cache and the miniature corpus used here. The slow tests run by hand with DATA_ROOT set: one rebuilds the stored Qwen3-Embedding-8B cache on a CPU, one embeds a sample on a GPU.
"""

import hashlib
import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from cli_helpers import run_cli
from phase3_fixtures import (
    CACHE_BATCHES,
    CACHE_MODEL,
    CACHE_ROOT,
    cache_rows,
    cache_texts,
    write_cache,
)

from mbdef import embed, files
from mbdef.postprocess.embeddings import EmbeddingCache

REPO = Path(__file__).resolve().parents[1]
FILES = ("keys.txt", "vectors.npy", "manifest.json")
# keys.txt of the fixture cache: the sha1 of each text (the first two checked with sha1sum)
FIXTURE_KEYS = [
    "53d10a31e3e976843fb9c4f1229c44c735d513fb",  # "media bias: A slant in news coverage."
    "a67b95a680c9e83d30d8bdb5faf9da1e8361479e",  # "framing"
    "3c6d554012cd1fddd0fadd11b3d95b45198e75d5",
    "3a5abeb15eef7910b16e596d8c5cb2a3f8a84a04",
]
# sha256 of the stored qwen3-emb-8b manifest.json (data/data-root.tsv)
STORED_MANIFEST_SHA256 = (
    "43fe979b55aeb31984d69f2fed2f9878192b2a1dc7c1dd166cf5b55834a03369"
)


def all_texts():
    return [t for batch in CACHE_BATCHES for t in cache_texts(embed, batch)]


def test_a_written_cache_reads_back_its_float16_rows_by_text(tmp_path):
    write_cache(embed, np, tmp_path)
    texts = all_texts()

    cache = embed.Cache(tmp_path, CACHE_MODEL)

    assert cache.keys == FIXTURE_KEYS
    assert cache.manifest == embed.config(CACHE_MODEL)
    assert cache.vectors is not None
    assert cache.vectors.dtype == np.float16
    stored = cache_rows(np, 0, len(texts)).astype(np.float16).astype(np.float32)
    assert np.array_equal(cache.get(texts[::-1]), stored[::-1])
    assert cache.missing([*texts, "framing", "new text"]) == ["new text"]


def test_the_files_are_byte_identical_to_the_fixture_cache(tmp_path):
    """The fixture holds the files the original implementation of the cache wrote for the same texts and rows."""
    write_cache(embed, np, tmp_path)

    for name in FILES:
        written = (tmp_path / CACHE_MODEL / name).read_bytes()
        assert written == (CACHE_ROOT / CACHE_MODEL / name).read_bytes(), name


def test_step5_reads_a_cache_written_here(tmp_path):
    """Step 5's reader (`postprocess.embeddings.EmbeddingCache`: a memory map, keys split on newlines, the sha1 of 'term: text') finds the rows the cache wrote."""
    write_cache(embed, np, tmp_path)
    text = embed.definition_text("media bias", "A slant in news coverage.")

    reader = EmbeddingCache(tmp_path / CACHE_MODEL)

    assert isinstance(reader.matrix, np.memmap)
    assert len(reader.index) == reader.matrix.shape[0] == 4
    assert reader.index[hashlib.sha1(text.encode()).hexdigest()] == 0
    want = embed.Cache(tmp_path, CACHE_MODEL).get([text])
    assert np.array_equal(reader.vectors([text]), want)


def test_the_manifest_written_for_qwen3_is_the_stored_caches_manifest(tmp_path):
    cache = embed.Cache(tmp_path, "qwen3-emb-8b")
    cache.check_writable(embed.config("qwen3-emb-8b"))

    written = (tmp_path / "qwen3-emb-8b" / "manifest.json").read_bytes()
    assert hashlib.sha256(written).hexdigest() == STORED_MANIFEST_SHA256


def test_the_cache_refuses_to_mix_configurations(tmp_path):
    c = embed.Cache(tmp_path, "f2llm-v2-14b")
    c.check_writable(embed.config("f2llm-v2-14b"))  # FP8 manifest written
    c.add(["a"], np.ones((1, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="quantization"):
        embed.Cache(tmp_path, "f2llm-v2-14b").check_writable(
            embed.config("f2llm-v2-14b", bf16=True)
        )
    legacy = embed.Cache(tmp_path, "bge-m3")
    legacy.add(["b"], np.ones((1, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="manifest"):
        embed.Cache(tmp_path, "bge-m3").check_writable(embed.config("bge-m3"))


def stored_cache():
    root = Path(os.environ.get("DATA_ROOT", "")).expanduser() / "embeddings"
    if not (root / "qwen3-emb-8b" / "vectors.npy").is_file():
        pytest.skip("needs the stored cache: mbdef data fetch embeddings/")
    return embed.Cache(root, "qwen3-emb-8b")


@pytest.mark.slow
def test_the_stored_cache_opens_with_its_own_settings():
    cache = stored_cache()

    assert len(cache.keys) == cache.vectors.shape[0] == 380_882
    assert cache.vectors.shape[1] == 4096
    cache.check_writable(embed.config("qwen3-emb-8b"))  # raises if the settings differ


def test_environment_switches_do_not_change_a_vector(monkeypatch):
    """Method settings are arguments with pinned defaults, not environment keys: TAXONOMY_BF16 and TAXONOMY_EMBED_GPU_SHARE in the environment are ignored."""
    pinned = embed.config("f2llm-v2-14b")
    monkeypatch.setenv("TAXONOMY_BF16", "1")
    monkeypatch.setenv("TAXONOMY_EMBED_GPU_SHARE", "0.5")

    assert embed.config("f2llm-v2-14b") == pinned
    assert pinned["quantization"] == "fp8"
    assert embed.config("f2llm-v2-14b", bf16=True)["quantization"] is None


# `embed build` on a miniature corpus store: two seed papers and one closure paper, with synthetic text. Occurrences are inserted in this order (the store's row order), and their ids sort differently, so the build must follow the row order, not the ids.
STORE_PAPERS = [("s1", "seeds"), ("s2", "seeds"), ("c1", "closure")]
STORE_MENTIONS = [
    ("m1", "s1", "media bias"),
    (
        "m2",
        "s1",
        "framing",
    ),  # a term without definitions: its bare term is still embedded
    ("m3", "c1", "echo chamber"),
    ("m4", "c1", "media bias"),
    ("m5", "s2", "agenda setting"),
]
STORE_OCCURRENCES = [
    ("o9", "m3", "c1", "definitions", "A closed space where beliefs are amplified."),
    ("o5", "m1", "s1", "definitions", "A slant in news coverage."),
    ("o4", "m1", "s1", "synonyms", "slant"),  # not a definition
    ("o1", "m5", "s2", "definitions", "Choosing which issues the news covers."),
    ("o2", "m4", "c1", "definitions", "A slant in news coverage."),  # a seed text again
]
GT_RELEVANT = {
    "papers": [
        {
            "paper_id": "g1",
            "split": "train",
            "definition_records": [
                {
                    "term": "media bias",
                    "definitions": [
                        {"text": "A slant in news coverage."},
                        {"text": "  "},
                    ],
                },
                {
                    "term": "hostile media effect",
                    "definitions": [
                        {"text": "Partisans see coverage as biased against them."}
                    ],
                },
            ],
        }
    ]
}
GT_NOT_RELEVANT = {
    "papers": [
        {
            "paper_id": "g2",
            "split": "dev",
            "definition_records": [
                {
                    "term": "word embeddings",
                    "definitions": [],
                },  # no definition: no text at all
                {
                    "term": "tokenization",
                    "definitions": [{"text": "Splitting text into tokens."}],
                },
            ],
        }
    ]
}
COMPARISON = ["[X]: a slant in news coverage.", "framing"]
# The cache's rows, worked out by hand from the rules in docs/step-5-post-processing/embed.md: pass 1 (seed papers) then pass 2 (seed and closure papers); each pass embeds the store's definitions in row order, the ground truth's definitions, every term sorted, then the comparison texts, skipping texts the cache already has.
BUILT_TEXTS = [
    "media bias: A slant in news coverage.",
    "agenda setting: Choosing which issues the news covers.",
    "hostile media effect: Partisans see coverage as biased against them.",
    "tokenization: Splitting text into tokens.",
    "agenda setting",
    "framing",
    "hostile media effect",
    "media bias",
    "tokenization",
    "[X]: a slant in news coverage.",
    "echo chamber: A closed space where beliefs are amplified.",  # pass 2
    "echo chamber",
]


def mini_inputs(folder):
    """The miniature store, ground truth and comparison texts; the CLI flags that name them."""
    folder.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(folder / "library.sqlite3")
    db.executescript(
        "CREATE TABLE papers(paper_id TEXT PRIMARY KEY, title TEXT, doi TEXT, corpus TEXT);"
        "CREATE TABLE mentions(mention_id TEXT PRIMARY KEY, paper_id TEXT, term_loose TEXT,"
        " display_term TEXT, base_key TEXT, acronym TEXT, first_seen TEXT);"
        "CREATE TABLE occurrences(occ_id TEXT PRIMARY KEY, mention_id TEXT, paper_id TEXT,"
        " field TEXT, text TEXT, text_loose TEXT, citations TEXT);"
    )
    db.executemany("INSERT INTO papers(paper_id, corpus) VALUES (?, ?)", STORE_PAPERS)
    db.executemany(
        "INSERT INTO mentions(mention_id, paper_id, display_term) VALUES (?, ?, ?)",
        STORE_MENTIONS,
    )
    db.executemany(
        "INSERT INTO occurrences(occ_id, mention_id, paper_id, field, text) VALUES (?, ?, ?, ?, ?)",
        STORE_OCCURRENCES,
    )
    db.commit()
    db.close()
    for name, data in (
        ("relevant.json", GT_RELEVANT),
        ("not-relevant.json", GT_NOT_RELEVANT),
        ("comparison.json", COMPARISON),
    ):
        (folder / name).write_text(json.dumps(data), encoding="utf-8")
    return [
        "--store",
        str(folder / "library.sqlite3"),
        "--ground-truth",
        str(folder / "relevant.json"),
        "--ground-truth",
        str(folder / "not-relevant.json"),
        "--comparison",
        str(folder / "comparison.json"),
    ]


def recorded_vector(text):
    """A distinct unit row of four numbers for each text (exact in float16)."""
    digits = [int(c, 16) + 1 for c in hashlib.sha1(text.encode()).hexdigest()[:4]]
    return np.array(digits, dtype=np.float16) / np.float16(16)


def recorded_cache(folder, texts):
    """A recorded cache folder (keys.txt, vectors.npy) holding TEXTS in the given order."""
    folder.mkdir(parents=True)
    keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
    (folder / "keys.txt").write_text("\n".join(keys))
    np.save(folder / "vectors.npy", np.stack([recorded_vector(t) for t in texts]))
    return folder


def test_build_fills_the_cache_in_two_passes_seed_papers_first(tmp_path):
    flags = mini_inputs(tmp_path / "in")
    # recorded in another order, with one text the build does not need
    replay = recorded_cache(tmp_path / "rec", [*BUILT_TEXTS[::-1], "unused: text"])

    done = run_cli(
        tmp_path,
        "embed",
        "build",
        "--out",
        str(tmp_path / "out"),
        *flags,
        "--replay",
        str(replay),
    )

    assert done.returncode == 0, done.stderr
    cache = tmp_path / "out" / "qwen3-emb-8b"
    keys = [hashlib.sha1(t.encode()).hexdigest() for t in BUILT_TEXTS]
    assert (cache / "keys.txt").read_bytes() == "\n".join(keys).encode()
    vectors = np.load(cache / "vectors.npy")
    assert vectors.dtype == np.float16
    assert np.array_equal(vectors, np.stack([recorded_vector(t) for t in BUILT_TEXTS]))
    written = (cache / "manifest.json").read_bytes()
    assert hashlib.sha256(written).hexdigest() == STORED_MANIFEST_SHA256
    assert "pass 1 (seeds): 12 texts, 10 new" in done.stdout
    assert "pass 2 (seeds, closure): 15 texts, 2 new" in done.stdout


def test_build_stops_when_the_recorded_cache_lacks_a_text(tmp_path):
    flags = mini_inputs(tmp_path / "in")
    replay = recorded_cache(tmp_path / "rec", BUILT_TEXTS[:-1])  # no "echo chamber"

    done = run_cli(
        tmp_path,
        "embed",
        "build",
        "--out",
        str(tmp_path / "out"),
        *flags,
        "--replay",
        str(replay),
    )

    assert done.returncode == 1
    assert "has no recorded vector" in done.stderr and "'echo chamber'" in done.stderr
    keys = (tmp_path / "out" / "qwen3-emb-8b" / "keys.txt").read_text().split("\n")
    assert len(keys) == 10  # pass 1 is kept; pass 2 added nothing


def test_build_on_a_complete_cache_adds_nothing_and_continues_an_unfinished_one(
    tmp_path,
):
    flags = mini_inputs(tmp_path / "in")
    out = tmp_path / "out"
    first = run_cli(
        tmp_path,
        "embed",
        "build",
        "--out",
        str(out),
        *flags,
        "--replay",
        str(recorded_cache(tmp_path / "short", BUILT_TEXTS[:-1])),
    )
    assert first.returncode == 1  # stopped in pass 2

    again = run_cli(
        tmp_path,
        "embed",
        "build",
        "--out",
        str(out),
        *flags,
        "--replay",
        str(recorded_cache(tmp_path / "rec", BUILT_TEXTS)),
    )
    complete = (out / "qwen3-emb-8b" / "keys.txt").read_bytes()
    once_more = run_cli(
        tmp_path, "embed", "build", "--out", str(out), *flags
    )  # no GPU needed

    assert again.returncode == 0, again.stderr
    assert "pass 1 (seeds): 12 texts, 0 new" in again.stdout
    assert "pass 2 (seeds, closure): 15 texts, 2 new" in again.stdout
    assert (
        complete
        == "\n".join(hashlib.sha1(t.encode()).hexdigest() for t in BUILT_TEXTS).encode()
    )
    assert once_more.returncode == 0, once_more.stderr
    assert "pass 2 (seeds, closure): 15 texts, 0 new" in once_more.stdout
    assert (out / "qwen3-emb-8b" / "keys.txt").read_bytes() == complete


def test_build_sample_embeds_a_fixed_draw_of_the_texts_in_cache_order(tmp_path):
    flags = mini_inputs(tmp_path / "in")
    replay = str(recorded_cache(tmp_path / "rec", BUILT_TEXTS))
    position = {
        hashlib.sha1(t.encode()).hexdigest(): i for i, t in enumerate(BUILT_TEXTS)
    }

    runs = [
        run_cli(
            tmp_path,
            "embed",
            "build",
            "--out",
            str(tmp_path / name),
            *flags,
            "--replay",
            replay,
            "--sample",
            "3",
        )
        for name in ("a", "b")
    ]

    assert all(done.returncode == 0 for done in runs), runs[0].stderr
    assert "pass 1 (a sample of 12): 3 texts, 3 new" in runs[0].stdout
    keys = (tmp_path / "a" / "qwen3-emb-8b" / "keys.txt").read_text().split("\n")
    assert len(keys) == 3 and all(k in position for k in keys)
    assert [position[k] for k in keys] == sorted(position[k] for k in keys)
    assert (tmp_path / "b" / "qwen3-emb-8b" / "keys.txt").read_text().split(
        "\n"
    ) == keys


def test_build_writes_a_receipt_with_the_settings_and_the_caches_hashes(tmp_path):
    flags = mini_inputs(tmp_path / "in")
    replay = recorded_cache(tmp_path / "rec", BUILT_TEXTS)

    done = run_cli(
        tmp_path,
        "embed",
        "build",
        "--out",
        str(tmp_path / "out"),
        *flags,
        "--replay",
        str(replay),
    )

    assert done.returncode == 0, done.stderr
    receipt = json.loads((tmp_path / "out" / "receipt.json").read_text())
    assert receipt["command"] == "mbdef embed build"
    assert receipt["settings"]["passes"] == [["seeds"], ["seeds", "closure"]]
    assert receipt["settings"]["manifest"] == embed.config("qwen3-emb-8b")
    assert receipt["settings"]["replay"] == str(replay)
    store = (tmp_path / "in" / "library.sqlite3").read_bytes()
    assert receipt["inputs"]["store"]["sha256"] == hashlib.sha256(store).hexdigest()
    keys = (tmp_path / "out" / "qwen3-emb-8b" / "keys.txt").read_bytes()
    assert receipt["outputs"]["keys.txt"]["sha256"] == hashlib.sha256(keys).hexdigest()
    assert f"keys.txt sha256 {hashlib.sha256(keys).hexdigest()}" in done.stdout


# sha256 of the stored cache's files (data/data-root.tsv)
STORED_SHA256 = {
    "keys.txt": "65ebe17668f856bc1fe1ca0c50c63143b7a8eb613a0ede186fa65029cb8479dd",
    "vectors.npy": "37456fbb95c619d9b63aaa82fcbd41d69a21edc3b868ef8665332fe42fa305c3",
    "manifest.json": STORED_MANIFEST_SHA256,
}


def build_inputs():
    """DATA_ROOT with the stored cache, the corpus store and the comparison texts."""
    from mbdef.postprocess import corpus

    root = Path(os.environ.get("DATA_ROOT", "")).expanduser()
    needed = ("embeddings/qwen3-emb-8b/vectors.npy", corpus.STORE, embed.COMPARISON)
    if not all((root / path).is_file() for path in needed):
        pytest.skip(f"needs in DATA_ROOT: {', '.join(needed)} (mbdef data fetch)")
    return root


@pytest.mark.slow
def test_build_replaying_the_stored_cache_rebuilds_its_three_files_byte_for_byte(
    tmp_path,
):
    """On a CPU, about 40 s; it writes 3.1 GB into the temporary folder and needs about 8 GB of memory. The pinned inputs from the data root, every vector taken from the stored cache."""
    root = build_inputs()
    out = tmp_path / "out"

    done = run_cli(
        tmp_path,
        "embed",
        "build",
        "--out",
        str(out),
        "--replay",
        str(root / "embeddings" / "qwen3-emb-8b"),
        env={"DATA_ROOT": str(root)},
    )

    assert done.returncode == 0, done.stderr
    assert "pass 1 (seeds): 226,725 texts, 178,937 new" in done.stdout
    assert "pass 2 (seeds, closure): 429,518 texts, 201,945 new" in done.stdout
    for name, sha256 in STORED_SHA256.items():
        assert files.file_sha256(out / "qwen3-emb-8b" / name) == sha256, name
    (out / "qwen3-emb-8b" / "vectors.npy").unlink()  # 3.1 GB that pytest would keep


@pytest.mark.slow
def test_a_sample_embedded_on_a_gpu_matches_the_stored_vectors(tmp_path):
    """On a GPU with about 21 GB free (one at a time: run it under flock). Embeds 256 texts of the build drawn with the pinned seed and compares each with its stored float16 row."""
    root = build_inputs()
    out = tmp_path / "out"

    done = run_cli(
        tmp_path,
        "embed",
        "build",
        "--out",
        str(out),
        "--sample",
        "256",
        env={"DATA_ROOT": str(root)},
    )

    assert done.returncode == 0, done.stderr
    fresh = embed.Cache(out, "qwen3-emb-8b")
    stored = root / "embeddings" / "qwen3-emb-8b"
    index = {k: i for i, k in enumerate((stored / "keys.txt").read_text().split("\n"))}
    rows = np.load(stored / "vectors.npy", mmap_mode="r")
    old = np.stack([rows[index[k]] for k in fresh.keys])  # float16
    new = fresh.vectors  # float16
    assert new is not None
    cosine = np.sum(new.astype(np.float32) * old.astype(np.float32), axis=1)
    steps = np.abs(new.astype(np.float32) - old.astype(np.float32)) / np.spacing(
        old
    ).astype(np.float32)  # how many float16 steps apart, element by element
    print(
        f"256 texts: cosine min {cosine.min():.6f}, median {np.median(cosine):.6f};"
        f" elements equal {np.mean(steps == 0):.4f}, within 1 float16 step"
        f" {np.mean(steps <= 1):.4f}, max {steps.max():.0f} steps;"
        f" max |difference| {np.abs(new.astype(np.float32) - old.astype(np.float32)).max():.2e}"
    )
    assert cosine.min() >= 0.999
