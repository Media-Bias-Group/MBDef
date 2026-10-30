"""Inputs of the committed embedding-cache fixture in tests/fixtures/phase3/embed-cache/.

The fixture holds the files the original implementation of the cache wrote for these inputs; tests/test_embed.py checks that `embed.Cache` writes the same bytes.
"""

from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "phase3"
CACHE_ROOT = FIXTURES / "embed-cache"
CACHE_MODEL = "qwen3-emb-8b"

# The cache fixture: two appends of texts in the text views the cache uses, and their rows.
CACHE_BATCHES = [
    [("definition", "media bias", "A slant in news coverage."), ("term", "framing")],
    [
        ("definition", "Medienverzerrung", "Schräge Berichterstattung – „Bias“."),
        ("definition", "echo chamber", "A closed space where beliefs are amplified."),
    ],
]
CACHE_DIM = 4


def cache_texts(embed, batch):
    """The texts of one batch, in the text views of the `embed` module given."""
    return [
        embed.definition_text(*item[1:])
        if item[0] == "definition"
        else embed.term_text(item[1])
        for item in batch
    ]


def cache_rows(numpy, start, count):
    """Fixed float32 rows `start..start+count`, scaled to unit length like real vectors."""
    rows = numpy.arange(
        start * CACHE_DIM, (start + count) * CACHE_DIM, dtype=numpy.float32
    )
    rows = (rows + 1).reshape(count, CACHE_DIM)
    return rows / numpy.linalg.norm(rows, axis=1, keepdims=True)


def write_cache(embed, numpy, root):
    """Write the fixture cache under `root` with the `embed` module given."""
    cache = embed.Cache(root, CACHE_MODEL)
    cache.check_writable(embed.config(CACHE_MODEL))
    start = 0
    for batch in CACHE_BATCHES:
        cache.add(cache_texts(embed, batch), cache_rows(numpy, start, len(batch)))
        start += len(batch)
    return cache
