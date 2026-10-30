"""The embedding cache: rows are read by the sha1 of the embedded text, in any order, and a missing text is an error."""

import hashlib
import json

import numpy as np
import pytest

from mbdef.postprocess.embeddings import EmbeddingCache


def write_cache(directory, texts, vectors, manifest=None):
    directory.mkdir(parents=True)
    np.save(directory / "vectors.npy", np.asarray(vectors, dtype=np.float16))
    (directory / "keys.txt").write_text(
        "\n".join(hashlib.sha1(t.encode()).hexdigest() for t in texts)
    )
    (directory / "manifest.json").write_text(
        json.dumps(manifest or {"model": "m", "prefix": "p"})
    )


def test_rows_come_back_in_the_order_asked_as_float32(tmp_path):
    write_cache(tmp_path / "c", ["a: x", "b: y", "c: z"], [[1, 0], [0, 1], [0.6, 0.8]])
    got = EmbeddingCache(tmp_path / "c").vectors(["c: z", "a: x", "c: z"])
    assert got.dtype == np.float32
    assert np.allclose(got, [[0.6, 0.8], [1, 0], [0.6, 0.8]], atol=1e-3)


def test_a_text_not_in_the_cache_is_an_error(tmp_path):
    write_cache(tmp_path / "c", ["a: x"], [[1, 0]])
    with pytest.raises(KeyError, match="d: w"):
        EmbeddingCache(tmp_path / "c").vectors(["a: x", "d: w"])


def test_the_default_cache_is_qwen_under_the_data_root(tmp_path, monkeypatch):
    write_cache(tmp_path / "embeddings/qwen3-emb-8b", ["a: x"], [[1, 0]])
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    assert np.allclose(EmbeddingCache().vectors(["a: x"]), [[1, 0]])
