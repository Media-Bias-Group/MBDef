"""Read-only access to the cached embedding vectors of definitions.

Step 5's model is Qwen3-Embedding-8B: its cache (built for the earlier curation pipeline, never written here) stores L2-normalised float16 rows in `vectors.npy`, one per text, with the sha1 of the embedded text in `keys.txt`. The embedded text of a definition is "<term>: <definition>" (the instruction prefix is applied at encoding time and is not part of the key). A cache of another model with the same layout is read with `EmbeddingCache(directory)`.
"""

from pathlib import Path

import numpy as np

from mbdef import files
from mbdef.embed import text_hash

# The cache is in the data root (data/data-root.tsv, docs/step-5-post-processing/embed.md), as a path relative to it.
CACHE = "embeddings/qwen3-emb-8b"
CHUNK = 20000  # rows read from the memory map at a time


def in_data_root(path):
    """A cache folder: a path relative to the data root, or any absolute path as given."""
    path = Path(path)
    return path if path.is_absolute() else files.data_root() / path


class EmbeddingCache:
    """The cache in `directory` (by default Step 5's Qwen3-Embedding-8B cache), read-only from a memory map."""

    def __init__(self, directory=CACHE):
        self.directory = in_data_root(directory)
        self.matrix = np.load(self.directory / "vectors.npy", mmap_mode="r")
        self.index = {
            k: i
            for i, k in enumerate((self.directory / "keys.txt").read_text().split("\n"))
        }

    def vectors(self, keys):
        """float32 rows for the given embedded texts, read from the memory map in file order."""
        hashes = [text_hash(k) for k in keys]
        missing = next(
            (k for k, h in zip(keys, hashes, strict=True) if h not in self.index), None
        )
        if missing is not None:
            raise KeyError(f"no embedding for {missing!r}")
        out = np.empty((len(hashes), self.matrix.shape[1]), dtype=np.float32)
        rows = np.array([self.index[h] for h in hashes], dtype=np.int64)
        order = np.argsort(rows)
        for s in range(0, len(order), CHUNK):
            o = order[s : s + CHUNK]
            out[o] = self.matrix[rows[o]]
        return out
