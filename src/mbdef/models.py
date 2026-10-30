"""The pinned models of the released runs, at the revisions those runs used.

This is the one table of them: `mbdef cluster download` fetches exactly these, and every step that loads or names one of them reads it from here, so the two cannot drift apart. A revision is a commit of the model's Hugging Face repository. (The embedding models of Step 5 have their own table in embed.py, with the settings that make each vector.)
"""

# name: (Hugging Face repository, revision)
PINNED = {
    # the extractor model of the released variants (Step 3) and of the seed run
    "gemma": (
        "google/gemma-4-12B-it-qat-w4a16-ct",
        "1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee",
    ),
    # the verifier, the other extractor of the grid, and the closure's PDF identity check
    "qwen": ("Qwen/Qwen3.6-35B-A3B", "995ad96eacd98c81ed38be0c5b274b04031597b0"),
    # the embedding model of the DefScore pair score and of the relevance filter
    "bge": ("BAAI/bge-base-en-v1.5", "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"),
    # the entailment (NLI) model of the DefScore pair score
    "nli": (
        "tasksource/ModernBERT-large-nli",
        "ca476cb923a8637073d4ceb0f19f7fc236e260d4",
    ),
}
