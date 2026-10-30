"""What every Step 5.2 relevance path does, and two helpers they share.

A path learns only in `fit`, which receives labelled training Records (never test) and, optionally, the unlabelled corpus.
`score` returns one number per item, higher = more Relevant. An item is anything with a `term` (or `label`) and `definitions` (ground-truth Records hold definition texts; corpus Extracted Concepts hold Definitions with a `.text`).
"""


def term(item) -> str:
    """The term of a ground-truth Record (`term`) or a corpus Extracted Concept (`label`)."""
    return item.term if hasattr(item, "term") else item.label


def texts(item) -> list[str]:
    """Definition texts of a ground-truth Record or a corpus Extracted Concept."""
    return [d if isinstance(d, str) else d.text for d in item.definitions]
