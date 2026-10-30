"""Paper step 4, closure: the works the seed papers cite, one level deep, and the release bundle (docs/step-4-closure/closure.md).

Resolve the citation strings of the seed papers' extracted definitions to their reference-list entries (`extend_gt.py`, `citations.py`, `bibliography.py`, `manual_citations.py`), decide which cited works are already in the corpus and write the queue of those still to collect (`corpus_check.py`; `run.py` is the whole run), check that a downloaded PDF is the cited work with a language model (`pdf_identity.py`), and build the release folder of the corpus, its links and its provenance (`definition_citations.py`, `exclusions.py`, `existing.py`, `release_folder.py`). The identity rules (DOIs, compared text, TEI headers, conversion receipts) are the collect step's: `mbdef.collect.identity`.
"""

import hashlib
from pathlib import Path

# Names the resolver's output ("resolver" in a report). The number goes up when its results change. The id keeps the name the reports on disk carry (files.py).
RESOLVER = "taxonomy_pipeline.closure.extend_gt/5"
# The code the resolver's results depend on.
RESOLVER_FILES = ("collect/identity.py",)


def resolver_fingerprint() -> str:
    """Bind generated reports to their resolver code, even if a version bump is missed."""
    root = Path(__file__).parent.parent
    digest = hashlib.sha256()
    for path in [
        *sorted((root / "closure").glob("*.py")),
        *(root / f for f in RESOLVER_FILES),
    ]:
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()
