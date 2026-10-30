"""Every step page under docs/ (in its paper step's folder) has the same six parts, so a reader finds them in the same place on each page.

The parts are a row of the page's first table (paper step, command) or a heading (inputs, outputs, how it was checked, how to adapt it).
A page that has no such thing, for example a library with no command, says so in the row.
"""

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parent.parent / "docs"

# Each page by its path under docs/, in its paper step's folder.
STEP_PAGES = [
    "step-1-2-papers/relevance",
    "step-1-2-papers/seeds",
    "step-1-2-papers/collect",
    "step-3-extraction/groundtruth",
    "step-3-extraction/extract",
    "step-3-extraction/evaluate",
    "step-3-extraction/selection",
    "step-4-closure/closure",
    "step-5-post-processing/embed",
    "step-5-post-processing/postprocess",
    "dataset/dataset",
    "setup/cluster",
    "setup/llm",
    "setup/data-fetch",
]

ROWS = {
    "paper step": r"\*\*Paper step\*\*",
    "command": r"\*\*Commands?\*\*",
}
HEADINGS = {
    "inputs": r"Inputs",
    "outputs": r"Outputs",
    "how it was checked": r"How (?:it|the \w+(?: \w+)*) was checked",
    "how to adapt it": r"How to adapt",
}


def page(name: str) -> str:
    return (DOCS / f"{name}.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("name", STEP_PAGES)
def test_the_page_has_the_six_parts(name):
    text = page(name)
    missing = []
    for part, pattern in ROWS.items():
        # a table row: | **Paper step** | what it says |
        if not re.search(rf"^\|\s*{pattern}\s*\|\s*\S", text, re.MULTILINE):
            missing.append(part)
    for part, pattern in HEADINGS.items():
        if not re.search(rf"^#{{2,4}} .*{pattern}", text, re.MULTILINE | re.IGNORECASE):
            missing.append(part)
    assert not missing, f"docs/{name}.md has no: {', '.join(missing)}"
