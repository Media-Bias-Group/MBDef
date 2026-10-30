"""A paper's text from its GROBID TEI, and the windows of it that the model reads.

- `read_teis` reads TEI files from a folder or a zip. A paper's id is its file name without `.tei.xml`, URL-decoded (`doi%3A10.1145%2F3677092.tei.xml` is paper `doi:10.1145/3677092`), or, in a folder of a collection run's GROBID output, the `paperid` of the conversion receipt beside the TEI.
- `parse_tei` keeps the title, the DOI and the text: the abstract, then every text block of the body and the back matter (paragraphs, headings, figure captions, formulas, tables, lists, notes, quotes), each with its white space collapsed and separated by a blank line. The bibliography is left out.
- `windows` cuts the text into overlapping character windows (the released runs used 8,000/1,000, 16,000/2,000 and 32,000/4,000). A window ends at the last line break in its second half if there is one, so it rarely cuts a block; the next window starts `overlap` characters before that end.
  Nothing is ever cut off to fit a model's context.
- `token_windows` is the same idea counted in a tokenizer's tokens (not used by the released runs).
"""

import json
import xml.etree.ElementTree as ET
import zipfile
from itertools import pairwise
from pathlib import Path
from urllib.parse import unquote

from mbdef.collect.identity import CONVERSION_RECEIPTS, converted_tei
from mbdef.extract.records import clean

TEI = "{http://www.tei-c.org/ns/1.0}"
TEI_SUFFIX = ".tei.xml"
BLOCKS = {"head", "p", "figDesc", "formula", "table", "list", "note", "quote", "ab"}
# The 150 ground-truth papers' TEIs, the default of `extract run`, `train` and `check-programs`. The file holds paper text and is not part of this repository; give `--teis` your own GROBID TEIs instead.
GROUND_TRUTH_TEIS = Path("data/paper_sources.zip")


def read_teis(path: Path, paper_ids: list[str] | None = None) -> dict[str, bytes]:
    """{paper id: TEI bytes} from a folder or a zip of `<paper id>.tei.xml` files, or from a folder that GROBID wrote for a collection run (`collect grobid`, a closure run's `grobid/`): there each TEI is named by the acquisition, below a sub-folder, beside its conversion receipt (`CONVERSION_RECEIPTS`), whose `paperid` names the paper.

    With `paper_ids`, only those papers, and every one of them must be there.
    """
    if not Path(path).exists():
        if Path(path).as_posix().endswith(GROUND_TRUTH_TEIS.as_posix()):
            raise SystemExit(
                f"{path}, the ground-truth papers' TEIs, holds paper text and is not part of "
                "this repository. Give --teis a folder or zip of <paper id>.tei.xml files, the "
                "GROBID TEIs of the papers you fetched (docs/step-3-extraction/extract.md)."
            )
        raise SystemExit(f"{path}: no such folder or zip of TEIs.")
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as bundle:
            names = {
                unquote(Path(n).name.removesuffix(TEI_SUFFIX)): n
                for n in bundle.namelist()
                if n.endswith(TEI_SUFFIX)
            }
            wanted = paper_ids if paper_ids else sorted(names)
            missing = [p for p in wanted if p not in names]
            if missing:
                raise SystemExit(f"{path} has no TEI for: {', '.join(missing)}")
            return {p: bundle.read(names[p]) for p in wanted}
    files = {
        unquote(p.name.removesuffix(TEI_SUFFIX)): p
        for p in Path(path).glob("*" + TEI_SUFFIX)
    }
    by_receipt: dict[str, Path] = {}
    for receipt in sorted(Path(path).rglob(CONVERSION_RECEIPTS)):
        paper = json.loads(receipt.read_text(encoding="utf-8"))["paperid"]
        if paper in by_receipt:
            raise SystemExit(
                f"{paper}: two conversion receipts in {path} name it "
                f"({by_receipt[paper].relative_to(path)}, {receipt.relative_to(path)}); keep one."
            )
        by_receipt[paper] = receipt
    for paper, receipt in by_receipt.items():
        files.setdefault(paper, converted_tei(receipt))
    wanted = paper_ids if paper_ids else sorted(files)
    missing = [p for p in wanted if p not in files]
    if missing:
        raise SystemExit(f"{path} has no TEI for: {', '.join(missing)}")
    return {p: files[p].read_bytes() for p in wanted}


def _tag(node) -> str:
    return node.tag.rsplit("}", 1)[-1]


def _blocks(node) -> list[str]:
    values = []
    for child in node:
        tag = _tag(child)
        if tag == "listBibl":
            continue
        if tag in BLOCKS:
            text = clean("".join(child.itertext()))
            if text:
                values.append(text)
        else:
            values.extend(_blocks(child))
    return values


def parse_tei(source: bytes) -> dict:
    """The title, DOI and text of one TEI document (module docstring)."""
    root = ET.fromstring(source)
    title_node = root.find(f".//{TEI}titleStmt/{TEI}title[@level='a']")
    if title_node is None:
        title_node = root.find(f".//{TEI}titleStmt/{TEI}title")
    title = clean("".join(title_node.itertext())) if title_node is not None else ""
    doi = ""
    for node in root.findall(f".//{TEI}idno"):
        if (node.get("type") or "").casefold() == "doi":
            doi = (
                clean("".join(node.itertext()))
                .casefold()
                .removeprefix("https://doi.org/")
            )
            break
    parts = []
    abstract = root.find(f".//{TEI}profileDesc/{TEI}abstract")
    if abstract is not None:
        parts.extend(_blocks(abstract))
    for location in (f".//{TEI}body", f".//{TEI}back"):
        node = root.find(location)
        if node is not None:
            parts.extend(_blocks(node))
    return {"title": title, "doi": doi, "text": "\n\n".join(parts)}


def windows(text: str, size: int, overlap: int) -> list[str]:
    """Overlapping windows of at most `size` characters (module docstring)."""
    if size < 1 or not 0 <= overlap < size:
        raise ValueError("window size must be positive and 0 <= overlap < size")
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    values = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            boundary = text.rfind("\n", start + size // 2, end)
            if boundary > start:
                end = boundary
        values.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return values


def token_windows(
    text: str, tokenizer, size: int = 8192, overlap: int = 1024
) -> list[dict]:
    """Windows of at most `size` tokens that keep the exact character offsets of `text`.

    `tokenizer` must be a fast Hugging Face tokenizer (it needs character offsets). Each window is tokenized again on its own to enforce the budget, so the overlap is approximate. Returns dicts with `start`, `end`, `text` and `tokens`.
    """
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer with character offsets is required")
    if size < 1 or not 0 <= overlap < size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")
    if not text:
        return []
    chunks: list[dict] = []
    start = 0
    while start < len(text):
        encoded = tokenizer(
            text[start:], add_special_tokens=False, return_offsets_mapping=True
        )
        offsets = encoded["offset_mapping"]
        end = len(text) if len(offsets) <= size else start + offsets[size][0]
        # Byte fallback can give one character several tokens: never split a character, and stop
        # if a single character is over the budget.
        local = {"input_ids": [], "offset_mapping": []}
        while end > start:
            local = tokenizer(
                text[start:end], add_special_tokens=False, return_offsets_mapping=True
            )
            if len(local["input_ids"]) <= size:
                break
            end -= 1
        if end <= start:
            raise ValueError("Token budget cannot contain the next source character")
        chunks.append(
            dict(
                start=start,
                end=end,
                text=text[start:end],
                tokens=len(local["input_ids"]),
            )
        )
        if end == len(text):
            break
        if overlap and len(local["offset_mapping"]) > overlap:
            next_start = start + local["offset_mapping"][-overlap][0]
            start = next_start if next_start > start else end
        else:
            start = end
    assert chunks[0]["start"] == 0 and chunks[-1]["end"] == len(text)
    assert all(a["end"] >= b["start"] > a["start"] for a, b in pairwise(chunks))
    return chunks
