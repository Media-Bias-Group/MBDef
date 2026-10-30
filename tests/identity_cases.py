"""The identity rule of `collect grobid`, case by case, as fixtures for its tests.

Each paper asks for a work (DOI, title, authors, year); its fake TEI shows a title page; the expected result is the rule that matched, or the reason the PDF was refused.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from http_replay import Answer
from mini_collect import downloads, fake_grobid, tei

ASKED = {"title": "Cited title", "authors": "Someone, A.", "year": "2019"}
PAGE = {"title": "Cited title", "surnames": ("Someone",), "year": "2019"}
OTHER_PAGE = {"title": "Another header title", "surnames": ("Else",), "year": "2019"}
VERDICT = {"verdict": "same_work", "model": "m", "revision": "r", "prompt_version": "p"}
UNVERIFIED = "grobid_identity_unverified: no matching DOI or title/year/author"
CONFLICT = "grobid_identity_conflict: DOI differs from requested work"
# paper id: (asked, title page (the keyword arguments of `mini_collect.tei`), verdict or None, expected)
CASES: dict[str, tuple[dict[str, str], dict[str, Any], dict[str, str] | None, str]] = {
    "doi-same": (
        ASKED | {"doi": "10.1234/requested"},
        OTHER_PAGE | {"doi": "10.1234/requested"},
        None,
        "doi",
    ),
    "doi-trailing-dot": (
        ASKED | {"doi": "10.1234/x"},
        OTHER_PAGE | {"doi": "10.1234/x)."},
        None,
        "doi",
    ),
    "title-year-author": (ASKED, PAGE, None, "title_year_author"),
    "year-missing": (ASKED, PAGE | {"year": ""}, None, "title_author_year_missing"),
    "year-off-by-one": (
        ASKED,
        PAGE | {"year": "2020"},
        None,
        "title_author_year_off_by_one",
    ),
    "year-two-off": (ASKED, PAGE | {"year": "2021"}, None, UNVERIFIED),
    "no-shared-author": (ASKED, PAGE | {"surnames": ("Else",)}, None, UNVERIFIED),
    "title-in-text": (
        ASKED,
        PAGE | {"header_title": "", "body": "Cited title. A. Someone, University."},
        None,
        "title_in_text",
    ),
    "unverifiable": (ASKED, OTHER_PAGE, None, UNVERIFIED),
    "verdict-accepts": (ASKED, OTHER_PAGE, VERDICT, "llm_verified"),
    "verdict-different-work": (
        ASKED,
        OTHER_PAGE,
        VERDICT | {"verdict": "different_work"},
        UNVERIFIED,
    ),
    "verdict-cannot-tell": (
        ASKED,
        OTHER_PAGE,
        VERDICT | {"verdict": "cannot_tell"},
        UNVERIFIED,
    ),
    "verdict-other-bytes": (
        ASKED,
        OTHER_PAGE,
        VERDICT | {"pdf_sha256": "0" * 64},
        UNVERIFIED,
    ),
    "verdict-loses-to-a-doi-conflict": (
        ASKED | {"doi": "10.1234/requested"},
        OTHER_PAGE | {"doi": "10.1234/other"},
        VERDICT,
        CONFLICT,
    ),
    "verdict-loses-to-a-doi-match": (
        ASKED | {"doi": "10.1234/requested"},
        OTHER_PAGE | {"doi": "10.1234/requested"},
        VERDICT,
        "doi",
    ),
}
# A DOI on the title page that says nothing about the work (the misplaced-DOI fix): the title, author and year decide. (requested DOI, DOI on the PDF, why it is no evidence)
NO_EVIDENCE = [
    ("10.1145/3409116", "10.1145/nnnnnnn.nnnnnnn", "template DOI"),
    ("10.1145/3479855", "10.1145/1122445.1122456", "ACM's sample DOI"),
    ("10.48550/arxiv.2107.07045", "10.1109/access.2017.doi", "template DOI"),
    (
        "10.1177/1464884915610988",
        "10.1177/1464884915610988jou.sagepub.com",
        "text glued on",
    ),
    ("10.1073/pnas.2209384120", "10.1073/pnas.22093841201of5", "text glued on"),
    (
        "10.11591/ijeecs.v22.i3.pp1731-1738",
        "10.11591/ijeecs.v22i3.pp1731-1738",
        "punctuation",
    ),
    ("10.1609/icwsm.v17i1.22209", "10.5281/zenodo.7799469", "dataset DOI"),
    ("10.48550/arxiv.1602.04938", "10.1145/2939672.2939778", "preprint DOI"),
    ("10.31390/jcdl.2.1.04", "10.1007/978-3-", "cut short"),
    (
        "10.1080/1369183x.2015.1133279",
        "10.10800/133693183x.20105.11303279.htp://www.tandfonline.com/doi/abs/10.10800/1336",
        "a web address read into it",
    ),
]
# Real evidence of a different work stays a conflict.
EVIDENCE = [
    (
        "10.4324/9781003163237",
        "10.4324/9781003163237-11",
        "a chapter of the requested book",
    ),
    ("10.1017/9781108560573", "10.1017/9781108560573.001", "a part of it"),
    ("10.1017/s003329172000224x", "10.1017/s0033291721000593", "a later correction"),
    ("10.1080/17512786.2012.667272", "10.5209/clac.52770", "another work"),
]
for n, (asked, seen, _why) in enumerate(NO_EVIDENCE):
    CASES[f"no-evidence-{n}"] = (
        ASKED | {"doi": asked},
        PAGE | {"doi": seen},
        None,
        "title_year_author",
    )
for n, (asked, seen, _why) in enumerate(EVIDENCE):
    CASES[f"evidence-{n}"] = (
        ASKED | {"doi": asked},
        PAGE | {"doi": seen},
        None,
        CONFLICT,
    )


def prepare(folder: Path, replay) -> Path:
    """The download ledger and PDFs of every case, the verdicts file, and a GROBID that answers each PDF with its case's TEI. Returns the verdicts file."""
    pdfs = {paperid: f"%PDF-1.4 {paperid}".encode() for paperid in CASES}
    downloads(folder, {p: asked for p, (asked, _, _, _) in CASES.items()}, pdfs)
    verdicts = folder / "accepted.json"
    accepted = [
        {"paperid": p, "pdf_sha256": hashlib.sha256(pdfs[p]).hexdigest()} | verdict
        for p, (_, _, verdict, _) in CASES.items()
        if verdict
    ]
    verdicts.write_text(json.dumps({"accepted": accepted}))
    replay.answer("GET /api/isalive", lambda request: Answer(body="true"))
    replay.answer(
        "POST /api/processFulltextDocument",
        fake_grobid({p: tei(**page) for p, (_, page, _, _) in CASES.items()}),
    )
    replay.answer("GET /api/version", lambda request: Answer(body="0.9.1"))
    return verdicts
