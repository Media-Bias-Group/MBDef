"""Synthetic papers: made-up text that the tests read in place of real papers.

This repository holds no paper text, so the tests of `extract run`, `extract verify`, `extract train`, `extract check-programs` and `closure identity` run on the papers below: every sentence here was written for the tests, about made-up media-bias terms.
`tests/fixtures/make_synthetic_replays.py` recorded the commands' runs on them, answered by the stand-in models below, into `tests/fixtures/synthetic-*-replay.zip`; a test writes the papers' TEIs (`write_teis`) and replays those runs.

The world, which the tests assert (not the code's own output):

- **Papers.** `t1` and `t2` (TEST) are the ones extracted and verified; `s1` to `s4` (TRAIN) and `s5`, `s6` (DEV) make the training dataset.
- **Terms.** Each paper defines its terms in one sentence each (`DEFINITIONS`). Each TEST paper also holds one sentence that only gives an example of a term (`EXAMPLES`).
- **Stand-in extractor.** It proposes every sentence of `DEFINITIONS` and `EXAMPLES` that stands whole in the window it reads, each as the definition of its term.
- **Stand-in verifier.** A proposed definition is a definition when it is a sentence of `DEFINITIONS`; the sentences of `EXAMPLES` are not. Every term is relevant.

Every text stays below 2,000 characters.
"""

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import quote
from xml.sax.saxutils import escape

TEST = ("t1", "t2")
TRAIN = ("s1", "s2", "s3", "s4")
DEV = ("s5", "s6")

# paper id: (title, paragraphs). The first paragraph is the abstract.
PAPERS = {
    "t1": (
        "Headlines That Move: Revision Practices in Online News",
        [
            "We follow forty online news stories through every revision of their headlines and "
            "images. The changes are small one by one, yet together they turn neutral reports "
            "into pointed ones.",
            "Headline drift is the gradual change of a story's headline toward a stronger slant "
            "across its online revisions. Editors rarely announce such changes, and archives "
            "keep only the last version.",
            "Quote pruning is the shortening of a source's quotation so that only the part "
            "supporting the story's angle remains. Readers who follow the link to the full "
            "interview often find a more careful speaker.",
            "Photo cropping bias is the selection of an image section that changes how large or "
            "hostile a crowd appears. A wide frame shows a calm square, a tight one shows a "
            "pushing line of people.",
            "An outrage loop happened when the evening edition repeated the morning's angriest "
            "reader comments. We count such loops but do not define them further here.",
            "Our coding scheme and the revision archive are described in the appendix. All "
            "stories were coded twice, and the coders agreed on most of the changes.",
        ],
    ),
    "t2": (
        "Who Speaks for the Region? Source Selection in Local Newsrooms",
        [
            "We interviewed editors of twelve regional newsrooms about how they choose experts, "
            "label sponsored pages and report refusals to comment.",
            "Source tilt is the uneven choice of interviewed experts that favours one side of a "
            "debate. It grows when a newsroom keeps a short list of reliable voices.",
            "Byline masking is the publication of sponsored content under the name of a staff "
            "reporter. Several editors called it a matter of layout rather than of honesty.",
            "Silence weighting is the habit of reporting a party's refusal to comment as if it "
            "were an admission. The phrase did not respond appears far more often for some "
            "parties than for others.",
            "Echo booking was seen when three talk shows invited the same commentator on one "
            "evening. The editors described it as a scheduling accident.",
            "We close with advice for newsrooms that want to widen their lists of sources.",
        ],
    ),
    "s1": (
        "Numbers Without Denominators",
        [
            "Counts in crime and health news often come without the group they were counted in.",
            "Denominator neglect is the reporting of a count without the size of the group it "
            "was counted in. A city of two million and a village of two thousand look alike.",
            "Scale stretching is the drawing of a chart axis so that a small change looks like a "
            "large one. Both habits make rare events look common.",
        ],
    ),
    "s2": (
        "Who Gets the Last Word",
        [
            "We look at the order in which a story presents the two sides of a dispute.",
            "Last-word bias is the placement of one side's argument at the end of a story so "
            "that it frames the reader's memory. It is hard to see in any single story.",
            "A rebuttal gap is the time between an accusation and the publication of the "
            "accused party's reply. Long gaps leave the accusation alone for a day or more.",
        ],
    ),
    "s3": (
        "Words for Protest",
        [
            "Protest coverage chooses between many words for the same afternoon in the street.",
            "Riot labelling is the description of a mostly peaceful protest with words of "
            "disorder and violence. One broken window can name a whole march.",
            "Crowd minimizing is the reporting of a large demonstration as a small gathering. "
            "Police and organizers often give very different counts.",
        ],
    ),
    "s4": (
        "The Weather of the Economy",
        [
            "Economic news borrows the language of storms, clouds and sunshine.",
            "Forecast framing is the presentation of an economic estimate as if it were a "
            "certain outcome. The word expected quietly disappears from the headline.",
            "Gloom priming is the opening of economic stories with the worst figure of the "
            "report. The better figures follow in the last paragraphs.",
        ],
    ),
    "s5": (
        "Lost in Translation",
        [
            "Foreign statements reach readers through translations chosen by the newsroom.",
            "Translation slant is the choice of a harsher or softer word when a foreign "
            "statement is translated. Agencies and newsrooms rarely publish the original.",
        ],
    ),
    "s6": (
        "Below the Line",
        [
            "Reader comments are moderated, ranked and sometimes quoted in the story itself.",
            "Comment amplification is the quoting of a few reader comments as if they spoke "
            "for the public. The loudest comments are the easiest to quote.",
        ],
    ),
}

# term: the sentence that defines it. Each is the first sentence of its paragraph and reads "<Term> is the …" (or "A <term> is the …"); 16 terms, three in each TEST paper.
DEFINITIONS = {
    sentence.split(" is ", 1)[0].removeprefix("A ").lower(): sentence
    for _title, paragraphs in PAPERS.values()
    for paragraph in paragraphs
    for sentence in re.findall(r"[^.]+ is (?:the|a|an) [^.]+\.", paragraph)
    if sentence == paragraph.split(". ")[0] + "."
}
# term: a sentence that only gives an example of it
EXAMPLES = {
    "outrage loop": PAPERS["t1"][1][4].split(". ")[0] + ".",
    "echo booking": PAPERS["t2"][1][4].split(". ")[0] + ".",
}


# The training dataset's reference records are the papers' definitions, except for s1, the first TRAIN paper in the training order: its references paraphrase them, so the stand-in's quotes do not match there, s1 scores 0, and Bootstrap goes on to the next paper.
PARAPHRASED = {
    "denominator neglect": "Reporting how many cases there were but not out of how many people.",
    "scale stretching": "Choosing chart axes that make modest movements look dramatic.",
}


def text_of(paper_id: str) -> str:
    """The paper's text as `extract.papers.parse_tei` reads it: the paragraphs, a blank line apart."""
    return "\n\n".join(PAPERS[paper_id][1])


def terms_of(paper_id: str) -> dict[str, str]:
    """{term: its defining sentence} for the terms the paper defines."""
    text = text_of(paper_id)
    return {
        term: sentence for term, sentence in DEFINITIONS.items() if sentence in text
    }


def gold(paper_id: str) -> list[dict]:
    """The reference records of a TRAIN or DEV paper in the synthetic training dataset."""
    return [record(t, PARAPHRASED.get(t, s)) for t, s in terms_of(paper_id).items()]


def tei(paper_id: str) -> bytes:
    """The paper as GROBID would write it: its title, the abstract, and the body's paragraphs."""
    title, paragraphs = PAPERS[paper_id]
    body = "".join(f"<div><p>{escape(p)}</p></div>" for p in paragraphs[1:])
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc><titleStmt>'
        f'<title level="a" type="main">{escape(title)}</title></titleStmt></fileDesc>'
        f"<profileDesc><abstract><p>{escape(paragraphs[0])}</p></abstract></profileDesc>"
        f"</teiHeader><text><body>{body}</body></text></TEI>\n"
    ).encode()


def write_teis(folder: Path, paper_ids=tuple(PAPERS)) -> Path:
    """Write `<paper id>.tei.xml` for each paper into FOLDER (made if missing); returns FOLDER."""
    folder.mkdir(parents=True, exist_ok=True)
    for paper_id in paper_ids:
        (folder / f"{quote(paper_id, safe='')}.tei.xml").write_bytes(tei(paper_id))
    return folder


# The stand-in models -------------------------------------------------------------------------------

FIELDS = (
    "definitions",
    "synonyms",
    "description",
    "related_terms",
    "subtypes",
    "supertypes",
    "examples",
)


def record(term: str, sentence: str) -> dict:
    """A definition record as the extractor answers it."""
    return {
        "term": term,
        **{field: [] for field in FIELDS},
        "definitions": [{"text": sentence, "citation": []}],
    }


def field(content: str, name: str) -> str:
    """The value of one `[[ ## name ## ]]` field of a DSPy chat message."""
    found = re.search(
        rf"\[\[ ## {name} ## \]\]\n(.*?)(?:\n\n\[\[ ## |\n\nRespond with |$)",
        content,
        re.S,
    )
    assert found, (name, content[:200])
    return found[1]


def extractor_answer(body: dict) -> str:
    """The stand-in extractor's answer to a chat request: every proposed sentence whole in the window."""
    window = field(body["messages"][-1]["content"], "paper_text")
    proposed = {**EXAMPLES, **DEFINITIONS}
    found = [(t, s) for t, s in proposed.items() if s in window]
    found.sort(key=lambda pair: window.index(pair[1]))
    reasoning = (
        "The window defines " + ", ".join(t for t, _ in found) + "."
        if found
        else "The window defines no term."
    )
    records = json.dumps([record(t, s) for t, s in found], ensure_ascii=False)
    return (
        f"[[ ## reasoning ## ]]\n{reasoning}\n\n[[ ## definition_records ## ]]\n{records}\n\n"
        "[[ ## completed ## ]]"
    )


def verifier_answer(body: dict) -> str:
    """The stand-in verifier's answer: a definition when the quote is a sentence of DEFINITIONS."""
    candidates = json.loads(field(body["messages"][-1]["content"], "candidates"))
    verdicts = {}
    for candidate in candidates:
        defines = candidate["definition"] in DEFINITIONS.values()
        verdicts[candidate["id"]] = {
            "definition": "yes" if defines else "no",
            "relevance": "yes",
            "reason": (
                f"The source defines '{candidate['term']}'."
                if defines
                else f"The source only gives an example of '{candidate['term']}'."
            ),
        }
    return json.dumps({"verdicts": verdicts}, indent=2)


def completion(body: dict, content: str) -> dict:
    """An OpenAI chat completion of CONTENT, as a vLLM server answers it (its id is made from the request, so a second recording gives the same ids)."""
    key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return {
        "id": f"chatcmpl-synthetic-{key[:16]}",
        "object": "chat.completion",
        "created": 0,
        "model": body["model"],
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": sum(len(m["content"].split()) for m in body["messages"]),
            "completion_tokens": len(content.split()),
            "total_tokens": sum(len(m["content"].split()) for m in body["messages"])
            + len(content.split()),
        },
    }


def similarity(pairs: list[tuple[str, str]]) -> list[float]:
    """The stand-in for BGE's cosine: the share of words the two texts have in common."""
    scores = []
    for left, right in pairs:
        a, b = set(left.lower().split()), set(right.lower().split())
        scores.append(round(len(a & b) / len(a | b), 6) if a | b else 0.0)
    return scores


def entailment(pairs: list[tuple[str, str]]) -> list[float]:
    """The stand-in for the entailment model: 0.98 for the same text, else 0.02."""
    return [0.98 if left == right else 0.02 for left, right in pairs]
