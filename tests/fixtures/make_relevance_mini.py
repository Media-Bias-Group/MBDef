"""Make `tests/fixtures/relevance-mini/`: eight made-up papers and the recorded answers of the three classifiers.

The relevance step's tests replay these recordings (tests/test_relevance.py). The papers and answers are made up, because the real ones hold the papers' abstracts. The recordings have the released runs' form (docs/step-1-2-papers/relevance.md):
one `normalized_predictions.csv` and one `metadata.json` per classifier, rendered from the step's prompt templates as the released runs rendered them (Jinja, with the result stripped). The cases:

- `doi_10.1000_agree`: both first classifiers answer relevant.
- `doi_10.1000_tie`: Qwen not relevant, Gemma relevant; the arbiter answers relevant.
- `doi_10.1000_unsure`: Qwen relevant, Gemma "not sure"; the arbiter answers not relevant.
- `doi_10.1000_neither`: both answer not relevant.
- `hash_da39a3ee5e6b4b0d`: no title and no abstract, so no classifier is asked.
- `hash_shared` twice (`shared_a`, `shared_b`): two papers under one id. Qwen's answers are recorded in the order they came back (`shared_b`'s first), Gemma's in the papers' order. The released rule pairs the k-th answer under an id with the k-th paper under it, so `shared_a` gets Qwen's answer to `shared_b`, and the other way round.
- `doi_10.1000_unasked`: Gemma answered it, Qwen has no answer (as for 80 papers of the released Qwen run).

Run it from the repository root: `uv run python tests/fixtures/make_relevance_mini.py`.
"""

import csv
import json
from pathlib import Path

from jinja2 import Environment, StrictUndefined

HERE = Path(__file__).resolve().parent
OUT = HERE / "relevance-mini"
PROMPTS = HERE.parent.parent / "src/mbdef/relevance/prompts"

PAPERS = [
    (
        "doi_10.1000_agree",
        "doi_10.1000_agree",
        "Framing of migration in national newspapers",
        "We compare how four newspapers frame migration and find a consistent slant.",
        "10.1000/agree",
    ),
    (
        "hash_shared_a",
        "hash_shared",
        "Soil chemistry of alpine meadows",
        "We measure nitrogen in alpine soils over three summers.",
        "",
    ),
    (
        "doi_10.1000_tie",
        "doi_10.1000_tie",
        "Spreading false stories online",
        "A review of how false information travels between platforms.",
        "10.1000/tie",
    ),
    ("hash_da39a3ee5e6b4b0d", "hash_da39a3ee5e6b4b0d", "", "", ""),
    (
        "doi_10.1000_unsure",
        "doi_10.1000_unsure",
        "Audience trust after corrections",
        "Survey of readers after a newspaper printed a correction.",
        "10.1000/unsure",
    ),
    (
        "hash_shared_b",
        "hash_shared",
        "Partisan slant in evening television news",
        "Coding of two networks' evening news for partisan tone.",
        "",
    ),
    (
        "doi_10.1000_neither",
        "doi_10.1000_neither",
        "A faster sorting network",
        "We give a sorting network of smaller depth.",
        "10.1000/neither",
    ),
    (
        "doi_10.1000_unasked",
        "doi_10.1000_unasked",
        "Headline wording and clicks",
        "An experiment with two headline wordings for the same story.",
        "10.1000/unasked",
    ),
]
EXAMPLES = [
    ("Bias in election coverage", "A content analysis of election news.", "relevant"),
    ("Protein folding at speed", "A simulation method for proteins.", "not relevant"),
]

# Each classifier's recorded answers in the order they were recorded: the corpus_id whose text was asked, the response, and the label the released runs' parser gave it.
QWEN = [
    (
        "doi_10.1000_agree",
        "The article studies newspaper slant.\n</think>\n\nrelevant",
        "relevant",
    ),
    (
        "hash_shared_b",
        "Partisan tone of news is media bias.\n</think>\n\nrelevant",
        "relevant",
    ),
    (
        "doi_10.1000_tie",
        "No media angle is clear.\n</think>\n\nnot relevant",
        "not relevant",
    ),
    (
        "doi_10.1000_unsure",
        "Corrections shape trust in news.\n</think>\n\nrelevant",
        "relevant",
    ),
    ("hash_shared_a", "Soil science.\n</think>\n\nnot relevant", "not relevant"),
    ("doi_10.1000_neither", "Algorithms.\n</think>\n\nnot relevant", "not relevant"),
]
GEMMA = [
    ("doi_10.1000_agree", "relevant", "relevant"),
    ("hash_shared_a", "not relevant", "not relevant"),
    ("doi_10.1000_tie", "relevant", "relevant"),
    ("doi_10.1000_unsure", "not sure", "can't determine"),
    ("hash_shared_b", "relevant", "relevant"),
    ("doi_10.1000_neither", "not relevant", "not relevant"),
    ("doi_10.1000_unasked", "relevant", "relevant"),
]
# The arbiter is asked about the disagreements, in the order of Qwen's answers.
GLM = [
    ("hash_shared_a", "Final answer: not relevant", "not relevant"),
    (
        "doi_10.1000_tie",
        "Information dissemination is in scope. Label: relevant",
        "relevant",
    ),
    ("doi_10.1000_unsure", "not relevant", "not relevant"),
    ("hash_shared_b", "relevant", "relevant"),
]
RUNS = {
    "qwen": ("qwen3_6_35b_a3b", "zero_shot", QWEN, False),
    "gemma": ("gemma_4_12b_it", "zero_shot", GEMMA, False),
    "glm": ("z-ai/glm-5.2", "few_shot_diverse", GLM, True),
}
COLUMNS = [
    "paperid",
    "template",
    "model",
    "system_prompt",
    "prompt",
    "response",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cached",
    "cache_key",
    "clean_label",
]


def render(name: str, context: dict) -> str:
    env = Environment(undefined=StrictUndefined, autoescape=False)
    return (
        env.from_string((PROMPTS / f"{name}.j2").read_text(encoding="utf-8"))
        .render(**context)
        .strip()
    )


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    write_csv(
        OUT / "papers.csv",
        ["corpus_id", "paperid", "title", "abstract", "doi"],
        [list(p) for p in PAPERS],
    )
    examples = [
        {
            "paperid": f"ex{i}",
            "title": t,
            "abstract": a,
            "manual_label": label,
            "strategy": "diverse",
            "rank": i,
            "cluster": i,
        }
        for i, (t, a, label) in enumerate(EXAMPLES, 1)
    ]
    write_csv(
        OUT / "examples.csv", list(examples[0]), [list(e.values()) for e in examples]
    )
    papers = {p[0]: {"paperid": p[1], "title": p[2], "abstract": p[3]} for p in PAPERS}
    system = render("system_prompt", {})
    for name, (model, template, answers, with_examples) in RUNS.items():
        rows = []
        for corpus_id, response, label in answers:
            paper = papers[corpus_id]
            prompt = render(
                template, {**paper, "examples": examples if with_examples else []}
            )
            rows.append(
                [
                    paper["paperid"],
                    template,
                    model,
                    system,
                    prompt,
                    response,
                    100,
                    10,
                    110,
                    False,
                    "",
                    label,
                ]
            )
        write_csv(OUT / name / "normalized_predictions.csv", COLUMNS, rows)
        metadata = {
            "model": model,
            "templates": [f"templates/relevance/{template}.j2"],
            "max_tokens": 2048,
            "temperature": None,
        }
        (OUT / name / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
