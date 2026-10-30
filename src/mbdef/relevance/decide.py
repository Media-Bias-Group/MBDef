"""`relevance compare` and `relevance export`: from the answers of the two classifiers and the arbiter to the list of relevant papers.

`compare` pairs Qwen's and Gemma's answers by paper (`corpus_id`), in the order of Qwen's answers. A paper both answered `relevant` is **agreed**; a paper with two different labels (one of them may be `can't determine`) is a **disagreement**, and the disagreements file is the arbiter's papers file. Any other pair (both `not relevant`, or both the same other label) is not relevant. A paper only one of the two answered is not compared, as in the released step.

`export` keeps the agreed papers and the disagreements the arbiter answered `relevant` (**tie-break**), in that order. Worked example (the released run): 2,963 agreed and 925 tie-break papers, 3,888 in all.
"""

from collections import Counter
from pathlib import Path

from mbdef import files
from mbdef.relevance.classify import (
    ANSWERS,
    read_csv,
    read_papers,
    write_csv,
)

AGREED = "agreed.csv"
DISAGREEMENTS = "disagreements.csv"
RELEVANT = "relevant-papers.csv"
AGREED_COLUMNS = ["corpus_id", "paperid", "doi", "title", "qwen_label", "gemma_label"]
DISAGREEMENT_COLUMNS = [
    "corpus_id",
    "paperid",
    "title",
    "abstract",
    "doi",
    "qwen_label",
    "gemma_label",
]
RELEVANT_COLUMNS = [
    "corpus_id",
    "paperid",
    "doi",
    "title",
    "qwen_label",
    "gemma_label",
    "glm_label",
    "route",
]


def new_folder(out: str) -> Path:
    folder = Path(out)
    if folder.exists() and any(folder.iterdir()):
        raise SystemExit(f"{folder} is not empty; give a new or empty folder.")
    return folder


def answers_of(folder: str) -> list[dict]:
    path = Path(folder) / ANSWERS
    if not path.is_file():
        raise SystemExit(
            f"{folder} has no {ANSWERS}; give the output folder of `relevance classify`."
        )
    return read_csv(path)


def write_receipt(
    folder: Path, command: str, inputs: dict, outputs: dict, summary: dict
) -> None:
    receipt = files.receipt(f"mbdef relevance {command}", {}, inputs, outputs)
    receipt["summary"] = summary
    files.write_json(folder / files.RECEIPT, receipt)
    print("; ".join(f"{k}: {v}" for k, v in summary.items()))


def compare_command(papers_arg: str | None, qwen: str, gemma: str, out: str) -> int:
    """`relevance compare`: the agreed papers and the disagreements of Qwen and Gemma."""
    out_dir = new_folder(out)
    papers_path, _ = files.located(papers_arg, "relevance/papers.csv")
    papers = {p["corpus_id"]: p for p in read_papers(papers_path)}
    gemma_label = {a["corpus_id"]: a["label"] for a in answers_of(gemma)}
    agreed, disagreements, pairs = [], [], Counter()
    for answer in answers_of(qwen):
        if answer["corpus_id"] not in gemma_label:
            continue
        paper = papers[answer["corpus_id"]]
        row = {
            **paper,
            "doi": paper.get("doi", ""),
            "qwen_label": answer["label"],
            "gemma_label": gemma_label[answer["corpus_id"]],
        }
        if row["qwen_label"] != row["gemma_label"]:
            disagreements.append(row)
            pairs["disagreements"] += 1
        elif row["qwen_label"] == "relevant":
            agreed.append(row)
            pairs["agreed"] += 1
        else:
            pairs[f"both {row['qwen_label']}"] += 1
    write_csv(out_dir / AGREED, AGREED_COLUMNS, agreed)
    write_csv(out_dir / DISAGREEMENTS, DISAGREEMENT_COLUMNS, disagreements)
    summary = {"answered by both": sum(pairs.values()), **dict(pairs.most_common())}
    write_receipt(
        out_dir,
        "compare",
        {
            "papers": papers_path,
            "qwen": Path(qwen) / ANSWERS,
            "gemma": Path(gemma) / ANSWERS,
        },
        {"agreed": out_dir / AGREED, "disagreements": out_dir / DISAGREEMENTS},
        summary,
    )
    return 0


def export_command(compare: str, glm: str, out: str) -> int:
    """`relevance export`: the agreed papers and the disagreements the arbiter marked relevant."""
    out_dir = new_folder(out)
    agreed = read_csv(Path(compare) / AGREED)
    disagreements = {d["corpus_id"]: d for d in read_csv(Path(compare) / DISAGREEMENTS)}
    arbiter = answers_of(glm)
    strangers = [a["corpus_id"] for a in arbiter if a["corpus_id"] not in disagreements]
    if strangers:
        raise SystemExit(
            f"{glm}: the arbiter answered about {strangers[0]}, which is not a disagreement of {compare}."
        )
    relevant = [{**row, "glm_label": "", "route": "agreed"} for row in agreed]
    relevant += [
        {**disagreements[a["corpus_id"]], "glm_label": a["label"], "route": "tie-break"}
        for a in arbiter
        if a["label"] == "relevant"
    ]
    write_csv(out_dir / RELEVANT, RELEVANT_COLUMNS, relevant)
    routes = Counter(r["route"] for r in relevant)
    summary = {
        "agreed": routes["agreed"],
        "tie-break": routes["tie-break"],
        "relevant": len(relevant),
        "disagreements": len(disagreements),
        "arbiter answers": len(arbiter),
    }
    write_receipt(
        out_dir,
        "export",
        {
            "agreed": Path(compare) / AGREED,
            "disagreements": Path(compare) / DISAGREEMENTS,
            "glm": Path(glm) / ANSWERS,
        },
        {"relevant": out_dir / RELEVANT},
        summary,
    )
    return 0
