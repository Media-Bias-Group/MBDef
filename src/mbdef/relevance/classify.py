"""`relevance classify`: ask one classifier whether each paper is relevant to media bias research, or replay its recorded answers.

A paper is a row of a papers file (`corpus_id`, `paperid`, `title`, `abstract`, optionally `doi`). For each paper with a title or an abstract, the classifier gets one chat request: the system prompt (three definitions of media bias and the rule) and the user prompt (the paper's title and abstract, for the arbiter after eight labelled examples), rendered from the templates in `prompts/` as the released runs rendered them. Its answer is reduced to one label by `label`: `relevant`, `not relevant`, `can't say` or `can't determine`.

The three classifiers are `CLASSIFIERS`. Worked example: for Gemma, the paper *Sharing Lies: A Socio-Technical Review on False Information Dissemination* is one request to model `gemma_4_12b_it` with `max_tokens` 2048 and no temperature; it answered `relevant`, and Qwen answered `not relevant`, so the paper went to the arbiter.

**Replay.** A recording is a folder with the answers (`answers.csv`, or `normalized_predictions.csv` in the released runs) and `metadata.json` (the model and settings). Each recorded answer must answer a request this module builds for a paper of the papers file with the same paper id, compared by `request_sha256` (the key of the repository's model layer, `mbdef.llm`); a recorded answer that does not stops the replay, so a changed prompt, example, setting or text cannot pass unnoticed. A paper with a request but no recorded answer is listed as not answered. Answers keep their recorded order.

**Which paper an answer belongs to.** An answer of this module names its paper (`corpus_id`). The released answers name only the paper id, which several papers can share; for them the released rule holds: the k-th answer under an id belongs to the k-th paper under that id in the papers file. The released Qwen run recorded its answers in the order they came back, so the rule gave a few of its answers to another paper of the same id; the replay keeps that pairing, which made the released list, and counts it (`paired with another paper's text`).
"""

import csv
import functools
import json
import re
import string
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, StrictUndefined, Template

from mbdef import files, llm

csv.field_size_limit(sys.maxsize)  # an answer with its reasoning can be long

PROMPTS = Path(__file__).resolve().parent / "prompts"
ANSWERS = "answers.csv"
RECORDED_ANSWERS = (
    "normalized_predictions.csv"  # the released runs' name for their answers
)
METADATA = "metadata.json"
NOT_ANSWERED = "not-answered.csv"
MAX_TOKENS = 2048
TIMEOUT_SECONDS = 600
RETRIES = 3  # per request, after the first try


@dataclass(frozen=True)
class Classifier:
    model: str  # the model's name on the server, as the released run recorded it
    template: str  # the user prompt's template in prompts/
    examples: bool  # whether the prompt shows the labelled examples
    workers: int  # requests in parallel, as in the released run


CLASSIFIERS = {
    # Qwen3.6-35B-A3B on a vLLM server. The released run used the few-shot template with "similar examples", but its examples were chosen for the hand-labelled papers, and no corpus paper is one of them: every prompt had no examples, which is the zero-shot prompt's text, so that is the template here.
    "qwen": Classifier("qwen3_6_35b_a3b", "zero_shot", False, 32),
    # Gemma 4 12B (instruction-tuned) on a vLLM server, zero-shot.
    "gemma": Classifier("gemma_4_12b_it", "zero_shot", False, 32),
    # GLM 5.2 through OpenRouter, the arbiter: few-shot with the eight "diverse" labelled examples.
    "glm": Classifier("z-ai/glm-5.2", "few_shot_diverse", True, 8),
}
PAPER_COLUMNS = ("corpus_id", "paperid", "title", "abstract")
ANSWER_COLUMNS = [
    "corpus_id",
    "paperid",
    "template",
    "model",
    "system_prompt",
    "prompt",
    "response",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "label",
]

# The answer parser of the released runs.
FINAL_ANSWER = re.compile(
    r"\b(?:final answer|answer|label|classification)\s*(?:is|:)?\s*(not relevant|irrelevant|relevant)\b"
)
ANY_LABEL = re.compile(r"\bnot relevant\b|\birrelevant\b|\brelevant\b")
UNSURE = re.compile(r"\b(can't|cannot|can not|cant) say\b|\binsufficient\b|\bunclear\b")
PUNCTUATION = str.maketrans("", "", string.punctuation.replace("'", ""))


def label(response: str) -> str:
    """The label of an answer: the text after the reasoning (`<think>…</think>`), lower case and without punctuation but the apostrophe; then its explicit final answer ("final answer", "answer", "label" or "classification" followed by a label), else the last label it names, else `can't say` if it says it cannot say, else `can't determine`. "irrelevant" counts as `not relevant`. The prompt's "not sure" is `can't determine`."""
    text = re.sub(r"(?is)<think>.*?</think>", " ", response.lower().strip())
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    text = re.sub(r"\s+", " ", text.translate(PUNCTUATION)).strip()
    found = FINAL_ANSWER.search(text)
    word = found.group(1) if found else None
    if word is None and (labels := ANY_LABEL.findall(text)):
        word = labels[-1]
    if word is not None:
        return "not relevant" if word in ("not relevant", "irrelevant") else "relevant"
    return "can't say" if UNSURE.search(text) else "can't determine"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, columns, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def read_papers(path: Path) -> list[dict]:
    """The papers file's rows, in its order. Every row needs `PAPER_COLUMNS`, and `corpus_id` must be unique."""
    papers = read_csv(path)
    missing = [c for c in PAPER_COLUMNS if papers and c not in papers[0]]
    if missing:
        raise SystemExit(f"{path} has no column {', '.join(missing)}.")
    repeated = [c for c, n in Counter(p["corpus_id"] for p in papers).items() if n > 1]
    if repeated:
        raise SystemExit(
            f"{path}: corpus_id {repeated[0]} is in {Counter(p['corpus_id'] for p in papers)[repeated[0]]} rows; it must name one paper."
        )
    return papers


def has_text(paper: dict) -> bool:
    """A paper without a title and without an abstract is not sent (the released runs skipped it too)."""
    return bool(paper["title"].strip() or paper["abstract"].strip())


@functools.cache
def template(name: str) -> Template:
    """A prompt template of prompts/, as the released runs loaded it (an undefined name is an error)."""
    source = (PROMPTS / f"{name}.j2").read_text(encoding="utf-8")
    return Environment(undefined=StrictUndefined, autoescape=False).from_string(source)


def render(name: str, context: dict) -> str:
    return template(name).render(**context).strip()


def messages(classifier: Classifier, paper: dict, examples: list[dict]) -> list[dict]:
    """The chat messages for one paper: the system prompt and the paper's prompt."""
    prompt = render(
        classifier.template,
        {**paper, "examples": examples if classifier.examples else []},
    )
    return [
        {"role": "system", "content": render("system_prompt", {})},
        {"role": "user", "content": prompt},
    ]


def settings(max_tokens: int, temperature: float | None) -> dict:
    """The request settings besides the model and the messages: only those that are set."""
    return {"max_tokens": max_tokens} | (
        {} if temperature is None else {"temperature": temperature}
    )


def post(api_base: str, api_key: str, body: dict) -> dict:
    """One chat completion from an OpenAI-compatible server, with up to `RETRIES` retries: after a rate limit (429) as long as the server asks (or 65 s), after another error 1, 2, 4 … s. An empty answer counts as an error."""
    request = urllib.request.Request(
        api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    for attempt in range(RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as handle:
                answer = json.loads(handle.read())
            text = (answer["choices"][0]["message"].get("content") or "").strip()
            if not text:
                raise ValueError(
                    f"empty answer, finish_reason {answer['choices'][0].get('finish_reason')!r}"
                )
            return {
                "response": text,
                **{
                    k: (answer.get("usage") or {}).get(k, "")
                    for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                },
            }
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as error:
            if attempt == RETRIES:
                raise
            wait = min(2**attempt, 30)
            if isinstance(error, urllib.error.HTTPError) and error.code == 429:
                wait = float(error.headers.get("Retry-After") or 63) + 2
            print(
                f"request failed ({error}); trying again in {wait:.0f}s",
                file=sys.stderr,
            )
            time.sleep(wait)
    raise AssertionError("unreachable")


def answer_row(
    paper: dict, classifier: Classifier, model: str, sent: list[dict], reply: dict
) -> dict:
    return {
        "corpus_id": paper["corpus_id"],
        "paperid": paper["paperid"],
        "template": classifier.template,
        "model": model,
        "system_prompt": sent[0]["content"],
        "prompt": sent[1]["content"],
        **reply,
        "label": label(reply["response"]),
    }


def live(
    classifier, model, sent_settings, papers, examples, api_base, api_key, workers
):
    """Ask the server about every paper with text, in the papers' order: (answers, failures)."""
    asked = [p for p in papers if has_text(p)]

    def one(paper: dict):
        sent = messages(classifier, paper, examples)
        try:
            reply = post(
                api_base, api_key, {"model": model, "messages": sent, **sent_settings}
            )
        except Exception as error:  # noqa: BLE001 - a failed paper is listed, the others go on
            return paper, None, f"failed: {error}"
        return paper, answer_row(paper, classifier, model, sent, reply), None

    with ThreadPoolExecutor(workers) as pool:
        done = list(pool.map(one, asked))
    answers = [row for _, row, _ in done if row is not None]
    failures = {paper["corpus_id"]: error for paper, _, error in done if error}
    return answers, failures


def recording(folder: Path) -> tuple[dict, list[dict]]:
    """A recording's metadata and answers."""
    meta_path = folder / METADATA
    if not meta_path.is_file():
        raise SystemExit(f"{folder} has no {METADATA}: not a recording of answers.")
    path = (
        folder / ANSWERS if (folder / ANSWERS).is_file() else folder / RECORDED_ANSWERS
    )
    if not path.is_file():
        raise SystemExit(f"{folder} has neither {ANSWERS} nor {RECORDED_ANSWERS}.")
    return json.loads(meta_path.read_text(encoding="utf-8")), read_csv(path)


def replay(classifier, model, sent_settings, papers, examples, folder: Path):
    """The recorded answers in FOLDER, each checked against the request this module builds (module docstring):
    (answers, the number paired by order with another paper's text)."""
    meta, recorded = recording(folder)
    wanted = {
        "model": model,
        "max_tokens": sent_settings["max_tokens"],
        "temperature": sent_settings.get("temperature"),
    }
    got = {k: meta.get(k) for k in wanted}
    if got != wanted:
        raise SystemExit(
            f"{folder} was recorded with {got}, not {wanted}; give the same --model, --max-tokens and --temperature."
        )
    key_of: dict[str, str] = {}  # corpus_id -> request_sha256
    under_id: dict[str, list[str]] = defaultdict(
        list
    )  # paper id -> its corpus_ids, in the papers' order
    for paper in papers:
        under_id[paper["paperid"]].append(paper["corpus_id"])
        if has_text(paper):
            key_of[paper["corpus_id"]] = llm.request_key(
                model, messages(classifier, paper, examples), sent_settings
            )
    taken: Counter = Counter()
    answers, other_text = [], 0
    for number, row in enumerate(recorded, 1):
        sent = [
            {"role": "system", "content": row["system_prompt"]},
            {"role": "user", "content": row["prompt"]},
        ]
        key = llm.request_key(row["model"], sent, sent_settings)
        same_id = under_id.get(row["paperid"], [])
        if row.get("corpus_id"):
            corpus_id = row["corpus_id"]
            known = key_of.get(corpus_id) == key
        else:  # a released answer: the released rule pairs it by order
            known = key in {key_of.get(c) for c in same_id}
            corpus_id = (
                same_id[taken[row["paperid"]]]
                if taken[row["paperid"]] < len(same_id)
                else None
            )
            taken[row["paperid"]] += 1
        if not known or corpus_id is None:
            raise SystemExit(
                f"{folder}: recorded answer {number} (paper id {row['paperid']}, request_sha256 {key}) answers no "
                "request this step builds from the papers file: its title, abstract, prompt, examples or settings "
                "differ. Replay with the papers and settings it was recorded with, or classify live."
            )
        recorded_label = row.get("label") or row.get("clean_label")
        if recorded_label and recorded_label != label(row["response"]):
            raise SystemExit(
                f"{folder}: recorded answer {number} was labelled {recorded_label!r}; the parser here gives {label(row['response'])!r}."
            )
        other_text += key_of.get(corpus_id) != key
        answers.append(
            answer_row(
                {"corpus_id": corpus_id, "paperid": row["paperid"]},
                classifier,
                row["model"],
                sent,
                {
                    k: row.get(k, "")
                    for k in (
                        "response",
                        "prompt_tokens",
                        "completion_tokens",
                        "total_tokens",
                    )
                },
            )
        )
    return answers, other_text


def classify_command(
    name: str,
    papers_arg: str | None,
    out: str,
    replay_from: str | None,
    examples_arg: str | None,
    model: str | None,
    max_tokens: int,
    temperature: float | None,
    api_base: str | None,
    workers: int | None,
) -> int:
    """`relevance classify`: the answers of classifier NAME about the papers, live or replayed."""
    classifier = CLASSIFIERS[name]
    model = model or classifier.model
    out_dir = Path(out)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} is not empty; give a new or empty folder.")
    papers_path, papers_name = files.located(papers_arg, "relevance/papers.csv")
    papers = read_papers(papers_path)
    inputs = {"papers": papers_path}
    examples: list[dict] = []
    if classifier.examples:
        examples_path, _ = files.located(examples_arg, "relevance/examples.csv")
        examples = read_csv(examples_path)
        inputs["examples"] = examples_path
    sent_settings = settings(max_tokens, temperature)
    started = time.time()
    other_text = 0
    if replay_from:
        answers, other_text = replay(
            classifier, model, sent_settings, papers, examples, Path(replay_from)
        )
        failures: dict[str, str] = {}
        inputs["recording"] = Path(replay_from)
    else:
        base = api_base or files.env("DSPY_API_BASE")
        answers, failures = live(
            classifier,
            model,
            sent_settings,
            papers,
            examples,
            base,
            files.env("DSPY_API_KEY"),
            workers or classifier.workers,
        )
    answered = {a["corpus_id"] for a in answers}
    not_answered = [
        {
            "corpus_id": p["corpus_id"],
            "paperid": p["paperid"],
            "reason": "no title or abstract"
            if not has_text(p)
            else failures.get(p["corpus_id"], "not answered"),
        }
        for p in papers
        if p["corpus_id"] not in answered
    ]
    write_csv(out_dir / ANSWERS, ANSWER_COLUMNS, answers)
    write_csv(out_dir / NOT_ANSWERED, ["corpus_id", "paperid", "reason"], not_answered)
    metadata = {
        "classifier": name,
        "model": model,
        "template": classifier.template,
        **sent_settings,
        "temperature": temperature,
    }
    files.write_json(out_dir / METADATA, metadata)
    labels = Counter(a["label"] for a in answers)
    summary = {
        "papers": len(papers),
        "without title or abstract": sum(not has_text(p) for p in papers),
        "answers": len(answers),
        "not answered": sum(
            r["reason"] != "no title or abstract" for r in not_answered
        ),
        "labels": dict(labels.most_common()),
    }
    if other_text:
        summary["paired with another paper's text"] = other_text
    receipt = files.receipt(
        f"mbdef relevance classify --classifier {name}",
        {
            **metadata,
            "papers": papers_name,
            "replay": replay_from,
            "api_base": None if replay_from else api_base or "DSPY_API_BASE",
        },
        inputs,
        {"answers": out_dir / ANSWERS, "not_answered": out_dir / NOT_ANSWERED},
    )
    receipt["summary"] = summary
    files.write_json(out_dir / files.RECEIPT, receipt)
    print(
        f"{name}: {len(answers)} answers ({', '.join(f'{n} {k}' for k, n in labels.most_common())}) in {time.time() - started:.0f}s; "
        + "; ".join(
            f"{k}: {v}" for k, v in summary.items() if k not in ("labels", "answers")
        )
    )
    return 0
