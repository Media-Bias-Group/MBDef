"""`mbdef postprocess judge`: the scores of the ensemble's GT-RAG judge (a judge retrieval-augmented with the ground truth, GT; Gemma-4-12B, 32 neighbours).

The judge is a language model (google/gemma-4-12B-it-qat-w4a16-ct, served by vLLM) that is shown a paper's title, up to 4 of its statements (a term and one of its definitions) and, for each statement, the 32 most similar labelled train records of the ground truth ("retrieval-augmented"), and answers for each statement whether it is a definition and whether it is relevant to a media-bias taxonomy (yes, no or uncertain), as JSON in a fixed schema, thinking off, temperature 0. A statement's score is P(yes) + 0.5 P(uncertain) of its relevance answer, read from the log-probabilities of the same request; a term's score is the highest score of its statements. This is the adopted setting (Qwen3-Embedding-8B retrieval, 32 neighbours, thinking off). Three steps:

  1. `package`: the corpus's statements (every definition of the 67,394 corpus terms) and the ground truth's train + dev statements, each with its 32 nearest train records (exact inner product of the cached Qwen3-Embedding-8B vectors of "term: definition", a record scoring its best definition, never the statement's own paper; at most one record per paper), and the train records themselves;
  2. `ask`: one request per 4 statements of a paper, sent to the model's server; the answers and their probabilities go into a run folder (decisions.json, as the recorded runs wrote it);
  3. `table`: the score table of the run folders, for the corpus or for the ground truth.

Worked example (the recorded corpus run): the term "observer strategies" (`m0000fd9a4a4afc72`) has 3 definitions; the judge answered "yes" for each, with P(yes), P(uncertain) = (0.94620, 0.02858), (0.99994, 0.00003) and (0.99960, 0.00011), so they score 0.960, 0.99996 and 0.99965, and the term scores the highest, 0.99996.

The code reproduces the recorded runs exactly: their requests and tables come back byte for byte (docs/step-5-post-processing/postprocess.md, "Rerun the judge").
"""

import json
import math
import random
import re
import sqlite3
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

# llm first: it imports NumPy before DSPy
from mbdef import files, llm, models

# isort: split
import dspy
import numpy as np
from dspy.adapters.json_adapter import JSONAdapter
from pydantic import BaseModel, Field, create_model

from mbdef.closure.pdf_identity import lower_ascii_words
from mbdef.extract.extractor import output_schema
from mbdef.extract.verifier import strict_verdicts

from . import groundtruth

# The package's name of the neighbour lists (whatever --neighbours is)
ARM = "qwen8b-combined-nearest32"
# Neighbours per statement (`package --neighbours`), and statements per request, all of one paper (`package --batch` and `ask --batch`)
K, BATCH = 32, 4
SHARDS = (
    8  # the corpus package's papers are split into 8 shards, balanced by statements
)
CORPUS_BLOCK = 4096  # corpus statements searched per faiss call; the ground truth's one at a time, as recorded
SHUFFLE_SEED = "20261016"  # a ground-truth paper's statements are shuffled with random.Random(f"{seed}:{paper id}")
# The two ground-truth files (their names in the --ground-truth-folder), the digit of their statement ids, their label
GROUNDTRUTH_FILES = tuple(
    (name, digit, label)
    for (name, digit), label in zip(
        groundtruth.FILES, ("relevant", "not-relevant"), strict=True
    )
)


def groundtruth_folder_default() -> Path:
    """The folder that holds them: MBDef-Gold's in the data root."""
    return groundtruth.paths()[0][0].parent


# The ground-truth package holds no TEST statement
REFERENCE_SPLITS = groundtruth.REFERENCE_SPLITS
PACKAGE, REQUESTS = "package.zip", "requests.jsonl"
GROUNDTRUTH_PACKAGE, GROUNDTRUTH_REQUESTS = (
    "groundtruth-package.zip",
    "groundtruth-requests.jsonl",
)
MEMBER = "inputs.json"  # the package's one file inside its zip
ZIP_DATE = (
    2026,
    10,
    21,
    0,
    0,
    0,
)  # a fixed date in the zip, so the same package is the same zip

MODEL, REVISION = models.PINNED["gemma"]  # `ask --model`
ROUTE = "openai/"  # before the model's name in the recorded request hashes (an OpenAI-compatible route)
# `ask --temperature --max-tokens --seed`
SETTINGS = {"temperature": 0.0, "max_tokens": 4096, "seed": 9}
CHAT_TEMPLATE = {"enable_thinking": False}
TOP_LOGPROBS = (
    20  # alternatives per generated token, for the read-out of the relevance answer
)
SERVER = "http://127.0.0.1:8000"
CONCURRENCY = 128  # requests in flight, as the recorded runs
ATTEMPTS = 5  # a request that fails on the network or the server is tried again after 5, 10, ... seconds
TIMEOUT_SECONDS = 3600
CLASSES = ("yes", "no", "uncertain")
RELEVANCE = {"yes": 1.0, "uncertain": 0.5, "no": 0.0}
METHOD = "judge prob: gemma, qwen8b-combined-nearest32, thinking off, temperature 0"
CORPUS_METHOD = METHOD + " (corpus)"
ANSWERS = "answers.jsonl"  # one line per request in a run folder, appended as answers come back
DECISIONS = "decisions.json"  # every statement's verdicts, written once every request has an answer
COMPLETE = "generation-complete.json"

INSTRUCTIONS = """Judge proposed definition statements extracted from a scientific paper for a
media-bias taxonomy. You see only the paper title, each candidate's term and
quoted definition, and annotated examples from OTHER training papers selected by
semantic similarity. No paper source text is supplied.

definition: yes if the quotation states what the term means (what it is, its
essential properties, or how it is operationalised); no if it is a fragment, an
example, a result, a list of mentions, an assertion that does not explain the
concept's meaning, or it defines a different term; uncertain when the quotation
alone cannot establish this.

relevance: interpret broadly: media/news bias and framing, misinformation,
persuasion, cognitive/audience/social biases, and bias/fairness in data or
algorithms can qualify. Do not require the literal words 'media bias'. A generic
technical or social term is not relevant merely because the paper is. Mark no
only with a candidate-specific reason; mark uncertain for borderline concepts.

Similar wording does not transfer a label automatically, and the examples are not
exhaustive. Explain the closest relevant analogy and a material difference when
it affects your judgment. Put definition validity in `reason` and the scope
judgment in `relevance_reason`. Return one judgment for every ID; never rewrite,
add or combine candidates.

In this study each training example carries a `label` from human annotation of
that concept in its own paper:
- relevant: relevant to the media-bias taxonomy;
- not-relevant: NOT relevant to the media-bias taxonomy.
Use not-relevant examples to locate the scope boundary. In stratified lists the label
mix is fixed by design (4 relevant, 4 not-relevant) and is not a prior probability.
"""


class ScopeVerdict(BaseModel):
    definition: Literal["yes", "no", "uncertain"]
    relevance: Literal["yes", "no", "uncertain"]
    reason: str = Field(max_length=240)
    relevance_reason: str = Field(max_length=240)


class GTOnlyJudge(dspy.Signature):
    title: str = dspy.InputField()
    candidates: str = dspy.InputField(
        desc="JSON array of candidate IDs, terms and definition quotations."
    )
    training_examples: str = dspy.InputField(
        desc="Other training papers only. Neighbors illustrate annotation scope, not answers for this paper."
    )
    verdicts: dict[str, ScopeVerdict] = dspy.OutputField()


# --- 1. the packages -------------------------------------------------------------------------------------------------


def norm_title(title) -> str:
    """The title's lower-case ASCII words: the closure's identity check's rule (`lower()`, not `casefold()`)."""
    return lower_ascii_words(title)


def statement_text(term, definition) -> str:
    """The embedded text of a statement (Step 5's cache keys)."""
    return f"{term}: {definition}"


def definitions(record) -> list[str]:
    return [d["text"] for d in record["definitions"] if d.get("text", "").strip()]


def training_examples(folder) -> list[dict]:
    """The ground truth's train records, relevant file first, papers and records in file order."""
    out = []
    for name, _, label in GROUNDTRUTH_FILES:
        path = Path(folder) / name
        for p in json.loads(path.read_text())["papers"]:
            if p["split"] != "train":
                continue
            for i, r in enumerate(p["definition_records"]):
                out.append(
                    dict(
                        paper_id=p["paper_id"],
                        split="train",
                        title=p["metadata"]["title"],
                        term=r["term"],
                        definitions=definitions(r),
                        label=label,
                        annotation_key=f"{Path(path).name}:{p['paper_id']}:{i}",
                    )
                )
    return out


def groundtruth_papers(folder, splits=REFERENCE_SPLITS) -> list[dict]:
    """The ground truth's papers of `splits`, by paper id, each with one statement per definition, id "<1|0>-<record>-<definition>", shuffled per paper (so a request never holds only one file's records)."""
    papers = {}
    for name, y, _ in GROUNDTRUTH_FILES:
        for p in json.loads((Path(folder) / name).read_text())["papers"]:
            entry = papers.setdefault(
                p["paper_id"],
                dict(
                    paper_id=p["paper_id"],
                    title=p["metadata"]["title"],
                    split="test" if "test" in p["split"] else p["split"],
                    source_sha256="",
                    candidates=[],
                ),
            )
            for i, r in enumerate(p["definition_records"]):
                entry["candidates"] += [
                    dict(id=f"{y}-{i}-{j}", term=r["term"], definition=t)
                    for j, t in enumerate(definitions(r))
                ]
    for p in papers.values():
        random.Random(f"{SHUFFLE_SEED}:{p['paper_id']}").shuffle(p["candidates"])
    return [papers[k] for k in sorted(papers) if papers[k]["split"] in splits]


def corpus_papers(training, store=None, links=None):
    """The corpus view's papers in the order of their first term (mention id), each with one statement per definition (id "<mention id>-<definition index>") and its shard; and, per paper, the train paper of the same normalised title, if any (its records are never its neighbours)."""
    from .corpus import default_store, load_view

    view = load_view(store, links)
    db = sqlite3.connect(f"file:{store or default_store()}?immutable=1", uri=True)
    try:
        titles = dict(db.execute("SELECT paper_id, title FROM papers"))
    finally:
        db.close()
    by_title = {}
    for r in training:
        by_title.setdefault(norm_title(r["title"]), set()).add(r["paper_id"])
    papers = {}
    for c in sorted(view.concepts, key=lambda c: c.mention_id):
        p = papers.setdefault(
            c.paper_id,
            dict(
                paper_id=c.paper_id,
                title=titles.get(c.paper_id) or "",
                source_sha256="",
                corpus=c.corpus,
                candidates=[],
            ),
        )
        for j, d in enumerate(c.definitions):
            p["candidates"].append(
                dict(id=f"{c.mention_id}-{j}", term=c.label, definition=d.text)
            )
    same_title = {}
    for pid, p in papers.items():
        hits = (
            by_title.get(norm_title(p["title"]), set())
            if norm_title(p["title"])
            else set()
        )
        if len(hits) > 1:
            raise SystemExit(
                f"{pid}: its title is the title of {len(hits)} train papers"
            )
        same_title[pid] = next(iter(hits)) if hits else None
    load = [0] * SHARDS  # largest papers first, each to the lightest shard
    for pid in sorted(papers, key=lambda q: -len(papers[q]["candidates"])):
        s = load.index(min(load))
        papers[pid]["split"] = f"shard{s}"
        load[s] += len(papers[pid]["candidates"])
    return papers, same_title


def nearest(similarities, training, excluded_paper, k=K) -> list[dict]:
    """The k train records most similar to a statement (stable order), never from `excluded_paper`, at most ceil(k / papers) per paper (1 while k fits the papers available)."""
    papers = len({r["paper_id"] for r in training} - {excluded_paper})
    per_paper = -(-k // papers)
    result, used = [], Counter()
    for index in np.argsort(-np.asarray(similarities), kind="stable"):
        record = training[int(index)]
        if record["split"] != "train":
            raise ValueError("a neighbour that is not a train record")
        if (
            record["paper_id"] == excluded_paper
            or used[record["paper_id"]] == per_paper
        ):
            continue
        result.append(dict(train_index=int(index), cosine=float(similarities[index])))
        used[record["paper_id"]] += 1
        if len(result) == k:
            return result
    raise ValueError(f"fewer than {k} train records outside the paper {excluded_paper}")


def neighbour_lists(queries, training, cache, block, k=K) -> list[list[dict]]:
    """For each (excluded paper, statement text), its k nearest train records: faiss's exact inner product over the train records' definitions, `block` statements per search, a record scoring its best definition."""
    import faiss

    rows = [
        (i, statement_text(r["term"], d))
        for i, r in enumerate(training)
        for d in r["definitions"]
    ]
    texts = sorted({t for _, t in rows} | {t for _, t in queries})
    vectors = dict(zip(texts, cache.vectors(texts), strict=True))
    index = faiss.IndexFlatIP(len(vectors[texts[0]]))
    index.add(np.stack([vectors[t] for _, t in rows]))
    owners = np.array([i for i, _ in rows])
    out = []
    for s in range(0, len(queries), block):
        chunk = queries[s : s + block]
        scores, ids = index.search(
            np.stack([vectors[t] for _, t in chunk]), index.ntotal
        )
        for (excluded, _), sc, ix in zip(chunk, scores, ids, strict=True):
            sim = np.full(len(training), -np.inf)
            np.maximum.at(sim, owners[ix], sc)
            out.append(nearest(sim, training, excluded, k))
    return out


def write_package(
    path: Path, union: list[dict], training: list[dict], neighbors: dict
) -> str:
    """The package (one JSON file in a zip, as the recorded runs read it); returns the JSON's sha256."""
    package = dict(
        union=dict(papers=union),
        training_examples=training,
        neighbors=neighbors,
        recovery_windows=[],
    )
    text = files.json_text(package, indent=None, end="")
    info = zipfile.ZipInfo(MEMBER, date_time=ZIP_DATE)
    info.compress_type = zipfile.ZIP_DEFLATED
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(info, text)
    return files.sha256(text.encode())


def write_index(path: Path, union: list[dict], key: str, batch=BATCH) -> int:
    """One line per request: its number, the paper's shard (or split), the paper and its statements' ids."""
    n = 0
    with path.open("w") as f:
        for p in union:
            for start in range(0, len(p["candidates"]), batch):
                ids = [c["id"] for c in p["candidates"][start : start + batch]]
                f.write(
                    json.dumps(
                        {
                            "request": n,
                            key: p["split"],
                            "paper_id": p["paper_id"],
                            "candidates": ids,
                        }
                    )
                    + "\n"
                )
                n += 1
    return n


def read_package(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read(MEMBER))


def package_command(
    out: Path,
    store=None,
    links=None,
    cache=None,
    groundtruth_folder=None,
    k=K,
    batch=BATCH,
) -> int:
    """`postprocess judge package`: the corpus package and the ground truth's train + dev package, with their request lists."""
    from .embeddings import CACHE, EmbeddingCache

    cache = EmbeddingCache(cache or CACHE)
    groundtruth_folder = groundtruth_folder or groundtruth_folder_default()
    training = training_examples(groundtruth_folder)
    papers, same_title = corpus_papers(training, store, links)
    queries = [
        (same_title[pid] or pid, statement_text(c["term"], c["definition"]))
        for pid, p in papers.items()
        for c in p["candidates"]
    ]
    lists = iter(neighbour_lists(queries, training, cache, CORPUS_BLOCK, k))
    neighbors = {
        pid: {c["id"]: {ARM: next(lists)} for c in p["candidates"]}
        for pid, p in papers.items()
    }
    union = [
        dict(
            **{k: v for k, v in p.items() if k != "candidates"},
            candidates=p["candidates"],
        )
        for _, p in sorted(papers.items())
    ]
    digest = write_package(Path(out) / PACKAGE, union, training, neighbors)
    n = write_index(Path(out) / REQUESTS, union, "shard", batch)
    statements = sum(len(p["candidates"]) for p in union)
    print(
        f"corpus: {len(union)} papers, {statements} statements, {n} requests -> {Path(out) / PACKAGE} ({MEMBER} sha256 {digest})"
    )
    gt = groundtruth_papers(groundtruth_folder)
    queries = [
        (p["paper_id"], statement_text(c["term"], c["definition"]))
        for p in gt
        for c in p["candidates"]
    ]
    lists = iter(neighbour_lists(queries, training, cache, 1, k))
    neighbors = {
        p["paper_id"]: {c["id"]: {ARM: next(lists)} for c in p["candidates"]}
        for p in gt
    }
    digest = write_package(Path(out) / GROUNDTRUTH_PACKAGE, gt, training, neighbors)
    n = write_index(Path(out) / GROUNDTRUTH_REQUESTS, gt, "split", batch)
    statements = sum(len(p["candidates"]) for p in gt)
    print(
        f"ground truth (train + dev): {len(gt)} papers, {statements} statements, {n} requests -> "
        f"{Path(out) / GROUNDTRUTH_PACKAGE} ({MEMBER} sha256 {digest})"
    )
    from .corpus import default_links, default_store

    receipt = files.receipt(
        "mbdef postprocess judge package",
        dict(neighbours=k, batch=batch, shards=SHARDS, shuffle_seed=SHUFFLE_SEED),
        {
            "store": Path(store or default_store()),
            "links": Path(links or default_links()),
            "cache": cache.directory,
            "ground_truth_folder": Path(groundtruth_folder),
        },
        {
            name: Path(out) / name
            for name in (PACKAGE, REQUESTS, GROUNDTRUTH_PACKAGE, GROUNDTRUTH_REQUESTS)
        },
    )
    files.write_json(Path(out) / files.RECEIPT, receipt)
    return 0


# --- 2. the requests and the answers ---------------------------------------------------------------------------------


def requests_for(package: dict, split=None, size=BATCH) -> list[dict]:
    """The requests of a package (of one shard or split): `size` (4) statements of a paper each, in package order, numbered from 0. The model sees the statements as c1..c4 (`shown_ids` maps them back), so an id never shows its label."""
    out = []
    for paper in package["union"]["papers"]:
        if split is not None and paper.get("split") != split:
            continue
        for start in range(0, len(paper["candidates"]), size):
            batch = [
                {k: c[k] for k in ("id", "term", "definition")}
                for c in paper["candidates"][start : start + size]
            ]
            shown = {c["id"]: f"c{j + 1}" for j, c in enumerate(batch)}
            examples = [
                dict(
                    candidate_id=shown[c["id"]],
                    examples=[
                        {
                            **package["training_examples"][r["train_index"]],
                            "cosine": r["cosine"],
                        }
                        for r in package["neighbors"][paper["paper_id"]][c["id"]][ARM]
                    ],
                )
                for c in batch
            ]
            inputs = dict(
                title=paper["title"],
                candidates=json.dumps(
                    [{**c, "id": shown[c["id"]]} for c in batch], ensure_ascii=False
                ),
                training_examples=json.dumps(examples, ensure_ascii=False),
            )
            out.append(
                dict(
                    paper_id=paper["paper_id"],
                    title=paper["title"],
                    sha256=paper["source_sha256"],
                    candidates=batch,
                    inputs=inputs,
                    split=paper.get("split"),
                    shown_ids={v: k for k, v in shown.items()},
                )
            )
    return [dict(**r, request_id=i) for i, r in enumerate(out)]


def judge_signature(ids):
    fields: dict[str, Any] = {cid: (ScopeVerdict, ...) for cid in ids}
    output = create_model("GTOnlyVerdicts", **fields)
    return GTOnlyJudge.with_instructions(INSTRUCTIONS).with_updated_fields(
        "verdicts", type_=output
    )


def render(request: dict):
    """A request's signature, its chat messages (DSPy's JSON adapter, as recorded) and its response format."""
    signature = judge_signature(list(request["shown_ids"]))
    messages = JSONAdapter(use_native_function_calling=False).format(
        signature, [], request["inputs"]
    )
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "extraction",
            "strict": True,
            "schema": output_schema(signature),
        },
    }
    return signature, messages, response_format


def request_sha256(messages, response_format, model=MODEL, settings=SETTINGS) -> str:
    """The hash the recorded runs logged for each request: `llm.request_key` of the model (by its OpenAI-compatible route), the messages and the settings, the thinking switch and the answer's schema among them."""
    return llm.request_key(
        ROUTE + model,
        messages,
        {
            **settings,
            "extra_body": {"chat_template_kwargs": CHAT_TEMPLATE},
            "response_format": response_format,
        },
    )


_VERDICT_HEAD = re.compile(
    r'"([^"\\]+)":\{"definition":"(?:yes|no|uncertain)","relevance":"'
)


def relevance_readout(choice, candidate_ids):
    """The model's own probabilities for each statement's relevance answer, from the token log-probabilities of the same request (before the JSON grammar or the temperature act on them).

    The answer is compact JSON in schema order, so each verdict starts `"<id>":{"definition":"…","relevance":"`. The token holding the first character of the relevance value is found in the generated token stream; its top alternatives count for "yes" if, after the characters they share with the chosen token before the value, their text is a start of `yes` (or `yes` followed by more), and the same for "no" and "uncertain"; alternatives that fit no answer are dropped and the three answers renormalised. `mass` is their total before renormalising.
    Returns {id: dict(yes, no, uncertain, mass, chosen)} and the token counts."""
    content = choice["message"]["content"]
    stream = ((choice.get("logprobs") or {}).get("content")) or []
    if not stream:
        raise ValueError("No token log-probabilities in the response")
    starts, text = [], ""
    for t in stream:
        starts.append(len(text))
        text += t["token"]
    base = text.rfind(content)
    if base < 0:
        raise ValueError("Answer text not found in the token stream")
    out = {}
    for m in _VERDICT_HEAD.finditer(content):
        pos = base + m.end()
        i = max(j for j, s in enumerate(starts) if s <= pos)
        lead = text[starts[i] : pos]
        alternatives = {
            a["token"]: a["logprob"] for a in stream[i].get("top_logprobs") or []
        }
        alternatives.setdefault(stream[i]["token"], stream[i]["logprob"])
        probs = dict.fromkeys(CLASSES, 0.0)
        for token, logprob in alternatives.items():
            if not token.startswith(lead) or len(token) == len(lead):
                continue
            rest = token[len(lead) :].split('"')[0]
            hits = [
                c
                for c in CLASSES
                if rest and (c.startswith(rest) or rest.startswith(c))
            ]
            if len(hits) == 1:
                probs[hits[0]] += math.exp(logprob)
        mass = sum(probs.values())
        if mass <= 0:
            raise ValueError(
                f"No allowed relevance answer among the top alternatives of {m.group(1)}"
            )
        chosen = content[m.end() :].split('"')[0]
        out[m.group(1)] = dict(
            **{c: v / mass for c, v in probs.items()}, mass=mass, chosen=chosen
        )
    if set(out) != set(candidate_ids):
        raise ValueError(
            f"Relevance read-out found {sorted(out)} for {sorted(candidate_ids)}"
        )
    first_answer_token = max(j for j, s in enumerate(starts) if s <= base)
    return out, dict(
        generated_tokens=len(stream),
        answer_tokens=len(stream) - first_answer_token,
        thinking_tokens=first_answer_token,
    )


def post(url: str, body: dict) -> dict:
    request = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as r:
        return json.load(r)


def ask_one(url: str, request: dict, model=MODEL, settings=SETTINGS) -> dict:
    """One request's line: its verdicts with their relevance probabilities, under the statements' own ids; or the error (a network or server error after ATTEMPTS tries, or an answer that fails a check)."""
    signature, messages, response_format = render(request)
    body = dict(
        model=model,
        messages=messages,
        **settings,
        response_format=response_format,
        logprobs=True,
        top_logprobs=TOP_LOGPROBS,
        chat_template_kwargs=CHAT_TEMPLATE,
    )
    row = dict(
        request_id=request["request_id"],
        paper_id=request["paper_id"],
        source_sha256=request["sha256"],
        request_sha256=request_sha256(messages, response_format, model, settings),
    )
    for attempt in range(ATTEMPTS):
        try:
            reply = post(url, body)
            break
        except (
            urllib.error.URLError,
            OSError,
        ) as e:  # the network or the server: tried again
            error = f"{type(e).__name__}: {e}"[:2000]
            time.sleep(5 * (attempt + 1))
    else:
        return row | dict(error=dict(type="server", message=error), value=None)
    try:
        value = strict_verdicts(reply, signature)
        readout, tokens = relevance_readout(reply["choices"][0], list(value))
        for cid, probs in readout.items():
            if probs["chosen"] != value[cid]["relevance"]:
                raise ValueError(
                    f"Read-out answer {probs['chosen']} differs from the verdict of {cid}"
                )
            value[cid]["relevance_probs"] = probs
    except Exception as e:  # noqa: BLE001  a malformed answer: recorded, asked again by the next run
        return row | dict(
            response_id=reply.get("id"),
            error=dict(type=type(e).__name__, message=str(e)[:2000]),
            value=None,
        )
    value = {request["shown_ids"][cid]: v for cid, v in value.items()}
    return row | dict(
        response_id=reply.get("id"),
        usage=reply.get("usage"),
        tokens=tokens,
        error=None,
        value=value,
    )


def ask_command(
    package_path: Path,
    out: Path,
    split=None,
    server=SERVER,
    concurrency=CONCURRENCY,
    model=MODEL,
    settings=SETTINGS,
    batch=BATCH,
) -> int:
    """`postprocess judge ask`: every request of the package (or of one shard or split) to the server; one line per request in OUT/answers.jsonl as the answers come back (a rerun asks only the requests without an answer), and, once every request has one, OUT/decisions.json and OUT/generation-complete.json."""
    package = read_package(package_path)
    requests = requests_for(package, split, batch)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    answers = out / ANSWERS
    done = {}
    if answers.exists():
        for line in answers.read_text().splitlines():
            row = json.loads(line)
            if row["value"] is not None:
                done[row["request_id"]] = row
    todo = [r for r in requests if r["request_id"] not in done]
    print(f"{len(todo)} requests to send ({len(done)} already answered)", flush=True)
    url, errors = server.rstrip("/") + "/v1/chat/completions", 0
    with answers.open("a") as f, ThreadPoolExecutor(concurrency) as pool:
        for n, future in enumerate(
            as_completed([pool.submit(ask_one, url, r, model, settings) for r in todo]),
            1,
        ):
            row = future.result()
            errors += row["error"] is not None
            if row["error"] is None:
                done[row["request_id"]] = row
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if n % 100 == 0 or n == len(todo):
                print(f"{n}/{len(todo)}  {errors} errors", flush=True)
    if errors:
        raise SystemExit(
            f"{errors} request(s) without an answer; run the same command again to send them"
        )
    judgments = {p["paper_id"]: {} for p in package["union"]["papers"]}
    for rid in sorted(done):
        row = done[rid]
        for cid, verdict in row["value"].items():
            judgments[row["paper_id"]].setdefault(cid, []).append(
                {**verdict, "source_sha256": row["source_sha256"]}
            )
    # As the recorded runs wrote them: indented by 2, UTF-8, a final newline; never half-written
    files.write_atomic({out / DECISIONS: files.json_text(judgments)})
    files.write_atomic(
        {out / COMPLETE: files.json_text(dict(requests=len(requests), failures=0))}
    )
    receipt = files.receipt(
        "mbdef postprocess judge ask",
        dict(
            model=model,
            **settings,
            batch=batch,
            split=split,
            thinking=CHAT_TEMPLATE["enable_thinking"],
            top_logprobs=TOP_LOGPROBS,
            server=server,
            concurrency=concurrency,
        ),
        {"package": Path(package_path)},
        {name: out / name for name in (ANSWERS, DECISIONS, COMPLETE)},
    )
    files.write_json(out / files.RECEIPT, receipt)
    print(f"{len(requests)} requests answered -> {out / DECISIONS}")
    return 0


# --- 3. the score tables ---------------------------------------------------------------------------------------------


def prob(verdict) -> float:
    return (
        verdict["relevance_probs"]["yes"]
        + 0.5 * verdict["relevance_probs"]["uncertain"]
    )


def run_decisions(runs):
    for run in runs:
        if not (Path(run) / COMPLETE).exists():
            raise SystemExit(f"{run}: not a complete run (no {COMPLETE})")
        yield json.loads((Path(run) / DECISIONS).read_text())


def corpus_scores(runs, package: dict) -> dict[str, float]:
    """{mention id: the highest score of its statements}, every statement of the package judged exactly once."""
    expected = {c["id"] for p in package["union"]["papers"] for c in p["candidates"]}
    scores, seen = {}, set()
    for decisions in run_decisions(runs):
        for cands in decisions.values():
            for cid, verdicts in cands.items():
                if not verdicts:
                    continue
                if cid in seen:
                    raise SystemExit(f"{cid} judged twice")
                seen.add(cid)
                m = cid.rsplit("-", 1)[0]
                scores[m] = max(scores.get(m, -math.inf), prob(verdicts[0]))
    missing = expected - seen
    if missing:
        raise SystemExit(
            f"{len(missing)} statements have no verdict, e.g. {sorted(missing)[0]}"
        )
    return {m: scores[m] for m in sorted({cid.rsplit("-", 1)[0] for cid in expected})}


def groundtruth_scores(runs) -> dict[str, float]:
    """{Step 5 record id: the highest score of its statements} (a statement's verdicts averaged), in the runs' order; record id "<paper>:<1|0>:<record>" from statement id "<1|0>-<record>-<definition>"."""
    per, seen = {}, set()
    for decisions in run_decisions(runs):
        for pid, cands in decisions.items():
            for cid, verdicts in cands.items():
                if (pid, cid) in seen:
                    raise SystemExit(f"{pid} {cid} judged twice")
                seen.add((pid, cid))
                y, i, _ = cid.split("-")
                per.setdefault(f"{pid}:{y}:{i}", []).append(
                    sum(prob(v) for v in verdicts) / len(verdicts)
                )
    return {rid: max(v) for rid, v in per.items()}


def table_command(package_path: Path, runs, out: Path, method=None) -> int:
    """`postprocess judge table`: the score table of the runs of a package, for the corpus (split "corpus") or the ground truth (each record's split; a record outside train + dev is refused)."""
    from .evaluate import ScoreTable

    package = read_package(package_path)
    if "corpus" in package["union"]["papers"][0]:
        scores = corpus_scores(runs, package)
        table = ScoreTable(
            method or CORPUS_METHOD, "term", scores, dict.fromkeys(scores, "corpus")
        )
    else:
        scores = groundtruth_scores(runs)
        papers = {p["paper_id"]: p["split"] for p in package["union"]["papers"]}
        split = {rid: papers.get(rid.rsplit(":", 2)[0], "") for rid in scores}
        unknown = [rid for rid, s in split.items() if s not in REFERENCE_SPLITS]
        if unknown:
            raise SystemExit(
                f"{len(unknown)} scores are not of a train or dev record, e.g. {unknown[0]}"
            )
        table = ScoreTable(
            method or METHOD, "term", scores, {rid: split[rid] for rid in scores}
        )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(out)
    print(f"{len(scores)} rows -> {out}, sha256 {files.file_sha256(out)}")
    return 0
