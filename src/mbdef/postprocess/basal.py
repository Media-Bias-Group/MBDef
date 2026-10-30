"""`mbdef postprocess basal`: the scores of the ensemble's decision model, basal-omni-1.0-ultra (FP8).

basal is a language model that answers a typed question about a "state" (here: a term and up to 3 of its definitions) with the probability of each option. Step 5 asks it which kind of term this is for a taxonomy of media bias: core (a type of bias), satellite (a concept needed to describe, detect or explain bias) or irrelevant; a term's score is P(core) + P(satellite). The model runs on a GPU behind its own server (`basal.server` of the `basal` package, with vLLM; docs/step-5-post-processing/postprocess.md gives the install). Three steps:

  1. `items`: one item {id, split, state} per Extracted Concept of the corpus (items.jsonl) and per train + dev record of the ground truth (groundtruth-items.jsonl, the reference its corpus scores are ranked against). The state is the term and its first 3 definitions, each with its whitespace collapsed and U+FFFD removed, and cut to 599 characters + "…" when longer than 600.
  2. `ask`: send every item to the server (POST /v1/systemone) and append each answer to a JSONL file as it comes back; a rerun asks only the items without an answer.
  3. `table`: the score table (method, unit, item_id, split, score) of the items, from the answers.

Worked example (the recorded corpus run, one H200 GPU): the first corpus item, `m0000fd9a4a4afc72` ("observer strategies" in a seed paper), got P(core) 0.136, P(satellite) 0.583 and P(irrelevant) 0.281, so its score is 0.136 + 0.583 = 0.719.
"""

import csv
import json
import math
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from mbdef import files
from mbdef.files import file_sha256

from . import groundtruth

MAX_DEFINITIONS = (
    3  # the definitions of a term that the state shows (`items --max-definitions`)
)
MAX_CHARS = 600  # a longer definition is cut to MAX_CHARS - 1 characters + "…" (`items --max-chars`)
ITEMS = "items.jsonl"  # the corpus's Extracted Concepts
# the ground truth's train + dev records (the reference)
GROUNDTRUTH_ITEMS = "groundtruth-items.jsonl"
# The ground truth's splits that `items` writes. The recorded runs also asked about the TEST records, for the one logged TEST read of the ensemble; the ensemble needs only train + dev, and a TEST record's id names its label.
REFERENCE_SPLITS = groundtruth.REFERENCE_SPLITS

# The question, as every recorded run asked it
KEYS = ("core", "satellite", "irrelevant")
QUESTION = {
    "type": "choice",
    "instructions": "For a taxonomy of media bias, which kind of term is this, as defined here?",
    "criteria": {
        "core": "a type of bias, including stereotypes and prejudice (e.g. confirmation bias, gender bias, selection bias, "
        "hostile media effect, stereotyping)",
        "satellite": "not itself a type of bias, but a concept the taxonomy needs to describe, detect or explain bias (e.g. "
        "stance detection, gender/sex, framing, ideology, news values, fact-checking, misinformation, "
        "propaganda, echo chambers, polarisation)",
        "irrelevant": "anything else (e.g. LLM, VLM, tokenization, generic research methods)",
    },
}
# How the recorded runs asked: basal's server on this machine, 32 requests in flight, the items in the order of random.Random(20261021).shuffle (so the server's batches mix items from all over the file), labelled run 1
SERVER = "http://127.0.0.1:8000"
ENDPOINT = "/v1/systemone"
REQUEST_MODEL = "basal"  # the request's model field
CONCURRENCY = 32
ORDER_SEED = 20261021
REPEAT = 1
ATTEMPTS = 5  # a network or server error is retried, after 2, 4, ... seconds; a malformed answer is not
TIMEOUT_SECONDS = 600
# The score table's method column (`table --method`); the recorded tables name their run there
METHOD = "basal-omni-1.0-ultra · FP8"


def clean(text) -> str:
    """Whitespace runs to one space, U+FFFD (a character the PDF conversion could not read) removed."""
    return " ".join(str(text).replace("\ufffd", "").split())


def state(
    term, definitions, max_definitions=MAX_DEFINITIONS, max_chars=MAX_CHARS
) -> dict:
    """What basal is shown: the term and its first `max_definitions` definitions, cleaned and cut."""
    defs = [clean(t) for t in list(definitions)[:max_definitions]]
    return {
        "term": clean(term),
        "definitions in this paper": [
            d if len(d) <= max_chars else d[: max_chars - 1] + "…" for d in defs
        ],
    }


def item_line(item: dict) -> str:
    return json.dumps(item, ensure_ascii=False) + "\n"


def corpus_items(store=None, links=None, **shown) -> list[dict]:
    """One item per Extracted Concept of the corpus view, in mention-id order, split "corpus" (SHOWN: `state`'s max_definitions and max_chars)."""
    from .corpus import load_view

    view = load_view(store, links)
    return [
        dict(
            id=c.mention_id,
            split="corpus",
            state=state(c.label, [d.text for d in c.definitions], **shown),
        )
        for c in view.concepts
    ]


def groundtruth_items(splits=REFERENCE_SPLITS, **shown) -> list[dict]:
    """One item per ground-truth record of `splits`, in the ground truth's order, with the record's split."""
    return [
        dict(id=r.id, split=r.split, state=state(r.term, r.definitions, **shown))
        for r in groundtruth.load(splits=splits)
    ]


def write_items(items: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(map(item_line, items)), encoding="utf-8")
    return path


def items_command(
    store, links, out: Path, max_definitions=MAX_DEFINITIONS, max_chars=MAX_CHARS
) -> int:
    """`postprocess basal items`: write OUT/items.jsonl (the corpus) and OUT/groundtruth-items.jsonl (train + dev)."""
    from .corpus import default_links, default_store

    shown = dict(max_definitions=max_definitions, max_chars=max_chars)
    for what, items, name in (
        ("corpus", corpus_items(store, links, **shown), ITEMS),
        ("ground-truth train + dev", groundtruth_items(**shown), GROUNDTRUTH_ITEMS),
    ):
        path = write_items(items, Path(out) / name)
        print(f"{len(items)} {what} items -> {path}")
    inputs = {
        "store": Path(store or default_store()),
        "links": Path(links or default_links()),
        **{f"ground_truth:{path.name}": path for path, _ in groundtruth.paths()},
    }
    if groundtruth.VOTES.exists():
        inputs["votes"] = groundtruth.VOTES
    receipt = files.receipt(
        "mbdef postprocess basal items",
        shown,
        inputs,
        {name: Path(out) / name for name in (ITEMS, GROUNDTRUTH_ITEMS)},
    )
    files.write_json(Path(out) / files.RECEIPT, receipt)
    return 0


def probabilities(answer) -> dict:
    """The answer's probabilities, or ValueError: exactly the three keys, each finite and nonnegative, summing to 1 within 0.001."""
    p = answer["probabilities"]
    if set(p) != set(KEYS):
        raise ValueError(f"keys {sorted(p)}")
    p = {k: float(p[k]) for k in KEYS}
    if (
        not all(math.isfinite(v) and v >= 0 for v in p.values())
        or abs(sum(p.values()) - 1) > 1e-3
    ):
        raise ValueError(f"probabilities {p}")
    return p


def ask_one(url: str, item: dict) -> dict:
    """One item's answer line: {id, p, choice, model}, or {id, error} after ATTEMPTS failed requests or for an answer that fails `probabilities`."""
    body = json.dumps(
        {
            "model": REQUEST_MODEL,
            "state": item["state"],
            "questions": {"kind": QUESTION},
        }
    ).encode()
    error = ""
    for attempt in range(ATTEMPTS):
        try:
            request = urllib.request.Request(
                url, body, {"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as r:
                reply = json.load(r)
            answer = reply["answers"]["kind"]
            break
        except Exception as e:  # noqa: BLE001  network or server error: retried, then recorded
            error = f"{type(e).__name__}: {e}"[:300]
            time.sleep(2 * (attempt + 1))
    else:
        return {"id": item["id"], "error": error}
    try:  # a malformed answer is not retried: the server did answer
        return {
            "id": item["id"],
            "p": probabilities(answer),
            "choice": answer["choice"],
            "model": reply.get("model"),
        }
    except (KeyError, TypeError, ValueError) as e:
        return {"id": item["id"], "error": f"bad answer: {type(e).__name__}: {e}"[:300]}


def read_items(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines()]


def ask_command(
    items_path: Path,
    out: Path,
    server: str = SERVER,
    concurrency: int = CONCURRENCY,
    order_seed: int = ORDER_SEED,
    repeat: int = REPEAT,
    limit: int = 0,
) -> int:
    """`postprocess basal ask`: append one answer line per item to OUT as the answers come back. Items that already have an answer in OUT are skipped, so a rerun asks only the rest; an error line is asked again."""
    items = read_items(items_path)
    items = items[:limit] if limit else items
    random.Random(order_seed).shuffle(items)
    out = Path(out)
    done = set()
    if out.exists():
        done = {
            r["id"] for r in map(json.loads, out.read_text().splitlines()) if "p" in r
        }
    todo = [x for x in items if x["id"] not in done]
    print(f"{len(todo)} items to ask ({len(done)} already answered)", flush=True)
    url, t0, errors = server.rstrip("/") + ENDPOINT, time.time(), 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f, ThreadPoolExecutor(concurrency) as pool:
        futures = [pool.submit(ask_one, url, x) for x in todo]
        for n, future in enumerate(as_completed(futures), 1):
            r = future.result() | {"repeat": repeat}
            errors += "error" in r
            f.write(json.dumps(r) + "\n")
            f.flush()
            if n % 200 == 0 or n == len(todo):
                rate = n / (time.time() - t0)
                print(
                    f"{n}/{len(todo)}  {rate:.1f} items/s  {errors} errors", flush=True
                )
    if errors:
        raise SystemExit(
            f"{errors} item(s) without an answer; run the same command again to ask them"
        )
    return 0


def read_answers(path: Path) -> tuple[dict[str, dict], int]:
    """{item id: probabilities} of the first answer of each item (two servers may answer the same item), and the number of error lines skipped."""
    out, errors = {}, 0
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        if "p" not in r:
            errors += 1
            continue
        try:
            probabilities({"probabilities": r["p"]})
        except ValueError as e:
            raise SystemExit(
                f"{path}: the answer of {r['id']} is malformed: {e}"
            ) from None
        out.setdefault(r["id"], r["p"])
    return out, errors


def method_of(table: Path) -> str:
    """The method column of a score table's first row."""
    with Path(table).open(newline="") as f:
        return next(csv.DictReader(f))["method"]


def table_command(
    items_path: Path, answers_path: Path, out: Path, method: str = METHOD
) -> int:
    """`postprocess basal table`: the score table of the items, P(core) + P(satellite) of each item's answer."""
    from .evaluate import ScoreTable

    items = read_items(items_path)
    answers, errors = read_answers(answers_path)
    missing = [x["id"] for x in items if x["id"] not in answers]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(items)} items have no answer in {answers_path}, e.g. "
            f"{missing[0]}; run `postprocess basal ask` again"
        )
    table = ScoreTable(
        method,
        "term",
        {
            x["id"]: answers[x["id"]]["core"] + answers[x["id"]]["satellite"]
            for x in items
        },
        {x["id"]: x["split"] for x in items},
    )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(out)
    print(
        f"{len(items)} rows ({errors} error lines skipped) -> {out}, sha256 {file_sha256(out)}"
    )
    return 0
