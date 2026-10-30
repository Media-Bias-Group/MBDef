"""`closure identity`: a language model checks the closure PDFs that the deterministic identity rule could not verify.

`collect grobid` keeps a converted PDF only if its title page shows the cited work (docs/step-1-2-papers/collect.md). Some correct PDFs still fail that rule (for example a garbled title, or a year more than one off). This check asks a model whether each such PDF is the cited work, and lets a verdict count only if a program can confirm it from the PDF's own text. Four actions, in order (only `judge` needs a model server):

- `evidence`: for each PDF the rule left `identity_unverified` (the candidates), the cited reference and the text of the PDF's first two pages; plus controls: PDFs already verified (the model must accept them, `control_same`) and the cited reference paired with a different paper's PDF (the model must reject them, `control_different_hard`: the most similar other title; `control_different_random`).
  A later round passes the earlier evidence (`--earlier`): it keeps only candidates whose PDF bytes (by sha256) were not judged before, and repeats no controls.
- `judge`: the model answers each item as JSON (`SCHEMA`) through an OpenAI-compatible server (vLLM).
  Every request is recorded as a receipt in `requests/`; with `--replay` the answers come from recorded receipts and nothing is sent, and a request that was not recorded stops the run (`NotRecorded`: the guard that makes a changed prompt, setting or evidence fail loudly instead of calling the model).
- `decide`: program checks (`checks`) turn each answer into accept or reject, and the controls give the error rates. A verdict counts only if the model says `same_work` and a cited author surname appears in the PDF text and either the cited title appears in the text, or the title the model read from the PDF appears in the text and is close to the cited title. (On the released rounds' controls, a verbatim-quote requirement only rejected correct PDFs, because the model tidies quotes, so the quote is recorded but not required.)
  Writes `accepted.json`, the input of `collect grobid --identity-verdicts`, and `identity-decisions.json`.
- `merge`: the accepted lists of two rounds in one (a paper judged again takes the later verdict).

The decision rule is the released rounds' rule: on their saved evidence and verdicts, `decide` writes the released files byte for byte (`closure bundle --identity-reference` compares them; `tests/test_closure_identity.py` replays synthetic checks).
"""

import collections
import difflib
import hashlib
import json
import random
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from email.message import Message
from pathlib import Path
from typing import Any

from mbdef import files, llm, models
from mbdef.collect import download, grobid, ledger
from mbdef.collect.identity import receipt_path

# Pinned defaults: the model and prompt of the released closure's identity check (2 H100 GPUs via vLLM).
PROMPT_VERSION = "identity-llm/1"
MODEL, REVISION = models.PINNED["qwen"]
BASE_URL = "http://127.0.0.1:8123/v1"
WORKERS = 16
TEXT_CHARS = 3500  # of the PDF's first two pages
CONTROLS = 150  # pairs of each control kind's pool, as the released rounds drew them
SEED = 20260929
MAX_TOKENS = 500
ATTEMPTS = 3
TIMEOUT_SECONDS = 300
# The pdftotext call: pages 1 to 2, UTF-8, at most this long.
PDFTOTEXT_SECONDS = 60
VERIFIED_BY = (
    "doi",
    "title_year_author",
)  # identity rules that make a PDF a positive control

EVIDENCE = "evidence.json"
EVIDENCE_FULL = "evidence-full.json"  # every item of a later round before the earlier ones are removed
VERDICTS = "verdicts.json"
ACCEPTED = "accepted.json"
DECISIONS = "identity-decisions.json"
REQUESTS = "requests"  # the receipts of `judge`

SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["same_work", "different_work", "cannot_tell"],
        },
        "pdf_title": {"type": "string"},
        "pdf_authors": {"type": "array", "items": {"type": "string"}},
        "pdf_year": {"type": ["integer", "null"]},
        "evidence": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": [
        "verdict",
        "pdf_title",
        "pdf_authors",
        "pdf_year",
        "evidence",
        "reasoning",
    ],
    "additionalProperties": False,
}
SYSTEM = (
    "You check whether a PDF is a specific cited publication. You are strict: a wrong match would put the wrong paper "
    "into a research corpus. Answer with JSON only."
)


def lower_ascii_words(text: str) -> str:
    """Lower-case runs of ASCII letters and digits, anything else a space. The released identity check's program checks (evidence in the text, title closeness) were made with this rule:
    `lower()`, not `casefold()` as in `groundtruth.ascii_words` ("Straße" gives "stra e" here, "strasse" there), so its released decisions are reproduced only with it."""
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def surnames(authors: str) -> set[str]:
    out = set()
    for a in re.split(r"[;|]", authors or ""):
        a = a.strip()
        if not a:
            continue
        s = a.split(",")[0] if "," in a else a.split()[-1]
        s = lower_ascii_words(s)
        if len(s) >= 3:
            out.add(s)
    return out


def user_prompt(ref: dict, text: str) -> str:
    return (
        "Cited reference (as recorded in the citing paper):\n"
        f"  title: {ref['title']}\n  authors: {ref['authors']}\n  year: {ref['year']}\n"
        + (f"  doi: {ref['doi']}\n" if ref.get("doi") else "")
        + "\nText extracted from the first two pages of the PDF (may be noisy, columns may be interleaved):\n<<<\n"
        + text
        + "\n>>>\n\nIs this PDF the cited work itself? Any version of the same work counts (preprint, accepted manuscript, "
        "journal or conference version, translation). It is NOT a match if it is a different paper, a proceedings "
        "or issue front matter, a review or commentary about the work, or a document that only cites it. "
        "The year may differ between versions. Use 'cannot_tell' when the text does not show the title.\n"
        "Return JSON with: verdict (same_work | different_work | cannot_tell); pdf_title (the title exactly as printed "
        "in the PDF text, or empty); pdf_authors (author names as printed); pdf_year (integer or null); evidence (a short "
        "passage copied verbatim from the PDF text that supports your verdict, usually the title line); reasoning (one or "
        "two sentences)."
    )


# --------------------------------------------------------------------------------------- evidence


def pdf_text(archive: str, member: str) -> tuple[str, str]:
    """The first two pages' text of a PDF in a zip (at most `TEXT_CHARS`), and the PDF's sha256."""
    with zipfile.ZipFile(archive) as z:
        data = z.read(member)
    with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
        f.write(data)
        f.flush()
        out = subprocess.run(
            ["pdftotext", "-f", "1", "-l", "2", "-enc", "UTF-8", f.name, "-"],
            capture_output=True,
            timeout=PDFTOTEXT_SECONDS,
        )
    text = " ".join(out.stdout.decode("utf-8", "replace").replace("�", "").split())
    return text[:TEXT_CHARS], hashlib.sha256(data).hexdigest()


def evidence(
    folder: Path, controls: int, seed: int, earlier: list[Path] | None = None
) -> tuple[dict, dict | None]:
    """The evidence of a round from the collect step's folder: (evidence, the unfiltered evidence).

    The second is None in the first round. With `earlier` evidence files, the first is that round's candidates not judged before (by the PDF's sha256), without controls.
    """
    snap = download.latest_status(folder / download.STATUS)
    latest = grobid.load_converted(folder / grobid.STATUS)
    rng = random.Random(seed)
    cand = sorted(
        p
        for p, r in latest.items()
        if "identity_unverified" in r["error"]
        and p in snap
        and ledger.truthy(snap[p]["downloaded"])
    )
    verified = []
    for p, r in latest.items():
        if r["error"] or p not in snap:
            continue
        receipt_file = receipt_path(Path(r["grobid_xml_path"]))
        if not receipt_file.is_file():
            receipt_file = Path(
                str(r["grobid_xml_path"])[: -len(".xml")] + ".closure.json"
            )
        if receipt_file.is_file():
            if (
                json.loads(receipt_file.read_text()).get("identity_match")
                in VERIFIED_BY
            ):
                verified.append(p)
    verified.sort()
    pos = rng.sample(verified, min(controls, len(verified)))
    neg_pool = [p for p in verified if p not in pos]
    items: list[dict] = []
    cache: dict[str, tuple[str, str]] = {}

    def text_of(p):
        if p not in cache:
            r = snap[p]
            cache[p] = pdf_text(r["pdf_archive_path"], r["pdf_archive_member"])
        return cache[p]

    def ref(p):
        r = snap[p]
        return {
            "title": r["title"],
            "authors": r["authors"],
            "year": r["year"],
            "doi": r["doi"],
        }

    for p in cand:
        text, sha = text_of(p)
        items.append(
            {
                "id": f"cand:{p}",
                "kind": "candidate",
                "paperid": p,
                "pdf_paperid": p,
                "ref": ref(p),
                "text": text,
                "pdf_sha256": sha,
            }
        )
    for p in pos:
        text, sha = text_of(p)
        items.append(
            {
                "id": f"pos:{p}",
                "kind": "control_same",
                "paperid": p,
                "pdf_paperid": p,
                "ref": ref(p),
                "text": text,
                "pdf_sha256": sha,
            }
        )
    titles = {p: lower_ascii_words(snap[p]["title"]) for p in neg_pool}
    hard = rng.sample(pos, min(controls * 2 // 3, len(pos)))
    for p in pos:
        t = lower_ascii_words(snap[p]["title"])
        if p in hard:
            # the most similar other paper (different title), so the pair looks plausible
            pool = [q for q in neg_pool if titles[q] != t]
            if (
                not pool
            ):  # a folder with no other verified paper has no negative control to draw
                continue
            w = set(t.split())
            best = max(
                pool,
                # most word overlap; the same year breaks ties
                key=lambda q: (
                    len(w & set(titles[q].split()))
                    / max(1, len(w | set(titles[q].split()))),
                    snap[q]["year"] == snap[p]["year"],
                ),
            )
            kind = "control_different_hard"
        else:
            pool = [q for q in neg_pool if q != p]
            if not pool:
                continue
            best, kind = rng.choice(pool), "control_different_random"
        if lower_ascii_words(snap[best]["title"]) == t:
            continue
        text, sha = text_of(best)
        items.append(
            {
                "id": f"neg:{p}:{best}",
                "kind": kind,
                "paperid": p,
                "pdf_paperid": best,
                "ref": ref(p),
                "text": text,
                "pdf_sha256": sha,
            }
        )
    full = {
        "schema": "identity-evidence/1",
        "prompt_version": PROMPT_VERSION,
        "items": items,
    }
    if not earlier:
        return full, None
    seen = {
        i["pdf_sha256"]
        for path in earlier
        for i in json.loads(Path(path).read_text())["items"]
        if i["kind"] == "candidate"
    }
    new = [i for i in items if i["kind"] == "candidate" and i["pdf_sha256"] not in seen]
    return {
        **full,
        "items": new,
        "round": len(earlier) + 1,
        "controls": "reused from round 1",
    }, full


# ----------------------------------------------------------------------------------------- judge


def settings(schema: bool) -> dict:
    """The request settings that identify a request (besides the model and the messages)."""
    found: dict = {
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    if schema:
        found["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "identity", "schema": SCHEMA, "strict": True},
        }
    return found


def post(
    base: str,
    model: str,
    messages: list[dict],
    schema: bool,
    replay: dict[str, dict] | None = None,
    requests: Path | None = None,
) -> str:
    """One chat completion's answer text: from the recorded receipt (`replay`), or from the server at `base` (then its receipt goes to `requests`). A replayed request that was not recorded raises `llm.NotRecorded`; one that failed when recorded fails again."""
    sent = settings(schema)
    key = llm.request_key(model, messages, sent)
    receipt = {
        "schema": llm.RECEIPT_SCHEMA,
        "request_sha256": key,
        "formatted_request": {"model": model, "messages": messages, **sent},
        "boundary": "closure identity: the chat-completions request body",
    }
    if replay is not None:
        if key not in replay:
            raise llm.NotRecorded(f"request_sha256 {key} was not recorded", model=model)
        recording = replay[key]
        if recording.get("response") is None:
            error = recording["error"]
            if error["type"] == "HTTPError":
                raise urllib.error.HTTPError(
                    base, int(error["message"]), "recorded", Message(), None
                )
            raise llm.RecordedFailure(f"{error['type']}: {error['message']}")
        return recording["response"]["choices"][0]["message"]["content"]
    # The server's own body: the request settings besides `extra_body`, whose keys sit beside them.
    body = {
        "model": model,
        "messages": messages,
        **{k: v for k, v in sent.items() if k != "extra_body"},
        **sent["extra_body"],
    }
    req = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as handle:
            answer = json.loads(handle.read())
    except urllib.error.HTTPError as error:
        if requests:
            llm.write_new(
                requests / f"{key}.json",
                receipt
                | {
                    "response": None,
                    "error": {"type": "HTTPError", "message": str(error.code)},
                },
            )
        raise
    if requests:
        llm.write_new(
            requests / f"{key}.json",
            receipt | {"response": {"choices": answer["choices"]}, "error": None},
        )
    return answer["choices"][0]["message"]["content"]


def parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def judge(
    evidence_path: Path,
    base: str,
    model: str,
    revision: str,
    workers: int,
    replay: dict[str, dict] | None = None,
    requests: Path | None = None,
) -> dict:
    """The verdicts file's content: the model's answer to every item of the evidence.

    Requests ask for the answer's JSON schema until the server refuses it once (HTTP 400); from then on every request of every worker asks for plain JSON, as the released run did. So the file records, besides the mode at the end (`schema_mode`), the mode each item's answer was asked with (`schema_mode_by_item`)."""
    doc = json.loads(evidence_path.read_text())
    use_schema = True
    asked_with: dict[
        str, bool
    ] = {}  # item id -> whether its last request had the schema

    def one(item):
        nonlocal use_schema
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt(item["ref"], item["text"])},
        ]
        out = ""
        for _ in range(ATTEMPTS):
            schema = asked_with[item["id"]] = use_schema
            try:
                out = post(base, model, msgs, schema, replay, requests)
            except llm.NotRecorded:
                raise  # the guard: a request that was not recorded stops the run
            except urllib.error.HTTPError as e:
                if e.code == 400 and use_schema:
                    use_schema = False  # this server build rejects json_schema: fall back to plain JSON
                    continue
                out = f"ERROR {e.code}"
            except Exception as e:  # noqa: BLE001
                out = f"ERROR {e}"
            v = parse_json(out)
            if v and v.get("verdict") in ("same_work", "different_work", "cannot_tell"):
                return {"id": item["id"], "raw": out, "answer": v}
            time.sleep(0 if replay is not None else 1)
        return {"id": item["id"], "raw": out, "answer": None}

    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(one, doc["items"]))
    return {
        "model": model,
        "revision": revision,
        "prompt_version": PROMPT_VERSION,
        "schema_mode": use_schema,
        "schema_mode_by_item": {
            item["id"]: asked_with[item["id"]] for item in doc["items"]
        },
        "system_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
        "verdicts": results,
    }


# ---------------------------------------------------------------------------------------- decide


def checks(item: dict, ans: dict | None) -> dict:
    """Program checks that turn a model answer into an accept/reject decision, with the reasons."""
    out = {"same_work": bool(ans) and ans.get("verdict") == "same_work"}
    text_n = " " + lower_ascii_words(item["text"]) + " "
    ev = lower_ascii_words((ans or {}).get("evidence", ""))
    out["evidence_in_text"] = len(ev) >= 15 and (" " + ev + " ") in text_n
    pt, ct = (
        lower_ascii_words((ans or {}).get("pdf_title", "")),
        lower_ascii_words(item["ref"]["title"]),
    )
    ratio = difflib.SequenceMatcher(None, pt, ct).ratio() if pt and ct else 0.0
    out["title_ratio"] = round(ratio, 3)
    out["title_close"] = bool(pt) and (
        ratio >= 0.85 or (len(ct) >= 20 and (ct in pt or pt in ct))
    )
    out["title_in_text"] = bool(ct) and (" " + ct + " ") in text_n
    out["pdf_title_in_text"] = bool(pt) and (" " + pt + " ") in text_n
    out["author_in_text"] = any(
        f" {s} " in text_n for s in surnames(item["ref"]["authors"])
    )
    out["accepted"] = (
        out["same_work"]
        and out["author_in_text"]
        and (out["title_in_text"] or (out["title_close"] and out["pdf_title_in_text"]))
    )
    return out


def decide(evidence_path: Path, verdicts_path: Path) -> tuple[dict, list[dict]]:
    """(the accepted file's content, the decisions): every verdict checked, the candidates accepted."""
    ev = {i["id"]: i for i in json.loads(evidence_path.read_text())["items"]}
    vd = json.loads(verdicts_path.read_text())
    rows = []
    for v in vd["verdicts"]:
        item = ev[v["id"]]
        rows.append(
            {
                "id": v["id"],
                "kind": item["kind"],
                "answer": v["answer"],
                "checks": checks(item, v["answer"]),
            }
        )
    by = collections.defaultdict(list)
    for r in rows:
        by[r["kind"]].append(r)
    summary = {}
    for kind, rs in by.items():
        summary[kind] = {
            "n": len(rs),
            "model_same_work": sum(r["checks"]["same_work"] for r in rs),
            "accepted_by_all_checks": sum(r["checks"]["accepted"] for r in rs),
            "unparsed": sum(r["answer"] is None for r in rs),
        }
    accepted = []
    for r in rows:
        if r["kind"] != "candidate" or not r["checks"]["accepted"]:
            continue
        item = ev[r["id"]]
        accepted.append(
            {
                "paperid": item["paperid"],
                "pdf_sha256": item["pdf_sha256"],
                "model": vd["model"],
                "revision": vd["revision"],
                "prompt_version": vd["prompt_version"],
                "verdict": r["answer"]["verdict"],
                "pdf_title": r["answer"]["pdf_title"],
                "evidence": r["answer"]["evidence"],
                "reasoning": r["answer"]["reasoning"],
                "checks": r["checks"],
            }
        )
    return {
        "schema": "identity-accepted/1",
        "summary": summary,
        "accepted": accepted,
    }, rows


def merge(earlier: dict, new: dict, round_number: int) -> dict:
    """The accepted lists of two rounds in one: `earlier` may already hold several rounds' summaries.
    A paper accepted in both keeps its place and takes the later verdict."""
    by = {x["paperid"]: x for x in earlier["accepted"]}
    by.update({x["paperid"]: x for x in new["accepted"]})
    rounds = earlier.get("rounds") or {"1": earlier["summary"]}
    return {
        "schema": "identity-accepted/1",
        "rounds": {**rounds, str(round_number): new["summary"]},
        "accepted": list(by.values()),
    }


# -------------------------------------------------------------------------------------- commands

JSON_FORMAT: dict[str, Any] = {
    "indent": 1,
    "ensure_ascii": False,
}  # the byte format of the released files


def read(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    """The released files' bytes: indent 1, non-ASCII kept, and no final newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, **JSON_FORMAT), encoding="utf-8")


def evidence_command(
    collect_folder: str | None,
    out: str,
    controls: int,
    seed: int,
    earlier: list[str],
) -> int:
    """`closure identity evidence`: write the round's evidence into OUT."""
    folder = ledger.folder(collect_folder)
    out_dir = Path(out)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} is not empty; give a new or empty folder.")
    kept, full = evidence(folder, controls, seed, [Path(p) for p in earlier])
    write(out_dir / EVIDENCE, kept)
    outputs = {"evidence": out_dir / EVIDENCE}
    if full is not None:
        write(out_dir / EVIDENCE_FULL, full)
        outputs["evidence_full"] = out_dir / EVIDENCE_FULL
    kinds = collections.Counter(i["kind"] for i in kept["items"])
    summary = {
        "items": len(kept["items"]),
        "by_kind": dict(kinds),
        "little_or_no_text": sum(1 for i in kept["items"] if len(i["text"]) < 200),
    }
    inputs = {
        "downloads": folder / download.STATUS,
        "converted": folder / grobid.STATUS,
    } | {f"earlier_{n}": Path(p) for n, p in enumerate(earlier, 1)}
    _receipt(
        out_dir,
        "closure identity evidence",
        {
            "controls": controls,
            "seed": seed,
            "text_chars": TEXT_CHARS,
            "prompt_version": PROMPT_VERSION,
        },
        inputs,
        outputs,
        summary,
    )
    print(json.dumps(summary))
    return 0


def judge_command(
    evidence_path: str,
    out: str,
    base: str,
    model: str,
    revision: str,
    workers: int,
    replay: list[str],
) -> int:
    """`closure identity judge`: write the model's verdicts on EVIDENCE into OUT, with request receipts."""
    out_dir = Path(out)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} is not empty; give a new or empty folder.")
    recorded = llm.recorded([Path(p) for p in replay]) if replay else None
    started = time.time()
    try:
        verdicts = judge(
            Path(evidence_path),
            base,
            model,
            revision,
            workers,
            recorded,
            None if replay else out_dir / REQUESTS,
        )
    except llm.NotRecorded as error:
        raise SystemExit(
            f"{error}. Record it with a live run, or fix the evidence."
        ) from None
    write(out_dir / VERDICTS, verdicts)
    unparsed = sum(1 for v in verdicts["verdicts"] if v["answer"] is None)
    summary = {"judged": len(verdicts["verdicts"]), "unparsed": unparsed}
    outputs = {"verdicts": out_dir / VERDICTS}
    if not replay and (out_dir / REQUESTS).exists():
        outputs["requests"] = out_dir / REQUESTS
    _receipt(
        out_dir,
        "closure identity judge",
        {
            "model": model,
            "revision": revision,
            "prompt_version": PROMPT_VERSION,
            "base_url": None if replay else base,
            "replay": replay,
            "workers": workers,
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
            "schema_mode": verdicts["schema_mode"],
        },
        {"evidence": Path(evidence_path)},
        outputs,
        summary,
    )
    print(
        f"judged {summary['judged']} in {time.time() - started:.0f}s; unparsed {unparsed}"
    )
    return 0


def decide_command(evidence_path: str, verdicts_path: str, out: str) -> int:
    """`closure identity decide`: accepted.json and identity-decisions.json into OUT."""
    out_dir = Path(out)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} is not empty; give a new or empty folder.")
    accepted, rows = decide(Path(evidence_path), Path(verdicts_path))
    write(out_dir / ACCEPTED, accepted)
    write(out_dir / DECISIONS, rows)
    _receipt(
        out_dir,
        "closure identity decide",
        {"prompt_version": PROMPT_VERSION},
        {"evidence": Path(evidence_path), "verdicts": Path(verdicts_path)},
        {"accepted": out_dir / ACCEPTED, "decisions": out_dir / DECISIONS},
        {
            "summary": accepted["summary"],
            "accepted_candidates": len(accepted["accepted"]),
        },
    )
    print(json.dumps(accepted["summary"], indent=1))
    print("accepted candidates:", len(accepted["accepted"]))
    return 0


def merge_command(earlier: str, new: str, round_number: int, out: str) -> int:
    """`closure identity merge`: the accepted lists of EARLIER (all earlier rounds) and NEW in OUT."""
    out_file = Path(out)
    merged = merge(read(earlier), read(new), round_number)
    write(out_file, merged)
    print(
        f"accepted verdicts, all rounds: {len(merged['accepted'])} "
        f"(before: {len(read(earlier)['accepted'])}, round {round_number}: {len(read(new)['accepted'])})"
    )
    return 0


def _receipt(
    folder: Path, command: str, settings: dict, inputs, outputs, summary
) -> None:
    receipt = files.receipt(f"mbdef {command}", settings, inputs, outputs)
    receipt["summary"] = summary
    files.write_json(folder / files.RECEIPT, receipt)
