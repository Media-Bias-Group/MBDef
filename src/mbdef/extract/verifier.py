"""The candidate verifier: a second model checks every definition the extractor kept.

Each definition of a paper's union is a **candidate**. The verifier shows the model the window the candidate was found in (its **context**, `records.contexts`), the paper's title and up to 16 candidates of that window at once (a **batch**), and asks for a verdict per candidate: is the quoted text a definition of the term (`definition`), is the term relevant to media bias (`relevance`), and a short reason. A candidate found in two windows is asked about in both.

The released rule (`POLICY`, `keep`): a candidate is kept if one of its verdicts says `definition: yes`; relevance is not used here (the relevance filter is the next stage). A candidate whose verdicts are not all there (its batch failed, or the answer left it out) is rejected. A record whose definitions are all rejected is dropped; the other lists of a kept record stay as they are.

- `VerifyCandidates` is the instruction and `signature_for` its version for one batch's ids; `inputs_for` is all the model sees.
- `make_batches` cuts a context's candidates so that each batch's prompt, counted with the verifier model's own tokenizer, plus the output allowance fits `BATCH_TOKENS`; a window is never cut.
- `strict_verdicts` checks the raw text the server sent (not DSPy's parse) against the strict JSON schema of exactly the batch's ids. `partial_verdicts` keeps the verdicts of an answer that is valid but leaves some ids out; any other bad answer fails the whole batch (`ask`).
- `verify_paper` does one paper and writes its files (`verify.py` has the command and the layout).

Worked example (released TEST run, part w8k, paper 73bfad11…, Qwen verifier): its union has 9 candidates, one per record, in 3 windows; one candidate is in two windows, so the 3 batches ask for 5, 3 and 2 verdicts. All 10 came back. 8 candidates have a `definition: yes` (the other got `no`), so `verified.json` keeps 8 of the 9 records.
"""

import copy
import json
from pathlib import Path
from typing import Literal

from mbdef import files, llm  # llm first: it imports NumPy before DSPy

# isort: split
import dspy
import jsonschema
from pydantic import BaseModel, Field, create_model

from mbdef.extract.extractor import output_schema
from mbdef.extract.records import contexts, merge

MAX_CANDIDATES = 16  # per batch
BATCH_TOKENS = (
    32768  # a batch's prompt plus the output allowance must fit in this many tokens
)
POLICY = "definitions-strict"  # keep a candidate that a verdict calls a definition
MISSING = "missing_verdict"  # the verdict written for a candidate whose verdicts are not all there
JSON_FORMAT = {"indent": 2, "sort_keys": True}  # the released runs' per-paper files


class Verdict(BaseModel):
    definition: Literal["yes", "no", "uncertain"]
    relevance: Literal["yes", "no", "uncertain"]
    reason: str = Field(max_length=240)


# The instruction, field names and descriptions are part of every request: a change here changes every request, and a replay of the released runs stops at its first call.
class VerifyCandidates(dspy.Signature):
    """Review proposed definition statements against their supplied paper source.

    Judge each candidate independently; do not aim for any number of accepted items.
    definition=yes: the quoted text explains what the named scholarly concept means,
    including an explicit characterization or operational definition. It may be a
    source-exact clause, not a standalone sentence. Distinct definitions of one term
    can all be valid. definition=no: it is only an example, study result, application,
    list of mentions, or an assertion that does not explain that concept's meaning;
    also no if it defines a different term. Use uncertain when context is insufficient.

    Interpret relevance broadly: media/news bias and framing, misinformation,
    persuasion, cognitive/audience/social biases, and bias/fairness in data or
    algorithms can qualify. Do not require the literal words 'media bias'. The
    paper's relevance alone does not make every generic technical or social term
    relevant. Mark no only when the concept has no meaningful connection to this
    scope; mark uncertain for borderline concepts instead of inventing a connection.

    Use only the source and title, not outside knowledge. The source may quote other
    authors' definitions. Do not reject a valid definition just because it is quoted,
    negative, disputed, or its term has another definition. Do not infer invalidity
    from possible absence in a reference dataset: no reference answers are supplied.
    Give separate definition and relevance labels and one short source-based reason
    for every supplied ID. Never rewrite, add or silently combine candidates.
    """

    title: str = dspy.InputField()
    source_text: str = dspy.InputField(
        desc="Exact original extraction window; not retrieved using reference labels."
    )
    candidates: str = dspy.InputField(
        desc="JSON array of candidate IDs, terms and source quotations."
    )
    verdicts: dict[str, Verdict] = dspy.OutputField()


def signature_for(candidates: list[dict]) -> type[dspy.Signature]:
    """The signature of one batch: one required verdict per candidate id, and no other."""
    ids = [c["id"] for c in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate candidate IDs")
    batch: type[BaseModel] = create_model(  # ty: ignore[no-matching-overload]
        "CandidateVerdicts",
        **{cid: (Verdict, ...) for cid in ids},
    )
    return VerifyCandidates.with_updated_fields("verdicts", type_=batch)


def inputs_for(batch: dict) -> dict:
    """What the model sees: the title, the window and the candidates (never other records)."""
    return dict(
        title=batch["title"],
        source_text=batch["source_text"],
        candidates=json.dumps(batch["candidates"], ensure_ascii=False),
    )


def answer(response: dict | None):
    """The JSON of a finished answer, refusing duplicate keys and NaN or infinite numbers."""

    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate JSON key: " + key)
            value[key] = item
        return value

    def refuse(constant):
        raise ValueError(constant)

    choices = (response or {}).get("choices", [])
    if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
        raise ValueError("Missing, multiple, or unfinished provider response")
    return json.loads(
        choices[0]["message"]["content"],
        object_pairs_hook=unique,
        parse_constant=refuse,
    )


def strict_verdicts(response: dict | None, signature) -> dict:
    """The verdicts of a raw answer that is valid under the batch's strict schema."""
    value = answer(response)
    jsonschema.validate(value, output_schema(signature))
    return value["verdicts"]


def partial_verdicts(response: dict | None, ids: list[str]) -> tuple[dict, list[str]]:
    """The verdicts of a valid answer that leaves some ids out, and the ids it left out.

    The answer must be a finished `{"verdicts": {...}}` with no unknown id and at least one verdict, and every verdict it has must be valid; anything else fails the whole batch.
    """
    value = answer(response)
    if not (
        isinstance(value, dict)
        and set(value) == {"verdicts"}
        and isinstance(value["verdicts"], dict)
    ):
        raise ValueError("Verifier response is not a verdict object")
    present = [cid for cid in ids if cid in value["verdicts"]]
    if not set(value["verdicts"]) <= set(ids):
        raise ValueError("Unknown candidate ID in verifier response")
    if not present:
        raise ValueError("Verifier response contains no verdict")
    verdicts = {cid: value["verdicts"][cid] for cid in present}
    schema = output_schema(signature_for([{"id": cid} for cid in present]))
    jsonschema.validate({"verdicts": verdicts}, schema)
    return verdicts, [cid for cid in ids if cid not in present]


def make_batches(context: dict, count_tokens, adapter, max_tokens: int) -> list[dict]:
    """A context's candidates in batches of at most `MAX_CANDIDATES`, each fitting `BATCH_TOKENS`.

    `count_tokens(messages)` counts the prompt of a batch. A batch that does not fit loses its last candidate until it fits; a window with a single candidate that does not fit raises ValueError.
    """
    batches = []
    remaining = list(context["candidates"])
    while remaining:
        count = min(MAX_CANDIDATES, len(remaining))
        while True:
            batch = {**context, "candidates": remaining[:count]}
            signature = signature_for(batch["candidates"])
            tokens = count_tokens(adapter.format(signature, [], inputs_for(batch)))
            if tokens + max_tokens <= BATCH_TOKENS:
                break
            if count == 1:
                raise ValueError(
                    "Source plus one candidate exceeds context; no source truncation allowed"
                )
            count -= 1
        batches.append({**batch, "estimated_prompt_tokens": tokens})
        remaining = remaining[count:]
    return batches


def keep(union: dict, decisions: dict[str, list[dict]]) -> list[dict]:
    """The union's records with only the candidates that a verdict calls a definition (`POLICY`)."""
    kept: dict[int, set[int]] = {}
    for candidate in union["candidates"]:
        verdicts = decisions.get(candidate["id"])
        if not verdicts:
            raise ValueError("Unreviewed candidate: " + candidate["id"])
        if any(v["definition"] == "yes" for v in verdicts):
            kept.setdefault(candidate["record_index"], set()).add(
                candidate["definition_index"]
            )
    records = []
    for index, record in enumerate(copy.deepcopy(union["records"])):
        if index in kept:
            record["definitions"] = [
                d for i, d in enumerate(record["definitions"]) if i in kept[index]
            ]
            records.append(record)
    return records


def windows_of(paper_dir: Path) -> list[dict]:
    """A paper's saved windows in reading order: by program (branch), then by window."""
    paths = sorted(
        (paper_dir / "extraction").glob("*.json"),
        key=lambda p: (int(p.name.split("-")[0]), p.name),
    )
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths]


def ask(
    lm, adapter, batch: dict
) -> tuple[dict | None, str | None, dict, list[str], str]:
    """One batch's model call: (response, error, verdicts, missing ids, status)."""
    ids = [c["id"] for c in batch["candidates"]]
    signature = signature_for(batch["candidates"])
    llm.LAST_RESPONSE.value = None
    try:
        with dspy.context(lm=lm, adapter=adapter):
            dspy.Predict(signature)(**inputs_for(batch))
        response = llm.LAST_RESPONSE.value
        return response, None, strict_verdicts(response, signature), [], "complete"
    except Exception as error:
        if llm.stops_the_run(error):
            raise  # a failed or unrecorded call stops the run
        # Anything else is an answer that is not a valid verdict object.
        response = llm.LAST_RESPONSE.value
        error = llm.unwrapped(error)
        message = f"{type(error).__name__}: {error}"
        try:
            verdicts, missing = partial_verdicts(response, ids)
            return response, message, verdicts, missing, "partial_missing_verdicts"
        except Exception as partial:
            message += f" | partial parse: {type(partial).__name__}: {partial}"
            return response, message, {}, list(ids), "failed"


def verify_paper(
    paper_dir: Path, out_dir: Path, lm, adapter, count_tokens, max_tokens: int
) -> dict:
    """Verify one paper's union (module docstring); write its files and return its summary."""
    union = json.loads((paper_dir / "union.json").read_text(encoding="utf-8"))
    windows = windows_of(paper_dir)
    if merge([r for w in windows for r in w["definition_records"]]) != union["records"]:
        raise SystemExit(f"{paper_dir}: union.json does not hold its windows' records.")
    paper_id = union["paper_id"]
    files.write_json(out_dir / "union.json", union, **JSON_FORMAT)
    batches, unbatchable = [], []
    for context in contexts(union, windows):
        try:
            for batch in make_batches(context, count_tokens, adapter, max_tokens):
                batches.append({**batch, "batch_id": len(batches)})
        except ValueError as error:
            unbatchable.append(
                {
                    "source_sha256": context["sha256"],
                    "ids": [c["id"] for c in context["candidates"]],
                    "message": str(error),
                }
            )
    expected: dict[str, list[str]] = {}
    received: dict[str, list[dict]] = {}
    counts = dict.fromkeys(
        ("batches", "failed_batches", "missing_verdicts", "failed_slots"), 0
    )
    for item in unbatchable:
        counts["failed_batches"] += 1
        for cid in item["ids"]:
            expected.setdefault(cid, []).append(item["source_sha256"])
    for batch in batches:
        ids = [c["id"] for c in batch["candidates"]]
        for cid in ids:
            expected.setdefault(cid, []).append(batch["sha256"])
        response, error, verdicts, missing, status = ask(lm, adapter, batch)
        counts["batches"] += 1
        counts["failed_batches"] += status == "failed"
        if status == "partial_missing_verdicts":
            counts["missing_verdicts"] += len(missing)
        if status == "failed":
            counts["failed_slots"] += len(ids)
        files.write_json(
            out_dir / "verification" / f"{batch['batch_id']:05d}.json",
            {
                "batch_id": batch["batch_id"],
                "source_sha256": batch["sha256"],
                "ids": ids,
                "status": status,
                "error": error,
                "response": response,
                "verdicts": verdicts,
                "missing_ids": missing,
                "estimated_prompt_tokens": batch["estimated_prompt_tokens"],
            },
            **JSON_FORMAT,
        )
        for cid, verdict in verdicts.items():
            received.setdefault(cid, []).append(
                {**verdict, "source_sha256": batch["sha256"]}
            )
    decisions, lacking = {}, []
    for candidate in union["candidates"]:
        cid = candidate["id"]
        got = received.get(cid, [])
        if not got or len(got) != len(expected.get(cid, [])):
            lacking.append(cid)
            decisions[cid] = [
                {
                    "definition": MISSING,
                    "relevance": MISSING,
                    "reason": "expected verdict missing; candidate rejected",
                }
            ]
        else:
            decisions[cid] = got
    verified = keep(union, decisions)
    files.write_json(
        out_dir / "decisions.json",
        {
            "paper_id": paper_id,
            "received": received,
            "expected_contexts": expected,
            "rejected_for_missing_verdict": lacking,
            "unbatchable_contexts": unbatchable,
        },
        **JSON_FORMAT,
    )
    files.write_json(
        out_dir / "verified.json",
        {"paper_id": paper_id, "definition_records": verified},
        **JSON_FORMAT,
    )
    summary = {
        "paper_id": paper_id,
        "successful_windows": len(windows),
        "candidates": len(union["candidates"]),
        "batches": counts["batches"],
        "unbatchable_contexts": len(unbatchable),
        "failed_batches": counts["failed_batches"],
        "missing_verdicts": counts["missing_verdicts"],
        "verdict_slots_in_failed_batches": counts["failed_slots"],
        "candidates_rejected_for_missing_verdict": len(lacking),
        "of_which_had_another_yes_verdict": sum(
            any(v["definition"] == "yes" for v in received.get(cid, []))
            for cid in lacking
        ),
    }
    files.write_json(out_dir / "paper-summary.json", summary, **JSON_FORMAT)
    return {
        **summary,
        "kept_records": len(verified),
        "kept_definitions": sum(len(r["definitions"]) for r in verified),
    }
