"""The DSPy extractor: what the model is asked, and the saved programs that ask it.

- `ExtractDefinitionRecords` is the signature: a paper's title and one window in, a list of definition records out. Its docstring is the base instruction; a saved program replaces it with its trained instruction and adds one worked example (a demo).
- `DefinitionExtractor` runs the signature on one window (with chain of thought) and applies `sanitize`; `PaperExtractor` runs it on every window of a paper and merges the windows' records.
  A model answer that cannot be parsed becomes an empty result with an `output_error`, never a repair.
- `load_program` loads a program from a folder of `data/extractors/` and refuses it unless its `program.json` has the sha256 its `manifest.json` records.
- `ADAPTERS` are the two output formats of the released runs: `chat` (DSPy's field markers, falling back to JSON once if the answer cannot be parsed or the call failed outside the server, as the released runs did: `ChatAdapter`) and `json` (a JSON object).
- `ControlledJSONAdapter` formats like the JSON adapter and, when `constrained`, also asks the server to decode against the output's JSON schema (`output_schema`); the candidate verifier uses it.
"""

import contextvars
import copy
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from mbdef import llm  # first: it imports NumPy before DSPy (see llm.py)

# isort: split
import dspy
from dspy.adapters.base import Adapter
from dspy.adapters.json_adapter import JSONAdapter
from dspy.utils.exceptions import AdapterParseError
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    create_model,
    field_validator,
    model_validator,
)

from mbdef.extract.papers import windows
from mbdef.extract.records import FIELDS, as_dict, merge, sanitize


class ChatAdapter(dspy.ChatAdapter):
    """DSPy's chat format with the released runs' JSON retry: it asks once more in JSON after an answer that cannot be parsed, and after a call that failed outside the server, such as the released runs' client-side `KeyError: 'temperature'` or its replay (`llm.stops_the_run` is False for it). DSPy before 3.4 did both; DSPy 3.4 asks again only after an unparsable answer.

    Why it skips its parent: `dspy.ChatAdapter.__call__` (3.4) holds the narrower retry rule, so this calls the grandparent `Adapter.__call__` (the formatting and parsing, unchanged) and puts the released rule around it. Live runs follow the same rule, so a replay and a live run of the same requests take the same path (tests/test_extract.py replays a recorded failure). Check this override when DSPy is upgraded."""

    def __call__(self, lm, lm_kwargs, signature, demos, inputs):
        try:
            return Adapter.__call__(self, lm, lm_kwargs, signature, demos, inputs)
        except Exception as error:
            if llm.stops_the_run(error) or not self.use_json_adapter_fallback:
                raise
            fallback = self._make_json_adapter_fallback()
            return fallback(lm, lm_kwargs, signature, demos, inputs)

    async def acall(self, lm, lm_kwargs, signature, demos, inputs):
        try:
            return await Adapter.acall(self, lm, lm_kwargs, signature, demos, inputs)
        except Exception as error:
            if llm.stops_the_run(error) or not self.use_json_adapter_fallback:
                raise
            fallback = self._make_json_adapter_fallback()
            return await fallback.acall(lm, lm_kwargs, signature, demos, inputs)


ADAPTERS = {"chat": ChatAdapter, "json": dspy.JSONAdapter}
RELEVANT_ONLY = (
    "\nFocus on concepts relevant to media bias, "
    "including news framing, misinformation, audience and cognitive biases, "
    "and bias or fairness in algorithms and data. Omit unrelated definitions."
)


# The schema the model is asked to fill (one record: a term and the seven lists of `FIELDS`). Its names and descriptions are part of every prompt, so a change here changes every request, and a replay stops at the first call.
class Evidence(BaseModel):
    text: str = Field(
        default="", description="Exact contiguous wording from the paper."
    )
    citation: list[str] = Field(
        default_factory=list,
        description="Citation markers attached to this statement.",
    )

    @model_validator(mode="before")
    @classmethod
    def accept_string(cls, value):
        return {"text": value, "citation": []} if isinstance(value, str) else value

    @field_validator("citation", mode="before")
    @classmethod
    def citation_list(cls, value):
        if value is None:
            return []
        return [value] if isinstance(value, str) else value


class DefinitionRecord(BaseModel):
    term: str = Field(
        default="", description="Explicitly named reusable scholarly term."
    )
    definitions: list[Evidence] = Field(default_factory=list)
    synonyms: list[Evidence] = Field(default_factory=list)
    description: list[Evidence] = Field(default_factory=list)
    related_terms: list[Evidence] = Field(default_factory=list)
    subtypes: list[Evidence] = Field(default_factory=list)
    supertypes: list[Evidence] = Field(default_factory=list)
    examples: list[Evidence] = Field(default_factory=list)


class ExtractDefinitionRecords(dspy.Signature):
    """Extract source-grounded definition records from one paper window.

    Extract reusable scholarly terms, not paper-local names, datasets, experiment
    groups, variables, or system components. Copy every string exactly and
    contiguously from the supplied text. Never paraphrase or use outside knowledge.
    Prefer the complete definitional sentence, including its subject, over a short
    clause when the complete sentence is present. Do not invent a missing subject.

    A definition states what a term means. Put explanation or qualification in
    description, equivalent terms in synonyms, associated terms in related_terms,
    narrower terms in subtypes, broader terms in supertypes, and concrete cases,
    textual illustrations, or manifestations in examples. Copy attached citation
    markers into citation. Leave unsupported fields empty.

    A window may contribute evidence to a term defined elsewhere in the same paper,
    so a partial record may have no definition. Paper-level aggregation discards any
    term that has no definition anywhere in the paper.
    """

    title: str = dspy.InputField(desc="Paper title.")
    paper_text: str = dspy.InputField(desc="One exact window from the GROBID paper.")
    definition_records: list[DefinitionRecord] = dspy.OutputField()


def output_error(error, title: str, text: str) -> dict:
    """What a model answer that failed to parse looked like (kept, never repaired)."""
    row = {
        "title": title,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "error_type": type(error).__name__,
        "response": getattr(llm.LAST_RESPONSE, "value", None),
    }
    if hasattr(error, "errors"):
        row["validation_errors"] = error.errors(
            include_url=False, include_context=False
        )
    else:
        row["lm_response"] = error.lm_response
    return row


class DefinitionExtractor(dspy.Module):
    """One window in, its sanitized records out (and the model's raw records)."""

    def __init__(self, relevant_only: bool = False):
        super().__init__()
        signature = ExtractDefinitionRecords
        if relevant_only:
            signature = signature.with_instructions(
                signature.instructions + RELEVANT_ONLY
            )
        self.extract_v3 = dspy.ChainOfThought(signature)  # the name saved programs use

    def forward(self, paper_text: str, title: str = ""):
        llm.LAST_RESPONSE.value = None
        try:
            prediction = self.extract_v3(paper_text=paper_text, title=title)
        except (ValidationError, AdapterParseError) as error:
            return dspy.Prediction(
                definition_records=[],
                raw_definition_records=[],
                output_error=output_error(error, title, paper_text),
            )
        raw = [as_dict(item) for item in prediction.definition_records or []]
        return dspy.Prediction(
            definition_records=sanitize(raw, paper_text), raw_definition_records=raw
        )


def run_windows(extractor, chunks: list[str], title: str, workers: int) -> list:
    """`extractor` on every window at the same time; its outputs, and its DSPy trace steps added to the caller's trace, in window order, as the one-by-one loop gives them.

    Each window runs in a copy of the caller's context (its language model and settings) with a trace of its own. The first window, in window order, that failed raises its error.
    """
    caller = contextvars.copy_context()

    def one(text: str):
        with dspy.context(trace=[]):
            output = extractor(paper_text=text, title=title)
            return output, list(dspy.settings.trace)

    with ThreadPoolExecutor(max_workers=min(workers, len(chunks))) as pool:
        futures = [pool.submit(caller.copy().run, one, text) for text in chunks]
        done: list[tuple[Any, list]] = [future.result() for future in futures]
    trace = dspy.settings.trace
    if trace is not None:
        for _output, steps in done:
            for step in steps:
                if len(trace) >= dspy.settings.max_trace_size:
                    trace.pop(0)
                trace.append(step)
    return [output for output, _steps in done]


class PaperExtractor(dspy.Module):
    """A paper's windows in, their records merged; the unit the saved programs were trained as.
    `workers` > 1 reads the windows at the same time (training, `extract train`); the records and
    the trace are those of the one-by-one loop."""

    def __init__(self, relevant_only: bool = False, workers: int = 1):
        super().__init__()
        self.chunk_extractor = DefinitionExtractor(relevant_only=relevant_only)
        self.workers = workers

    def forward(self, chunks, title=""):
        if self.workers > 1 and len(chunks) > 1:
            outputs = run_windows(
                self.chunk_extractor, list(chunks), title, self.workers
            )
        else:
            outputs = [
                self.chunk_extractor(paper_text=text, title=title) for text in chunks
            ]
        clean = [copy.deepcopy(r) for p in outputs for r in p.definition_records]
        raw = [copy.deepcopy(r) for p in outputs for r in p.raw_definition_records]
        errors = [p.output_error for p in outputs if getattr(p, "output_error", None)]
        return dspy.Prediction(
            definition_records=merge(clean),
            raw_definition_records=merge(raw, require_definition=False),
            output_error=errors or None,
        )


def extract(program: PaperExtractor, lm, fmt: str, source: str, title: str):
    """Run a program's window extractor on one window with `lm` and the output format `fmt`."""
    with dspy.context(lm=lm, adapter=ADAPTERS[fmt]()):
        return program.chunk_extractor(paper_text=source, title=title)


def paper_examples(
    papers: list[dict], size: int, overlap: int, evaluated_fields=FIELDS
):
    """One training example per paper (its windows in, its ground-truth records out)."""
    return [
        dspy.Example(
            paper_id=p["paper_id"],
            title=p["title"],
            split=p["split"],
            paper_text=p["text"],
            chunks=windows(p["text"], size, overlap),
            gold_records=p["gold"],
            definition_records=p["gold"],
            evaluated_fields=evaluated_fields,
            positive=bool(p["gold"]),
            index=0,
        ).with_inputs("chunks", "title")
        for p in papers
    ]


def not_saved_with_other_versions(record: logging.LogRecord) -> bool:
    """False for DSPy's warning that a program was saved with other library versions (a logging filter, `load_program`)."""
    return not record.getMessage().startswith("There is a mismatch of ")


def load_program(folder: Path) -> tuple[PaperExtractor, dict]:
    """The program in `folder` and its manifest, after checking the program's sha256.

    `folder` holds `program.json` (the DSPy program) and `manifest.json` (its training receipt, with `program_sha256`, `window_chars` and `overlap_chars`).
    """
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    program_path = folder / manifest["program"]
    found = hashlib.sha256(program_path.read_bytes()).hexdigest()
    if found != manifest["program_sha256"]:
        raise SystemExit(
            f"{program_path}: its sha256 is {found}, not the {manifest['program_sha256']} "
            f"that {folder / 'manifest.json'} records. Refusing to run a changed program."
        )
    program = PaperExtractor(relevant_only=True)
    # DSPy warns on each load that the program was saved with other Python and DSPy versions (the released programs: 3.13 and 3.3.1). That is expected, and the program file records them, so the warning is left out of the run's output.
    versions = logging.getLogger("dspy.primitives.base_module")
    versions.addFilter(not_saved_with_other_versions)
    try:
        program.load(str(program_path))
    finally:
        versions.removeFilter(not_saved_with_other_versions)
    if program.chunk_extractor.extract_v3.predict.lm is not None:
        raise SystemExit(f"{program_path} has a language model saved in it.")
    return program, manifest


def output_schema(signature) -> dict:
    """The strict JSON schema of a signature's outputs (every property required, no others)."""
    fields: dict[str, Any] = {
        name: (field.annotation, ...) for name, field in signature.output_fields.items()
    }
    schema = create_model("ExtractionResponse", **fields).model_json_schema()

    def strict(node):
        if isinstance(node, dict):
            node.pop("default", None)
            if "properties" in node:
                node["required"] = list(node["properties"])
                node["additionalProperties"] = False
            for value in node.values():
                strict(value)
        elif isinstance(node, list):
            for value in node:
                strict(value)

    strict(schema)
    return schema


class ControlledJSONAdapter(JSONAdapter):
    """JSON formatting and parsing; with `constrained`, the request also carries the JSON schema."""

    def __init__(self, constrained: bool = False):
        super().__init__(use_native_function_calling=False)
        self.constrained = constrained

    def request_kwargs(self, lm_kwargs, signature):
        kwargs = copy.deepcopy(lm_kwargs)
        kwargs.pop("response_format", None)
        if self.constrained:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "extraction",
                    "strict": True,
                    "schema": output_schema(signature),
                },
            }
        return kwargs

    def __call__(self, lm, lm_kwargs, signature, demos, inputs):
        return Adapter.__call__(
            self,
            lm,
            self.request_kwargs(lm_kwargs, signature),
            signature,
            demos,
            inputs,
        )

    async def acall(self, lm, lm_kwargs, signature, demos, inputs):
        return await Adapter.acall(
            self,
            lm,
            self.request_kwargs(lm_kwargs, signature),
            signature,
            demos,
            inputs,
        )
