"""The language-model layer: one LM for every model call, its setup, its usage log and recorded replay.

Every step that asks a language model something (extraction, the verifier, training) builds its LM with `make_lm` and calls it through DSPy. The LM is a plain `dspy.LM` whose engine (`Engine`, DSPy 3.4's `complete(Request) -> Response` interface) sends each call through LiteLLM with exactly the request settings the released runs sent, and does three things around it:

- **Usage log.** One JSON line per call in `usage_log`, with the same fields the released runs logged (`USAGE_FIELDS`): token counts, finish reasons, the public request settings and `request_sha256`.
  No prompt text, key or response text goes into it.
- **Request receipts.** Each call's exact request and the full response (or the error of a failed call) go into `requests/<request_sha256>.json` (`RECEIPT_SCHEMA`, the released runs' format).
- **Replay.** Given `replay` (from `recorded(folders)`), the LM answers each call from the recorded receipt of the identical request and never contacts a server; a call that failed when it was recorded fails again in the same way (`replayed_failure`). A call that was not recorded raises `NotRecorded`, which names its `request_sha256`: the guard that makes a changed prompt, program, setting or library fail loudly instead of silently calling a model.

  Worked example: the released TEST run's first 8k window of paper 4243aa9b… was request `4afab7d2…`; a replay answers it with that receipt's response, while the same window under another title is a different request, which stops the replay with "request_sha256 … was not recorded".

A request is identified by `request_key`: the sha256 of the JSON of the model name, the messages and the public settings (`PUBLIC_SETTINGS`), with sorted keys. It is the key the released runs recorded, so their receipts replay unchanged. `make_lm` sends exactly the settings it is given: it adds no temperature, so a model that is given none samples with its own generation config, as the released thinking runs did.

Why an engine of our own: DSPy 3.4 deprecates `forward()` overrides of an LM, and 3.5 removes them. It hands an engine a canonical `dspy.lm15.Request`, which cannot carry `extra_body` (vLLM's `chat_template_kwargs`, the thinking switch of every released call), and DSPy's own LiteLLM engine (`engine="litellm"`) takes such a request only through a dictionary shortcut that is deprecated too. So the engine keeps `extra_body` itself and maps the request back to the OpenAI chat form (`openai_request`): on the 28,126 request receipts of the released TEST and training runs this gives every recorded `request_sha256`. It calls LiteLLM, as `engine="litellm"` would, so the usage log keeps LiteLLM's response id and provider.
"""

import os

# Two LiteLLM import-time habits are switched off. It downloads its model price list unless told to use its own copy (the released runs ran on compute nodes without internet, so they used that copy too). And in its default mode it loads the first `.env` it finds above its own install folder, which from a git worktree is the main checkout's: machine settings must come only from this repository's `.env` (mbdef.files).
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
os.environ.setdefault("LITELLM_MODE", "PRODUCTION")

import asyncio  # noqa: E402
import dataclasses  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

# NumPy must be imported before DSPy: DSPy installs a lazy NumPy stand-in, and NumPy then fails to start when something imports it later (for example joblib: "data type 'bool' not understood").
import numpy  # noqa: E402, F401

# isort: split
import dspy  # noqa: E402
import litellm  # noqa: E402
from dspy.clients.engines.litellm_errors import litellm_errors  # noqa: E402
from dspy.lm15 import Request, Response, response_from_openai_chat  # noqa: E402
from dspy.utils.exceptions import LMError, LMUnexpectedError  # noqa: E402
from litellm import ModelResponse  # noqa: E402

# The request settings that identify a request, besides the model and the messages. Transport settings (endpoint, key, timeout) are left out, so a replay does not depend on the machine.
PUBLIC_SETTINGS = (
    "max_tokens",
    "max_completion_tokens",
    "temperature",
    "seed",
    "rollout_id",
    "extra_body",
    "response_format",
)
RECEIPT_SCHEMA = "qwen-two-request-receipt/1"  # the stored schema name of the released runs' receipts
RECEIPT_BOUNDARY = "DSPy LoggedLM input, before LiteLLM HTTP serialization"
# The fields of one usage-log line, in order (as the released runs' usage logs have them).
USAGE_FIELDS = (
    "stage",
    "started",
    "seconds",
    "model",
    "request_sha256",
    "response_id",
    "provider",
    "reported_cost_usd",
    "dspy_cache_hit",
    "prompt_tokens",
    "completion_tokens",
    "cached_prompt_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
    "message_chars",
    "finish_reasons",
    "settings",
)

LOCK = threading.Lock()  # one writer at a time for the usage logs
# The raw response of this thread's last call ({"response_id", "choices"}), for the caller's receipt.
LAST_RESPONSE = threading.local()
# The request_sha256 of each call this thread made since the caller set `CALLS.value = []`.
CALLS = threading.local()


class NotRecorded(LMError):
    """A replayed run asked for a call whose request was not recorded."""


class RecordedFailure(Exception):
    """A replayed call that failed, not in the LM, when it was recorded (`replayed_failure`)."""


def request_key(model: str, messages: list[dict], settings: dict) -> str:
    """The sha256 that identifies a request: model, messages and the public settings."""
    public = {k: v for k, v in settings.items() if k in PUBLIC_SETTINGS}
    payload = json.dumps(
        {"model": model, "messages": messages, **public}, sort_keys=True, default=str
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def recorded(folders: list[Path]) -> dict[str, dict]:
    """{request_sha256: receipt} from the request receipts (`*.json`) in `folders`.

    A receipt holds the recorded `response`, or the `error` of a call that failed when it was recorded; a replay fails that call again in the same way (`RecordedFailure`). Receipts that a replay wrote for an unrecorded call are skipped. A receipt whose key does not match its request is refused, and so is a folder without receipts. If a request was recorded more than once, its first answered receipt wins.
    """
    receipts: dict[str, dict] = {}
    for folder in folders:
        paths = sorted(Path(folder).glob("*.json"))
        if not paths:
            raise SystemExit(f"{folder}: no recorded requests (*.json) to replay.")
        for path in paths:
            receipt = json.loads(path.read_text(encoding="utf-8"))
            if receipt.get("schema") != RECEIPT_SCHEMA:
                raise SystemExit(f"{path}: not a request receipt ({RECEIPT_SCHEMA}).")
            request = receipt["formatted_request"]
            settings = {
                k: v for k, v in request.items() if k not in ("model", "messages")
            }
            key = request_key(request["model"], request["messages"], settings)
            if key != receipt["request_sha256"]:
                raise SystemExit(
                    f"{path}: its request_sha256 does not match its request."
                )
            if (receipt.get("error") or {}).get("type") == NotRecorded.__name__:
                continue
            if key not in receipts or receipts[key].get("response") is None:
                receipts[key] = receipt
    return receipts


def replayed_failure(error: dict) -> Exception:
    """What a call that failed when it was recorded raises again.

    An LM error (the server failed) is raised as an LM error, so it stops the run. Any other error (for example the released runs' `KeyError: 'temperature'`) is a `RecordedFailure`: DSPy's chat format then asks again in JSON, and an extractor counts the window as failed, as when recorded.
    """
    message = f"the recorded call failed: {error['type']}: {error['message']}"
    lm_errors: set[str] = set()
    stack: list[type[LMError]] = [LMError]
    while stack:
        cls = stack.pop()
        lm_errors.add(cls.__name__)
        stack.extend(cls.__subclasses__())
    return LMError(message) if error["type"] in lm_errors else RecordedFailure(message)


def stops_the_run(error: BaseException) -> bool:
    """Whether a model call's error stops the run: an LM error (the server failed, or a replayed call was not recorded).

    Any other error is an answer the step could not use: the chat format asks again in JSON, a window gives no records. DSPy 3.4 passes such an error on as `LMUnexpectedError` (for example a replayed `RecordedFailure`, or the released runs' client-side `KeyError: 'temperature'`). The released runs used a DSPy that passed such an error on unwrapped; both forms are treated alike.
    """
    return isinstance(error, LMError) and not isinstance(error, LMUnexpectedError)


def unwrapped(error: BaseException) -> BaseException:
    """The error DSPy 3.4 wrapped as `LMUnexpectedError` (its cause), else the error itself."""
    if isinstance(error, LMUnexpectedError) and error.__cause__ is not None:
        return error.__cause__
    return error


def write_new(path: Path, data: dict) -> None:
    """Write a JSON file unless it exists (the same request twice keeps its first receipt)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(data, ensure_ascii=False, indent=1) + "\n")
    except FileExistsError:
        pass


# The settings of a canonical request that an OpenAI chat request carries, in the order the released runs' settings had; any other setting is refused (`openai_request`).
CARRIED = ("temperature", "max_tokens", "seed", "response_format")


def openai_request(
    request: Request, extra_body: dict | None
) -> tuple[list[dict], dict]:
    """A DSPy request in the OpenAI chat form the released runs sent and keyed: (messages, settings).

    DSPy 3.4 hands an engine a canonical `dspy.lm15.Request` (the system text, the messages and a `Config`). This maps it back: the system text as the first message, each message as {"role", "content"}, the settings temperature, max_tokens and seed, `extra_body` (which the engine keeps, because a canonical request cannot carry it), and the response format in its OpenAI form. A tool, a message that is not text or any other setting is refused, so nothing is dropped. Worked example: the verifier's request with system text S, one user message U, temperature 0.0, max_tokens 4096, seed 9 and the `extraction` JSON schema becomes messages [{"role": "system", "content": S}, {"role": "user", "content": U}] and settings {"temperature": 0.0, "max_tokens": 4096, "seed": 9, "extra_body": {"chat_template_kwargs":
    {"enable_thinking": False}}, "response_format": {"type": "json_schema", "json_schema":
    {"name": "extraction", "strict": True, "schema": ...}}}, its recorded form.
    """
    if request.tools:
        raise ValueError("This LM sends no tools.")
    messages = []
    if request.system is not None:
        if not isinstance(request.system, str):
            raise ValueError("This LM sends a text system message only.")
        messages.append({"role": "system", "content": request.system})
    for message in request.messages:
        if message.text is None or len(message.parts) != 1:
            raise ValueError(
                f"This LM sends text messages only, not {message.parts!r}."
            )
        messages.append({"role": message.role, "content": message.text})
    config = request.config
    other = [
        f.name
        for f in dataclasses.fields(config)
        if f.name not in CARRIED and getattr(config, f.name) != f.default
    ]
    if other:
        raise ValueError(f"This LM does not send the settings {other}.")
    settings = {
        name: getattr(config, name)
        for name in CARRIED[:3]
        if getattr(config, name) is not None
    }
    if extra_body is not None:
        settings["extra_body"] = extra_body
    if config.response_format is not None:
        form = dict(config.response_format)
        kind = form.pop("type")
        settings["response_format"] = (
            {"type": kind, kind: form}
            if kind == "json_schema"
            else {"type": kind, **form}
        )
    return messages, settings


class Engine:
    """One OpenAI-compatible chat call per DSPy request, with the usage log, request receipts and replay (module docstring), behind DSPy 3.4's engine interface: `complete(Request) -> Response`.

    It owns the connection (`api_base`, `api_key`, `timeout`) and `extra_body`. `usage_log`, `requests` and `replay` may each be None (no log, no receipts, live calls). `stage` names the step in the usage log, for example `w8k/chat:extract`. A live call goes through LiteLLM with no retries, and LiteLLM's failures come out as DSPy's LM errors, as with `engine="litellm"`.
    """

    def __init__(
        self,
        *,
        api_base: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        extra_body: dict | None = None,
        usage_log: Path | None = None,
        requests: Path | None = None,
        replay: dict[str, dict] | None = None,
        stage: str = "unspecified",
    ):
        self.transport: dict[str, Any] = {"api_base": api_base, "api_key": api_key}
        if timeout is not None:
            self.transport["timeout"] = timeout
        self.extra_body = extra_body
        self.usage_log, self.requests, self.replay, self.stage = (
            usage_log,
            requests,
            replay,
            stage,
        )

    def complete(self, request: Request) -> Response:
        started = time.time()
        model = request.model
        content, public = openai_request(request, self.extra_body)
        key = request_key(model, content, public)
        if isinstance(getattr(CALLS, "value", None), list):
            CALLS.value.append(key)
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "request_sha256": key,
            "formatted_request": {"model": model, "messages": content, **public},
            "boundary": RECEIPT_BOUNDARY,
        }
        if self.replay is not None:
            if key not in self.replay:
                message = f"request_sha256 {key} was not recorded"
                if self.requests:
                    receipt.update(
                        error={"type": "NotRecorded", "message": message}, response=None
                    )
                    write_new(self.requests / f"{key}.json", receipt)
                    message += f"; its request is in {self.requests / f'{key}.json'}"
                raise NotRecorded(message, model=model)
            recording = self.replay[key]
            if recording.get("response") is None:
                if self.requests:
                    receipt.update(error=recording["error"], response=None)
                    write_new(self.requests / f"{key}.json", receipt)
                raise replayed_failure(recording["error"])
            response = ModelResponse(**recording["response"])
        else:
            with litellm_errors(model=model):
                try:
                    response = litellm.completion(
                        model=model,
                        messages=content,
                        **public,
                        **self.transport,
                        num_retries=0,
                    )
                except Exception as error:
                    if self.requests:
                        receipt.update(
                            error={"type": type(error).__name__, "message": str(error)},
                            response=None,
                        )
                        write_new(self.requests / f"{key}.json", receipt)
                    raise
            assert isinstance(response, ModelResponse)  # no stream was asked for
        LAST_RESPONSE.value = {
            "response_id": getattr(response, "id", None),
            "choices": [c.model_dump() for c in response.choices],
        }
        body = response.model_dump(mode="json")
        if self.requests:
            receipt.update(response=body, error=None)
            write_new(self.requests / f"{key}.json", receipt)
        if self.usage_log:
            self.log_usage(
                self.usage_log, model, key, response, content, public, started
            )
        return response_from_openai_chat(body, model=model)

    def close(self) -> None:
        """Nothing to close: LiteLLM's clients are shared by the process."""

    def log_usage(
        self, path: Path, model: str, key, response, content, public, started
    ) -> None:
        usage = getattr(response, "usage", {})
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        completion_details = getattr(usage, "completion_tokens_details", None)
        row = {
            "stage": self.stage,
            "started": started,
            "seconds": time.time() - started,
            "model": model,
            "request_sha256": key,
            "response_id": getattr(response, "id", None),
            "provider": getattr(response, "provider", None),
            "reported_cost_usd": getattr(usage, "cost", None),
            "dspy_cache_hit": bool(getattr(response, "cache_hit", False)),
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "cached_prompt_tokens": getattr(prompt_details, "cached_tokens", 0) or 0,
            "cache_write_tokens": getattr(prompt_details, "cache_write_tokens", 0) or 0,
            "reasoning_tokens": getattr(completion_details, "reasoning_tokens", 0) or 0,
            "message_chars": sum(len(str(m.get("content", ""))) for m in content),
            "finish_reasons": [
                getattr(c, "finish_reason", None) for c in response.choices
            ],
            "settings": public,
        }
        assert tuple(row) == USAGE_FIELDS
        with LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, default=str) + "\n")


class AsyncEngine:
    """The async counterpart DSPy asks for (`await lm.acall(...)`): the same engine, run in a worker thread."""

    def __init__(self, engine):
        self.engine = engine

    async def complete(self, request: Request) -> Response:
        return await asyncio.to_thread(self.engine.complete, request)

    async def aclose(self) -> None:
        """Nothing to close (`Engine.close`)."""


class LM(dspy.LM):
    """`dspy.LM` on `Engine`, which does the usage log, request receipts and replay.

    It overrides none of `dspy.LM`'s calls; it only builds the engine from the connection, `extra_body` and log settings (`Engine`), and passes the request settings to `dspy.LM`.
    """

    def __init__(
        self,
        model: str,
        *,
        api_base: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        extra_body: dict | None = None,
        usage_log: Path | None = None,
        requests: Path | None = None,
        replay: dict[str, dict] | None = None,
        stage: str = "unspecified",
        **kwargs,
    ):
        engine = Engine(
            api_base=api_base,
            api_key=api_key,
            timeout=timeout,
            extra_body=extra_body,
            usage_log=usage_log,
            requests=requests,
            replay=replay,
            stage=stage,
        )
        super().__init__(
            model, engine=engine, async_engine=AsyncEngine(engine), **kwargs
        )
        # dspy.LM stores unset sampling settings as None; a None would still change the key.
        self.kwargs = {k: v for k, v in self.kwargs.items() if v is not None}


def make_lm(
    model: str,
    *,
    api_base: str | None,
    api_key: str | None,
    max_tokens: int,
    seed: int | None = None,
    temperature: float | None = None,
    extra_body: dict | None = None,
    timeout: float | None = None,
    usage_log: Path | None = None,
    requests: Path | None = None,
    replay: dict[str, dict] | None = None,
    stage: str = "unspecified",
) -> LM:
    """An LM for `model` on an OpenAI-compatible endpoint (vLLM, OpenRouter, ...).

    It sends exactly the settings given here: `seed`, `temperature` and `extra_body` only when they are set. DSPy's response cache and retries and LiteLLM's retries are off, so every call reaches the server (or the replay) once. In a replay `api_base` and `api_key` are not used.
    """
    settings: dict = {"max_tokens": max_tokens}
    for name, value in (("seed", seed), ("temperature", temperature)):
        if value is not None:
            settings[name] = value
    return LM(
        "openai/" + model,  # LiteLLM's OpenAI-compatible adapter
        api_base=api_base,
        api_key=api_key,
        timeout=timeout,
        extra_body=extra_body,
        cache=False,
        num_retries=0,
        usage_log=usage_log,
        requests=requests,
        replay=replay,
        stage=stage,
        **settings,
    )


def usage_totals(path: Path) -> dict:
    """Calls and token sums of a usage log (empty if there is none)."""
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return {
        "calls": len(rows),
        "dspy_cache_hits": sum(r["dspy_cache_hit"] for r in rows),
        "api_calls": sum(not r["dspy_cache_hit"] for r in rows),
        "truncated_calls": sum("length" in r["finish_reasons"] for r in rows),
        **{
            k: sum(r[k] for r in rows)
            for k in (
                "prompt_tokens",
                "completion_tokens",
                "cached_prompt_tokens",
                "cache_write_tokens",
                "reasoning_tokens",
            )
        },
    }
