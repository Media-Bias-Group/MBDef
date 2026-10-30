# The language-model layer: `mbdef.llm`

| | |
|---|---|
| **Paper step** | Shared by every step that asks a language model: extraction and the candidate verifier ([docs/step-3-extraction/extract.md](../step-3-extraction/extract.md)) and training; the closure's PDF identity check uses its request key, receipts and replay ([docs/step-4-closure/closure.md](../step-4-closure/closure.md)); no command of its own |
| **Code** | [`src/mbdef/llm.py`](../../src/mbdef/llm.py) |
| **Command** | none: a library, called by `extract run`, `extract verify` and `extract train` through `make_lm`, and by `closure identity judge` for its recorded requests |
| **Checked by** | the extraction tests [`tests/test_extract.py`](../../tests/test_extract.py) (replay, the guard, the usage log) |

## What it does: inputs and outputs

A step builds its LM with `make_lm(model, api_base=…, api_key=…, max_tokens=…, seed=…, extra_body=…, usage_log=…, requests=…, replay=…, stage=…)` and calls it through DSPy (`dspy.context(lm=lm)`). The LM sends exactly the settings it is given: no temperature unless one is given, so a thinking model samples with its own settings, as the released runs did. Each call:

- appends one line to `usage_log` with the fields the released runs logged (`USAGE_FIELDS`: stage, start, seconds, model, `request_sha256`, response id, provider, cost, cache hit, the token counts, the characters sent, the finish reasons and the public settings), and no text;
- writes its exact request and the full response (or the error) to `requests/<request_sha256>.json`;
- in a replay (`replay=recorded(folders)`), is answered from the recorded receipt of the identical request and contacts no server. A call that failed when it was recorded fails again the same way.
  A call that was not recorded raises `NotRecorded`, naming its `request_sha256` and keeping its request next to the outputs, so it can be compared with the recorded ones.

**The request key.** `request_sha256` is the sha256 of `json.dumps({"model": …, "messages": […], <public settings>}, sort_keys=True, default=str)`; the public settings are `max_tokens`, `max_completion_tokens`, `temperature`, `seed`, `rollout_id`, `extra_body` and `response_format`. Worked example: the first 8k window of paper `4243aa9b…` in the released TEST run was sent as model `openai/google/gemma-4-12B-it-qat-w4a16-ct`, four messages (the instruction, the worked example's question and answer, the window), `max_tokens` 32768, `seed` 9 and `extra_body` `{"chat_template_kwargs": {"enable_thinking": true}}`: key `4afab7d2…`. The same window with another title, or the same request with `temperature` 0.0 added, has another key, and a replay of the released run stops at it.

**How a call reaches the model.** The LM is a plain `dspy.LM` on an engine of its own (`Engine`, DSPy's `complete(Request) -> Response` interface); it does not override `forward()`, which DSPy 3.4 deprecates and 3.5 removes. DSPy hands the engine a canonical request that cannot carry `extra_body`, so the engine keeps `extra_body` itself, maps the request back to the OpenAI chat form (`openai_request`) and sends exactly that through LiteLLM. DSPy's own engine, `engine="litellm"`, cannot carry `extra_body` either, which is why the LM has an engine of its own. DSPy 3.4 passes an error that is not the server's on as `LMUnexpectedError`; `stops_the_run` treats it as the released runs did, as an answer that failed (the chat format asks again in JSON, a window gives no records), while a server error or an unrecorded call stops the run. Live runs follow the same rule.

## For the steps that use it

- **Replay in tests.** Copy the recorded receipts of a few inputs into a fixture and pass their folder as `replay`. No network, no model, and any change of prompt, program, setting or library version shows up as `NotRecorded` at the first changed call (a change of library versions is such a change and is found the same way).
- **Live runs record themselves.** The `requests` folder of a live run is a valid replay folder.
- **Errors.** `NotRecorded` and the server's errors are DSPy `LMError`s, which DSPy's adapters do not retry in another format; a step should stop on them. Any other exception from a call is what DSPy and the step see as a failed answer.
- **Thread-local helpers.** `LAST_RESPONSE.value` is this thread's last raw response, and `CALLS.value` (set it to `[]` first) collects the `request_sha256` of this thread's calls, for the step's own receipts.
- LiteLLM is kept from downloading its price list and from loading a `.env` of its own when this module is imported; NumPy is imported before DSPy, which needs that order.

## How it was checked

- **Extraction tests** ([`tests/test_extract.py`](../../tests/test_extract.py)): a replay of a run recorded on synthetic papers answers every call from the recorded receipts, the usage log has the fields of the recorded log and, but for the timing, the same values, a call that was not recorded stops the run and names its `request_sha256`, and a call that failed when it was recorded fails again in the replay.
- **The released receipts:** the 28,126 request receipts of the released TEST and training runs were checked once with `openai_request`: each maps back to its recorded `request_sha256`. They hold paper text and are not public.
- **The engine** ([`tests/test_extract.py`](../../tests/test_extract.py), the last test): a DSPy program's call reaches the LM's engine, its receipt holds exactly the settings sent (with `extra_body`), the usage log the recorded response id and provider, and DSPy's warnings about its deprecated interfaces are errors there.

## How to adapt it

- **Another model or endpoint:** the `model`, `api_base` and `api_key` arguments of `make_lm`; the steps pass them from their flags and from `.env` ([docs/setup/data-root.md](data-root.md)).
- **Other settings the request should carry:** `make_lm` takes `max_tokens`, `seed`, `temperature`, `extra_body` (passed on as it is, for example a chat template's thinking switch) and `timeout`, and sends only those that are set. The public settings (`PUBLIC_SETTINGS`; `timeout` is not one) enter the request key, so a replay of recorded runs stops at the first call that differs.
- **A paid API:** none is wired in (the repository makes no paid call). To call one, give `make_lm` its model name, endpoint and key; the usage log records the cost LiteLLM reports for each call.
