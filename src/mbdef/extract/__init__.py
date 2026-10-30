"""Step 3, information extraction: definition records from the full text of papers (docs/step-3-extraction/extract.md).

- `papers.py`: a paper's text from its GROBID TEI, and the character windows the model reads.
- `records.py`: the rules applied to the records after the model: every string must be copied from the window (`exact`, `sanitize`), and a paper's windows are merged into one union (`merge`, `union`).
- `extractor.py`: the DSPy extractor (record schema, signature, window and paper modules), the saved programs in `data/extractors/`, and the JSON adapter with schema-constrained decoding.
- `run.py`: `mbdef extract run`, the step itself, with its pinned defaults.
- `verifier.py`: the candidate verifier (a second model checks each kept definition), and `verify.py`, its command `extract verify`.
- `relevance.py`: the frozen relevance filter (BGE vectors and a frozen classifier; `apply_filter`), and `filter.py`, its command `extract filter`.
- `plan.py`: `mbdef extract plan`, a corpus run's shards (balanced by their windows) and the jobs (`extract run`, `verify`, `filter`) that make each stage its variants need.
- `derive.py`: `mbdef extract derive`, a run's variants (P≥20 … P≥28) pooled from its per-paper outputs or its saved part tarballs, their union, and their comparison without gold (`derive`, `compare`).

The language model is set up by `mbdef.llm` (docs/setup/llm.md).
"""
