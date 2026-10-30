"""The evaluate step: how good the ground truth and the extractions are (docs/step-3-extraction/evaluate.md).

- `agreement.py`: `mbdef evaluate agreement`, the annotators' agreement and the curation numbers of the ground truth (paper Table 3). Its statistics are in `stats.py`.
- `metric.py`: DefScore, the definition matcher: the pair score (token F1, BGE similarity and ModernBERT entailment), the one-to-one matching of definitions, the full-record score, and the recorded pair scores that let a scoring run without the models.
- `score.py`: the general scorer (`score`, one function), and `mbdef evaluate score`, which scores a list of scorings (predictions of one setup on one reference).
- `heldout.py`: `mbdef evaluate heldout`, the held-out TEST rescoring from the saved outputs, with its frozen eligibility and pick rules.
- `paired.py`: `mbdef evaluate paired`: our P≥28 extractor against Opus, paper by paper, from saved per-paper counts.
- `grid.py` and `selection.py`: `mbdef evaluate grid` and `evaluate select`, the extractor selection: every setup of the grid scored, then the audit and the frozen pick rule (docs/step-3-extraction/selection.md).
"""
