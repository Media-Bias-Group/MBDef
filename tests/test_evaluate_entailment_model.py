"""The entailment model of DefScore is pinned: another model needs its own pinned revision, and a model whose labels name no entailment class is refused.

No command chooses the entailment model (Step 5 builds `NLIScorer` in Python), so these tests call `evaluate.metric` directly; they load no model (no weights are needed).
"""

import pytest

from mbdef.evaluate import metric


def test_another_entailment_model_needs_a_pinned_revision():
    with pytest.raises(SystemExit) as stopped:
        metric.NLIScorer("someone/another-nli-model")

    assert str(stopped.value) == (
        "The entailment model someone/another-nli-model needs a pinned revision (a commit "
        "of its Hugging Face repository): pass revision=…. Only "
        f"{metric.NLI_MODEL} has one by default ({metric.NLI_REVISION})."
    )


def test_the_entailment_class_is_the_label_that_says_entailment():
    labels = {"0": "contradiction", "1": "Entailment", "2": "neutral"}

    assert metric.entailment_class(labels) == 1


def test_a_model_whose_labels_name_no_entailment_class_is_refused():
    with pytest.raises(SystemExit) as stopped:
        metric.entailment_class({"0": "LABEL_0", "1": "LABEL_1", "2": "LABEL_2"})

    assert str(stopped.value) == (
        "The entailment model's labels (LABEL_0, LABEL_1, LABEL_2) name no single entailment class, "
        "so its entailment probability cannot be read."
    )
