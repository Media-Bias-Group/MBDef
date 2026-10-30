# Which terms are kept

Part of [MBDef]({{repo_url}}). Step 5 decides for every term-in-paper whether it belongs to a media-bias taxonomy. This page says how; the [paper]({{paper_url}}) says how well it works.

## Three scores

Three methods score each term-in-paper; higher means more relevant.

- **Embedding classifier.** Each definition is embedded with Qwen3-Embedding-8B, and the term-in-paper's vectors are averaged. A logistic regression trained on the labelled records turns that vector into a probability.
- **LLM decision model** (basal-omni-1.0-ultra, 8-bit weights). For each term and its definitions it answers one question: is this a type of bias (*core*), a concept needed to describe, detect or explain bias (*satellite*), or neither (*irrelevant*)? The score is the probability of core plus satellite.
- **Retrieval judge.** Gemma 4 12B is shown the 32 most similar labelled records (found with Qwen3-Embedding-8B) and asked whether the term is relevant. Thinking is off and the temperature is 0.

## Combining them

1. **Put each score on a shared scale.** A member's score is replaced by the share of that member's own scores on the 1,700 labelled training and development records that lie below it (ties count half). {{ex_rank_sentence}} This makes each term's score independent of which other terms are scored with it.
2. **Average them.** The weights are classifier {{w_classifier}}, decision model {{w_basal}} and judge {{w_judge}}. They were learned on the training and development records: a stacked logistic regression, chosen from 315 candidate combinations. {{ex_combo_sentence}}
3. **Keep or drop.** A term-in-paper is kept when its combined score is at least **{{cut}}**. That cut keeps 95% of the relevant training and development records. It was set with each record scored by a model that had not seen its paper.

The kept rows are the `instances` table; their three scores, reference ranks and combined score are columns of it ([Columns]({{files_url}}/docs/columns.md)).
