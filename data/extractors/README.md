# The four selected extractor variants

The released extractor variants P≥20, P≥24, P≥28 and P≥28 Gemma-only (each a selected extractor's configuration) use the same three prompt programs, Gemma 4 12B's Bootstrap programs for 8k, 16k and 32k character windows:

- `gemma12b-bootstrap/w8k/`, `w16k/`, `w32k/`: `program.json` (the DSPy program: instruction and one worked example) and `manifest.json` (the training run's receipt). All 388 part receipts of the seed run record these hashes.
- `picks.json`: the recipe of the four variants: which reading passes and output formats each keeps, which verifier it applies and whether the relevance filter runs. Plan amendment 8 chose P≥20, P≥24 and P≥28; amendment 9 added Gemma-only (`data/frozen/results/grid-20260925/`).

The files are byte-for-byte copies of the released run's files.
The manifests and `picks.json` hold the paths of the machine the runs ran on; they are records, not settings.
