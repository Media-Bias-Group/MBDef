"""Build tests/fixtures/score-sample/, the scorer tests' sample: two TEST papers, two released TEST rows, recorded pair scores.

Run from the repository root, with the data root fetched (`mbdef data fetch frozen/`) and the two scoring models on a GPU:

    uv run python tests/fixtures/make_score_sample.py tests/fixtures/score-sample

- `reference.json`: the two papers' entries of the frozen TEST reference (data/frozen/results/test-eval-20260924/inputs/test-reference.json), unchanged.
- `rows/<row>/papers/<paper>.json`: byte copies of the released rows' saved outputs ($DATA_ROOT/frozen/artifacts/experiments/test-eval-20260924/collected/rows).
- `scorings.json`: the two scorings, for `mbdef evaluate score`.
- `pair-scores.json`: every BGE similarity and ModernBERT entailment the scorer asked for, written by `evaluate score --record-scores` with the real models.
- `SOURCES.tsv`: the sha256 and origin of every copied file.
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

out = Path(sys.argv[1])
REFERENCE = Path("data/frozen/results/test-eval-20260924/inputs/test-reference.json")
ROWS = "frozen/artifacts/experiments/test-eval-20260924/collected/rows"
PAPERS = [
    "4243aa9bacd51bc02d4240378e7528912962b53e",
    "73bfad11b96a69cb882028ead115751adb55252d",
]
SCORED_ROWS = ["Rdb54680", "QWEN-POOL3"]

from mbdef import files  # noqa: E402

files.load_env()
rows = files.data_root() / ROWS
reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
members: dict[str, bytes] = {
    "reference.json": (
        json.dumps(
            {"papers": [p for p in reference["papers"] if p["paper_id"] in PAPERS]},
            indent=1,
            ensure_ascii=False,
        )
        + "\n"
    ).encode()
}
sources = [("reference.json", "", f"the papers {', '.join(PAPERS)} of {REFERENCE}")]
for row in SCORED_ROWS:
    for pid in PAPERS:
        name = f"rows/{row}/papers/{pid}.json"
        data = (rows / row / "papers" / f"{pid}.json").read_bytes()
        members[name] = data
        sources.append(
            (
                name,
                hashlib.sha256(data).hexdigest(),
                f"$DATA_ROOT/{ROWS}/{row}/papers/{pid}.json",
            )
        )
members["scorings.json"] = (
    json.dumps(
        [
            {"id": row, "reference": "reference.json", "papers": f"rows/{row}/papers"}
            for row in SCORED_ROWS
        ],
        indent=1,
    )
    + "\n"
).encode()

with tempfile.TemporaryDirectory() as tmp:
    folder = Path(tmp) / "sample"
    for name, data in members.items():
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(data)
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "mbdef",
            "evaluate",
            "score",
            "--scorings",
            str(folder / "scorings.json"),
            "--out",
            str(Path(tmp) / "out"),
            "--record-scores",
            str(folder / "pair-scores.json"),
        ],
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    if done.returncode:
        raise SystemExit(done.returncode)
    members["pair-scores.json"] = (folder / "pair-scores.json").read_bytes()
sources.append(("pair-scores.json", "", "evaluate score --record-scores (real models)"))
members["SOURCES.tsv"] = (
    "member\tsha256\tsource\n"
    + "".join(f"{name}\t{digest}\t{origin}\n" for name, digest, origin in sources)
).encode()

for name, data in sorted(members.items()):
    (out / name).parent.mkdir(parents=True, exist_ok=True)
    (out / name).write_bytes(data)
print(f"wrote {out}: {len(members)} files")
