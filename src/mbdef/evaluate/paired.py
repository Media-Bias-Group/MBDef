"""`mbdef evaluate paired`: our P≥28 extractor against Opus, paper by paper.

The paper says our self-hosted extractor finds about as many reference definitions as Opus. The comparison uses saved per-paper counts only, no model:

- **ours**: the P≥28 variant of the selection (`OURS`), its per-paper found / predicted / reference counts on DEV (21 papers) and TEST (29 papers), from the selection's results.json;
- **Opus**: on TEST the row Rdb54680 of the held-out TEST rescoring (Opus extraction with the Qwen verifier), on DEV its parent run R59b5d1d (definitions-strict).

For each split it prints the pooled counts and three differences (ours − Opus): definitions found, recall in points (found difference / reference definitions) and precision in points, each with a 95% interval from a paired bootstrap over papers (both systems resampled together, 20,000 draws, Python `random.Random(20261013)`), the papers where ours finds more, fewer or the same; then a paired permutation test (20,000 swaps within papers, `random.Random(20261014)`) and a sign test on the papers. Worked example (TEST, the released outputs): ours 294 of 365, Opus 290 of 365, so recall differs by +4 / 365 = +1.1 points, 95% interval −7.9 to +9.3.

output.txt is byte for byte what the original comparison printed for the same inputs: the same draws in the same order, the same interval indices (the test fixtures hold its output for small made-up inputs).
"""

import json
import random
from pathlib import Path

from mbdef import files, stats

# Inputs under DATA_ROOT (data/data-root.tsv).
SELECTION = "frozen/results/production-20260925/report-20260927/selection/results.json"
TEST_RESULTS = "frozen/artifacts/experiments/test-eval-20260924/collected/results.json"
OPUS_DEV = (
    "frozen/artifacts/experiments/matched-winners-v1/refinement/R59b5d1d/"
    "definitions-strict/evaluation.json"
)
# The two systems.
OURS = "gemma12b/TR-bootstrap-8k+16k32k/chat-default-think/filtered-ordinary-qv"
OPUS_TEST_ROW = "Rdb54680"
SPLITS = ("dev", "test")
# The resampling.
BOOTSTRAP_SEED = 20261013
BOOTSTRAP_REPLICATES = 20_000
PERMUTATION_SEED = 20261014
PERMUTATION_SWAPS = 20_000
# Outputs, in --out.
OUTPUT = "output.txt"
NUMBERS = "paired.json"

Counts = dict[str, tuple[int, int, int]]  # paper id -> (found, predicted, reference)


def load_counts(selection: Path, test_results: Path, opus_dev: Path):
    """({split: ours}, {split: Opus}) per-paper counts from the three saved outputs."""
    configs = {
        c["id"]: c for c in json.loads(selection.read_text(encoding="utf-8"))["configs"]
    }
    ours = {
        s: {
            p["paper_id"]: (p["tp"], p["predicted"], p["gold"])
            for p in configs[f"{s}/{OURS}"]["per_paper"]
        }
        for s in SPLITS
    }
    rows = json.loads(test_results.read_text(encoding="utf-8"))["rows"]
    row = next(r for r in rows if r["row"] == OPUS_TEST_ROW)
    opus = {
        "test": {
            p["paper_id"]: tuple(
                p["binary_one_to_one"][k] for k in ("tp", "predicted", "gold")
            )
            for p in row["score"]["per_paper"]
        }
    }

    def dev_counts(paper: dict) -> tuple:
        m = (
            paper["M3"]["binary_one_to_one"]
            if "M3" in paper
            else paper["binary_one_to_one"]
        )
        return (m.get("tp", m.get("matched_gold")), m["predicted"], m["gold"])

    papers = json.loads(opus_dev.read_text(encoding="utf-8"))["biasdef"]["per_paper"]
    opus["dev"] = {p["paper_id"]: dev_counts(p) for p in papers}
    return ours, opus


def pooled(sample: list, ours: Counts, opus: Counts) -> tuple:
    """(found difference, recall difference in points, precision difference in points) of the papers in `sample`, pooled."""

    def total(counts: Counts, k: int):
        return sum(counts[i][k] for i in sample)

    reference = total(ours, 2)
    return (
        total(ours, 0) - total(opus, 0),
        100 * (total(ours, 0) - total(opus, 0)) / reference,
        100 * (total(ours, 0) / total(ours, 1) - total(opus, 0) / total(opus, 1)),
    )


def compare(ours: dict, opus: dict) -> tuple[list[str], dict]:
    """The lines of output.txt and the same numbers as data."""
    lines, numbers = [], {}
    rng = random.Random(BOOTSTRAP_SEED)
    for s in SPLITS:
        ids = sorted(ours[s])
        if set(ids) != set(opus[s]):
            different = ", ".join(sorted(set(ids) ^ set(opus[s])))
            raise SystemExit(
                f"{s}: the two systems have different papers: {different}."
            )
        for i in ids:
            if ours[s][i][2] != opus[s][i][2]:
                raise SystemExit(f"{s}: paper {i} has different reference counts.")
        observed = pooled(ids, ours[s], opus[s])
        draws = stats.paired_bootstrap(
            ids,
            lambda sample, s=s: pooled(sample, ours[s], opus[s]),
            BOOTSTRAP_REPLICATES,
            rng,
        )
        ci = [stats.index_interval([d[k] for d in draws]) for k in range(3)]
        wins = sum(ours[s][i][0] > opus[s][i][0] for i in ids)
        losses = sum(ours[s][i][0] < opus[s][i][0] for i in ids)

        def tot(counts: Counts):
            return tuple(sum(v[k] for v in counts.values()) for k in range(3))

        lines += [
            f"{s}: papers={len(ids)} ours={tot(ours[s])} opus={tot(opus[s])}",
            f"  found diff (ours-opus) {observed[0]:+d}  95% CI {ci[0]}",
            f"  recall diff pp {observed[1]:+.1f}  95% CI ({ci[1][0]:+.1f}, {ci[1][1]:+.1f})",
            f"  precision diff pp {observed[2]:+.1f}  95% CI ({ci[2][0]:+.1f}, {ci[2][1]:+.1f})",
            f"  papers where ours finds more / fewer / same: {wins} / {losses} / "
            f"{len(ids) - wins - losses}",
        ]
        names = ("found", "predicted", "reference")
        numbers[s] = {
            "papers": len(ids),
            "ours": dict(zip(names, tot(ours[s]), strict=True)),
            "opus": dict(zip(names, tot(opus[s]), strict=True)),
            **{
                name: {"observed": observed[k], "ci95": list(ci[k])}
                for k, name in enumerate(
                    ("found_diff", "recall_diff_pp", "precision_diff_pp")
                )
            },
            "papers_more_fewer_same": [wins, losses, len(ids) - wins - losses],
        }

    # Paired permutation test: under "no difference", which system produced a paper's result is exchangeable, so swap ours/Opus within each paper at random and recompute the pooled difference.
    prng = random.Random(PERMUTATION_SEED)
    for s in SPLITS:
        ids = sorted(ours[s])

        def recall_precision(a: Counts, b: Counts, ids=ids) -> tuple:
            g = sum(a[i][2] for i in ids)
            rec = 100 * (sum(a[i][0] for i in ids) - sum(b[i][0] for i in ids)) / g
            prec = 100 * (
                sum(a[i][0] for i in ids) / sum(a[i][1] for i in ids)
                - sum(b[i][0] for i in ids) / sum(b[i][1] for i in ids)
            )
            return rec, prec

        p_recall, p_precision = stats.paired_permutation_p(
            ours[s], opus[s], ids, recall_precision, PERMUTATION_SWAPS, prng
        )
        w, l_ = numbers[s]["papers_more_fewer_same"][:2]
        sign_p = stats.sign_test(w, l_)
        lines.append(
            f"{s}: permutation p (two-sided, {PERMUTATION_SWAPS} swaps): recall "
            f"{p_recall:.3f}, precision {p_precision:.3f}; sign test on papers ({w} more / "
            f"{l_} fewer, {len(ids) - w - l_} ties): p = {sign_p:.3f}"
        )
        numbers[s]["permutation_p"] = {"recall": p_recall, "precision": p_precision}
        numbers[s]["sign_test_p"] = sign_p
    return lines, numbers


def paired_command(
    out: str, selection: str | None, test_results: str | None, opus_dev: str | None
) -> int:
    """Compare ours with Opus and write output.txt, paired.json and a receipt into OUT."""
    target = Path(out)
    if target.exists() and any(target.iterdir()):
        raise SystemExit(f"{out} is not empty; choose a new folder for --out.")
    inputs = {
        "selection": Path(selection) if selection else files.data_root() / SELECTION,
        "test_results": Path(test_results)
        if test_results
        else files.data_root() / TEST_RESULTS,
        "opus_dev": Path(opus_dev) if opus_dev else files.data_root() / OPUS_DEV,
    }
    ours, opus = load_counts(*inputs.values())
    lines, numbers = compare(ours, opus)
    target.mkdir(parents=True, exist_ok=True)
    (target / OUTPUT).write_text("\n".join(lines) + "\n", encoding="utf-8")
    files.write_json(target / NUMBERS, numbers)
    receipt = files.receipt(
        "evaluate paired",
        settings={
            "ours": OURS,
            "opus_test_row": OPUS_TEST_ROW,
            "bootstrap": {
                "rng": "Python random.Random",
                "seed": BOOTSTRAP_SEED,
                "replicates": BOOTSTRAP_REPLICATES,
                "unit": "paper",
                "interval": "sorted draws at index int(0.025 n) and int(0.975 n) - 1",
            },
            "permutation": {
                "rng": "Python random.Random",
                "seed": PERMUTATION_SEED,
                "swaps": PERMUTATION_SWAPS,
                "unit": "paper",
            },
            "splits": list(SPLITS),
        },
        inputs=inputs,
        outputs={name: target / name for name in (OUTPUT, NUMBERS)},
    )
    files.write_json(target / files.RECEIPT, receipt)
    print("\n".join(lines))
    print("wrote", out)
    return 0
