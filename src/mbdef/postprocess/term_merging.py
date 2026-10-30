"""`mbdef postprocess merge-terms`: merge the kept Extracted Concepts by Label Group, after Step 5.4.

The merging runs after the relevance decisions, before the dataset is released, so that the release holds no duplicates. Relevance was decided for each Extracted Concept (a term as one paper defines it), so a merged row is a Step 5.1 Label Group (the Label Match rule over all corpus Extracted Concepts, `postprocess label-groups`; the adopted run uses the C-20261021 groups) restricted to its kept Extracted Concepts; the dropped ones stay out. It is not a Media Bias Concept: a Label Group only says that the labels match.

Every merged row keeps its provenance, for traceability: each kept Extracted Concept carries its two keys and every normalisation step that changed its label (`label_match.trace`, read from the groups' labels.jsonl), and each row carries `joined_by` (the keys its labels share) and `merge_flags`, the kinds of join a person may want to check (word order, spacing, a dropped bracket, a chosen plural). Writes:

  concepts.jsonl              one row per Label Group with at least one kept Extracted Concept: its labels and how
                              often each is used, its papers, how many were dropped, `joined_by`, `merge_flags`, and
                              every kept one with its keys, normalisation and definitions (`dataset build` reads it)
  split-labels.csv            set aside for a look: every Extracted Concept of a Label Group that the decisions split
                              (some kept, some dropped), with its scores
  majority-disagreements.csv  set aside for a look: where a majority vote over a Label Group (kept share above 0.5
                              keeps the group, below 0.5 drops it, exactly 0.5 is no majority) would decide differently,
                              one row per Extracted Concept that would change
  summary.json                counts before and after, the merge flags, and a check: the Label Match rule rerun on the kept labels
                              alone, and how many kept Extracted Concepts land in another grouping there

Worked example (the miniature in tests/fixtures/term-merging-mini): the label "media bias" joins "bias in the media" (same content words) and "MediaBias" (same letters without spaces) into one Label Group. With every "media bias" dropped, the merged row still holds the kept "bias in the media" and "MediaBias", flagged "joined through another label", while the Label Match rule rerun on the kept labels alone would split them: the check counts those as in another grouping.
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from mbdef.files import file_sha256

from . import FROZEN, REPO, frozen_file, label_match, stored
from .compare import shown
from .label_groups import joined_by
from .label_match import STOP
from .provenance import code_version

# The adopted run's inputs (frozen files): the ensemble's decisions and the Label Groups of Label Match C-20261021; their jsonl files are over 10 MB, so in the data root
DECISIONS = "20261021/decisions/ensemble.csv"
LABEL_GROUPS = "20261021/label-groups"
# the other decision columns are copied as scores
NOT_SCORES = ("mention_id", "paper_id", "keep", "corpus", "label")


def default_inputs():
    """The adopted run: the ensemble's decisions and the Label Groups of Label Match C-20261021."""
    return dict(
        decisions=frozen_file(DECISIONS), label_groups=REPO / FROZEN / LABEL_GROUPS
    )


def truthy(v):
    """A decision file's `keep` value as a bool ("True", "1", "yes", "keep" or "kept" keep the term)."""
    return str(v).strip().lower() in ("1", "true", "yes", "keep", "kept")


def merge_flags(labels, joined, traced):
    """The kinds of join in a merged row that a person may want to check; empty when it has one label."""
    if len(labels) < 2:
        return []
    flags = set()
    for j in joined:
        if (
            j["key_type"] == "spacing"
            and len({traced[x]["word_set_key"] for x in j["labels"]}) > 1
        ):
            flags.add("joined by spacing (compound written apart or together)")
        if j["key_type"] == "word-set":
            orders = {
                tuple(w for w in traced[x]["words"] if w not in STOP)
                for x in j["labels"]
            }
            if len(orders) > 1:
                flags.add("joined by the same words in a different order")
    if len({traced[x]["word_set_key"] for x in labels}) > 1 and not any(
        j["key_type"] == "spacing" for j in joined
    ):
        flags.add("joined through another label")
    for x in labels:
        for s in traced[x]["steps"]:
            if "dropped" in s and ("bracket" in s or "acronym" in s):
                flags.add("a bracket or acronym dropped")
            elif s.startswith("plural") and ", ends " in s:
                flags.add("plural with two possible singulars")
            elif s.startswith("plural") and ("irregular" in s or "-ics" in s):
                flags.add("irregular or -ics plural")
            elif s.startswith("British spelling"):
                flags.add("British spelling")
    return sorted(flags)


def merge(*, decisions, label_groups, out):
    """Write the four files of the module docstring into `out`; return the summary. `label_groups` is a folder written by `postprocess label-groups` (each of its files in git or, over 10 MB, under DATA_ROOT/frozen/)."""
    lg = Path(label_groups)
    files = {
        n: stored(lg, n)
        for n in ("terms.jsonl", "groups.jsonl", "labels.jsonl", "run.json")
    }
    csv.field_size_limit(1 << 30)
    with Path(decisions).open(newline="") as f:
        rows = list(csv.DictReader(f))
    dec = {r["mention_id"]: r for r in rows}
    if len(dec) != len(rows):
        raise SystemExit(
            f"{decisions}: more than one decision for an Extracted Concept"
        )
    terms_rows = [json.loads(line) for line in files["terms.jsonl"].open()]
    by_id = {g["group_id"]: g for g in map(json.loads, files["groups.jsonl"].open())}
    traced = {x["label"]: x for x in map(json.loads, files["labels.jsonl"].open())}
    lg_rule = json.loads(files["run.json"].read_text())["rule"]
    if set(dec) != {t["mention_id"] for t in terms_rows}:
        raise SystemExit(
            f"decisions cover {len(dec)} terms, the label groups {len(terms_rows)}: not the same terms"
        )
    defs = defaultdict(list)
    for g in by_id.values():
        for d in g["definitions"]:
            defs[d["mention_id"]].append(d["text"])
    score_cols = [c for c in next(iter(dec.values())) if c not in NOT_SCORES]
    by_group = defaultdict(list)
    for t in terms_rows:
        row = dec[t["mention_id"]]
        by_group[t["group_id"]].append(
            dict(t, keep=truthy(row["keep"]), scores={c: row[c] for c in score_cols})
        )

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    concepts, split_rows, majority_rows, flag_counts = 0, [], [], Counter()
    split_groups, majority_keeps, majority_drops, ties = 0, 0, 0, 0
    with (out / "concepts.jsonl").open("w", encoding="utf-8") as f:
        for gid, inst in sorted(by_group.items()):
            kept = [t for t in inst if t["keep"]]
            n, k = len(inst), len(kept)
            key = by_id[gid]["key"]
            if 0 < k < n:
                split_groups += 1
                share = k / n
                for t in inst:
                    split_rows.append(
                        dict(
                            group_id=gid,
                            group_key=key,
                            label=t["term"],
                            mention_id=t["mention_id"],
                            paper_id=t["paper_id"],
                            keep=t["keep"],
                            kept_share=round(share, 4),
                            definition=(defs[t["mention_id"]] or [""])[0],
                            **t["scores"],
                        )
                    )
                if share > 0.5:
                    majority_keeps += 1
                    side = (
                        "majority keeps the group (brings back these dropped instances)"
                    )
                    lost = [t for t in inst if not t["keep"]]
                elif share < 0.5:
                    majority_drops += 1
                    side = "majority drops the group (loses these kept instances)"
                    lost = kept
                else:
                    ties += 1
                    side, lost = None, []
                for t in lost:
                    majority_rows.append(
                        dict(
                            group_id=gid,
                            group_key=key,
                            kept=k,
                            instances=n,
                            majority=side,
                            label=t["term"],
                            mention_id=t["mention_id"],
                            paper_id=t["paper_id"],
                            keep=t["keep"],
                            definition=(defs[t["mention_id"]] or [""])[0],
                            **t["scores"],
                        )
                    )
            if not kept:
                continue
            concepts += 1
            kept_labels = sorted({t["term"] for t in kept})
            joined = joined_by(kept_labels, traced)
            flags = merge_flags(kept_labels, joined, traced)
            flag_counts.update(flags)
            merged = dict(
                concept_id=gid,
                key=key,
                labels=dict(Counter(t["term"] for t in kept).most_common()),
                terms=k,
                papers=len({t["paper_id"] for t in kept}),
                dropped_instances=n - k,
                joined_by=joined,
                merge_flags=flags,
                instances=[
                    dict(
                        mention_id=t["mention_id"],
                        paper_id=t["paper_id"],
                        term=t["term"],
                        word_set_key=traced[t["term"]]["word_set_key"],
                        spacing_key=traced[t["term"]]["spacing_key"],
                        normalisation=traced[t["term"]]["steps"],
                        definitions=defs[t["mention_id"]],
                    )
                    for t in kept
                ],
            )
            f.write(json.dumps(merged, ensure_ascii=False) + "\n")
    for name, rows in (
        ("split-labels.csv", split_rows),
        ("majority-disagreements.csv", majority_rows),
    ):
        with (out / name).open("w", newline="", encoding="utf-8") as f:
            if rows:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)

    # The check: the Label Match rule (the groups' own version) on the kept labels alone. Each old group's main new group is the one most of its kept Extracted Concepts land in (the first such group on a tie); every other kept one has moved.
    kept_terms = [t for t in terms_rows if truthy(dec[t["mention_id"]]["keep"])]
    regroup = label_match.label_groups(
        [t["term"] for t in kept_terms], rule=lg_rule["version"]
    )
    pairs = Counter(
        (t["group_id"], r) for t, r in zip(kept_terms, regroup, strict=True)
    )
    main_of_old = {}
    for (old, new), _ in sorted(pairs.items(), key=lambda x: -x[1]):
        main_of_old.setdefault(old, new)
    moved = sum(c for (old, new), c in pairs.items() if main_of_old[old] != new)
    summary = dict(
        decisions=dict(path=shown(decisions), sha256=file_sha256(Path(decisions))),
        label_groups=dict(
            path=shown(lg),
            rule=lg_rule,
            groups_sha256=file_sha256(files["groups.jsonl"]),
            labels_sha256=file_sha256(files["labels.jsonl"]),
        ),
        before=dict(
            terms=len(terms_rows),
            groups=len(by_id),
            definitions=sum(len(v) for v in defs.values()),
        ),
        kept=dict(
            terms=len(kept_terms),
            definitions=sum(len(defs[t["mention_id"]]) for t in kept_terms),
            papers=len({t["paper_id"] for t in kept_terms}),
        ),
        after=dict(
            concepts=concepts,
            single_term_concepts=sum(
                1 for v in by_group.values() if sum(t["keep"] for t in v) == 1
            ),
        ),
        merge_flags=dict(flag_counts.most_common()),
        split_groups=split_groups,
        majority=dict(
            keeps_group=majority_keeps,
            drops_group=majority_drops,
            tie=ties,
            instances_that_would_change=len(majority_rows),
        ),
        check_rule_c_on_kept_only=dict(
            groups=len(set(regroup)), kept_instances_in_a_different_grouping=moved
        ),
        rule=f"concept = a Label Group of the Label Match rule ({lg_rule['version']}, over all corpus terms), "
        "restricted to its kept instances",
        provenance=code_version(),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def merge_command(decisions, label_groups, out):
    """`postprocess merge-terms`: merge into `out` and say what came out."""
    s = merge(decisions=decisions, label_groups=label_groups, out=out)
    print(
        f"{s['kept']['terms']:,} of {s['before']['terms']:,} Extracted Concepts kept; {s['before']['groups']:,} Label "
        f"Groups -> {s['after']['concepts']:,} merged rows; {s['split_groups']:,} groups split by the decisions "
        f"(set aside in split-labels.csv); wrote {out}"
    )
    return 0
