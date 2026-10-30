"""Step 5.1: the Label Groups of the corpus under the Label Match rule, written for every later step.

Writes to the folder `--out` names:

  terms.jsonl   one row per Extracted Concept: mention_id, paper_id, corpus, term, group_id, n_definitions
  groups.jsonl  one row per Label Group, most Definitions first: group_id, key (word-set key of its first label),
                labels (distinct spellings, most frequent first), terms, papers, n_definitions, n_unique_texts,
                definitions [{def_id, mention_id, paper_id, text}]
  labels.jsonl  one row per distinct label (provenance): label, group_id, terms, word_set_key, spacing_key, steps (every
                normalisation step that changed it, in plain words, from `label_match.trace`)
  summary.json  group counts, group sizes, and how many groups each normalisation step saves
  run.json      code version, rule version, input files and counts, output file hashes, timings

Each group in groups.jsonl also lists `joined_by`: every key that two or more of its distinct labels share, with those labels, so it is clear why the labels are one group ("media bias" and "bias in the media" share the word-set key "bias media"). With `--rule C-20261015` the files are written in that rule's format, without labels.jsonl and `joined_by` (they come with C-20261021, which keeps the provenance for traceability), so that the frozen C-20261015 groups rebuild byte for byte.

A group id is "g" + the first 12 hex digits of the SHA-1 of the group's key, so it stays the same across runs as long as the group's first Extracted Concept (by mention id) keeps its key.
"""

import hashlib
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from mbdef import files

from . import label_match
from .corpus import (
    CorpusView,
    ExtractedConcept,
    default_links,
    default_store,
    load_view,
)
from .provenance import code_version, file_record

OUTPUTS = ("terms.jsonl", "groups.jsonl", "labels.jsonl", "summary.json", "run.json")
PLAIN = "C-20261015"  # the rule version whose files have no provenance


def outputs(rule: str) -> tuple[str, ...]:
    return tuple(n for n in OUTPUTS if not (rule == PLAIN and n == "labels.jsonl"))


def _line(row: dict) -> str:
    """One JSON Lines row: compact, non-ASCII kept."""
    return files.json_text(row, indent=None)


def joined_by(labels: list[str], traced: dict[str, dict]) -> list[dict]:
    """Every key two or more of a group's distinct labels share, with those labels: why the labels are one group."""
    shared = []
    for kind, field in (("word-set", "word_set_key"), ("spacing", "spacing_key")):
        by_key: dict[str, list[str]] = {}
        for label in labels:
            by_key.setdefault(traced[label][field], []).append(label)
        shared += [
            dict(key_type=kind, key=k, labels=ls)
            for k, ls in sorted(by_key.items())
            if len(ls) > 1
        ]
    return shared


def write_label_groups(
    view: CorpusView, out: Path, rule: str = label_match.RULE_VERSION
) -> tuple[dict, dict]:
    """Write terms.jsonl, groups.jsonl, labels.jsonl (not under C-20261015) and summary.json; return the summary and the seconds each phase took."""
    concepts = view.concepts
    seconds = {}
    t = time.perf_counter()
    labels = [c.label for c in concepts]
    group = label_match.label_groups(labels, rule=rule)
    vocab = label_match.vocabulary(labels, rule)
    first: dict[int, int] = {}
    for i, g in enumerate(group):
        first.setdefault(g, i)
    key = {g: label_match.label_keys(labels[i], vocab)[0] for g, i in first.items()}
    gid = {g: "g" + hashlib.sha1(k.encode()).hexdigest()[:12] for g, k in key.items()}
    if len(set(gid.values())) != len(gid):
        raise RuntimeError("two Label Groups got the same id")
    provenance = rule != PLAIN
    traced = (
        {label: label_match.trace(label, vocab) for label in sorted(set(labels))}
        if provenance
        else {}
    )
    group_of = dict(zip(labels, group, strict=True))
    seconds["group"] = time.perf_counter() - t

    t = time.perf_counter()
    savings = label_match.step_savings(labels, rule)
    seconds["step_savings"] = time.perf_counter() - t

    t = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    members: dict[int, list[ExtractedConcept]] = {}
    with (out / "terms.jsonl").open("w") as f:
        for c, g in zip(concepts, group, strict=True):
            members.setdefault(g, []).append(c)
            f.write(
                _line(
                    dict(
                        mention_id=c.mention_id,
                        paper_id=c.paper_id,
                        corpus=c.corpus,
                        term=c.label,
                        group_id=gid[g],
                        n_definitions=len(c.definitions),
                    )
                )
            )
    sizes: Counter[int] = Counter()
    pairs_all = pairs_unique = 0
    n_defs = {g: sum(len(c.definitions) for c in cs) for g, cs in members.items()}
    with (out / "groups.jsonl").open("w") as f:
        for g in sorted(members, key=lambda g: (-n_defs[g], gid[g])):
            cs = members[g]
            definitions = [
                dict(
                    def_id=d.def_id,
                    mention_id=c.mention_id,
                    paper_id=c.paper_id,
                    text=d.text,
                )
                for c in cs
                for d in c.definitions
            ]
            n, u = len(definitions), len({d["text"] for d in definitions})
            pairs_all += n * (n - 1)
            pairs_unique += u * (u - 1)
            sizes[len(cs)] += 1
            f.write(
                _line(
                    dict(
                        group_id=gid[g],
                        key=key[g],
                        labels=[
                            x for x, _ in Counter(c.label for c in cs).most_common()
                        ],
                        terms=len(cs),
                        papers=len({c.paper_id for c in cs}),
                        n_definitions=n,
                        n_unique_texts=u,
                        **(
                            dict(
                                joined_by=joined_by(
                                    sorted({c.label for c in cs}), traced
                                )
                            )
                            if provenance
                            else {}
                        ),
                        definitions=definitions,
                    )
                )
            )
    if provenance:
        with (out / "labels.jsonl").open("w") as f:
            uses = Counter(labels)
            for label in sorted(traced, key=lambda x: (gid[group_of[x]], x)):
                f.write(
                    _line(
                        dict(
                            group_id=gid[group_of[label]],
                            terms=uses[label],
                            **traced[label],
                        )
                    )
                )
    groups = len(members)
    distinct = len(set(labels))
    summary = dict(
        rule=f"{label_match.RULE} ({rule})",
        view_papers=len(view.papers),
        papers_with_definitions=len({c.paper_id for c in concepts}),
        terms=len(concepts),
        definitions=sum(len(c.definitions) for c in concepts),
        distinct_labels=distinct,
        groups=groups,
        groups_saved_over_distinct_labels=distinct - groups,
        single_term_groups=sizes[1],
        terms_in_multi_term_groups=sum(k * v for k, v in sizes.items() if k > 1),
        groups_with_2plus_definitions=sum(n >= 2 for n in n_defs.values()),
        step_savings=[
            dict(
                step=s,
                name=label_match.STEPS[s],
                groups_without=groups + saved,
                saved=saved,
            )
            for s, saved in sorted(savings.items(), key=lambda kv: -kv[1])
        ],
        nli_pairs_all_ordered=pairs_all,
        nli_pairs_unique_texts_ordered=pairs_unique,
        group_size_terms_histogram={str(k): v for k, v in sorted(sizes.items())},
    )
    (out / "summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n"
    )
    seconds["write"] = time.perf_counter() - t
    return summary, seconds


def run(
    store: Path,
    links: Path,
    out: Path,
    argv: list[str],
    rule: str = label_match.RULE_VERSION,
) -> dict:
    """Build the Label Groups of the corpus view into `out` and record the run in run.json; return the summary."""
    started = datetime.now(UTC)
    t = time.perf_counter()
    view = load_view(store, links)
    load = time.perf_counter() - t
    summary, seconds = write_label_groups(view, out, rule)
    t = time.perf_counter()
    argv = ["label-groups", *argv]
    record = dict(
        command="mbdef postprocess " + " ".join(argv),
        argv=argv,
        started=started.isoformat(timespec="seconds"),
        code=code_version(),
        rule=dict(
            name=label_match.RULE,
            version=rule,
            source_sha256=file_record(Path(label_match.__file__))["sha256"],
        ),
        inputs=dict(store=file_record(store), links=file_record(links)),
        input_counts=dict(
            view_papers=summary["view_papers"],
            papers_with_definitions=summary["papers_with_definitions"],
            extracted_concepts=summary["terms"],
            definitions=summary["definitions"],
        ),
        outputs={
            name: {
                k: v for k, v in file_record(out / name).items() if k != "path"
            }  # next to run.json
            for name in outputs(rule)
            if name != "run.json"
        },
    )
    record["seconds"] = dict(load=load, **seconds, provenance=time.perf_counter() - t)
    record["finished"] = datetime.now(UTC).isoformat(timespec="seconds")
    (out / "run.json").write_text(
        json.dumps(record, indent=1, ensure_ascii=False) + "\n"
    )
    return summary


def label_groups_command(store, links, out, rule, overwrite, argv):
    """`postprocess label-groups`: check the inputs and the output folder, build the Label Groups, print the summary.
    `argv` is the command's own arguments, which run.json records."""
    store, links = store or default_store(), links or default_links()
    for path in (store, links):
        if not path.is_file():
            raise SystemExit(f"input not found: {path}")
    present = [name for name in outputs(rule) if (out / name).exists()]
    if present and not overwrite:
        raise SystemExit(
            f"{out} already holds {', '.join(present)}; pass --overwrite to replace them"
        )
    summary = run(store, links, out, argv, rule)
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "group_size_terms_histogram"},
            indent=1,
            ensure_ascii=False,
        )
    )
    print(f"wrote {out}")
    return 0
