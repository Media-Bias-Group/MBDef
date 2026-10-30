"""`mbdef explorer data`: the MBDef Explorer's page data for exactly what MBDef releases (docs/dataset/explorer.md).

It reads an MBDef dataset folder: its tables and the page data in its folder `explorer/`.
It writes page data that holds only the term mentions of the `instances` table, with only their definitions, fields, concepts and papers.
The relevance filter's dropped mentions, which earlier page data also held, are left out; only their number stays in `meta.json`.

What depends on which mentions are there is computed again on the released ones alone, from the Qwen3-Embedding-8B vectors of their definitions (a mention's vector is the L2-normalised mean of its definitions' vectors, each embedded as "term: definition"):
the map (PCA to 50 dimensions, then UMAP with cosine distance, 30 neighbours and minimum distance 0.15, to 3D and 2D), the topics (k-means, k = 36, on the 50 PCA dimensions, numbered by size and named by their most frequent concepts),
every mention's 8 nearest mentions (exact cosine), every concept's spread, and the random-pair and nearest-neighbour baselines.
These are the settings of the first map, which placed all corpus mentions before the relevance filter, with another seed for k-means (TOPIC_SEED, chosen among 30 for its media-bias topics); the seeds are fixed, so a second run writes the same files.
A concept's name and its merge trace are the dataset's (`concepts.label`, `concepts.joined_by`); the rest (scores, fields, merged views, links) is copied from the page data, with its indices renumbered.

The output is checked as `explorer build` checks page data (`explorer.disagreements`), and every stored neighbour similarity of two released mentions must come out of the vectors again; nothing is written when a check fails.
"""

import hashlib
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

from mbdef import explorer
from mbdef.embed import definition_text, text_hash

SEED = 20261005  # the seed of the first map (PCA and UMAP)
# k-means' seed: of 30 seeds tried, one of the five where a topic is led by the concept "media bias", and the one whose
# topics follow the definitions most closely (the nearest mention in the same topic for 72.2% of the mentions)
TOPIC_SEED = 20261031
BASELINE_SEED = 20261022  # the random pairs of the similarity baseline
TOPICS = 36
PCA_DIMENSIONS = 50
UMAP = dict(n_neighbors=30, min_dist=0.15, metric="cosine")
NEIGHBOURS = 8
RANDOM_PAIRS = 200_000
QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
TERMS_SHARD = 2000  # label groups per shard of terms/
# a stored neighbour similarity (×1000, rounded) and the one the vectors give may differ by rounding and float16 storage
NEIGHBOUR_TOLERANCE = 0.0015
CHUNK = 4096


def dump(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def shards(column: list, size: int) -> list[list]:
    return [column[s : s + size] for s in range(0, len(column), size)]


def mention_vectors(
    definitions: pl.DataFrame, ids: list[str], cache: Path
) -> np.ndarray:
    """One row per mention of IDS: the L2-normalised mean of its definitions' vectors, from the cache in CACHE."""
    keys = (cache / "keys.txt").read_text().split("\n")
    index = {k: i for i, k in enumerate(keys)}
    matrix = np.load(cache / "vectors.npy", mmap_mode="r")
    texts = [
        definition_text(t, x)
        for t, x in zip(definitions["term"], definitions["text"], strict=True)
    ]
    missing = next((t for t in texts if text_hash(t) not in index), None)
    if missing is not None:
        raise SystemExit(f"No page data: {cache} has no vector for {missing[:80]!r}.")
    rows = np.array([index[text_hash(t)] for t in texts], dtype=np.int64)
    row_of = {m: i for i, m in enumerate(ids)}
    owner = np.array([row_of[m] for m in definitions["mention_id"]])
    X = np.zeros((len(ids), matrix.shape[1]), np.float32)
    order = np.argsort(rows)
    for s in range(0, len(order), 20000):  # read the memory map in file order
        o = order[s : s + 20000]
        np.add.at(X, owner[o], matrix[rows[o]].astype(np.float32))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def cosines(X: np.ndarray):
    """A function of (rows, columns) giving the cosine matrix X[rows] @ X[columns].T, on the GPU when there is one."""
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        T = torch.from_numpy(X).to(device)
        return lambda a, b: (T[a] @ T[b].T).cpu().numpy()
    except ImportError:
        return lambda a, b: X[a] @ X[b].T


def nearest(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Every row's K most similar other rows by exact cosine, most similar first, and their cosines."""
    n = len(X)
    cos = cosines(X)
    every = np.arange(n)
    index, value = np.zeros((n, k), np.int64), np.zeros((n, k), np.float32)
    for s in range(0, n, CHUNK):
        rows = every[s : s + CHUNK]
        sim = cos(rows, every)
        sim[np.arange(len(rows)), rows] = -2
        top = np.argpartition(-sim, k, axis=1)[:, :k]
        part = np.take_along_axis(sim, top, axis=1)
        order = np.lexsort((top, -part), axis=1)  # by similarity, ties by index
        index[rows] = np.take_along_axis(top, order, axis=1)
        value[rows] = np.take_along_axis(part, order, axis=1)
    return index, value


def layout(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The 3D and 2D map and the topic of every row (0 = the largest topic)."""
    import umap
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from threadpoolctl import threadpool_limits

    # k-means on float32 coordinates gives other topics from run to run with the same seed (its threads sum in any order); in float64 on one thread it gives the same
    Z = (
        PCA(n_components=PCA_DIMENSIONS, random_state=SEED)
        .fit_transform(X)
        .astype(np.float64)
    )
    with threadpool_limits(1):
        labels = (
            KMeans(n_clusters=TOPICS, n_init=4, random_state=TOPIC_SEED).fit(Z).labels_
        )
    by_size = [
        t for t, _ in sorted(Counter(labels).items(), key=lambda x: (-x[1], x[0]))
    ]
    topic = np.array([by_size.index(t) for t in range(TOPICS)])[labels]
    maps = [
        umap.UMAP(
            n_components=d, random_state=SEED, low_memory=True, **UMAP
        ).fit_transform(Z)
        for d in (3, 2)
    ]

    def scaled(
        Y,
    ):  # centred on the median, the 99.5th percentile of |coordinate| at 1000
        Y = Y.astype(np.float64) - np.median(Y, 0)
        return np.round(1000 * Y / np.percentile(np.abs(Y), 99.5)).astype(int)

    return scaled(maps[0]), scaled(maps[1]), topic


def plain(label: str) -> str:
    """A concept's label for a topic name: a capital that only starts the label is lowered ("Metaphor" -> "metaphor")."""
    return label[0].lower() + label[1:] if label[1:] == label[1:].lower() else label


def topics(topic: np.ndarray, concept_labels: list[str]) -> list[dict]:
    """Each topic's id, name (its three most frequent concepts), size and four most frequent concepts."""
    out = []
    for t in range(TOPICS):
        count = Counter(concept_labels[i] for i in np.flatnonzero(topic == t))
        top = [plain(c) for c, _ in count.most_common(4)]
        out.append(
            dict(id=t, name=" · ".join(top[:3]), size=int(count.total()), top=top)
        )
    return out


def spread(X: np.ndarray, members: list[list[int]]) -> list[tuple | None]:
    """Per group of two or more rows: mean pairwise cosine, mean and least cosine to the normalised centroid, the row farthest from it."""
    out: list[tuple | None] = []
    for ix in members:
        if len(ix) < 2:
            out.append(None)
            continue
        v = X[ix].astype(np.float64)
        c = v.sum(0)
        n = len(ix)
        pair = float((c @ c - n) / (n * (n - 1)))
        to_c = v @ (c / np.linalg.norm(c))
        out.append(
            (pair, float(to_c.mean()), float(to_c.min()), ix[int(np.argmin(to_c))])
        )
    return out


def baseline(X: np.ndarray, nn1: np.ndarray) -> dict:
    rng = np.random.default_rng(BASELINE_SEED)
    a, b = rng.integers(0, len(X), RANDOM_PAIRS), rng.integers(0, len(X), RANDOM_PAIRS)
    a, b = a[a != b], b[a != b]
    rand = np.einsum("ij,ij->i", X[a], X[b])
    q = lambda x: {str(p): float(np.quantile(x, p)) for p in QUANTILES}  # noqa: E731
    return dict(
        random_pair_cos_quantiles=q(rand),
        random_pair_cos_mean=float(rand.mean()),
        nearest_neighbour_cos_quantiles=q(nn1),
        nearest_neighbour_cos_mean=float(nn1.mean()),
    )


def released(dataset: Path, cache: Path, out: Path) -> int:
    """Write to OUT the page data of the MBDef dataset in DATASET restricted to its released mentions (module docstring)."""
    t0 = time.time()
    bundle = explorer.checked_bundle(dataset / explorer.BUNDLE)
    tables = dataset / explorer.TABLES
    meta = json.loads(bundle["meta.json"])
    m = explorer.mentions(bundle, meta)
    defs = explorer.definitions(bundle, meta)
    shard = meta["shard"]
    old_n = len(m["mention_id"])
    fields = [
        f
        for s in range(-(-old_n // shard))
        for f in json.loads(bundle[f"fields/{s:02d}.json"])
    ]
    groups, papers = (
        json.loads(bundle["groups.json"]),
        json.loads(bundle["papers.json"]),
    )
    merges, relations = (
        json.loads(bundle["merges.json"]),
        json.loads(bundle["relations.json"]),
    )
    ref = json.loads(bundle["ensemble-ref.json"])
    n_groups = len(groups["name"])
    terms = [
        t
        for s in range(-(-n_groups // TERMS_SHARD))
        for t in json.loads(bundle[f"terms/{s:02d}.json"])
    ]

    instances = pl.read_parquet(tables / "instances.parquet")
    table_defs = pl.read_parquet(tables / "definitions.parquet")
    concepts = pl.read_parquet(tables / "concepts.parquet")
    in_table = set(instances["mention_id"])
    keep = [i for i, mid in enumerate(m["mention_id"]) if mid in in_table]
    if len(keep) != len(in_table):
        raise SystemExit(
            f"No page data: {len(in_table) - len(keep):,} mentions of instances are not in the page data."
        )
    dropped = old_n - len(keep) + meta["counts"].get("dropped", 0)
    new_of = {old: new for new, old in enumerate(keep)}
    N = len(keep)
    ids = [m["mention_id"][i] for i in keep]
    labels = [m["label"][i] for i in keep]

    # label groups and papers that keep a released mention, in their old order
    old_groups = sorted({m["group"][i] for i in keep})
    group_of = {g: k for k, g in enumerate(old_groups)}
    old_papers = sorted({m["paper"][i] for i in keep})
    paper_of = {p: k for k, p in enumerate(old_papers)}
    group = [group_of[m["group"][i]] for i in keep]
    paper = [paper_of[m["paper"][i]] for i in keep]
    members: list[list[int]] = [[] for _ in old_groups]
    for i, g in enumerate(group):
        members[g].append(i)
    by_paper: list[list[int]] = [[] for _ in old_papers]
    for i, p in enumerate(paper):
        by_paper[p].append(i)
    concept_of = {
        frozenset(ms): row for row, ms in enumerate(concepts["mention_ids"].to_list())
    }
    concept_row = []
    for ms in members:
        row = concept_of.get(frozenset(ids[i] for i in ms))
        if row is None:
            raise SystemExit(
                f"No page data: the released mentions of label group '{groups['name'][old_groups[len(concept_row)]]}' are no concept of concepts."
            )
        concept_row.append(row)
    names = [concepts["label"][r] for r in concept_row]

    # the vectors and everything computed from them
    X = mention_vectors(table_defs, ids, cache)
    print(
        f"{N:,} released mentions, {table_defs.height:,} definitions embedded ({time.time() - t0:.0f}s)",
        flush=True,
    )
    stored = [
        (new_of[i], new_of[j], m["nbs"][i * NEIGHBOURS + k] / 1000)
        for i in keep
        for k in range(NEIGHBOURS)
        if (j := m["nb"][i * NEIGHBOURS + k]) in new_of
    ]
    a, b, want = (np.array(x) for x in zip(*stored, strict=True))
    worst = float(np.abs(np.einsum("ij,ij->i", X[a], X[b]) - want).max())
    if worst > NEIGHBOUR_TOLERANCE:
        raise SystemExit(
            f"No page data: the vectors in {cache} are not those the page data was made with "
            f"(a stored neighbour similarity is off by {worst:.4f})."
        )
    nb, nbs = nearest(X, NEIGHBOURS)
    print(f"nearest mentions ({time.time() - t0:.0f}s)", flush=True)
    xyz, xy, topic = layout(X)
    print(f"map and topics ({time.time() - t0:.0f}s)", flush=True)
    concept_label = [names[g] for g in group]
    sp = spread(X, members)

    # mentions/, defs/, fields/
    column = lambda key: [m[key][i] for i in keep]  # noqa: E731
    table = dict(
        label=labels,
        paper=paper,
        group=group,
        ndef=column("ndef"),
        c=column("c"),
        b=column("b"),
        j=column("j"),
        s=column("s"),
        strict=column("strict"),
        xyz=xyz.ravel().tolist(),
        xy=xy.ravel().tolist(),
        nb=nb.ravel().tolist(),
        nbs=np.round(nbs * 1000).astype(int).ravel().tolist(),
        steps=column("steps"),
        mention_id=ids,
        topic=topic.tolist(),
    )
    per_row = dict(xyz=3, xy=2, nb=NEIGHBOURS, nbs=NEIGHBOURS)
    files: dict[str, bytes] = {}
    for s in range(0, N, shard):
        part = {"n": min(shard, N - s)}
        for key, col in table.items():
            w = per_row.get(key, 1)
            part[key] = col[s * w : (s + shard) * w]
        files[f"mentions/{s // shard:02d}.json"] = dump(part)
    for name, col in (
        ("defs", [defs[i] for i in keep]),
        ("fields", [fields[i] for i in keep]),
    ):
        for s, part in enumerate(shards(col, shard)):
            files[f"{name}/{s:02d}.json"] = dump(part)

    # groups.json
    G = len(old_groups)
    new_g = lambda g: group_of.get(g, -1) if g >= 0 else -1  # noqa: E731
    old_cnb = [
        groups["cnb"][g * NEIGHBOURS + k] for g in old_groups for k in range(NEIGHBOURS)
    ]
    cnb = [new_g(c) for c in old_cnb]
    if any(c < 0 <= o for c, o in zip(cnb, old_cnb, strict=True)):
        raise SystemExit(
            "No page data: a concept's similar concepts include a label group with no released mention."
        )
    s_ens = np.array(table["s"]) / 10000
    files["groups.json"] = dump(
        dict(
            name=names,
            n=[len(ms) for ms in members],
            papers=[len({paper[i] for i in ms}) for ms in members],
            ndef=[sum(table["ndef"][i] for i in ms) for ms in members],
            mean_s=[int(round(float(s_ens[ms].mean()) * 10000)) for ms in members],
            pair=[None if x is None else int(round(x[0] * 1000)) for x in sp],
            to_c=[None if x is None else int(round(x[1] * 1000)) for x in sp],
            min_c=[None if x is None else int(round(x[2] * 1000)) for x in sp],
            far=[None if x is None else x[3] for x in sp],
            labels=[
                dict(Counter(labels[i] for i in ms).most_common(6)) for ms in members
            ],
            cnb=cnb,
            cns=[
                groups["cns"][g * NEIGHBOURS + k]
                for g in old_groups
                for k in range(NEIGHBOURS)
            ],
        )
    )

    # merges.json: every concept with two or more released spellings, its spellings' keys and steps, and the dataset's joined_by
    spelling = {}
    for entry in merges.values():
        for lab, wsk, spk, steps, _ in entry["labels"]:
            spelling[lab] = (wsk, spk, steps)
    new_merges = {}
    for g, ms in enumerate(members):
        count = Counter(labels[i] for i in ms)
        if len(count) < 2:
            continue
        if any(lab not in spelling for lab in count):
            raise SystemExit(
                f"No page data: no merge trace for a spelling of concept '{names[g]}'."
            )
        row = concepts.row(concept_row[g], named=True)
        if set(row["labels"]) != set(count):
            raise SystemExit(
                f"No page data: concept '{names[g]}' has other spellings in concepts than on the page."
            )
        new_merges[str(g)] = dict(
            labels=[[lab, *spelling[lab], n] for lab, n in count.most_common()],
            joined_by=[
                [j["key_type"], j["key"], j["labels"]] for j in row["joined_by"]
            ],
        )
    files["merges.json"] = dump(new_merges)

    # papers.json
    files["papers.json"] = dump(
        dict(
            id=[papers["id"][p] for p in old_papers],
            title=[papers["title"][p] for p in old_papers],
            corpus=[papers["corpus"][p] for p in old_papers],
            n=[len(ix) for ix in by_paper],
            ndef=[sum(table["ndef"][i] for i in ix) for ix in by_paper],
        )
    )

    # terms/ and relations.json: the merged views of the released concepts, their links and sources renumbered
    def pack(item):
        text, n_papers, link, variants, sources = item
        if (link >= 0 and link not in group_of) or any(
            i not in new_of for i in sources
        ):
            raise SystemExit(
                "No page data: a merged view links or cites a mention that is not released."
            )
        return [
            text,
            n_papers,
            new_g(link),
            variants,
            sorted(new_of[i] for i in sources),
        ]

    views = []
    for g in old_groups:
        view = terms[g]
        if view is None:
            raise SystemExit(
                f"No page data: concept '{groups['name'][g]}' has no merged view."
            )
        views.append(
            {
                key: (value if key == "nd" else [pack(it) for it in value])
                for key, value in view.items()
            }
        )
    for s, part in enumerate(shards(views, TERMS_SHARD)):
        files[f"terms/{s:02d}.json"] = dump(part)
    edges = []
    for x, y, f, n in relations["edges"]:
        if x not in group_of or y not in group_of:
            raise SystemExit(
                "No page data: a link joins a label group with no released mention."
            )
        edges.append([group_of[x], group_of[y], f, n])
    files["relations.json"] = dump(dict(relations, edges=edges))

    # ensemble-ref.json: the ranks of the released mentions, in their order
    cut, n2 = ref["cut"], 2 * ref["n_ref"]
    w = {
        k: ref["weights"][name]
        for k, name in zip("cbj", ("classifier", "basal", "judge"), strict=True)
    }
    total = sum(abs(v) for v in w.values())
    ranks = {k: [ref["ranks"][k][i] for i in keep] for k in "cbj"}
    exact = sum((w[k] / total) * np.array(ranks[k]) / n2 for k in "cbj")
    if (exact < cut).any() or (
        (exact >= ref["best_f1_cut"]) != np.array(table["strict"], bool)
    ).any():
        raise SystemExit(
            "No page data: a released mention's exact score misses the cut, or its strict flag disagrees with it."
        )
    checks = dict(
        page_recompute={
            k: v
            for k, v in ref["checks"]["page_recompute"].items()
            if k.startswith("reference_")
        },
        released=dict(
            mentions=N,
            below_cut=int((exact < cut).sum()),
            max_abs_diff_vs_4dp_s=float(np.abs(exact - s_ens).max()),
            closest_to_cut=float((exact - cut).min()),
        ),
    )
    files["ensemble-ref.json"] = dump(
        dict(
            ref,
            ranks=ranks,
            ids_sha256=hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            checks=checks,
            ranks_note="every mention's exact ranks as integers r * 3400 = 2 #{ref < x} + #{ref == x}, in the order of mentions/; "
            "the raw scores there are rounded to 4 decimals, too coarse to rank against the references",
        )
    )

    # meta.json
    fcount = {code: [0, 0] for code in meta["fields"]["codes"].values()}
    for f in (fields[i] for i in keep):
        for code, items in f.items():
            fcount[code][0] += len(items)
            fcount[code][1] += 1
    corpus = papers["corpus"]
    new_meta = dict(
        cut=meta["cut"],
        best_f1_cut=meta["best_f1_cut"],
        weights=meta["weights"],
        counts=dict(
            terms=N,
            dropped=dropped,
            definitions=table_defs.height,
            concepts=G,
            papers=len(old_papers),
            **{
                c: dict(
                    terms=sum(1 for p in paper if corpus[old_papers[p]] == c),
                    papers=sum(1 for p in old_papers if corpus[p] == c),
                )
                for c in ("seeds", "closure")
            },
        ),
        shard=shard,
        concepts=meta["concepts"],
        topics=topics(topic, concept_label),
        baseline=baseline(X, nbs[:, 0]),
        method=explorer.METHOD,
        fields=dict(
            meta["fields"],
            counts=fcount,
            relations=len(edges),
            note=explorer.FIELDS_NOTE,
        ),
    )
    files["meta.json"] = dump(new_meta)

    problems, summary = explorer.disagreements(files, tables)
    if problems:
        raise SystemExit(
            "No page data: it disagrees with the MBDef tables:\n"
            + "\n".join(f"- {p}" for p in problems)
        )
    files = dict(sorted(files.items()))
    files[explorer.SUMS] = "".join(
        f"{explorer.sha256(c)}  {n}\n" for n, c in files.items()
    ).encode()
    staging = out.with_name(out.name + ".part")
    if staging.exists():
        shutil.rmtree(staging)
    for name, content in files.items():
        (staging / name).parent.mkdir(parents=True, exist_ok=True)
        (staging / name).write_bytes(content)
    if out.exists():
        shutil.rmtree(out)
    staging.replace(out)
    size = sum(len(c) for c in files.values()) / 2**20
    print(
        f"The page data agrees with the MBDef tables: {summary}. Left out: {dropped:,} dropped mentions.\n"
        f"Wrote {out}/ ({len(files):,} files, {size:.1f} MB) in {time.time() - t0:.0f}s."
    )
    return 0
