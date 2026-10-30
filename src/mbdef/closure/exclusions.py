"""Closure papers left out of the release because their text is another paper's.

Two steps of `closure bundle`:

1. `find_duplicates`: every TEI is reduced to its words (tags stripped); overlapping 8-word sequences (every 4th word) are hashed. Two papers are the same document when at least `SHARE` (80%) of the smaller paper's sequences occur in the other. The seed papers are indexed once; each candidate (a closure paper) is compared with all of them through the index, and the candidates with each other.
2. `exclusions`: a closure paper with a seed paper's text is left out: the work is a seed paper (the corpus check did not recognise it because the seed's TEI header was garbled). Of two closure papers with the same text, one is kept: the one already extracted, then the one with a published DOI (not an arXiv DOI, not none), then the one cited by more seed papers, then the smaller paper id; the other is left out and its citations count for the kept one. Papers that no seed reference cites (`uncited`: collected through a DOI GROBID had put on the wrong reference entry) are left out too.

A candidate's TEI is found by its file name in the given folders, so the list of candidates need not carry the paths of one machine (the released `all-candidates.json` names absolute paths of the machine that built it; six of its TEIs were moved out of the collected folder after the check).
"""

import collections
import json
import re
import zlib
from pathlib import Path

SHARE = 0.8
SHINGLE_WORDS = 8  # words per hashed sequence
SHINGLE_STEP = 4  # one sequence every this many words


def shingles(path: Path) -> set[int]:
    text = re.sub(r"<[^>]+>", " ", path.read_text(encoding="utf-8", errors="ignore"))
    words = re.findall(r"\w+", text.lower())
    return {
        zlib.crc32(" ".join(words[i : i + SHINGLE_WORDS]).encode())
        for i in range(0, max(0, len(words) - SHINGLE_WORDS), SHINGLE_STEP)
    }


def tei_by_name(folders: list[Path]) -> dict[str, Path]:
    """{file name: path} of every `*.tei.xml` below the folders (the first folder wins a repeated name)."""
    found: dict[str, Path] = {}
    for folder in folders:
        for path in sorted(Path(folder).rglob("*.tei.xml")):
            found.setdefault(path.name, path)
    return found


def find_duplicates(
    seed_tei: Path, candidates: list[dict], tei_folders: list[Path]
) -> dict:
    """The duplicate-text record: each candidate's near-copies among the seeds and among the other candidates.

    `candidates` is [{"paper_id": ..., "tei": path}, ...]; only the file name of `tei` is used, to find the file in `tei_folders`.
    """
    seeds = {
        f.name[: -len(".tei.xml")]: shingles(f)
        for f in sorted(Path(seed_tei).glob("*.tei.xml"))
    }
    index = collections.defaultdict(set)
    for sid, s in seeds.items():
        for h in s:
            index[h].add(sid)
    files = tei_by_name(tei_folders)
    missing = [c["paper_id"] for c in candidates if Path(c["tei"]).name not in files]
    if missing:
        raise SystemExit(
            f"No TEI file in {', '.join(map(str, tei_folders))} for {len(missing)} candidates, "
            f"for example {missing[0]}."
        )
    cand_sets = {
        c["paper_id"]: shingles(files[Path(c["tei"]).name]) for c in candidates
    }
    found = []
    for cid, s in cand_sets.items():
        hits = collections.Counter(sid for h in s for sid in index.get(h, ()))
        for sid, n in hits.most_common(3):
            share = n / max(1, min(len(s), len(seeds[sid])))
            if share >= SHARE:
                found.append(
                    {
                        "paper_id": cid,
                        "duplicate_of": sid,
                        "kind": "seed",
                        "share": round(share, 3),
                    }
                )
    return {
        "share_threshold": SHARE,
        "seeds": len(seeds),
        "candidates": len(candidates),
        "duplicates": found + copies_among(cand_sets),
    }


def copies_among(cand_sets: dict[str, set[int]]) -> list[dict]:
    """Pairs of papers with the same text among the candidates ({paper id: its hashed sequences}, in order): each paper's three best matches that share at least `SHARE` of the smaller paper's sequences, once per pair (the smaller paper id first)."""
    cindex = collections.defaultdict(set)
    for cid, s in cand_sets.items():
        for h in s:
            cindex[h].add(cid)
    found = []
    for cid, s in cand_sets.items():
        hits = collections.Counter(o for h in s for o in cindex.get(h, ()) if o != cid)
        for oid, n in hits.most_common(3):
            share = n / max(1, min(len(s), len(cand_sets[oid])))
            if share >= SHARE and cid < oid:
                found.append(
                    {
                        "paper_id": cid,
                        "duplicate_of": oid,
                        "kind": "closure",
                        "share": round(share, 3),
                    }
                )
    return found


def write_duplicates(path: Path, record: dict) -> None:
    """The released byte format: indent 1, ASCII only, no final newline."""
    Path(path).write_text(json.dumps(record, indent=1))


def rank(pid: str, info: dict) -> tuple:
    """Which copy stays: one already extracted (same text, no second extraction), then a published DOI, then citations."""
    doi = (info.get(pid) or {}).get("doi") or ""
    published = bool(doi) and not doi.lower().startswith("10.48550/")
    return (
        bool((info.get(pid) or {}).get("extracted")),
        published,
        int((info.get(pid) or {}).get("n_citing_seed_papers") or 0),
        [-ord(ch) for ch in pid],
    )


def exclusions(duplicates: list[dict], info: dict, uncited: list[str]) -> list[dict]:
    """The papers the release leaves out, sorted by paper id, each with its reason.

    `duplicates` is the record's list; `info` is {paper_id: {"doi", "n_citing_seed_papers", "extracted"}}; `uncited` lists collected papers that no seed reference cites."""
    out = {}
    for d in duplicates:
        if d["kind"] == "seed":
            out[d["paper_id"]] = {
                "paper_id": d["paper_id"],
                "reason": "same text as a seed paper",
                "seed_paper_id": d["duplicate_of"],
                "share": d["share"],
            }
    for d in duplicates:
        if d["kind"] != "closure" or d["paper_id"] in out or d["duplicate_of"] in out:
            continue
        a, b = d["paper_id"], d["duplicate_of"]
        keep, drop = (a, b) if rank(a, info) >= rank(b, info) else (b, a)
        out[drop] = {
            "paper_id": drop,
            "reason": "same text as another closure paper",
            "kept_paper_id": keep,
            "share": d["share"],
        }
    # a kept paper may itself be left out by another pair: follow to the paper that stays
    for e in out.values():
        seen = set()
        while e.get("kept_paper_id") in out and e["kept_paper_id"] not in seen:
            seen.add(e["kept_paper_id"])
            nxt = out[e["kept_paper_id"]]
            if "seed_paper_id" in nxt:
                e.pop("kept_paper_id")
                e.update(
                    reason="same text as a seed paper",
                    seed_paper_id=nxt["seed_paper_id"],
                )
            else:
                e["kept_paper_id"] = nxt["kept_paper_id"]
    ex = sorted(out.values(), key=lambda e: e["paper_id"])
    have = {e["paper_id"] for e in ex}
    # Collected through a DOI that GROBID had put on the wrong reference entry: once that DOI is ignored, no seed reference cites the paper, so it is not a cited work.
    ex += [
        {
            "paper_id": pid,
            "reason": "not cited: collected through a DOI GROBID had put on the wrong reference entry",
        }
        for pid in uncited
        if pid not in have
    ]
    ex.sort(key=lambda e: e["paper_id"])
    return ex


def write_exclusions(path: Path, rows: list[dict]) -> None:
    """The released byte format: indent 1, non-ASCII kept, a final newline."""
    Path(path).write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n")


def counts(rows: list[dict]) -> dict:
    return {
        "left_out": len(rows),
        "seed_copies": sum("seed_paper_id" in e for e in rows),
        "closure_copies": sum("kept_paper_id" in e for e in rows),
        "not_cited": sum(e["reason"].startswith("not cited") for e in rows),
    }
