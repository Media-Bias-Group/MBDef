"""A miniature closure, as `closure bundle` finds the closure run's files: two seed papers, five cited works.

The world, worked out by hand (the tests assert these numbers, not the code's own):

- **seed-a** cites W1 and W4; **seed-b** cites W1, seed-a (W3), W2 and W5.
- **W1**, "Child work", has an open-access PDF; it was collected, converted and extracted: a closure paper.
- **W2** has no open-access PDF: nothing was collected.
- **W3** is seed-a itself: present in the corpus, never collected.
- **W4** is in the project's existing corpus only (a zip of TEI files): `existing` makes it a closure paper.
- **W5** was collected, but its text is seed-b's text: the release leaves it out.
- The PDF identity check ran in three rounds over the first, second and third of the three candidates of tests/fixtures/synthetic-identity-replay.zip: only the first is accepted.

So the release holds 2 seed papers and 2 closure papers (W1 and W4), and leaves out 1 (W5).
`closure run` writes the world's `before.json` and queue; everything else is written literally below.
"""

import csv
import hashlib
import json
import re
import zipfile
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import quote
from xml.sax.saxutils import escape

from cli_helpers import run_cli, write_env

from mbdef.closure import bundle
from mbdef.collect import download, grobid, open_access, openalex

DOIS = {
    "w1": "10.9876/child",
    "w4": "10.9876/existing",
    "w5": "10.9876/preprint",
}
TITLES = {
    "seed-a": "Seed paper A",
    "seed-b": "Seed paper B",
    "w1": "Child work",
    "w2": "Another work without open access",
    "w4": "A work only in the existing corpus",
    "w5": "Seed paper B as a preprint",
}
SURNAMES = {
    "seed-a": "Alpha",
    "seed-b": "Beta",
    "w1": "Gamma",
    "w2": "Delta",
    "w4": "Epsilon",
    "w5": "Beta",
}
YEARS = {
    "seed-a": "2019",
    "seed-b": "2021",
    "w1": "2020",
    "w2": "2018",
    "w4": "2017",
    "w5": "2021",
}
VARIANTS = {
    "P20": "P>=20",
    "P24": "P>=24",
    "P28": "P>=28",
    "P28-single-model": "P>=28-single-model",
}


def words(topic: str, n: int = 40) -> str:
    """A body of `n` distinct words about `topic` (long enough to be compared by its 8-word sequences)."""
    return " ".join(f"{topic}{i}" for i in range(n))


BODIES = {
    "seed-a": words("alpha"),
    "seed-b": words("beta"),
    "w1": words("gamma"),
    "w4": words("epsilon"),
    "w5": words("beta"),  # the same text as seed-b
}


def reference(bibl: str, key: str) -> str:
    doi = f'<idno type="DOI">{DOIS[key]}</idno>' if key in DOIS else ""
    return (
        f'<biblStruct xml:id="{bibl}"><analytic><title level="a">{escape(TITLES[key])}</title>{doi}'
        f"<author><persName><surname>{SURNAMES[key]}</surname></persName></author></analytic>"
        f'<monogr><imprint><date when="{YEARS[key]}"/></imprint></monogr></biblStruct>'
    )


def tei(key: str, cites: Sequence[str] = ()) -> bytes:
    """A paper's TEI as GROBID writes it: title page (DOI if it has one), body, and the works it cites."""
    refs = " ".join(
        f'<ref type="bibr" target="#b{n}">[{n + 1}]</ref>' for n in range(len(cites))
    )
    listed = "".join(reference(f"b{n}", cited) for n, cited in enumerate(cites))
    doi = f'<idno type="DOI">{DOIS[key]}</idno>' if key in DOIS else ""
    return (
        '<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc>'
        f"<titleStmt><title>{escape(TITLES[key])}</title></titleStmt><sourceDesc><biblStruct><analytic>"
        f'<title level="a" type="main">{escape(TITLES[key])}</title>'
        f"<author><persName><surname>{SURNAMES[key]}</surname></persName></author></analytic>"
        f'<monogr><imprint><date when="{YEARS[key]}"/></imprint></monogr>{doi}'
        f"</biblStruct></sourceDesc></fileDesc></teiHeader><text><body><p>{BODIES[key]} {refs}</p></body>"
        f"<back><listBibl>{listed}</listBibl></back></text></TEI>"
    ).encode()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


BUILT = re.compile(r"(?m)^Built \d{4}-\d\d-\d\dT[\d:]{8}Z\.")
BUILD_TIME_FILES = ("README.md", "provenance/release-facts.json")


def without_build_time(release: Path) -> dict[str, object]:
    """A release folder file by file (each by its sha256), with the build time taken out: the `Built …` sentence of the README, `built_utc` of the facts, and the manifest lines of those two files (their hashes change with it)."""
    found: dict[str, object] = {}
    for path in sorted(release.rglob("*")):
        if not path.is_file():
            continue
        name = path.relative_to(release).as_posix()
        data = path.read_bytes()
        if name == "README.md":
            found[name] = BUILT.sub("Built X.", data.decode("utf-8"))
        elif name == "provenance/release-facts.json":
            facts = json.loads(data)
            facts.pop("built_utc")
            found[name] = facts
        elif name == "MANIFEST.sha256":
            found[name] = [
                line
                for line in data.decode("utf-8").splitlines()
                if line.split("  ", 1)[1] not in BUILD_TIME_FILES
            ]
        else:
            found[name] = sha(data)
    return found


def put(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    elif isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(
            json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return path


def put_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([{f: "" for f in fields} | r for r in rows])


def record(term: str, text: str, citations: list[str]) -> dict:
    """A definition record whose one definition carries `citations` (every other field empty)."""
    return {
        "term": term,
        "definitions": [{"text": text, "citation": citations}],
        "synonyms": [],
        "description": [],
        "related_terms": [],
        "subtypes": [],
        "supertypes": [],
        "examples": [],
    }


def paper(paper_id: str, key: str, records: list[dict]) -> dict:
    return {
        "paper_id": paper_id,
        "metadata": {"doi": DOIS.get(key, ""), "title": TITLES[key]},
        "definition_records": records,
    }


def counts(papers: list[dict]) -> dict:
    recs = [r for p in papers for r in p["definition_records"]]
    return {
        "papers": len(papers),
        "records": len(recs),
        "definitions": sum(len(r["definitions"]) for r in recs),
    }


def derived(folder: Path, papers: list[dict], created: str) -> dict:
    """A run's derived/ folder: the four variants (the same papers; P28 without the first one's first record), their union, per-paper files and DONE.json. Returns DONE.json's content."""
    picks = {}
    for tag, name in VARIANTS.items():
        kept = papers
        if tag == "P28":
            kept = [
                {**p, "definition_records": p["definition_records"][1:]}
                if i == 0
                else p
                for i, p in enumerate(papers)
            ]
        text = json.dumps(
            {"missing_part_outputs": [], "paper_count": len(kept), "papers": kept},
            ensure_ascii=False,
        )
        put(folder / tag / "predictions.json", text + "\n")
        for p in kept:
            put(folder / tag / "papers" / (quote(p["paper_id"], safe="") + ".json"), p)
        picks[name] = {
            **counts(kept),
            "missing_part_outputs": 0,
            "pick": f"test/{tag}",
            "predictions_sha256": sha((text + "\n").encode()),
        }
    union = json.dumps(
        {"paper_count": len(papers), "papers": papers}, ensure_ascii=False
    )
    put(folder / "union" / "predictions.json", union + "\n")
    done = {
        "created_utc": created,
        "papers": len(papers),
        "picks": picks,
        "problems": [],
        "union": {**counts(papers), "predictions_sha256": sha((union + "\n").encode())},
    }
    put(folder / "DONE.json", done)
    return done


class World:
    """The folders of a miniature closure, laid out like the data root's closure inputs."""

    def __init__(self, tmp_path: Path, data_root: Path | None = None):
        """With `data_root`, the folders are where `closure bundle` looks for them by default."""
        self.tmp = tmp_path
        self.root = data_root or tmp_path / "world"
        if data_root:
            self.seeds = data_root / bundle.SEEDS
            self.seed_inputs = data_root / bundle.SEED_INPUTS
            self.v11 = data_root / bundle.V11
            self.ledgers = data_root / bundle.LEDGERS
            self.closure_set = data_root / bundle.CLOSURE_SET
            self.closure_tei = data_root / bundle.CLOSURE_TEI
            self.left_out_tei = data_root / bundle.LEFT_OUT_TEI
            self.identity = data_root / bundle.IDENTITY
        else:
            self.seeds = self.root / "seed-run" / "corpus" / "seeds"
            self.seed_inputs = self.root / "seed-run" / "inputs"
            self.v11 = self.root / "closure-v11"
            self.ledgers = self.v11 / "links-inputs"
            self.closure_set = self.v11 / "extraction" / "unpacked"
            self.closure_tei = self.root / "run" / "grobid"
            self.left_out_tei = self.root / "run" / "grobid-left-out-v11"
            self.identity = self.root / "identity-llm"
        self.run = tmp_path / "closure-run-folder"
        self.loss_audit = tmp_path / "closure-loss-audit.md"
        self.work = tmp_path / "work"
        self.work.mkdir(parents=True)
        write_env(
            self.work, DATA_ROOT=str(self.root if data_root else tmp_path / "data-root")
        )
        self.ids: dict[str, str] = {}
        self.build()

    # ------------------------------------------------------------------------------- the seed run
    def build(self) -> None:
        seed_tei = self.seed_inputs / "seed-tei"
        put(seed_tei / "seed-a.tei.xml", tei("seed-a", ["w1", "w4"]))
        put(seed_tei / "seed-b.tei.xml", tei("seed-b", ["w1", "seed-a", "w2", "w5"]))
        # The extracted definitions of the seed papers (the closure run's input and the seeds' union).
        seeds = [
            paper(
                "seed-a",
                "seed-a",
                [
                    record("bias", "Bias is a tilt [1].", ["[1]"]),
                    record("framing", "Framing selects aspects [2].", ["[2]"]),
                ],
            ),
            paper(
                "seed-b",
                "seed-b",
                [
                    record(
                        "slant", "Slant is a choice of words [1] [2].", ["[1]", "[2]"]
                    ),
                    record("spin", "Spin presents facts [3] [4].", ["[3]", "[4]"]),
                ],
            ),
        ]
        self.dataset = put(
            self.v11 / "links-inputs" / "union-dataset.json",
            {"note": "mini", "paper_count": 2, "papers": seeds},
        )
        done = derived(self.seeds / "derived", seeds, "2026-09-26T19:21:02Z")
        self.seeds_done = done
        put(self.seeds / "compare" / "compare.json", {"pairwise": {}})
        put(
            self.seed_inputs / "runner-rows.json",
            [
                {
                    "paper_id": k,
                    "tei_path": f"/cluster/inputs/seed-tei/{k}.tei.xml",
                    "tei_sha256": sha((seed_tei / f"{k}.tei.xml").read_bytes()),
                    "tei_size_bytes": (seed_tei / f"{k}.tei.xml").stat().st_size,
                    "frozen_title": TITLES[k],
                }
                for k in ("seed-a", "seed-b")
            ],
        )
        # The existing corpus: a zip of TEI files and its sha256 list.
        member = "doi_10.9876_existing.grobid.tei.xml"
        w4 = tei("w4")
        self.seed_inputs.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(
            self.seed_inputs / "existing-corpus-tei.zip", "w"
        ) as archive:
            archive.writestr(member, w4)
        put(self.seed_inputs / "existing-corpus-tei.sha256", f"{sha(w4)}  {member}\n")

        # The closure run: its check report and queue.
        done = run_cli(
            self.work,
            "closure",
            "run",
            "--dataset",
            str(self.dataset),
            "--out",
            str(self.run / "closure-run"),
            "--corpus",
            str(seed_tei),
            "--corpus",
            str(self.seed_inputs / "existing-corpus-tei.zip"),
            "--tei-dir",
            str(seed_tei),
        )
        assert done.returncode == 0, done.stderr
        out = self.run / "closure-run"
        self.ledgers.mkdir(parents=True, exist_ok=True)
        (self.ledgers / "before.json").write_bytes((out / "before.json").read_bytes())
        (self.ledgers / "before-queue.csv").write_bytes(
            (out / "before-queue.csv").read_bytes()
        )
        self.ledger_files()

    def queue_id(self, key: str) -> str:
        """The collect queue's id of work `key`: its paper id and 16 hex digits of its row's sha256."""
        rows = list(
            csv.DictReader((self.ledgers / "before-queue.csv").open(encoding="utf-8"))
        )
        row = next(r for r in rows if r["title"] == TITLES[key])
        return self.ids[key] if key in self.ids else self.compute_id(row)

    @staticmethod
    def compute_id(row: dict) -> str:
        from mbdef.collect.identity import collection_paper_id

        doi = open_access.normalized_queue_doi(row["doi"])
        base = collection_paper_id(
            doi=doi, title=row["title"], year=row["year"], authors=row["authors"]
        )
        entry = {
            "paperid": base,
            "queue_status": row["queue_status"],
            "title": row["title"],
            "authors": row["authors"],
            "year": row["year"],
            "doi": doi,
            "source_url": row["url"].strip(),
        }
        return (
            base
            + "__"
            + sha(json.dumps(entry, sort_keys=True, ensure_ascii=False).encode())[:16]
        )

    # ------------------------------------------------------------------ the collected closure papers
    def ledger_files(self) -> None:
        keys = ("w1", "w2", "w5")
        ids = {k: self.queue_id(k) for k in keys}
        self.ids.update(ids)
        base = {"queue_status": "missing", "attempted_at": "2026-09-26T00:00:00+00:00"}

        def row(k: str) -> dict:
            return base | {
                "paperid": ids[k],
                "title": TITLES[k],
                "authors": f"{SURNAMES[k]}, A.",
                "year": YEARS[k],
                "doi": DOIS.get(k, ""),
            }

        has_pdf = {
            "oa_status": "has_pdf",
            "pdf_url": "https://example.org/x.pdf",
            "status_code": "200",
        }
        none = {"oa_status": "not_open_access", "status_code": "200"}
        put_csv(
            self.ledgers / "provider.csv",
            openalex.STATUS_FIELDS,
            [row("w1") | has_pdf, row("w2") | none, row("w5") | has_pdf],
        )
        put_csv(
            self.ledgers / "unpaywall.csv",
            open_access.STATUS_FIELDS,
            [row("w1") | has_pdf, row("w2") | none, row("w5") | has_pdf],
        )
        got = {
            "downloaded": "True",
            "status_code": "200",
            "pdf_archive_path": "/zip",
            "pdf_bytes": "10",
        }
        put_csv(
            self.ledgers / "download.csv",
            download.FIELDNAMES,
            [
                row("w1") | got | {"pdf_archive_member": "w1.pdf"},
                row("w5") | got | {"pdf_archive_member": "w5.pdf"},
            ],
        )
        # The papers collected, as `collect grobid` leaves them: a TEI with its receipt per work.
        rows_all = []
        for k in ("w1", "w5"):
            # W5's PDF is seed-b's own paper (the corpus check could not tell: the cited title differs).
            data = tei("seed-b", ["w1", "seed-a", "w2", "w5"]) if k == "w5" else tei(k)
            path = grobid.tei_path(
                self.closure_tei, paperid=ids[k], doi=DOIS.get(k, "")
            )
            put(path, data)
            put(
                path.with_suffix(".closure.json"),
                {
                    "schema": "corpus-closure/1",
                    "depth": 1,
                    "paperid": ids[k],
                    "requested_doi": DOIS.get(k, ""),
                    "identity_match": "doi",
                    "pdf_sha256": sha(k.encode()),
                    "tei_sha256": sha(data),
                },
            )
            rows_all.append(self.extraction_row(ids[k], k, path))
        put_csv(
            self.ledgers / "grobid-laptop-snapshot.csv",
            grobid.FIELDNAMES,
            [
                row("w1")
                | {"converted": "True", "status_code": "200", "grobid_xml_path": "x"},
                row("w5")
                | {"converted": "True", "status_code": "200", "grobid_xml_path": "y"},
            ],
        )
        put(self.ledgers / "download-dedup.json", {"duplicates_not_downloaded": []})
        # W4, from the existing corpus: written by the step that adds such works, as in the released closure.
        from mbdef.closure import existing

        before = json.loads((self.ledgers / "before.json").read_text())
        added = existing.add_existing(
            before,
            self.seed_inputs / "existing-corpus-tei.zip",
            self.seed_inputs / "existing-corpus-tei.sha256",
            self.closure_tei,
            set(),
        )
        assert [a["cited_title"] for a in added] == [TITLES["w4"]]
        self.ids["w4"] = added[0]["paper_id"]
        rows_all.append(
            self.extraction_row(self.ids["w4"], "w4", Path(added[0]["tei"]))
        )
        self.left_out_tei.mkdir(parents=True, exist_ok=True)
        self.extraction(rows_all)

    def extraction_row(self, paper_id: str, key: str, path: Path) -> dict:
        return {
            "paper_id": paper_id,
            "tei_path": f"/cluster/run/grobid/{path.parent.name}/{path.name}",
            "tei_sha256": sha(path.read_bytes()),
            "tei_size_bytes": path.stat().st_size,
            "frozen_title": TITLES[key],
        }

    def extraction(self, rows_all: list[dict]) -> None:
        """The closure papers' extraction: rows, derived/, batches, set, picks, metadata, compare/."""
        by_key = {self.ids[k]: k for k in ("w1", "w5", "w4")}
        papers = [
            paper(
                r["paper_id"],
                by_key[r["paper_id"]],
                [record("term", f"A definition of {by_key[r['paper_id']]}.", [])],
            )
            for r in rows_all
        ]
        put(self.closure_set / "rows-all.json", rows_all)
        derived(self.closure_set / "derived", papers, "2026-10-01T10:24:12Z")
        put(
            self.closure_set / "batches.json",
            [{"batch": 1, "excluded": [], "papers": 3, "shards": ["cb001-000"]}],
        )
        put(self.closure_set / "set.json", {"set": "closure", "papers": 3})
        put(self.closure_set / "picks.json", {"picks": []})
        put(
            self.closure_set / "paper-metadata.json",
            {
                r["paper_id"]: {
                    "doi": DOIS.get(by_key[r["paper_id"]], ""),
                    "openalex_id": "",
                    "depth": 1,
                }
                for r in rows_all
            },
        )
        put(self.closure_set / "compare" / "compare.json", {"pairwise": {}})
        # Audit inputs: the candidates compared for duplicate text, what is known of each, nobody uncited.
        put(
            self.v11 / "all-candidates.json",
            [
                {"paper_id": r["paper_id"], "tei": "/workstation" + r["tei_path"]}
                for r in rows_all
            ],
        )
        put(
            self.v11 / "papers-info.json",
            {
                r["paper_id"]: {
                    "doi": DOIS.get(by_key[r["paper_id"]], ""),
                    "n_citing_seed_papers": 2 if by_key[r["paper_id"]] == "w1" else 1,
                    "extracted": True,
                }
                for r in rows_all
            },
        )
        put(self.v11 / "uncited.json", [])
        put(self.v11 / "existing-exclude.json", [])
        put(self.loss_audit, "# Closure loss audit (a frozen text)\n")
        # The cited works as the closure audit grouped them (its cited-works.json): a work with a DOI is keyed by it, any other by its place in the check report; here no two references are the same work.
        before = json.loads((self.ledgers / "before.json").read_text())
        works = []
        for i, p in enumerate(before["papers"]):
            dois = p["reference"]["dois"]
            citing = sorted({o["paper_id"] for o in p["occurrences"]})
            works.append(
                {
                    "work_key": f"doi:{dois[0]}" if dois else f"group:{i}",
                    "n_reference_groups_merged": 1,
                    "n_citing_seed_papers": len(citing),
                    "citing_seed_papers": citing,
                    "n_citing_definitions": len(p["occurrences"]),
                    "n_citing_items": len(p["occurrences"]),
                    "outputs_citing": list(VARIANTS),
                }
            )
        put(self.ledgers / "cited-works.json", works)
        self.identity_rounds()

    # --------------------------------------------------------------------- the PDF identity check
    def identity_rounds(self) -> None:
        """Three rounds over the first, second and third candidate of the synthetic replay fixture."""
        with zipfile.ZipFile(
            Path(__file__).parent / "fixtures" / "synthetic-identity-replay.zip"
        ) as archive:
            evidence = json.loads(archive.read("evidence-5.json"))
            verdicts = json.loads(archive.read("expected-verdicts.json"))
        candidates = [i for i in evidence["items"] if i["kind"] == "candidate"]
        by_id = {v["id"]: v for v in verdicts}
        header = {
            "model": "Qwen/Qwen3.6-35B-A3B",
            "revision": "995ad96eacd98c81ed38be0c5b274b04031597b0",
            "prompt_version": "identity-llm/1",
            "schema_mode": True,
        }
        for folder, item in zip(("", "r2", "r3"), candidates, strict=True):
            where = self.identity / folder
            put(where / "evidence.json", {**evidence, "items": [item]})
            put(where / "verdicts.json", {**header, "verdicts": [by_id[item["id"]]]})
        self.candidate_ids = [c["paperid"] for c in candidates]
