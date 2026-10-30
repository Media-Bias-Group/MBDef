"""The miniature inputs of basal's scorer (tests/fixtures/basal-mini/) and a stand-in for its model server.

- `make_store`: a corpus store in the store's schema, from the made-up `inputs/store.json` (synthetic text only).
- `StandIn`: answers `POST /v1/systemone` from recorded replies, on this machine. The real server (docs/step-5-post-processing/postprocess.md, "Rerun basal") needs a large GPU. The stand-in knows each item's state (from an items file) and gives, for each request, the next recorded reply of that item (the last one again once they are used up). It keeps every request body it received, as text, in arrival order.

tests/test_postprocess_basal.py uses them; the expected files are the outputs of the original implementation on the same store and the same recorded replies.
"""

import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MINI = Path(__file__).parent / "fixtures/basal-mini"


def make_store(path: Path, source: Path = MINI / "inputs/store.json") -> Path:
    """Write the corpus store described by `source` (papers, mentions, occurrences with their sources) to `path`."""
    spec = json.loads(Path(source).read_text(encoding="utf-8"))
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE papers(paper_id TEXT PRIMARY KEY, title TEXT, doi TEXT, corpus TEXT);
        CREATE TABLE mentions(mention_id TEXT PRIMARY KEY, paper_id TEXT, term_loose TEXT, display_term TEXT,
                              base_key TEXT, acronym TEXT, first_seen TEXT);
        CREATE TABLE occurrences(occ_id TEXT PRIMARY KEY, mention_id TEXT, paper_id TEXT, field TEXT, text TEXT,
                                 text_loose TEXT, citations TEXT);
        CREATE TABLE occurrence_sources(occ_id TEXT, source_id TEXT, record_index INT, item_index INT, raw_text TEXT,
                                        raw_citations TEXT, PRIMARY KEY(occ_id, source_id, record_index, item_index));
    """)
    db.executemany("INSERT INTO papers(paper_id, corpus) VALUES (?, ?)", spec["papers"])
    db.executemany(
        "INSERT INTO mentions(mention_id, paper_id, display_term) VALUES (?, ?, ?)",
        spec["mentions"],
    )
    paper = {m: p for m, p, _ in spec["mentions"]}
    for occ, m, field, text, sources in spec["occurrences"]:
        db.execute(
            "INSERT INTO occurrences(occ_id, mention_id, paper_id, field, text) VALUES (?, ?, ?, ?, ?)",
            (occ, m, paper[m], field, text),
        )
        db.executemany(
            "INSERT INTO occurrence_sources(occ_id, source_id, record_index, item_index) VALUES (?, ?, 0, 0)",
            [(occ, s) for s in sources],
        )
    db.executemany(  # a store spec may name the papers' titles
        "UPDATE papers SET title = ? WHERE paper_id = ?",
        [(t, p) for p, t in spec.get("titles", {}).items()],
    )
    db.commit()
    db.close()
    return path


def state_key(state) -> str:
    return json.dumps(state, ensure_ascii=False, sort_keys=True)


class StandIn:
    """`with StandIn(items_file, replies_file) as server:` serves on `server.url`; `server.received` holds the bodies."""

    def __init__(self, items: Path, replies: Path):
        ids = {
            state_key(x["state"]): x["id"]
            for x in map(
                json.loads, Path(items).read_text(encoding="utf-8").splitlines()
            )
        }
        queue = {
            r["id"]: list(r["replies"])
            for r in map(
                json.loads, Path(replies).read_text(encoding="utf-8").splitlines()
            )
        }
        self.received: list[str] = []
        lock = threading.Lock()
        received = self.received

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 (the name http.server calls)
                body = self.rfile.read(int(self.headers["Content-Length"])).decode()
                if self.path != "/v1/systemone":
                    self.send_error(404)
                    return
                with lock:
                    received.append(body)
                    replies = queue[ids[state_key(json.loads(body)["state"])]]
                    reply = replies.pop(0) if len(replies) > 1 else replies[0]
                data = json.dumps(reply).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format: str, *args: object) -> None:  # quiet
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
