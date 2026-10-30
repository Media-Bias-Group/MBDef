"""The miniature inputs of the GT-RAG judge (tests/fixtures/judge-mini/) and a stand-in for its model server.

`ChatStandIn` answers `POST /v1/chat/completions` from recorded replies, keyed by the sha256 of the request's messages (json.dumps), so a request whose messages differ from the original's gets a 404. It keeps every request body it received.
"""

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MINI = Path(__file__).parent / "fixtures/judge-mini"


class ChatStandIn:
    """`with ChatStandIn(replies_file) as server:` serves on `server.url`; `server.received` holds the bodies."""

    def __init__(self, replies: Path = MINI / "inputs/replies.jsonl"):
        by_messages = {
            r["messages_sha256"]: r["reply"]
            for r in map(json.loads, Path(replies).read_text().splitlines())
        }
        self.received: list[dict] = []
        lock, received = threading.Lock(), self.received

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 (the name http.server calls)
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                key = hashlib.sha256(json.dumps(body["messages"]).encode()).hexdigest()
                with lock:
                    received.append(body)
                if self.path != "/v1/chat/completions" or key not in by_messages:
                    self.send_error(404)
                    return
                data = json.dumps(by_messages[key]).encode()
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
