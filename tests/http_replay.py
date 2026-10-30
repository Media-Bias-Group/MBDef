"""Recorded HTTP answers on a local port: the only network the collect tests use.

`Replay` serves the answers a test records, keyed by method and path (`"GET /v2/10.1234%2Fa"`), on 127.0.0.1. A request it has no answer for gets HTTP 599 and is listed in `replay.misses`, so a test can assert that nothing unrecorded was asked. `NO_NETWORK` points every HTTP proxy at a closed local port (as the cluster's GROBID job did), so a request to any other host fails instead of reaching the internet.
"""

import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qsl, urlsplit

CLOSED_PORT = "http://127.0.0.1:9"
NO_NETWORK = {
    name: CLOSED_PORT
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy")
} | {"NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}


@dataclass
class Answer:
    status: int = 200
    body: bytes | str | dict | list = b""
    headers: dict[str, str] = field(default_factory=dict)

    def payload(self) -> tuple[bytes, str]:
        if isinstance(self.body, (dict, list)):
            return json.dumps(self.body).encode(), "application/json"
        if isinstance(self.body, str):
            return self.body.encode(), "text/plain"
        return self.body, "application/octet-stream"


@dataclass
class Request:
    method: str
    path: str
    query: dict[str, str]
    headers: dict[str, str]
    body: bytes


class Replay:
    """`answer(key, *answers)` records answers for one key; they are served in order, each once.

    An answer may be a function of the `Request` (for example a fake GROBID that converts each PDF it is sent); a function answers every request for its key.
    """

    def __init__(self):
        self.answers: dict[str, list] = {}
        self.requests: list[Request] = []
        self.misses: list[str] = []
        replay = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: object) -> None:
                pass

            def handle(self):
                try:
                    super().handle()
                except (
                    ConnectionError
                ):  # the client gave up on a response, e.g. one too large
                    pass

            def _serve(self):
                size = int(self.headers.get("Content-Length") or 0)
                parts = urlsplit(self.path)
                request = Request(
                    method=self.command,
                    path=parts.path,
                    query=dict(parse_qsl(parts.query)),
                    headers={k.lower(): v for k, v in self.headers.items()},
                    body=self.rfile.read(size),
                )
                key = f"{self.command} {parts.path}"
                with lock:
                    replay.requests.append(request)
                    queue = replay.answers.get(key, [])
                    if queue and callable(queue[0]):
                        answer = queue[0](request)
                    elif queue:
                        answer = queue.pop(0)
                    else:
                        replay.misses.append(key)
                        answer = Answer(599, f"no recorded answer for {key}")
                body, content_type = answer.payload()
                self.send_response(answer.status)
                headers = {"Content-Type": content_type, **answer.headers}
                for name, value in headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            do_GET = do_POST = _serve

        lock = threading.Lock()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def answer(self, key: str, *answers) -> None:
        self.answers.setdefault(key, []).extend(answers)

    def asked(self, key: str) -> list[Request]:
        method, path = key.split(" ", 1)
        return [r for r in self.requests if r.method == method and r.path == path]

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
