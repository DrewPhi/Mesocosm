"""Local spectator server: saved snapshots stay readable during model inference."""
import json
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import terrain


def serve(engine, host="127.0.0.1", port=8765, readonly=False):
    token = secrets.token_urlsafe(32)
    static = Path(__file__).parent / "static"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, body, status=200, content_type="application/json"):
            raw = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if not self.valid_host():
                return self.send({"error": "Loopback Host required"}, 403)
            try:
                parsed = urlparse(self.path)
                query = parse_qs(parsed.query)
                path = parsed.path
                if path in {"/", "/app.js", "/style.css"}:
                    name = "index.html" if path == "/" else path[1:]
                    mime = {"index.html": "text/html; charset=utf-8", "app.js": "application/javascript", "style.css": "text/css"}[name]
                    return self.send((static / name).read_bytes(), content_type=mime)
                if path == "/api/session":
                    return self.send({"admin_token": None if readonly else token, "readonly": readonly})
                if path == "/api/state":
                    at = int(query["at"][0]) if "at" in query else None
                    revision, state = engine.store.latest(at)
                    if state is None:
                        raise ValueError("No recorded event at that cursor")
                    state = dict(state)
                    # Full observation/referee audit remains in event details; canvas gets compact terrain.
                    agents = [e for e in state["entities"].values() if e["kind"] == "agent"]
                    selected = next((a for a in agents if a["id"] == query.get("follow", [None])[0]), agents[0])
                    x, _, z = selected["pos"]
                    step = 2
                    tiles = []
                    for px in range(int(x) - 24, int(x) + 25, step):
                        for pz in range(int(z) - 24, int(z) + 25, step):
                            cell = terrain.cell(state["seed"], px, pz)
                            cell["y"] = terrain.surface(state, px, pz) - 1
                            tiles.append(cell)
                    return self.send({"revision": revision, "head": engine.store.latest()[0], "state": state,
                                      "terrain": tiles, "status": "replay" if readonly else engine.status,
                                      "paused": engine.paused, "error": engine.last_error})
                if path == "/api/events":
                    return self.send(engine.store.events(int(query.get("after", [0])[0]), int(query.get("limit", [200])[0]), int(query["through"][0]) if "through" in query else None))
                if path == "/api/calls":
                    return self.send(engine.store.calls())
                if path == "/api/verify":
                    return self.send(engine.store.verify())
                return self.send({"error": "Not found"}, 404)
            except (ValueError, KeyError, TypeError) as exc:
                self.send({"error": str(exc)}, 400)

        def do_POST(self):
            if not self.valid_host():
                return self.send({"error": "Loopback Host required"}, 403)
            if readonly or self.headers.get("X-Eden-Token") != token:
                return self.send({"error": "Operator token required"}, 403)
            # No CORS. Require same-origin browser requests when Origin is present.
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc != self.headers.get("Host"):
                return self.send({"error": "Cross-origin control refused"}, 403)
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 <= length <= 4096:
                    raise ValueError("Request too large")
                body = json.loads(self.rfile.read(length) or b"{}")
                path = urlparse(self.path).path
                if path == "/admin/pause":
                    engine.pause()
                elif path == "/admin/resume":
                    engine.resume()
                elif path == "/admin/step":
                    engine.request_step()
                elif path == "/admin/speed":
                    engine.configure(speed=float(body["speed"]))
                elif path == "/admin/checkpoint":
                    destination = engine.store.directory / "checkpoints" / f"event-{engine.store.latest()[0]}-{secrets.token_hex(3)}.sqlite"
                    engine.store.backup(destination)
                    return self.send({"checkpoint": str(destination)})
                else:
                    return self.send({"error": "Unknown command"}, 404)
                return self.send({"ok": True})
            except (ValueError, KeyError, TypeError) as exc:
                self.send({"error": str(exc)}, 400)

        def valid_host(self):
            try:
                return urlparse("http://" + self.headers.get("Host", "")).hostname in {"localhost", "127.0.0.1", "::1"}
            except ValueError:
                return False

    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Bind to loopback; use an SSH tunnel for remote access")
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server
