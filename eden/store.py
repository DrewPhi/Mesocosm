"""SQLite is authoritative, including private Markdown and its complete history."""
import hashlib
import json
import sqlite3
import threading
import zlib
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def decode_state(raw):
    return json.loads(zlib.decompress(raw) if isinstance(raw, bytes) else raw)


class Store:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.directory / "world.sqlite", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY, time REAL NOT NULL, kind TEXT NOT NULL,
                actor TEXT, payload TEXT NOT NULL, state TEXT NOT NULL,
                state_hash TEXT NOT NULL, prev_hash TEXT NOT NULL, hash TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS calls (
                id INTEGER PRIMARY KEY, time REAL, actor TEXT, role TEXT,
                request TEXT, response TEXT, error TEXT, latency REAL);
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
        ''')
        self.db.commit()

    def close(self):
        with self.lock:
            self.db.close()

    def latest(self, at=None):
        with self.lock:
            row = self.db.execute("SELECT * FROM events " + ("WHERE id <= ? " if at is not None else "") + "ORDER BY id DESC LIMIT 1", () if at is None else (at,)).fetchone()
            return (row["id"], decode_state(row["state"])) if row else (0, None)

    def commit(self, expected, state, kind, actor=None, payload=None):
        with self.lock, self.db:
            self.db.execute("BEGIN IMMEDIATE")
            row = self.db.execute("SELECT id, hash FROM events ORDER BY id DESC LIMIT 1").fetchone()
            actual = row["id"] if row else 0
            if actual != expected:
                raise ValueError("Stale world revision; reload before committing")
            event_id = actual + 1
            previous = row["hash"] if row else "0" * 64
            state_hash = digest(state)
            body = {"id": event_id, "time": float(state["time"]), "kind": kind, "actor": actor,
                    "payload": payload or {}, "state_hash": state_hash, "prev_hash": previous}
            event_hash = digest(body)
            self.db.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?)", (event_id, state["time"], kind, actor, canonical(payload or {}), zlib.compress(canonical(state).encode(), level=3), state_hash, previous, event_hash))
            return event_id

    def events(self, after=0, limit=200, through=None):
        with self.lock:
            rows = self.db.execute("SELECT id,time,kind,actor,payload,state_hash,prev_hash,hash FROM events WHERE id>? AND id<=? ORDER BY id LIMIT ?", (after, through if through is not None else 2**63-1, min(limit, 2000))).fetchall()
        return [{**dict(row), "payload": json.loads(row["payload"])} for row in rows]

    def record_call(self, sim_time, actor, role, request, response, error, latency):
        with self.lock, self.db:
            cur = self.db.execute("INSERT INTO calls(time,actor,role,request,response,error,latency) VALUES (?,?,?,?,?,?,?)", (sim_time, actor, role, canonical(request), canonical(response), error, latency))
            return cur.lastrowid

    def calls(self, limit=50):
        with self.lock:
            rows = self.db.execute("SELECT * FROM calls ORDER BY id DESC LIMIT ?", (min(limit, 200),)).fetchall()
        return [{**dict(r), "request": json.loads(r["request"]), "response": json.loads(r["response"])} for r in rows]

    def verify(self):
        previous = "0" * 64
        count = 0
        last_time = -1
        with self.lock:
            for row in self.db.execute("SELECT * FROM events ORDER BY id"):
                count += 1
                state = decode_state(row["state"])
                body = {key: row[key] for key in ("id", "time", "kind", "actor", "state_hash", "prev_hash")}
                body["payload"] = json.loads(row["payload"])
                if row["id"] != count or row["prev_hash"] != previous or digest(state) != row["state_hash"] or digest(body) != row["hash"] or row["time"] < last_time:
                    raise ValueError(f"History verification failed at event {row['id']}")
                previous, last_time = row["hash"], row["time"]
        return {"events": count, "head_hash": previous, "valid": True}

    def backup(self, target):
        target = Path(target)
        if target.resolve() == (self.directory / "world.sqlite").resolve() or target.exists():
            raise ValueError("Checkpoint destination must be a new file")
        target.parent.mkdir(parents=True, exist_ok=True)
        with self.lock, sqlite3.connect(target) as dest:
            self.db.backup(dest)
