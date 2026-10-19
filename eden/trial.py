"""Bounded research runner with progress reports and durable rolling checkpoints."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import signal
import time
import hashlib

from .cli import writer_lock
from .config import Config
from .engine import Engine
from .store import Store


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    with temp.open("w") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)
        file.flush()
        os.fsync(file.fileno())
    temp.replace(path)


def report(engine, reason, wall_seconds):
    s = engine.state
    agents = [{"id": a["id"], "name": a["name"], "alive": a["alive"], "position": a["pos"],
               "needs": a["needs"], "conditions": a["conditions"], "activations": a["activations"],
               "reported_deliberation": a["deliberation"], "last_result": a["last_result"],
               "files": a["files"], "death_reason": a.get("death_reason")} for a in s["entities"].values() if a["kind"] == "agent"]
    with engine.store.lock:
        events = dict(engine.store.db.execute("SELECT kind,count(*) FROM events GROUP BY kind"))
        calls = dict(engine.store.db.execute("SELECT role,count(*) FROM calls GROUP BY role"))
        failed_calls = engine.store.db.execute("SELECT count(*) FROM calls WHERE error IS NOT NULL").fetchone()[0]
    return {"status": reason, "revision": engine.revision, "sim_seconds": s["time"], "sim_days": s["time"] / 86400,
            "wall_seconds": wall_seconds, "error": engine.last_error, "agents": agents,
            "events": events, "calls": calls, "failed_calls": failed_calls, "facts": len(s["facts"]),
            "new_types": s["types"], "pending_events": len(s["queue"])}


def checkpoint(engine, archive):
    # One self-contained SQLite file; replace only after backup has completed.
    destination = archive / f".checkpoint-{os.getpid()}-{time.time_ns()}.sqlite"
    engine.store.backup(destination)
    with destination.open("rb") as file:
        os.fsync(file.fileno())
    destination.replace(archive / "latest.sqlite")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--until", type=float, default=3 * 86400)
    parser.add_argument("--max-events", type=int, default=30000)
    parser.add_argument("--checkpoint-seconds", type=float, default=60)
    parser.add_argument("--max-wall-seconds", type=float, default=0, help="Stop cleanly after this wall-time budget; zero disables")
    parser.add_argument("--intervention-note", help="Record an explicit operator change before resuming")
    args = parser.parse_args()
    archive = Path(args.archive)
    archive.mkdir(parents=True, exist_ok=True)
    with writer_lock(args.run):
        existing = (Path(args.run) / "world.sqlite").exists()
        engine = Engine(args.run, None if existing else Config.load(args.config))
        if args.intervention_note:
            provenance = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(Path(__file__).parent.glob("*.py"))}
            engine.state.setdefault("operator_interventions", []).append({"time": engine.state["time"],
                "note": args.intervention_note, "source_sha256": provenance})
            engine.commit("operator_intervention", payload={"note": args.intervention_note,
                          "source_sha256": provenance, "previous_revision": engine.revision})
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
            signal.signal(sig, lambda *_: engine.stop())
        started = last_checkpoint = time.monotonic()
        reason, count = "running", 0
        checkpoint(engine, archive)
        atomic_json(archive / "progress.json", report(engine, reason, 0))
        try:
            while not engine.stopped.is_set() and count < args.max_events:
                if args.max_wall_seconds and time.monotonic() - started >= args.max_wall_seconds:
                    reason = "wall_limit"
                    break
                if not engine.state["queue"] or engine.state["queue"][0]["time"] > args.until:
                    reason = "target_reached"
                    break
                if not engine.step():
                    reason = "extinct"
                    break
                count += 1
                if engine.last_error:
                    reason = "inference_paused"
                    break
                if time.monotonic() - last_checkpoint >= args.checkpoint_seconds or count % 25 == 0:
                    checkpoint(engine, archive)
                    value = report(engine, reason, time.monotonic() - started)
                    atomic_json(archive / "progress.json", value)
                    print(json.dumps({k: value[k] for k in ("revision", "sim_seconds", "sim_days", "calls", "failed_calls")}), flush=True)
                    last_checkpoint = time.monotonic()
            else:
                reason = "interrupted" if engine.stopped.is_set() else "event_limit"
        except Exception:
            reason = "error"
            raise
        finally:
            checkpoint(engine, archive)
            result = report(engine, reason, time.monotonic() - started)
            result["verification"] = engine.store.verify()
            atomic_json(archive / "progress.json", result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            engine.store.close()
        if reason == "inference_paused":
            raise SystemExit(2)


if __name__ == "__main__":
    main()
