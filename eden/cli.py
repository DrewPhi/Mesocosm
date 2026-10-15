import argparse
import contextlib
import fcntl
import json
from pathlib import Path
import signal
import sys
import threading

from .config import Config
from .engine import Engine
from .server import serve
from .store import Store, decode_state


@contextlib.contextmanager
def writer_lock(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)
    with (Path(directory) / ".writer.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another Eden writer owns this run; use replay for additional viewers") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def parser():
    p = argparse.ArgumentParser(description="Eden — persistent autonomous world, local model gateway, faithful replay")
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Create/resume a world; choose a model and go")
    run.add_argument("--run", default="runs/eden")
    run.add_argument("--config")
    run.add_argument("--model", help="Served model name; used for both inhabitants and referee by default")
    run.add_argument("--referee-model", help="Optional separate referee model on the same endpoint")
    run.add_argument("--endpoint", help="OpenAI-compatible API base URL ending /v1")
    run.add_argument("--mock", action="store_true", help="Scripted test fixture; no inference server")
    run.add_argument("--response-format", choices=["json_object", "none"])
    run.add_argument("--seed", type=int)
    run.add_argument("--steps", type=int, help="Stop after this many scheduled events")
    run.add_argument("--until", type=float, help="Stop before events beyond this simulation second")
    run.add_argument("--speed", type=float, help="Simulation seconds / real second; 0 = maximum")
    run.add_argument("--max-population", type=int, help="Operational cap; pauses at birth if reached")
    run.add_argument("--serve", action="store_true", help="Open local spectator/control server")
    run.add_argument("--paused", action="store_true", help="Start paused (requires --serve)")
    run.add_argument("--port", type=int, default=8765)
    replay = sub.add_parser("replay", help="View archived history without model calls")
    replay.add_argument("--run", default="runs/eden")
    replay.add_argument("--port", type=int, default=8765)
    for name, description in (("status", "Show run status"), ("verify", "Verify every state and event hash"), ("export", "Export a lossless JSONL history"), ("checkpoint", "Create a consistent SQLite checkpoint")):
        child = sub.add_parser(name, help=description)
        child.add_argument("--run", default="runs/eden")
        if name in {"export", "checkpoint"}:
            child.add_argument("--output", required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return execute(args)
    except (ValueError, OSError) as exc:
        print(f"Eden: {exc}", file=sys.stderr)
        raise SystemExit(2)


def execute(args):
    directory = Path(args.run)
    if args.command != "run" and not (directory / "world.sqlite").is_file():
        raise ValueError("Run does not exist")
    if args.command == "run":
        if args.paused and not args.serve:
            raise ValueError("--paused requires --serve")
        if args.steps is not None and args.steps <= 0:
            raise ValueError("--steps must be positive")
        with writer_lock(directory):
            existing = (directory / "world.sqlite").is_file()
            overrides = {name: getattr(args, name) for name in ("model", "referee_model", "endpoint", "response_format", "speed", "max_population") if getattr(args, name) is not None}
            if args.mock:
                overrides["backend"] = "mock"
            elif args.model or args.endpoint:
                overrides["backend"] = "openai"
            if existing:
                if args.config or args.seed is not None:
                    raise ValueError("Existing run retains its seed/config; choose a new --run for new initial conditions")
                engine = Engine(directory)
                if overrides:
                    engine.configure(**overrides)
            else:
                config = Config.load(args.config)
                for key, value in overrides.items():
                    setattr(config, key, value)
                if args.seed is not None:
                    config.seed = args.seed
                engine = Engine(directory, config)
            return run_engine(engine, args)
    if args.command == "replay":
        engine = Engine(directory)
        server = serve(engine, port=args.port, readonly=True)
        print(f"Replay: http://127.0.0.1:{args.port} (no inference)", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
            engine.store.close()
        return
    store = Store(directory)
    try:
        if args.command == "verify":
            print(json.dumps(store.verify(), indent=2))
        elif args.command == "status":
            revision, state = store.latest()
            print(json.dumps({"revision": revision, "sim_time": state["time"], "queued_events": len(state["queue"]),
                "model": state["config"]["model"], "backend": state["config"]["backend"],
                "agents": [{"name": e["name"], "alive": e["alive"], "position": e["pos"]} for e in state["entities"].values() if e["kind"] == "agent"]}, indent=2))
        elif args.command == "checkpoint":
            store.backup(args.output)
            print(args.output)
        elif args.command == "export":
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x") as file, store.lock:
                store.db.execute("BEGIN")
                # State snapshots include every private-file revision; calls include exact I/O.
                for row in store.db.execute("SELECT * FROM events ORDER BY id"):
                    value = dict(row)
                    value["state"] = decode_state(value["state"])
                    value["payload"] = json.loads(value["payload"])
                    file.write(json.dumps({"record": "event", **value}, ensure_ascii=False) + "\n")
                for row in store.db.execute("SELECT * FROM calls ORDER BY id"):
                    value = dict(row)
                    for key in ("request", "response"):
                        value[key] = json.loads(value[key])
                    file.write(json.dumps({"record": "model_call", **value}, ensure_ascii=False) + "\n")
                store.db.rollback()
            print(args.output)
    finally:
        store.close()


def run_engine(engine, args):
    server = None
    def shutdown(signum=None, frame=None):
        engine.stop()
    previous = {sig: signal.signal(sig, shutdown) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        if args.paused:
            engine.pause()
        if args.serve:
            server = serve(engine, port=args.port)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            print(f"Eden: http://127.0.0.1:{args.port}", flush=True)
        print(f"Run: {engine.store.directory} | backend: {engine.config.backend} | model: {engine.config.model or 'scripted mock'}", flush=True)
        if server:
            engine.run(steps=args.steps, until=args.until)
        else:
            # Headless failures stop with a useful exit code, leaving pending actions intact.
            count = 0
            while not engine.stopped.is_set() and (args.steps is None or count < args.steps):
                if args.until is not None and engine.state["queue"][0]["time"] > args.until:
                    break
                engine.run(steps=1)
                count += 1
                if engine.last_error:
                    raise ValueError("Inference paused: " + engine.last_error + ". Fix the server/config and resume this run.")
                if engine.status == "extinct":
                    break
        print(f"Saved event {engine.revision}; simulated {engine.state['time']:.1f}s; status {engine.status}", flush=True)
    finally:
        if server:
            server.shutdown()
            server.server_close()
        engine.store.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
