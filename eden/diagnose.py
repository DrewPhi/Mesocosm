"""Ask recorded agent contexts diagnostic questions without executing any intents."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
from urllib.parse import quote

from .config import Config
from .inference import Gateway
from .store import Store, decode_state, digest
from .trial import atomic_json


def diagnose(snapshot, config, work, output):
    original_hash = hashlib.sha256(Path(snapshot).read_bytes()).hexdigest()
    # A separate local copy is used as the read source; diagnostics have their own log.
    source = sqlite3.connect("file:" + quote(str(Path(snapshot).resolve())) + "?immutable=1", uri=True)
    source.row_factory = sqlite3.Row
    calls = Store(work)
    results = []
    try:
        latest = source.execute("SELECT id,state FROM events ORDER BY id DESC LIMIT 1").fetchone()
        revision, state = latest["id"], decode_state(latest["state"])
        gateway = Gateway(config, calls)
        for actor, agent in state["entities"].items():
            if agent["kind"] != "agent" or not agent["alive"]:
                continue
            row = source.execute("SELECT id,request FROM calls WHERE actor=? AND role='agent' ORDER BY id DESC LIMIT 1", (actor,)).fetchone()
            if not row:
                continue
            request = json.loads(row["request"])
            context = json.loads(request["messages"][1]["content"])
            visible = context["observation"]["visible"]
            partners = [e for e in visible if e["kind"] == "agent" and e.get("alive") and e.get("distance_m", 1e9) <= 20]
            if not partners:
                continue
            partner = partners[0]
            question = (f"Diagnostic question for {agent['name']}, not an in-world utterance: "
                f"Do you know how to talk to {partner['name']}, who is {partner['distance_m']:g} metres away? "
                "Do you need to move closer before speaking? Explain in your deliberation, and give the "
                "single intent you would use IF you chose to say something. Your proposed intent will "
                "NOT be executed in the simulation. Return the usual deliberation/intent JSON envelope.")
            for variant in ("recorded_context", "minimal_context"):
                selected = context
                if variant == "minimal_context":
                    obs = context["observation"]
                    selected = {"observation": {"self": obs["self"], "visible": [partner],
                        "age_years": obs["age_years"], "coordinate_convention": obs.get("coordinate_convention")},
                        "activation": context.get("activation")}
                answer, call_id = gateway.generate("agent", selected, actor, state["time"],
                    system_prompt=request["messages"][0]["content"], question=question)
                results.append({"actor": actor, "name": agent["name"], "variant": variant,
                    "source_call": row["id"], "diagnostic_call": call_id,
                    "question": question, "answer": answer, "executed": False})
                atomic_json(output, {"source_revision": revision, "results": results, "complete": False})
        after_hash = hashlib.sha256(Path(snapshot).read_bytes()).hexdigest()
        if original_hash != after_hash:
            raise ValueError("Diagnostic snapshot changed unexpectedly")
        calls.backup(Path(output).with_suffix(".sqlite"))
        result = {"source_revision": revision, "source_state_hash": digest(state),
                  "snapshot_bytes_before": original_hash, "snapshot_bytes_after": after_hash,
                  "results": results, "complete": True, "world_intents_executed": 0}
        atomic_json(output, result)
        return result
    finally:
        source.close()
        calls.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, help="Node-local copied world.sqlite")
    parser.add_argument("--config", required=True)
    parser.add_argument("--work", required=True, help="Separate local diagnostic database directory")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = diagnose(args.snapshot, Config.load(args.config).validate(), args.work, args.output)
    print(json.dumps({"diagnostic_answers": len(result["results"]), "world_intents_executed": 0}), flush=True)


if __name__ == "__main__":
    main()
