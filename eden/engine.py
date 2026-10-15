import copy
import json
import math
import threading
import time
import hashlib
import platform
from pathlib import Path

from .config import Config
from .inference import Gateway, ModelError
from . import kernel as k, terrain
from .store import Store

MEMORY = {"list_files", "read_file", "search_files", "write_file", "append_file", "edit_file", "move_file", "delete_file"}


class Engine:
    def __init__(self, directory, config=None):
        self.store = Store(directory)
        self.revision, self.state = self.store.latest()
        if self.state is None:
            self.config = (config or Config()).validate()
            self.state = k.initial(self.config)
            self.state["manifest"] = {"python": platform.python_version(), "platform": platform.platform(),
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))},
                "note": "Exact model artifacts and inference-server versions must be recorded by the operator; model aliases alone do not identify weights."}
            self.revision = self.store.commit(0, self.state, "world_created", payload={"backend": self.config.backend, "config": self.config.to_dict()})
        else:
            if self.state.get("schema") != 1 or self.state.get("terrain_version") != terrain.VERSION:
                raise ValueError("Unsupported run schema or terrain version")
            self.config = Config(**self.state["config"]).validate()
            if config and config.to_dict() != self.config.to_dict():
                raise ValueError("Existing run: use configure or --model/--endpoint overrides to record a model switch")
        self.gateway = Gateway(self.config, self.store)
        self.lock = threading.RLock()
        self.control = threading.Condition()
        self.paused = False
        self.step_budget = 0
        self.stopped = threading.Event()
        self.gateway.cancelled = self.stopped.is_set
        self.status = "ready"
        self.last_error = None

    def commit(self, kind, actor=None, payload=None):
        k.validate(self.state)
        self.revision = self.store.commit(self.revision, self.state, kind, actor, payload)

    def configure(self, **changes):
        with self.lock:
            allowed = {"model", "referee_model", "endpoint", "backend", "response_format", "speed", "timeout", "max_tokens", "max_population"}
            if not set(changes) <= allowed:
                raise ValueError("Unsupported runtime configuration change")
            config = Config(**{**self.config.to_dict(), **changes}).validate()
            previous = self.config.to_dict()
            self.config = config
            self.state["config"] = config.to_dict()
            self.gateway = Gateway(config, self.store)
            self.gateway.cancelled = self.stopped.is_set
            self.commit("configuration_changed", payload={"previous": previous, "current": config.to_dict()})

    def pause(self):
        with self.control:
            self.paused = True
            self.step_budget = 0
            self.control.notify_all()

    def resume(self):
        with self.control:
            self.paused = False
            self.control.notify_all()

    def request_step(self):
        with self.control:
            self.paused = True
            self.step_budget += 1
            self.control.notify_all()

    def stop(self):
        self.stopped.set()
        with self.control:
            self.control.notify_all()

    def run(self, steps=None, until=None):
        count = 0
        self.status = "running"
        try:
            while not self.stopped.is_set() and (steps is None or count < steps):
                with self.control:
                    while self.paused and self.step_budget == 0 and not self.stopped.is_set():
                        self.status = "paused"
                        self.control.wait(timeout=.5)
                    if self.stopped.is_set():
                        break
                    if self.step_budget:
                        self.step_budget -= 1
                if until is not None and (not self.state["queue"] or self.state["queue"][0]["time"] > until):
                    break
                started, before = time.monotonic(), self.state["time"]
                self.status = "running"
                if not self.step():
                    self.status = "extinct"
                    break
                count += 1
                if self.config.speed:
                    delay = (self.state["time"] - before) / self.config.speed - (time.monotonic() - started)
                    deadline = time.monotonic() + delay
                    while delay > 0 and not self.stopped.is_set() and not self.paused:
                        self.stopped.wait(min(.25, delay))
                        delay = deadline - time.monotonic()
        except Exception as exc:
            self.status, self.last_error = "error", str(exc)
            raise
        finally:
            if self.status not in {"error", "extinct"}:
                self.status = "stopped"
        return count

    def step(self):
        with self.lock:
            if not any(e.get("alive") for e in self.state["entities"].values() if e["kind"] == "agent"):
                return False
            if not self.state["queue"]:
                return False
            before = copy.deepcopy(self.state)
            event = self.state["queue"].pop(0)
            self.state["time"] = event["time"]
            actor = event["actor"]
            try:
                kind, payload = self.dispatch(event)
                self.commit(kind, actor, {"scheduled_event": event, **payload})
                self.last_error = None
            except ModelError as exc:
                # Keep the same pending event. Server faults must not starve inhabitants.
                self.state = before
                self.commit("inference_error", actor, {"error": str(exc), "pending_event": event["id"]})
                self.last_error = str(exc)
                self.pause()
            except (ValueError, KeyError, TypeError, OverflowError) as exc:
                # Roll back *all* partial effects, consume only the invalid event.
                self.state = before
                self.state["queue"].pop(0)
                self.state["time"] = event["time"]
                if actor and actor in self.state["entities"]:
                    a = self.state["entities"][actor]
                    a["busy"] = None
                    a["last_result"] = "Attempt failed validation: " + str(exc)
                    k.wake(self.state, actor, self.state["time"] + self.config.activation_seconds, self.config)
                self.commit("action_rejected", actor, {"reason": str(exc), "scheduled_event": event})
            except Exception:
                # Database failure: do not continue from uncommitted in-memory state.
                self.revision, self.state = self.store.latest()
                raise
            return True

    def dispatch(self, event):
        s, actor, data = self.state, event["actor"], event["data"]
        kind = event["kind"]
        a = s["entities"].get(actor)
        if actor and (not a or not a.get("alive")):
            return "event_cancelled", {"reason": "Actor no longer alive"}
        if kind == "activation":
            return self.activate(actor)
        if kind == "move_step":
            if a["busy"] != data["action"]:
                return "event_cancelled", {"reason": "Movement interrupted"}
            dest = data["pos"]
            # Recheck modified terrain before every path segment.
            terrain.path(s, a["pos"], dest[0], dest[2])
            a["pos"] = terrain.position(s, dest[0], dest[2])
            a["rev"] += 1
            k.realize(s, a["pos"][0], a["pos"][2])
            if data["last"]:
                a["busy"] = None
                a["last_result"] = "You arrived at your destination."
                k.wake(s, actor, s["time"], self.config)
            return "motion_updated", {"position": a["pos"], "arrived": data["last"]}
        if kind == "action_complete":
            if a["busy"] != data["action"]:
                return "event_cancelled", {"reason": "Action interrupted"}
            a["busy"] = None
            if data["mode"] == "rest":
                a["needs"]["fatigue"] = max(0, a["needs"]["fatigue"] - data["duration"] / (8 * 3600))
                a["last_result"] = "You finish resting."
            elif data["mode"] == "referee":
                # Any physical change or changed weather requires a fresh ruling.
                refs = data["revisions"]
                stale = any(s["entities"].get(i, {}).get("rev") != rev for i, rev in refs.items()) or data["weather"] != s["weather"]
                ruling = data["ruling"]
                call_id = data["call_id"]
                if stale:
                    ruling, call_id = self.judge(actor, data["intent"], elapsed=data["duration"])
                self.state = k.apply_delta(s, actor, ruling["delta"], self.config)
                s = self.state
                a = s["entities"][actor]
                a["last_result"] = k.text(ruling["public_result"])
                k.note(a, {"time": s["time"], "result": a["last_result"]})
                k.wake(s, actor, s["time"], self.config)
                return "action_completed", {"ruling": ruling, "call_id": call_id, "revalidated": stale}
            else:
                a["last_result"] = "You finish waiting."
            k.wake(s, actor, s["time"], self.config)
            return "action_completed", data
        if kind == "environment":
            return self.environment()
        if kind == "effect":
            e = s["entities"].get(data["target"])
            if not e:
                return "effect_expired", data
            effect = data["effect"]
            if effect["kind"] == "condition" and e.get("alive"):
                if effect["severity"]:
                    e["conditions"][effect["condition"]] = {"severity": effect["severity"], "since": s["time"], "description": effect["condition"]}
                else:
                    e["conditions"].pop(effect["condition"], None)
                e["last_result"] = "Your physical condition has changed."
                k.wake(s, e["id"], s["time"], self.config, interrupt=True)
            elif effect["kind"] == "decay":
                e.setdefault("attributes", {}).update({"active": False, "condition": effect.get("description", "decayed")})
            e["rev"] += 1
            return "effect_applied", data
        if kind == "birth":
            if not a["pregnancy"]:
                return "event_cancelled", {"reason": "No pregnancy"}
            alive = sum(bool(e.get("alive")) for e in s["entities"].values())
            if alive >= self.config.max_population:
                # Operational cap pauses BEFORE birth; retain the pending birth on disk.
                raise ModelError("Population cap reached; resume with --max-population increased")
            parents = a["pregnancy"]["parents"]
            child = k.agent(s, f"Child {s['counter'] + 1}", list(a["pos"]), s["time"], parents,
                            "gestation" if terrain.noise(s["seed"], s["counter"], "sex") < .5 else "sperm")
            a["pregnancy"] = None
            a["last_result"] = "Your child has been born."
            k.wake(s, actor, s["time"], self.config, interrupt=True)
            k.schedule(s, s["time"] + k.YEAR * 3, "activation", child, priority=4)
            return "birth", {"child": child, "parents": parents}
        raise ValueError(f"Unknown scheduled event {kind}")

    def activate(self, actor):
        s, c = self.state, self.config
        a = s["entities"][actor]
        if a["busy"]:
            return "activation_skipped", {"reason": "Busy"}
        a["activations"] += 1
        observation = k.observe(s, actor)
        output, call_id = self.gateway.generate("agent", {"observation": observation, "activation": a["activations"]}, actor, s["time"])
        a["deliberation"] = output["deliberation"]
        intent = output["intent"]
        payload = {"observation": observation, "output": output, "call_id": call_id}
        # Preserve genuine output even when intent validation fails; no invented thoughts.
        before_intent = copy.deepcopy(s)
        try:
            result = self.intent(actor, intent)
            payload.update(result)
        except ModelError:
            raise
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            self.state = before_intent
            a = self.state["entities"][actor]
            a["last_result"] = "Attempt rejected: " + str(exc)
            payload["rejection"] = str(exc)
            k.wake(self.state, actor, self.state["time"] + c.activation_seconds, c)
        return "agent_decision", payload

    def intent(self, actor, intent):
        s, c = self.state, self.config
        a = s["entities"][actor]
        kind = intent.get("kind")
        age = (s["time"] - a["born"]) / k.YEAR
        if age < 3 and kind not in {"wait", "rest", "communicate"}:
            raise ValueError("Your body is too young for independent action")
        if kind in MEMORY:
            result = k.memory(a, intent)
            a["last_result"] = json.dumps(result, ensure_ascii=False)[:16000]
        elif kind == "inspect":
            target = intent.get("target")
            if target and (target not in s["entities"] or not k.visible(s, a, s["entities"][target])):
                raise ValueError("Target is not visible")
            a["last_result"] = json.dumps(k.public(s["entities"][target])) if target else "You survey your visible surroundings."
        elif kind == "communicate":
            message = k.text(intent.get("text"), 3000)
            listeners = []
            for e in s["entities"].values():
                if e["kind"] == "agent" and e.get("alive") and e["id"] != actor and terrain.distance(a["pos"], e["pos"]) <= 20:
                    k.note(e, {"time": s["time"], "speaker": a["name"], "text": message})
                    k.wake(s, e["id"], s["time"], c, interrupt=True)
                    listeners.append(e["id"])
            a["last_result"] = "You spoke aloud."
            k.note(a, {"time": s["time"], "said": message})
            k.wake(s, actor, s["time"] + max(c.activation_seconds, len(message.split()) / 2), c)
            return {"speech": message, "listeners": listeners}
        elif kind == "move":
            points = terrain.path(s, a["pos"], intent["x"], intent["z"])
            action = k.new_id(s, "walk")
            a["busy"] = action
            duration = 0
            for previous, point in zip(points, points[1:]):
                duration += max(1, terrain.distance(previous, point) / max(.3, 1.3 * (1 - a["needs"]["fatigue"] * .6)))
                k.schedule(s, s["time"] + duration, "move_step", actor, {"action": action, "pos": point, "last": point is points[-1]}, priority=1)
            return {"trajectory": points, "duration": duration}
        elif kind in {"wait", "rest"}:
            duration = k.number(intent.get("seconds", c.wake_interval), 30, 86400)
            action = k.new_id(s, kind)
            a["busy"] = action
            k.schedule(s, s["time"] + duration, "action_complete", actor, {"mode": kind, "action": action, "duration": duration}, priority=1)
            return {"duration": duration}
        elif kind == "world_action":
            k.text(intent.get("description"))
            targets = intent.get("targets", [])
            if not isinstance(targets, list) or len(targets) > 24:
                raise ValueError("targets must be a list of at most 24 IDs")
            for target in targets:
                k.reachable(s, actor, target)
            ruling, call_id = self.judge(actor, intent)
            if ruling["outcome"] == "rejected":
                if ruling["delta"]:
                    raise ValueError("Rejected attempt cannot mutate state")
                a["last_result"] = k.text(ruling["public_result"])
                k.wake(s, actor, s["time"], c)
                return {"ruling": ruling, "referee_call_id": call_id}
            duration = k.number(ruling.get("duration"), 1, 86400, "duration")
            # Validate now without committing future effects early.
            k.apply_delta(s, actor, ruling["delta"], c)
            references = {actor, *targets}
            for op in ruling["delta"]:
                for key in ("id", "source", "scope", "other", "holder"):
                    if isinstance(op.get(key), str) and op[key] in s["entities"]:
                        references.add(op[key])
                references.update(source["id"] for source in op.get("sources", []) if source["id"] in s["entities"])
            action = k.new_id(s, "attempt")
            a["busy"] = action
            data = {"mode": "referee", "action": action, "intent": intent, "ruling": ruling,
                    "duration": duration, "call_id": call_id, "weather": s["weather"],
                    "revisions": {i: s["entities"][i]["rev"] for i in references}}
            k.schedule(s, s["time"] + duration, "action_complete", actor, data, priority=1)
            return {"ruling": ruling, "referee_call_id": call_id, "duration": duration}
        elif kind == "consent_reproduction":
            partner = k.reachable(s, actor, intent["partner"])
            if partner["kind"] != "agent" or age < 18 or (s["time"] - partner["born"]) / k.YEAR < 18 or partner["id"] == actor:
                raise ValueError("Requires two adults")
            a["consent"][partner["id"]] = s["time"] + 86400
            a["last_result"] = "Your willingness to reproduce with this partner is recorded for one day."
        elif kind == "reproduce":
            partner = k.reachable(s, actor, intent["partner"])
            if partner["kind"] != "agent" or partner["id"] == actor or age < 18 or (s["time"] - partner["born"]) / k.YEAR < 18:
                raise ValueError("Requires two adults")
            if a["consent"].get(partner["id"], -1) < s["time"] or partner["consent"].get(actor, -1) < s["time"]:
                raise ValueError("Requires independently recorded mutual consent")
            roles = {a["reproductive_role"], partner["reproductive_role"]}
            if roles != {"gestation", "sperm"}:
                raise ValueError("Biological reproduction is not feasible for this pairing")
            mother = a if a["reproductive_role"] == "gestation" else partner
            if mother["pregnancy"]:
                raise ValueError("Pregnancy is already ongoing")
            a["consent"].pop(partner["id"], None)
            partner["consent"].pop(actor, None)
            conceived = terrain.noise(s["seed"], s["counter"], actor, "conception") < .25
            if conceived:
                mother["pregnancy"] = {"parents": [actor, partner["id"]], "since": s["time"]}
                k.schedule(s, s["time"] + 280 * 86400, "birth", mother["id"], priority=0)
            a["last_result"] = "You attempted reproduction. The outcome is not immediately apparent."
            k.wake(s, actor, s["time"] + 1800, c)
            return {"conception": conceived, "parents": [actor, partner["id"]]}
        elif kind == "care":
            child = k.reachable(s, actor, intent["child"])
            if child["kind"] != "agent" or (s["time"] - child["born"]) / k.YEAR >= 3 or age < 12:
                raise ValueError("Care requires a dependent young child and capable caregiver")
            if max(a["needs"]["hunger"], a["needs"]["thirst"]) > .7:
                raise ValueError("You need nourishment before feeding a dependent child")
            for need in ("hunger", "thirst"):
                child["needs"][need] = max(0, child["needs"][need] - .3)
                a["needs"][need] += .1
            child["rev"] += 1
            a["last_result"] = "You care for the child."
        else:
            raise ValueError(f"Unknown transport kind {kind}; arbitrary physical attempts use world_action")
        k.wake(s, actor, s["time"] + c.activation_seconds, c)
        return {"result": a["last_result"]}

    def judge(self, actor, intent, elapsed=0):
        s = self.state
        nearby = [e for e in s["entities"].values() if terrain.distance(s["entities"][actor]["pos"], k.location(s, e)) <= 5]
        ids = {e["id"] for e in nearby}
        context = {"intent": intent, "actor": actor, "time": s["time"], "weather": s["weather"],
                   "elapsed_seconds": elapsed, "instruction": "Assess the outcome now after elapsed time; do not charge the original duration twice." if elapsed else "Assess the proposed attempt.",
                   "entities": [{**k.public(e), **({"needs": e["needs"], "conditions": e["conditions"]} if e["kind"] == "agent" else {})} for e in nearby],
                   "precedents": [f for f in s["facts"] if f["scope"] in ids][-80:]}
        for attempt in range(self.config.retries + 1):
            ruling, call_id = self.gateway.generate("referee", context, actor, s["time"])
            try:
                k.text(ruling.get("public_result"))
                k.text(ruling.get("rationale", ""), 8000)
                k.number(ruling.get("duration"), 1, 86400)
                if ruling["outcome"] == "rejected" and ruling["delta"]:
                    raise ValueError("Rejected actions cannot have effects")
                k.apply_delta(s, actor, ruling["delta"], self.config)
                return ruling, call_id
            except (ValueError, KeyError, TypeError) as exc:
                context["repair_error"] = str(exc)
                context["invalid_ruling"] = ruling
        return {"outcome": "failed", "public_result": "Your attempt could not produce a grounded outcome.", "rationale": "Referee validation retries exhausted: " + context["repair_error"], "duration": 30, "delta": []}, call_id

    def environment(self):
        s, c = self.state, self.config
        hour = int(s["time"] // 3600)
        old_weather = s["weather"]
        s["weather"] = "rain" if terrain.noise(s["seed"], hour // 6, "weather") < .25 else "clear"
        deaths = []
        for a in list(s["entities"].values()):
            if a["kind"] == "animal":
                # Cheap persistent wildlife; unusual interaction goes to the referee.
                dx = round(terrain.noise(s["seed"], a["id"], hour, "x") * 4 - 2)
                dz = round(terrain.noise(s["seed"], a["id"], hour, "z") * 4 - 2)
                try:
                    route = terrain.path(s, a["pos"], a["pos"][0] + dx, a["pos"][2] + dz)
                    a["pos"] = route[-1]
                    a["rev"] += 1
                except ValueError:
                    pass
            if a["kind"] != "agent" or not a["alive"]:
                continue
            before = max(a["needs"].values())
            age = (s["time"] - a["born"]) / k.YEAR
            for need, rate in (("hunger", 1/120), ("thirst", 1/60), ("fatigue", 1/48), ("exposure", .01 if s["weather"] == "rain" else -.02)):
                a["needs"][need] = min(1, max(0, a["needs"][need] + rate))
            a["rev"] += 1
            severe = max((v["severity"] for v in a["conditions"].values()), default=0)
            age_risk = max(0, age - c.lifespan_years + 15) / (15 * k.YEAR / 3600)
            reason = None
            if a["needs"]["hunger"] >= 1 or a["needs"]["thirst"] >= 1:
                reason = "unmet nutrition or hydration"
            elif severe >= .95 and terrain.noise(s["seed"], a["id"], hour, "illness") < .08:
                reason = "severe illness or injury"
            elif terrain.noise(s["seed"], a["id"], hour, "aging") < age_risk:
                reason = "age-related decline"
            if reason:
                k.die(s, a["id"], reason)
                deaths.append({"id": a["id"], "reason": reason})
                continue
            if before < .65 <= max(a["needs"].values()) or (old_weather != s["weather"] and s["weather"] == "rain"):
                a["last_result"] = "Your needs or the weather have changed noticeably."
                k.wake(s, a["id"], s["time"], c, interrupt=True)
        # Coarse renewable anchors, never replenished by repeated search prompts.
        if hour % 24 == 0:
            for e in s["entities"].values():
                if e["kind"] == "source" and e["type"] in {"tree", "vegetation", "water_source"}:
                    cap = 10000 if e["type"] == "water_source" else 40
                    e["mass"] = min(cap, e["mass"] + (100 if e["type"] == "water_source" else .1))
                    e["rev"] += 1
        k.schedule(s, s["time"] + 3600, "environment", priority=0)
        return "environment_changed", {"weather": s["weather"], "deaths": deaths}
