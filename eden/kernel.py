import copy
import json
import math
import posixpath

from . import terrain

YEAR = 365 * 86400
MAX_FILES_BYTES = 1024 * 1024


def number(value, low, high, label="value"):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{label} must be a finite number between {low} and {high}")
    return value


def text(value, limit=4000):
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError(f"Expected text of at most {limit} characters")
    return value


def new_id(state, prefix):
    state["counter"] += 1
    return f"{prefix}-{state['counter']}"


def schedule(state, time, kind, actor=None, data=None, priority=3):
    number(time, state["time"], 1e15, "event time")
    event = {"id": new_id(state, "scheduled"), "time": time, "priority": priority,
             "kind": kind, "actor": actor, "data": data or {}}
    state["queue"].append(event)
    state["queue"].sort(key=lambda e: (e["time"], e["priority"], int(e["id"].rsplit("-", 1)[1])))
    return event["id"]


def wake(state, actor, at, config, interrupt=False):
    agent = state["entities"].get(actor)
    if not agent or not agent.get("alive"):
        return
    if (state["time"] - agent["born"]) / YEAR < 3:
        return
    if agent.get("busy"):
        if not interrupt:
            return
        state["queue"] = [e for e in state["queue"] if not (e["actor"] == actor and e["kind"] in {"action_complete", "move_step"})]
        agent["busy"] = None
        agent["last_result"] = "Your ongoing action was interrupted."
    existing = [e for e in state["queue"] if e["actor"] == actor and e["kind"] == "activation"]
    if existing and min(e["time"] for e in existing) <= at:
        return
    state["queue"] = [e for e in state["queue"] if not (e["actor"] == actor and e["kind"] == "activation")]
    schedule(state, max(state["time"] + config.activation_seconds, at), "activation", actor, priority=4)


def agent(state, name, pos, born, parents=None, reproductive_role=None):
    ident = new_id(state, "agent")
    state["entities"][ident] = {
        "id": ident, "kind": "agent", "type": "human", "name": name, "pos": pos,
        "rev": 0, "alive": True, "born": born, "parents": parents or [],
        "reproductive_role": reproductive_role, "pregnancy": None,
        "needs": {"hunger": 0.0, "thirst": 0.0, "fatigue": 0.0, "exposure": 0.0},
        "conditions": {}, "busy": None, "files": {}, "recent": [], "last_result": "You awaken in Eden.",
        "deliberation": {}, "consent": {}, "activations": 0,
    }
    return ident


def initial(config):
    state = {"schema": 1, "terrain_version": terrain.VERSION, "seed": config.seed,
             "time": 0, "counter": 0, "entities": {}, "queue": [], "facts": [],
             "voxels": {}, "types": {}, "weather": "clear", "config": config.to_dict(), "relations": []}
    # Find a reproducible dry starting clearing.
    x = next(x for x in range(100) if terrain.height(config.seed, x, 0) >= 5)
    adam = agent(state, "Adam", terrain.position(state, x, 0), -config.initial_age_years * YEAR, reproductive_role="sperm")
    eve = agent(state, "Eve", terrain.position(state, x + 2, 0), -config.initial_age_years * YEAR, reproductive_role="gestation")
    for ident in (adam, eve):
        schedule(state, 0, "activation", ident, priority=4)
    realize(state, x, 0)
    schedule(state, 3600, "environment", priority=0)
    return state


def realize(state, x, z):
    """Finite natural anchors; stable IDs make revisits idempotent."""
    for gx in range(math.floor(x / 8) - 3, math.floor(x / 8) + 4):
        for gz in range(math.floor(z / 8) - 3, math.floor(z / 8) + 4):
            px, pz = gx * 8 + 3, gz * 8 + 3
            c = terrain.cell(state["seed"], px, pz)
            for kind in (["water_source"] if c["biome"] == "water" else ["tree" if c["biome"] == "forest" else "vegetation", "rock_outcrop"]):
                ident = f"natural:{gx}:{gz}:{kind}"
                if ident not in state["entities"]:
                    state["entities"][ident] = {"id": ident, "kind": "source", "type": kind,
                        "name": kind.replace("_", " "), "pos": terrain.position(state, px, pz),
                        "mass": 10000.0 if kind == "water_source" else 40.0, "rev": 0,
                        "attributes": {"description": "Coarse natural resource; reachable details require inspection."}}
            if c["biome"] != "water" and terrain.noise(state["seed"], gx, gz, "animal") > .95:
                ident = f"animal:{gx}:{gz}"
                state["entities"].setdefault(ident, {"id": ident, "kind": "animal", "type": "deer", "name": "deer", "pos": terrain.position(state, px, pz), "mass": 60.0, "rev": 0, "attributes": {}})


def location(state, entity):
    seen = set()
    while entity.get("holder"):
        if entity["id"] in seen:
            raise ValueError("Cyclic custody")
        seen.add(entity["id"])
        entity = state["entities"][entity["holder"]]
    return entity["pos"]


def visible(state, observer, entity):
    if entity.get("holder") == observer["id"]:
        return True
    a, b = observer["pos"], location(state, entity)
    hour = (state["time"] / 3600 + 7) % 24
    reach = 80 if 6 <= hour <= 19 else 15
    if state["weather"] == "rain":
        reach *= .6
    dist = terrain.distance(a, b)
    if dist > reach:
        return False
    n = max(1, math.ceil(dist / 2))
    for i in range(1, n):
        f = i / n
        px, pz = a[0] + (b[0] - a[0]) * f, a[2] + (b[2] - a[2]) * f
        eye = a[1] + 1.6 + (b[1] - a[1]) * f
        if terrain.surface(state, px, pz) > eye:
            return False
    return True


def public(entity):
    keys = ("id", "kind", "type", "name", "pos", "holder", "mass", "attributes", "alive")
    return {k: copy.deepcopy(entity[k]) for k in keys if k in entity}


def observe(state, actor):
    a = state["entities"][actor]
    seen = [{**public(e), "distance_m": round(terrain.distance(a["pos"], location(state, e)), 2)}
            for e in state["entities"].values() if e["id"] != actor and visible(state, a, e)]
    social = [{"id": e["id"], "name": e["name"], "distance_m": e["distance_m"],
               "can_hear_you": True} for e in seen
              if e["kind"] == "agent" and e.get("alive")
              and terrain.distance(a["pos"], location(state, state["entities"][e["id"]])) <= 20]
    history = a.get("action_history", [])[-6:]
    repeated = (len(history) == 6 and all(h["intent"].get("kind") == "move" for h in history)
                and len({json.dumps(h["intent"], sort_keys=True) for h in history}) <= 3)
    return {"sim_time": state["time"], "hour": (state["time"] / 3600 + 7) % 24,
            "weather": state["weather"], "self": {k: copy.deepcopy(a[k]) for k in ("id", "name", "pos", "needs", "conditions", "last_result", "deliberation")},
            "age_years": (state["time"] - a["born"]) / YEAR,
            "coordinate_convention": "pos is [x,y,z] metres; x/z horizontal, y elevation; move uses x,z",
            "visible": seen,
            "recent": a["recent"][-12:], "files": sorted(a["files"]),
            "recent_actions": copy.deepcopy(history),
            "operational_feedback": ("Your recent movement choices repeat a small set of destinations. Check whether they advance your intention. You may persist, reconsider, or choose another step." if repeated else None),
            "social_opportunities": social}


def social_question(observation):
    nearby = observation.get("social_opportunities", [])
    if not nearby:
        return None
    people = "; ".join(f"{p['name']} ({p['id']}) is {p['distance_m']:g} metres away" for p in nearby)
    return (f"{people}. They are already within speaking range and can hear you. "
            "Do you want to say anything to them? If yes, choose intent.kind=communicate "
            "and put your own words in intent.text; no further approach is needed to speak. "
            "If no, you may freely choose another action or wait.")


def note(a, value):
    a["recent"] = (a["recent"] + [value])[-12:]


def file_path(value):
    value = text(value, 240)
    if not value or value.startswith("/") or "\\" in value or "\x00" in value or any(p in {"", ".", ".."} for p in value.split("/")) or not value.lower().endswith(".md"):
        raise ValueError("Use a relative .md path with no empty, . or .. segments")
    return posixpath.normpath(value)


def memory(a, intent):
    kind = intent["kind"]
    files = a["files"]
    if kind == "list_files":
        return sorted(files)
    if kind == "search_files":
        query = text(intent.get("query", ""), 200).lower()
        return {p: content[:3000] for p, content in files.items() if query in content.lower() or query in p.lower()}
    path = file_path(intent.get("path"))
    if kind == "read_file":
        if path not in files:
            raise ValueError("File does not exist")
        start = int(number(intent.get("offset", 0), 0, MAX_FILES_BYTES))
        return {"content": files[path][start:start + 12000], "total_characters": len(files[path])}
    revised = dict(files)
    if kind in {"write_file", "append_file"}:
        content = text(intent.get("content"), 64000)
        revised[path] = (files.get(path, "") if kind == "append_file" else "") + content
    elif kind == "edit_file":
        old, new = text(intent.get("old"), 64000), text(intent.get("new"), 64000)
        if not old or files.get(path, "").count(old) != 1:
            raise ValueError("edit_file old text must match exactly once")
        revised[path] = files[path].replace(old, new, 1)
    elif kind == "move_file":
        dest = file_path(intent.get("destination"))
        if path not in files or dest in files:
            raise ValueError("Source missing or destination already exists")
        revised[dest] = revised.pop(path)
    elif kind == "delete_file":
        if path not in files:
            raise ValueError("File does not exist")
        del revised[path]
    else:
        raise ValueError("Unknown memory operation")
    if len(revised) > 256 or sum(len(c.encode()) for c in revised.values()) > MAX_FILES_BYTES:
        raise ValueError("Private memory quota exceeded (256 files / 1 MiB)")
    a["files"] = revised
    return {"path": path, "operation": kind, "ok": True}


def reachable(state, actor, ident, radius=3):
    if ident not in state["entities"]:
        raise ValueError(f"Unknown entity {ident}")
    a, e = state["entities"][actor], state["entities"][ident]
    if not a["alive"] or terrain.distance(a["pos"], location(state, e)) > radius:
        raise ValueError(f"Entity {ident} is out of reach")
    if e.get("holder") not in {None, actor}:
        raise ValueError("Item is in another entity's custody")
    return e


def validate(state):
    for ident, e in state["entities"].items():
        if e["id"] != ident:
            raise ValueError("Entity ID mismatch")
        pos = location(state, e)
        if len(pos) != 3 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in pos):
            raise ValueError("Invalid position")
        if "mass" in e:
            number(e["mass"], 0, 1e12, "mass")
        if e.get("holder"):
            holder = state["entities"][e["holder"]]
            if holder["kind"] != "agent" or not holder["alive"]:
                raise ValueError("Invalid holder")
    for a in state["entities"].values():
        if a["kind"] == "agent":
            mass = sum(e.get("mass", 0) for e in state["entities"].values() if e.get("holder") == a["id"])
            if mass > 30:
                raise ValueError("Carried mass exceeds 30 kg")
    for event in state["queue"]:
        number(event["time"], state["time"], 1e15, "queued time")


def die(state, ident, reason):
    a = state["entities"][ident]
    a["alive"], a["busy"], a["death_time"], a["death_reason"] = False, None, state["time"], reason
    a["rev"] += 1
    state["queue"] = [e for e in state["queue"] if e["actor"] != ident]
    for e in state["entities"].values():
        if e.get("holder") == ident:
            e["holder"], e["pos"] = None, list(a["pos"])
            e["rev"] += 1


def apply_delta(state, actor, operations, config):
    """Transactional candidate. Unknown operations and protected attributes fail closed."""
    if not isinstance(operations, list) or len(operations) > 24:
        raise ValueError("A ruling may contain at most 24 operations")
    result = copy.deepcopy(state)
    aliases = {}
    def ref(value):
        return aliases.get(value, value)
    for index, operation in enumerate(operations):
        if not isinstance(operation, dict):
            raise ValueError("Delta must contain objects")
        op = operation.get("op")
        if not isinstance(op, str) or not op:
            raise ValueError(f'delta[{index}] requires a nonempty "op" string, e.g. '
                             '{"op":"transfer_item","id":"object-id","holder":"agent-id"}; '
                             'use an operation name as its value, not as a wrapper key')
        required = {
            "create_entity": ("type", "mass", "sources"),
            "transfer_item": ("id", "holder"), "set_attribute": ("id", "key", "value"),
            "change_status": ("condition",), "satisfy_need": ("need", "amount", "source"),
            "record_fact": ("scope", "claim"), "schedule_effect": ("delay", "effect"),
            "add_relation": ("other", "relation"), "modify_voxel": ("position", "material"),
        }
        if op not in required:
            raise ValueError(f"delta[{index}]: Unsupported delta operation: {op}; "
                             f"supported op values: {', '.join(required)}")
        missing = [key for key in required[op] if key not in operation]
        if missing:
            raise ValueError(f"delta[{index}] ({op}) missing required fields: {', '.join(missing)}")
        if op == "create_entity":
            sources = operation.get("sources", [])
            if (not isinstance(sources, list) or not sources or len(sources) > 12
                    or any(not isinstance(item, dict) or not {"id", "mass"} <= item.keys() for item in sources)):
                raise ValueError("New matter requires finite source entities")
            mass = number(operation.get("mass"), .0001, 10000, "created mass")
            total = 0
            for source in sources:
                entity = reachable(result, actor, ref(source["id"]))
                amount = number(source["mass"], .0001, entity.get("mass", 0), "source mass")
                entity["mass"] -= amount
                entity["rev"] += 1
                total += amount
                if entity["mass"] < .00001 and entity["kind"] != "source":
                    del result["entities"][entity["id"]]
            if abs(total - mass) > .00001:
                raise ValueError("Created mass must equal debited source mass")
            ident = new_id(result, "object")
            alias = operation.get("alias", ident)
            if not isinstance(alias, str) or alias in aliases or alias in result["entities"]:
                raise ValueError("Creation alias must be unique")
            aliases[alias] = ident
            type_name = text(operation.get("type"), 120)
            attributes = operation.get("attributes", {})
            if not isinstance(attributes, dict) or len(str(attributes)) > 8000:
                raise ValueError("Invalid attributes")
            result["entities"][ident] = {"id": ident, "kind": "object", "type": type_name,
                "name": text(operation.get("name", type_name), 120), "pos": list(result["entities"][actor]["pos"]),
                "mass": mass, "holder": None, "attributes": copy.deepcopy(attributes), "rev": 0}
            result["types"].setdefault(type_name, {"description": text(operation.get("description", type_name)), "first_seen": result["time"]})
        elif op == "transfer_item":
            e = reachable(result, actor, ref(operation["id"]))
            if e["kind"] != "object":
                raise ValueError(f"delta[{index}]: Only portable objects can be transferred; "
                                 f"{e['id']} has kind={e['kind']}. If extraction is physically justified, "
                                 "use create_entity with sources debiting the extracted mass, then "
                                 "transfer_item using the new alias. This does not establish feasibility.")
            target = ref(operation.get("holder"))
            if target is not None:
                receiver = reachable(result, actor, target)
                if receiver["kind"] != "agent" or not receiver["alive"]:
                    raise ValueError("Receiver must be alive")
            e["holder"], e["pos"] = target, list(result["entities"][actor]["pos"])
            e["rev"] += 1
        elif op == "set_attribute":
            e = reachable(result, actor, ref(operation["id"]))
            key = text(operation["key"], 100)
            if key in {"pos", "holder", "mass", "alive", "files", "born", "needs", "pregnancy"}:
                raise ValueError("Protected attribute")
            if e["kind"] == "agent":
                raise ValueError("Use change_status for agent conditions")
            value = operation.get("value")
            if len(str(value)) > 8000:
                raise ValueError("Attribute too large")
            e.setdefault("attributes", {})[key] = value
            e["rev"] += 1
        elif op == "change_status":
            e = reachable(result, actor, ref(operation.get("id", actor)))
            if e["kind"] != "agent":
                raise ValueError("Status target must be an agent")
            name = text(operation["condition"], 120)
            severity = number(operation.get("severity", 0), 0, 1, "severity")
            if severity == 0:
                e["conditions"].pop(name, None)
            else:
                e["conditions"][name] = {"severity": severity, "since": result["time"], "description": text(operation.get("description", name))}
            e["rev"] += 1
        elif op == "satisfy_need":
            a = result["entities"][actor]
            need = operation["need"]
            if need not in {"hunger", "thirst", "exposure"}:
                raise ValueError("Invalid need (fatigue is handled by rest)")
            amount = number(operation["amount"], 0, .5, "need relief")
            source = reachable(result, actor, ref(operation["source"]))
            if need in {"hunger", "thirst"}:
                mass = number(operation.get("mass"), max(.001, amount * .5), source.get("mass", 0), "consumed mass")
                source["mass"] -= mass
                if source["mass"] <= .00001 and source["kind"] != "source":
                    del result["entities"][source["id"]]
            source["rev"] += 1
            a["needs"][need] = max(0, a["needs"][need] - amount)
            a["rev"] += 1
        elif op == "record_fact":
            scope = ref(operation.get("scope"))
            if scope not in result["entities"]:
                raise ValueError("Facts must be scoped to an existing entity")
            reachable(result, actor, scope)
            result["facts"].append({"id": new_id(result, "fact"), "scope": scope,
                "claim": text(operation["claim"]), "conditions": text(operation.get("conditions", "")),
                "confidence": text(operation.get("confidence", "uncertain"), 40), "time": result["time"]})
        elif op == "schedule_effect":
            delay = number(operation["delay"], 1, 365 * 86400, "delay")
            target = ref(operation.get("id", actor))
            reachable(result, actor, target)
            effect = operation["effect"]
            if not isinstance(effect, dict) or effect.get("kind") not in {"condition", "decay"}:
                raise ValueError("Future effects support condition or decay")
            if effect["kind"] == "condition":
                if result["entities"][target]["kind"] != "agent":
                    raise ValueError("Condition target must be an agent")
                text(effect["condition"], 120)
                number(effect["severity"], 0, 1)
            else:
                if result["entities"][target]["kind"] != "object":
                    raise ValueError("Decay target must be an object")
                text(effect.get("description", "decayed"))
            schedule(result, result["time"] + delay, "effect", data={"target": target, "effect": effect}, priority=0)
        elif op == "add_relation":
            other = ref(operation["other"])
            reachable(result, actor, other)
            result["relations"].append({"a": actor, "b": other, "relation": text(operation["relation"], 120), "time": result["time"]})
        elif op == "modify_voxel":
            xyz = operation["position"]
            if not isinstance(xyz, list) or len(xyz) != 3 or not all(type(v) is int for v in xyz):
                raise ValueError("Voxel position must contain three integers")
            if terrain.distance(result["entities"][actor]["pos"], xyz) > 3:
                raise ValueError("Voxel out of reach")
            material = operation["material"]
            if material != "air":
                raise ValueError("v0.1 voxel editing supports excavation; construction uses persistent objects")
            if terrain.voxel(result, *xyz) in {"air", "water"}:
                raise ValueError("No solid material to excavate")
            if any(e["kind"] == "agent" and math.floor(e["pos"][0]) == xyz[0] and math.floor(e["pos"][2]) == xyz[2] and abs(e["pos"][1] - xyz[1]) <= 1 for e in result["entities"].values()):
                raise ValueError("Cannot excavate supporting voxel beneath an agent")
            result["voxels"][",".join(map(str, xyz))] = "air"
            ident = new_id(result, "rubble")
            result["entities"][ident] = {"id": ident, "kind": "source", "type": "rubble", "name": "excavated rubble", "pos": list(xyz), "mass": 1500.0, "rev": 0, "attributes": {}}
        else:
            raise ValueError(f"Unsupported delta operation: {op}")
    validate(result)
    return result
