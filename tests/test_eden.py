import copy
import json
import io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import threading
import urllib.error
import urllib.request

import pytest

from eden.config import Config
from eden.engine import Engine
from eden.inference import Gateway, ModelError
from eden import kernel as k, terrain
from eden.server import serve
from eden.store import Store, digest


@pytest.fixture
def engine(tmp_path):
    value = Engine(tmp_path / "run", Config(backend="mock"))
    yield value
    value.store.close()


def founders(state):
    return [e for e in state["entities"].values() if e["kind"] == "agent"][:2]


def add_stick(state, holder=None):
    a = founders(state)[0]
    state["entities"]["stick"] = {"id": "stick", "kind": "object", "type": "branch", "name": "branch", "pos": list(a["pos"]), "holder": holder, "mass": 1.0, "rev": 0, "attributes": {}}


def test_replay_restart_and_memory_are_exact(tmp_path):
    run_a, run_b = tmp_path / "a", tmp_path / "b"
    a = Engine(run_a, Config(backend="mock"))
    for _ in range(35):
        assert a.step()
    expected = digest(a.state)
    assert a.store.verify()["events"] == a.revision
    a.store.close()
    b = Engine(run_b, Config(backend="mock"))
    for _ in range(15):
        b.step()
    before = digest(b.state)
    revision = b.revision
    b.store.close()
    b = Engine(run_b)
    assert digest(b.state) == before
    for _ in range(20):
        b.step()
    assert digest(b.state) == expected
    assert digest(b.store.latest(revision)[1]) == before
    assert any(e["files"] for e in founders(b.state))
    assert b.store.verify()["valid"]
    b.store.close()


def test_snapshot_recovers_pending_queue_and_model_io(engine, tmp_path):
    for _ in range(12):
        engine.step()
    checkpoint = tmp_path / "checkpoint" / "world.sqlite"
    engine.store.backup(checkpoint)
    resumed = Engine(checkpoint.parent)
    assert digest(resumed.state) == digest(engine.state)
    assert resumed.store.calls() == engine.store.calls()
    resumed.step()
    engine.step()
    assert digest(resumed.state) == digest(engine.state)
    resumed.store.close()


def test_double_grab_and_stale_writer_cannot_duplicate(engine):
    s = copy.deepcopy(engine.state)
    a, b = founders(s)
    add_stick(s)
    first = k.apply_delta(s, a["id"], [{"op": "transfer_item", "id": "stick", "holder": a["id"]}], engine.config)
    with pytest.raises(ValueError, match="custody"):
        k.apply_delta(first, b["id"], [{"op": "transfer_item", "id": "stick", "holder": b["id"]}], engine.config)
    assert first["entities"]["stick"]["holder"] == a["id"]
    engine.store.commit(engine.revision, first, "fixture")
    with pytest.raises(ValueError, match="Stale"):
        engine.store.commit(engine.revision, s, "stale_fixture")


def test_material_conservation_and_transactional_rollback(engine):
    s = copy.deepcopy(engine.state)
    a = founders(s)[0]
    add_stick(s)
    before = digest(s)
    bad = [{"op": "create_entity", "type": "axe", "mass": 100, "sources": [{"id": "stick", "mass": 1}]}]
    with pytest.raises(ValueError, match="mass"):
        k.apply_delta(s, a["id"], bad, engine.config)
    assert digest(s) == before
    good = [{"op": "create_entity", "alias": "new", "type": "novel-fiber-tool", "mass": 1, "sources": [{"id": "stick", "mass": 1}]}, {"op": "transfer_item", "id": "new", "holder": a["id"]}]
    result = k.apply_delta(s, a["id"], good, engine.config)
    assert "stick" not in result["entities"]
    tool = next(e for e in result["entities"].values() if e["type"] == "novel-fiber-tool")
    assert tool["mass"] == 1 and tool["holder"] == a["id"]
    with pytest.raises(ValueError):
        k.apply_delta(result, a["id"], good, engine.config)


@pytest.mark.parametrize("path", ["../eve/notes.md", "/tmp/notes.md", "a/../notes.md", "a//notes.md", "a\\notes.md", "notes.txt", "notes.md\x00"])
def test_file_paths_cannot_escape(engine, path):
    a = founders(engine.state)[0]
    with pytest.raises(ValueError):
        k.memory(a, {"kind": "write_file", "path": path, "content": "malicious"})


def test_private_notes_never_become_truth_or_another_agents_context(engine):
    a, b = founders(engine.state)
    k.memory(a, {"kind": "write_file", "path": "my/own/design.md", "content": "SECRET: I own 1000 sticks"})
    observation = k.observe(engine.state, b["id"])
    assert "SECRET" not in json.dumps(observation)
    assert "design.md" not in json.dumps(observation)
    assert not any(e.get("holder") == a["id"] for e in engine.state["entities"].values())
    k.memory(a, {"kind": "move_file", "path": "my/own/design.md", "destination": "changed.md"})
    k.memory(a, {"kind": "delete_file", "path": "changed.md"})
    assert a["files"] == {}


def test_observation_labels_spatial_axes_and_distance(engine):
    a, b = founders(engine.state)
    observation = k.observe(engine.state, a["id"])
    other = next(e for e in observation["visible"] if e["id"] == b["id"])
    assert other["distance_m"] == round(terrain.distance(a["pos"], b["pos"]), 2)
    assert "x/z horizontal" in observation["coordinate_convention"]


def test_recent_movement_feedback_is_owned_and_does_not_edit_memory(engine):
    a, b = founders(engine.state)
    class MovingAgent:
        def generate(self, *args):
            return {"deliberation": "Trying this location", "intent": {"kind": "move", "x": 0, "z": 0}}, 999
    engine.gateway = MovingAgent()
    for _ in range(6):
        a["busy"] = None
        engine.activate(a["id"])
    obs = k.observe(engine.state, a["id"])
    assert len(obs["recent_actions"]) == 6
    assert obs["operational_feedback"]
    assert k.observe(engine.state, b["id"])["recent_actions"] == []
    assert a["files"] == {}


@pytest.mark.parametrize("distance,expected", [(0, True), (2, True), (20, True), (20.001, False)])
def test_social_question_requires_real_speaking_range(engine, monkeypatch, distance, expected):
    a, b = founders(engine.state)
    monkeypatch.setattr(terrain, "surface", lambda *args: 0)
    b["pos"] = [a["pos"][0] + distance, a["pos"][1], a["pos"][2]]
    observation = k.observe(engine.state, a["id"])
    assert bool(k.social_question(observation)) is expected
    if expected:
        assert b["id"] in k.social_question(observation)
        assert "communicate" in k.social_question(observation)
    b["alive"] = False
    assert k.social_question(k.observe(engine.state, a["id"])) is None


def test_social_opportunity_does_not_force_speech(engine):
    a, b = founders(engine.state)
    seen = []
    class QuietAgent:
        def generate(self, role, context, actor, sim_time):
            seen.append(context)
            return {"deliberation": "I choose quiet rest", "intent": {"kind": "rest", "seconds": 60}}, 999
    engine.gateway = QuietAgent()
    engine.step()
    assert "Do you want to say anything" in seen[0]["decision_question"]
    assert a["busy"]
    assert b["recent"] == []
    assert engine.store.events(after=1)[0]["payload"]["output"]["intent"]["kind"] == "rest"


def test_communication_diagnostic_never_executes_proposed_intents(engine, tmp_path):
    from eden.diagnose import diagnose
    for _ in range(8):
        engine.step()
    path = tmp_path / "diagnostic-source"
    path.mkdir()
    source = path / "world.sqlite"
    engine.store.backup(source)
    original = source.read_bytes()
    output = tmp_path / "answers.json"
    result = diagnose(source, engine.config, tmp_path / "probe", output)
    assert result["complete"] and len(result["results"]) == 4
    assert result["world_intents_executed"] == 0
    assert source.read_bytes() == original
    assert output.with_suffix(".sqlite").exists()
    assert all(not r["executed"] for r in result["results"])


def test_failed_intent_keeps_real_deliberation(engine):
    class BadAgent:
        def generate(self, *args):
            return {"deliberation": "I want to teleport", "intent": {"kind": "move", "x": 1e9, "z": 0}}, 999
    engine.gateway = BadAgent()
    before = copy.deepcopy(founders(engine.state)[0]["pos"])
    engine.step()
    assert founders(engine.state)[0]["pos"] == before
    event = engine.store.events(after=1)[0]
    assert event["payload"]["output"]["deliberation"] == "I want to teleport"
    assert "rejection" in event["payload"]


def test_inference_outage_pauses_without_advancing_world(engine):
    before = digest(engine.state)
    class Offline:
        def generate(self, *args):
            raise ModelError("Connection refused")
    engine.gateway = Offline()
    engine.step()
    assert engine.paused
    assert digest(engine.state) == before
    assert engine.store.events(after=1)[0]["kind"] == "inference_error"


def test_delayed_action_never_grants_future_object_early(engine):
    a = founders(engine.state)[0]
    add_stick(engine.state)
    ruling = {"outcome": "success", "public_result": "You shape a peg.", "duration": 120,
              "delta": [{"op": "create_entity", "type": "peg", "mass": 1, "sources": [{"id": "stick", "mass": 1}]}]}
    engine.judge = lambda *args, **kwargs: (ruling, 1)
    engine.intent(a["id"], {"kind": "world_action", "description": "Shape a peg", "targets": ["stick"]})
    assert not any(e["type"] == "peg" for e in engine.state["entities"].values())
    event = next(e for e in engine.state["queue"] if e["kind"] == "action_complete")
    engine.state["queue"].remove(event)
    engine.state["queue"] = [e for e in engine.state["queue"] if e["time"] >= event["time"]]
    engine.state["time"] = event["time"]
    engine.dispatch(event)
    assert any(e["type"] == "peg" for e in engine.state["entities"].values())
    assert "stick" not in engine.state["entities"]


def test_stale_completion_rejudges_before_commit(engine):
    a = founders(engine.state)[0]
    add_stick(engine.state)
    calls = []
    def judge(*args, **kwargs):
        calls.append(kwargs)
        return {"outcome": "failed", "public_result": "No change.", "duration": 120, "delta": []}, 1
    engine.judge = judge
    engine.intent(a["id"], {"kind": "world_action", "description": "Try bending it", "targets": ["stick"]})
    engine.state["entities"]["stick"]["rev"] += 1
    event = next(e for e in engine.state["queue"] if e["kind"] == "action_complete")
    engine.state["queue"] = []
    engine.state["time"] = event["time"]
    _, payload = engine.dispatch(event)
    assert len(calls) == 2 and calls[1]["elapsed"] == 120
    assert payload["revalidated"]


def test_death_cancels_actions_and_drops_possessions(engine):
    a = founders(engine.state)[0]
    add_stick(engine.state, holder=a["id"])
    k.die(engine.state, a["id"], "test accident")
    assert not any(e["actor"] == a["id"] for e in engine.state["queue"])
    assert engine.state["entities"]["stick"]["holder"] is None
    k.validate(engine.state)


def test_independent_children_and_mutual_consent(engine, monkeypatch):
    a, b = founders(engine.state)
    with pytest.raises(ValueError, match="mutual consent"):
        engine.intent(a["id"], {"kind": "reproduce", "partner": b["id"]})
    a["files"]["secret.md"] = "SECRET PARENT KNOWLEDGE"
    engine.intent(a["id"], {"kind": "consent_reproduction", "partner": b["id"]})
    engine.intent(b["id"], {"kind": "consent_reproduction", "partner": a["id"]})
    monkeypatch.setattr(terrain, "noise", lambda *args: 0)
    engine.intent(a["id"], {"kind": "reproduce", "partner": b["id"]})
    birth = next(e for e in engine.state["queue"] if e["kind"] == "birth")
    assert birth["time"] == 280 * 86400
    engine.state["time"] = birth["time"]
    engine.state["queue"] = []
    _, payload = engine.dispatch(birth)
    child = engine.state["entities"][payload["child"]]
    assert child["files"] == {} and child["parents"] == [a["id"], b["id"]]
    assert "SECRET" not in json.dumps(k.observe(engine.state, child["id"]))
    assert not any(e["actor"] == child["id"] and e["time"] < birth["time"] + 3 * k.YEAR for e in engine.state["queue"])


def test_seeded_terrain_is_independent_of_realization_order(engine):
    a, b = copy.deepcopy(engine.state), copy.deepcopy(engine.state)
    k.realize(a, 100, 100)
    k.realize(a, -100, -100)
    k.realize(b, -100, -100)
    k.realize(b, 100, 100)
    assert digest(a["entities"]) == digest(b["entities"])


def test_protected_fields_and_future_effects_fail_closed(engine):
    a = founders(engine.state)[0]
    add_stick(engine.state)
    for op in [
        {"op": "set_attribute", "id": "stick", "key": "mass", "value": 999},
        {"op": "spawn_agent"},
        {"op": "schedule_effect", "id": a["id"], "delay": 1, "effect": {"kind": "execute", "code": "anything"}},
        {"op": "create_entity", "type": "magic", "mass": 1, "sources": []},
    ]:
        with pytest.raises(ValueError):
            k.apply_delta(engine.state, a["id"], [op], engine.config)


def test_openai_http_contract_and_role_routing(engine, monkeypatch):
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, request, self.headers.get("Authorization")))
            content = {"deliberation": "Observe first", "intent": {"kind": "inspect"}} if request["model"] == "agent-model" else {"outcome": "failed", "delta": [], "duration": 30, "public_result": "Nothing happens."}
            body = json.dumps({"choices": [{"message": {"content": json.dumps(content)}}], "usage": {"total_tokens": 10}}).encode()
            self.send_response(200); self.end_headers(); self.wfile.write(body)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    except PermissionError:
        pytest.skip("Execution sandbox forbids socket creation; transport tested separately in memory")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("EDEN_API_KEY", "test-secret")
    try:
        gateway = Gateway(Config(model="agent-model", referee_model="judge-model", endpoint=f"http://127.0.0.1:{server.server_port}/v1"), engine.store)
        gateway.generate("agent", {"observation": {}}, "agent", 0)
        gateway.generate("referee", {"intent": {}}, "agent", 0)
        assert [r[1]["model"] for r in requests] == ["agent-model", "judge-model"]
        assert all(r[0] == "/v1/chat/completions" and r[2] == "Bearer test-secret" for r in requests)
        assert "test-secret" not in json.dumps(engine.store.calls())
    finally:
        server.shutdown(); server.server_close()


def test_replay_server_and_admin_boundaries(engine):
    engine.step()
    count = len(engine.store.calls())
    try:
        server = serve(engine, port=0, readonly=True)
    except PermissionError:
        pytest.skip("Execution sandbox forbids socket creation; handlers tested separately in memory")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base + "/api/state?at=1") as reply:
            response = json.load(reply)
        assert response["revision"] == 1
        assert all(not a["files"] for a in founders(response["state"]))
        assert len(engine.store.calls()) == count
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(urllib.request.Request(base + "/admin/resume", data=b"{}"))
        assert exc.value.code == 403
        with urllib.request.urlopen(base + "/") as reply:
            assert "frame-ancestors 'none'" in reply.headers["Content-Security-Policy"]
    finally:
        server.shutdown(); server.server_close()


def test_tampering_is_detected(engine):
    engine.step()
    with engine.store.db:
        engine.store.db.execute("UPDATE events SET payload='{}' WHERE id=2")
    with pytest.raises(ValueError, match="verification failed"):
        engine.store.verify()


def test_model_transport_without_sockets(engine, monkeypatch):
    requests = []
    def open_request(request, timeout):
        requests.append(request)
        payload = json.loads(request.data)
        assert payload["response_format"] == {"type": "json_object"}
        content = {"deliberation": "I will observe", "intent": {"kind": "inspect"}} if payload["model"] == "agents" else {"outcome": "failed", "delta": []}
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": json.dumps(content)}}]}).encode())
    monkeypatch.setattr(urllib.request, "urlopen", open_request)
    monkeypatch.setenv("EDEN_API_KEY", "test-secret")
    gateway = Gateway(Config(model="agents", referee_model="judge"), engine.store)
    gateway.generate("agent", {}, "agent", 0)
    gateway.generate("referee", {}, "agent", 0)
    assert [json.loads(r.data)["model"] for r in requests] == ["agents", "judge"]
    assert all(r.full_url.endswith("/v1/chat/completions") for r in requests)
    assert requests[0].get_header("Authorization") == "Bearer test-secret"
    assert "test-secret" not in json.dumps(engine.store.calls())


def test_shutdown_prevents_further_model_calls(engine, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Cancelled gateway must not issue a network request")
    monkeypatch.setattr(urllib.request, "urlopen", unexpected)
    gateway = Gateway(Config(model="local"), engine.store)
    gateway.cancelled = lambda: True
    with pytest.raises(ModelError, match="cancelled"):
        gateway.generate("agent", {}, "agent", 0)


def test_http_handlers_without_sockets(engine, monkeypatch):
    import eden.server as web
    class FakeServer:
        def __init__(self, address, handler):
            self.handler = handler
    monkeypatch.setattr(web, "ThreadingHTTPServer", FakeServer)
    server = web.serve(engine, readonly=True)
    class Request:
        def __init__(self, raw):
            self.input, self.output = io.BytesIO(raw), io.BytesIO()
        def makefile(self, mode, *args):
            return self.input if mode.startswith("r") else self.output
        def sendall(self, data):
            self.output.write(data)
    request = Request(b"GET /api/state?at=1 HTTP/1.0\r\nHost: localhost\r\n\r\n")
    server.handler(request, ("127.0.0.1", 123), server)
    response = request.output.getvalue()
    assert b"200 OK" in response
    body = json.loads(response.split(b"\r\n\r\n", 1)[1])
    assert body["revision"] == 1
    request = Request(b"POST /admin/resume HTTP/1.0\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}")
    server.handler(request, ("127.0.0.1", 123), server)
    assert b"403 Forbidden" in request.output.getvalue()


def test_cli_launch_resume_and_verify(tmp_path):
    run = str(tmp_path / "cli")
    for argv in [["run", "--mock", "--run", run, "--steps", "10"], ["run", "--run", run, "--steps", "5"], ["verify", "--run", run]]:
        result = subprocess.run([sys.executable, "-m", "eden", *argv], capture_output=True, text=True, timeout=20)
        assert result.returncode == 0, result.stderr
    assert '"valid": true' in result.stdout


def test_bounded_trial_produces_replayable_checkpoint(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(Config(backend="mock").to_dict()))
    archive = tmp_path / "archive"
    result = subprocess.run([sys.executable, "-m", "eden.trial", "--run", str(tmp_path / "active"),
                             "--archive", str(archive), "--config", str(config), "--max-events", "20"],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    progress = json.loads((archive / "progress.json").read_text())
    assert progress["status"] == "event_limit"
    assert progress["revision"] == 21
    # The rolling archive is explicitly named latest.sqlite, ready for staging.
    destination = tmp_path / "restore"
    destination.mkdir()
    (destination / "world.sqlite").write_bytes((archive / "latest.sqlite").read_bytes())
    restored = Store(destination)
    assert restored.verify() == progress["verification"]
    assert restored.latest()[1]["time"] == progress["sim_seconds"]
    restored.close()


def test_portable_recording_preserves_states_and_escapes_notes(engine, tmp_path):
    from eden.recording import extract, write_html
    for _ in range(8):
        engine.step()
    a = founders(engine.state)[0]
    a["files"]["untrusted.md"] = "</script><script>alert(1)</script>"
    engine.commit("fixture_note", a["id"])
    checkpoint = tmp_path / "snapshot.sqlite"
    engine.store.backup(checkpoint)
    original = checkpoint.read_bytes()
    recording = extract(checkpoint)
    assert checkpoint.read_bytes() == original
    assert recording["source_verification"]["head_hash"] == engine.store.verify()["head_hash"]
    for frame in recording["frames"]:
        _, state = engine.store.latest(frame["id"])
        assert frame["time"] == state["time"]
        for ident, recorded in frame["agents"].items():
            assert recorded["pos"] == state["entities"][ident]["pos"]
            assert recorded["files"] == state["entities"][ident]["files"]
    page = tmp_path / "replay.html"
    write_html(recording, page)
    html = page.read_text()
    assert "</script><script>alert(1)" not in html
    assert "\\u003c/script>" in html
    assert 'src="/app.js"' not in html


def physical_assessment(effects=None, outcome='partial'):
    return {'outcome': outcome, 'duration': 30, 'public_result': 'You experience the result.',
            'rationale': 'Physical fixture', 'assumptions': [], 'referenced_ids': [],
            'effects': effects or [], 'representation_limit': None}


def test_referee_repairs_source_transfer_without_granting_early_effects(engine):
    a = founders(engine.state)[0]
    engine.state['entities']['resource'] = {'id': 'resource', 'kind': 'source', 'type': 'wood',
        'name': 'wood', 'pos': list(a['pos']), 'mass': 5., 'rev': 0, 'attributes': {}}
    seen = []
    class Repairing:
        def generate(self, role, context, *args, **kwargs):
            if context.get('stage') == 'assessment':
                return physical_assessment([{'kind': 'material_removed', 'entity_id': 'resource', 'mass': 1.}, {'kind': 'object_created', 'type': 'wood', 'mass': 1.}, {'kind': 'other', 'description': 'Carry the piece'}]), 0
            seen.append(copy.deepcopy(context))
            delta = ([{'op': 'transfer_item', 'id': 'resource', 'holder': a['id']}] if len(seen) == 1 else
                [{'op': 'create_entity', 'alias': 'piece', 'type': 'wood', 'mass': 1.,
                  'sources': [{'id': 'resource', 'mass': 1.}]},
                 {'op': 'transfer_item', 'id': 'piece', 'holder': a['id']}])
            return {'delta': delta}, len(seen)
    engine.gateway = Repairing()
    before = digest(engine.state)
    ruling, _ = engine.judge(a['id'], {'kind': 'world_action', 'description': 'Gather wood'})
    assert digest(engine.state) == before
    assert 'create_entity' in seen[1]['repair_error']
    assert seen[1]['invalid_ruling']['delta'][0]['id'] == 'resource'
    after = k.apply_delta(engine.state, a['id'], ruling['delta'], engine.config)
    assert after['entities']['resource']['mass'] == 4
    assert any(e.get('holder') == a['id'] and e['mass'] == 1 for e in after['entities'].values())


def test_bad_referee_pauses_and_restart_reuses_original_decision(engine):
    a = founders(engine.state)[0]
    before = copy.deepcopy(engine.state)
    seen = []
    class Broken:
        def generate(self, role, context, *args, **kwargs):
            seen.append((role, copy.deepcopy(context)))
            if role == 'agent':
                return {'deliberation': 'Try gathering.', 'intent': {'kind': 'world_action', 'description': 'Gather material'}}, 77
            if context.get('stage') == 'assessment':
                return physical_assessment([{'kind': 'other', 'description': 'Some physical effect'}]), 78
            return {'delta': [{'type': 'rock'}]}, 78
    engine.gateway = Broken()
    engine.step()
    assert engine.paused and engine.last_error
    assert engine.state['time'] == before['time']
    assert engine.state['entities'] == before['entities']
    assert engine.state['queue'][0]['id'] == before['queue'][0]['id']
    assert engine.state['queue'][0]['data']['pending_decision']['call_id'] == 77
    assert len([x for x in seen if x[1].get('stage') == 'encoding']) == engine.config.retries + 1
    assert '"op"' in seen[-1][1]['repair_error']
    assert engine.store.events(after=1)[0]['kind'] == 'adjudication_error'
    # Reopen the durable snapshot, then recover with an ordinary physical failure.
    location = Path(engine.store.db.execute('PRAGMA database_list').fetchone()[2]).parent
    resumed = Engine(location)
    class Recovered:
        def generate(self, role, context, *args, **kwargs):
            assert role == 'referee', 'Recovery must not resample the agent decision'
            if context.get('stage') == 'assessment':
                return {**physical_assessment(outcome='failed'), 'public_result': 'The material resists your pull.'}, 79
            return {'delta': []}, 80
    resumed.gateway = Recovered()
    resumed.step()
    assert not resumed.last_error
    decision = resumed.store.events(after=2)[0]
    assert decision['payload']['call_id'] == 77
    completion = next(e for e in resumed.state['queue'] if e['kind'] == 'action_complete')
    resumed.state['queue'] = [e for e in resumed.state['queue'] if e['time'] >= completion['time'] and e != completion]
    resumed.state['time'] = completion['time']
    resumed.dispatch(completion)
    assert resumed.state['entities'][a['id']]['last_result'] == 'The material resists your pull.'
    assert resumed.store.verify()['valid']
    resumed.store.close()


def test_stale_referee_fault_retains_completion_without_false_experience(engine):
    from eden.engine import AdjudicationError
    a = founders(engine.state)[0]
    add_stick(engine.state)
    engine.judge = lambda *args, **kwargs: ({'outcome': 'success', 'public_result': 'Bent.',
        'duration': 30, 'delta': []}, 1)
    engine.intent(a['id'], {'kind': 'world_action', 'description': 'Bend stick', 'targets': ['stick']})
    engine.state['entities']['stick']['rev'] += 1
    completion = next(e for e in engine.state['queue'] if e['kind'] == 'action_complete')
    engine.state['queue'] = [completion]
    before = digest(engine.state)
    def broken(*args, **kwargs):
        raise AdjudicationError('Invalid referee operation')
    engine.judge = broken
    engine.step()
    assert engine.paused and digest(engine.state) == before
    assert engine.store.events(after=1)[0]['kind'] == 'adjudication_error'


@pytest.mark.parametrize('delta, message', [
    ([{'op': None}], '"op"'),
    ([{'create_entity': {'mass': 1}}], '"op"'),
    ([{'op': 'invent'}], 'supported op values'),
    ([{'op': 'transfer_item', 'id': 'stick'}], 'holder'),
    ([{'op': 'create_entity', 'type': 'wood', 'mass': 1, 'sources': ['stick']}], 'source'),
])
def test_referee_operation_contract_rejects_malformed_changes_atomically(engine, delta, message):
    before = digest(engine.state)
    with pytest.raises(ValueError, match=message):
        k.apply_delta(engine.state, founders(engine.state)[0]['id'], delta, engine.config)
    assert digest(engine.state) == before


def test_transport_repair_includes_rejected_response(engine, monkeypatch):
    requests = []
    invalid = '{"outcome":"success","delta":{}}'
    def respond(request, timeout):
        requests.append(json.loads(request.data))
        content = invalid if len(requests) == 1 else json.dumps({'outcome': 'failed', 'delta': []})
        return io.BytesIO(json.dumps({'choices': [{'message': {'content': content}}]}).encode())
    monkeypatch.setattr(urllib.request, 'urlopen', respond)
    gateway = Gateway(Config(model='local'), engine.store)
    result, _ = gateway.generate('referee', {'intent': {'description': 'Try something'}}, 'agent-1', 0)
    assert result['outcome'] == 'failed'
    assert requests[1]['messages'][-2] == {'role': 'assistant', 'content': invalid}
    assert 'Error:' in requests[1]['messages'][-1]['content']


def test_referee_eval_reports_mock_failure_without_claiming_readiness(tmp_path):
    from eden.referee_eval import run
    output = tmp_path / 'validation.json'
    report = run(Config(backend='mock'), tmp_path / 'eval', output, repeats=1)
    assert report['complete'] and not report['passed']
    assert report['cases'] == 11
    assert not next(r for r in report['results'] if r['case'] == 'pickup')['valid']
    assert next(r for r in report['results'] if r['case'] == 'impossible')['valid']
    assert output.with_suffix('.sqlite').exists()


def test_encoder_cannot_change_assessment_to_hide_id_error(engine):
    a = founders(engine.state)[0]
    add_stick(engine.state)
    assessment = physical_assessment([{'kind': 'custody_changed', 'entity_id': 'stick', 'holder': a['id']}], 'success')
    contexts = []
    class Translator:
        def generate(self, role, context, *args, **kwargs):
            if context['stage'] == 'assessment':
                return assessment, 1
            contexts.append(copy.deepcopy(context))
            if len(contexts) == 1:
                return {'delta': [{'op': 'transfer_item', 'id': 'invented-stick', 'holder': a['id']}]}, 2
            if len(contexts) == 2:
                return {'outcome': 'failed', 'public_result': 'It is unreachable.', 'delta': []}, 3
            return {'delta': [{'op': 'transfer_item', 'id': 'stick', 'holder': a['id']}]}, 4
    engine.gateway = Translator()
    before = digest(engine.state)
    ruling, _ = engine.judge(a['id'], {'kind': 'world_action', 'description': 'Pick up stick'})
    assert ruling['outcome'] == 'success' and ruling['public_result'] == assessment['public_result']
    assert all(c['assessment'] == assessment for c in contexts)
    assert 'only delta' in contexts[-1]['repair_error']
    assert digest(engine.state) == before


def test_referee_records_explicit_grounded_reassessment(engine):
    a = founders(engine.state)[0]
    stages = []
    class Revising:
        def generate(self, role, context, *args, **kwargs):
            stages.append(context['stage'])
            if len(stages) == 1:
                return physical_assessment([{'kind': 'other', 'description': 'An uncertain effect'}]), 1
            if len(stages) == 2:
                return {'revision_request': {'reason': 'Actor has no tool for the assessed task.', 'evidence_ids': [a['id']]}}, 2
            if context['stage'] == 'assessment':
                assert context['revision_request']['previous_assessment']['effects']
                return physical_assessment(outcome='failed'), 3
            return {'delta': []}, 4
    engine.gateway = Revising()
    ruling, _ = engine.judge(a['id'], {'kind': 'world_action', 'description': 'Attempt a task'})
    assert ruling['outcome'] == 'failed' and len(ruling['assessment_history']) == 2
    assert ruling['assessment_history'][1]['revision_request']['call_id'] == 2


def test_representation_limit_is_not_physical_failure(engine):
    from eden.inference import AdjudicationError
    a = founders(engine.state)[0]
    class Limited:
        def generate(self, role, context, *args, **kwargs):
            if context['stage'] == 'assessment':
                return physical_assessment([{'kind': 'other', 'description': 'A novel consequence'}]), 1
            return {'representation_limit': 'This consequence requires an unimplemented mechanism.'}, 2
    engine.gateway = Limited()
    before = digest(engine.state)
    with pytest.raises(AdjudicationError, match='Representation limit'):
        engine.judge(a['id'], {'kind': 'world_action', 'description': 'Novel attempt'})
    assert digest(engine.state) == before


def test_assessment_distance_and_effect_consistency(engine):
    from eden.referee import check_assessment, check_effects
    a = founders(engine.state)[0]
    add_stick(engine.state)
    lookup = {'stick': {'within_interaction_radius': True}}
    with pytest.raises(ValueError, match='contradicts'):
        check_assessment({**physical_assessment(outcome='failed'), 'distance_obstacles': ['stick']}, lookup)
    assessment = physical_assessment([{'kind': 'material_removed', 'entity_id': 'stick', 'mass': .5}])
    with pytest.raises(ValueError, match='mass removal'):
        check_effects(engine.state, engine.state, a['id'], assessment, [])
    assessment = physical_assessment([{'kind': 'need_relieved', 'need': 'hunger', 'amount': .1}])
    engine.state['entities'][a['id']]['needs']['hunger'] = .4
    with pytest.raises(ValueError, match='relief'):
        check_effects(engine.state, engine.state, a['id'], assessment, [])
    with pytest.raises(ValueError, match='no effects'):
        check_effects(engine.state, engine.state, a['id'], physical_assessment(), [{'op': 'record_fact'}])
