# Eden / Mesocosm

A persistent world for autonomous model-driven inhabitants. Agents choose goals and
manage their own Markdown memory; a referee proposes consequences; a deterministic
kernel validates and commits reality. A local spectator app displays terrain,
reported deliberations, memories, and replayable history.

**Python 3.11+ on Linux/macOS. No runtime Python dependencies, Node build, model
download, Ollama installation, or cloud account required.** Run from this directory.
Optional installation: `python3 -m pip install -e .` provides the `eden` command.

## Choose a model and launch

Start your preferred inference server separately, then supply its OpenAI-compatible
Chat Completions API base URL and served model name:

```bash
python3 -m eden run \
  --run runs/first-world \
  --endpoint http://127.0.0.1:8000/v1 \
  --model YOUR_SERVED_MODEL_NAME \
  --serve
```

Open **http://127.0.0.1:8765**. One model controls both founders, all descendants, and
the referee. Add `--referee-model OTHER_NAME` for a separate referee model served at
the same endpoint. If authentication is required, set `EDEN_API_KEY` in your shell;
authorization headers are never archived.

The endpoint is a running inference service, not a checkpoint path. Eden does not
load weights or allocate GPUs. The server must provide the appropriate chat template
for your model. If it rejects `response_format`, add `--response-format none`; Eden
still validates the JSON response. Native function calling is not required.

For headless HPC operation, omit `--serve`. Add `--paused --serve` to start paused,
`--steps 100` for a bounded smoke run, or `--until 86400` to stop before events beyond
one simulated day. Without a limit, the run continues until interrupted or extinct.
`--speed 0` runs as fast as inference permits; `--speed 10` targets 10 simulated
seconds per real second. It cannot accelerate the inference server.

For additional settings, copy [configs/example.json](configs/example.json), set the
model/endpoint, and use `--config your-config.json` for a new run. Inference is serial
by default, so only one loaded model is necessary.

## Try without a model

```bash
python3 -m eden run --mock --run runs/demo --serve
```

The **SCRIPTED MOCK** exercises files, observations, conversation, failed attempts,
rest, persistence, and the interface. Its behavior is an integration fixture, not
autonomous intelligence. It does not feed itself and eventually dies.

## Resume, switch models, inspect and replay

Ctrl-C or SIGTERM stops at a safe boundary after any active inference request.
Every completed event is already durable. Resume using the same directory:

```bash
python3 -m eden run --run runs/first-world --serve
python3 -m eden run --run runs/first-world --model NEW_MODEL --endpoint http://127.0.0.1:9000/v1 --serve
python3 -m eden status --run runs/first-world
python3 -m eden verify --run runs/first-world
python3 -m eden replay --run runs/first-world
```

Model changes are recorded. Files, identities, and history survive the switch.
Changing a mock run to real inference requires `--model` or `--endpoint`; plain
resume preserves the previous backend. Use a new run directory for a new seed.

The viewer offers a monochrome isometric map, follow/pan/zoom, agent reports and
needs, private-file inspection, event details, pause/step/speed controls, checkpoints,
and a history slider. Selecting a historical event changes only the view. It never
rewinds an active simulation. Replay makes zero model calls and plays recorded events
at one event per second. “Live / latest” returns to the current head.

The server binds to loopback. Use an SSH tunnel for HPC viewing through your cluster's
supported access path, with localhost/127.0.0.1 as the browser address. It is a local
operator tool, not a public hosting service. Other local users with port access can
inspect the run. Controls require a per-process token and same-origin requests;
replay mode disables them.

## Persistence and HPC

`world.sqlite` contains authoritative state, the event hash chain, all Markdown
revisions, exact prompts/responses, and pending events. Private Markdown uses a
**virtual per-agent filesystem stored transactionally in SQLite**, without giving
models host filesystem access.

Each event stores a compressed full state checkpoint. This simplifies recovery and
exact historical seek but costs more storage than incremental projections. Benchmark
storage before long runs. Model-call records are currently uncompressed. Initial
events record Python source hashes. Archive model weight hashes, quantization, chat
templates and server versions separately: a served alias does not identify weights.

Run the live database on node-local scratch, with one writer. Do not place a live
SQLite WAL database on a cluster network filesystem. To back up a running database:

```bash
python3 -m eden checkpoint --run runs/first-world --output /durable/path/checkpoint-001.sqlite
python3 -m eden export --run runs/first-world --output /durable/path/history.jsonl
```

Output must be a new filename. Restore a checkpoint as `world.sqlite` in a new run
directory and launch Eden there. It includes everything needed for replay and resume.
Do not directly copy a live database while its WAL is active.

[configs/hpc.sbatch](configs/hpc.sbatch) stages to node-local scratch, checkpoints
periodically, and archives on exit. Supply your cluster allocation directives and
the environment variables documented in that file. Inference is managed separately;
the template does not allocate model GPUs. Allow enough scheduler signal lead time
for configured inference timeouts/retries to finish.

For an already installed Transformers/PyTorch environment and a cached instruction
checkpoint, `python -m eden.hf_server --model-path /path/to/snapshot` provides a small
single-GPU adapter. It loads only local files and binds to loopback; run it inside
your GPU allocation. This optional adapter requires the existing model dependencies;
the Eden simulation itself still has no runtime dependencies. Use
`--response-format none` with this adapter because it does not constrain JSON decoding.

[configs/bouchet_qwen.sbatch](configs/bouchet_qwen.sbatch) runs this adapter and a
bounded trial in one Bouchet allocation. Set `EDEN_PROJECT_DIR`, `EDEN_ARCHIVE`,
`EDEN_PYTHON`, and `EDEN_MODEL_PATH`. The default target is three simulated days,
with a 30,000-event ceiling. `EDEN_MAX_EVENTS=60` selects a short smoke trial;
`EDEN_UNTIL` changes the target in simulated seconds. The runner writes atomic
`progress.json` reports and a rolling `latest.sqlite` checkpoint in the archive.
Resubmitting against the same archive resumes the saved world and records the new
inference endpoint. It never silently restarts the inhabitants.

## Implemented scope

- Seeded voxel terrain, finite natural resource anchors, sparse excavation, checked
  movement, bounded perception, local speech, and basic wildlife.
- Independent agents with freely organized Markdown, structured intents, actual
  reported deliberations, bounded output, and swappable models.
- Arbitrary physical proposals, model adjudication, material-accounted object
  creation/transformation, custody, scoped facts, and delayed effects.
- Concurrent simulated activities under one event writer; completion-time checks,
  interruption, and re-adjudication when referenced state changes.
- Coarse needs, illness, rest, aging, death, consent, conception, gestation, births,
  dependent children, care, and independent descendants' private memory.
- Durable history, model-free replay, hash verification, backup/export, and browser UI.

This version has **not yet been evaluated with a real model**. Engineering tests
cannot establish referee scientific accuracy or useful autonomous behavior. The
system does not guarantee survival, reproduction, or civilization.

[docs/implementation.md](docs/implementation.md) describes current abstractions and
remaining parts of the broader proposal, including more general environmental
dynamics, pathfinding, developmental cognition, storage optimization, and evaluation.

## Portable wireframe replay

Create an offline HTML viewer from a standalone checkpoint copied to local storage:

```bash
python3 -m eden.recording --database /path/to/checkpoint.sqlite --output replay.html
```

Open the HTML file in a browser. It embeds recorded positions, agent reports,
private-file states, event details, wireframe terrain, and recent movement trails.
It needs no inference server, SSH tunnel, internet connection, or local web server.
The history slider selects the recorded event; Play replay advances through events.

For remote transfer, export `.json.gz` instead of `.html`, then build the viewer
locally with `--recording snapshot.json.gz --output replay.html`. The exporter
verifies the full source event/state hash chain. This compact spectator extract
omits raw model requests and observation payloads; retain the full SQLite archive
for complete auditing. Export only self-contained checkpoints, not a live database
whose active WAL contains newer events.

## Nearby conversation opportunities

Each decision now includes an optional social question when another visible living
inhabitant is within the 20-metre speech range: do they want to say anything, or
choose another action? The model still chooses whether to speak and supplies all
words. This cue does not create a conversation or edit anyone's private notes.

For intervention experiments, `eden.trial --intervention-note "..."` records the
operator's change and current source hashes before resuming the saved world. The
Bouchet script accepts `EDEN_INTERVENTION_NOTE` and, optionally,
`EDEN_DIAGNOSE_COMMUNICATION=1`. The latter asks diagnostic questions in copies of
the agents' last recorded contexts before continuing. Replies are archived separately
and their proposed intents are never executed in the world.

## Tests

```bash
python3 -m pytest -q
node --check eden/static/app.js
```

Only development tests require pytest. Node is optional for syntax checking and is
not an application dependency. Tests cover conservation, competing custody, stale
commits, delayed effects, independent children, memory isolation, inference failures,
model routing, checkpoint/restore, replay, HTTP handlers, and archive tampering.
Live socket tests skip only when the execution environment forbids socket creation.
