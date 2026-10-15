# Implementation boundaries

This executable first version implements launch → perceive → decide → adjudicate →
schedule → commit → inspect/replay, with an external model server. It does not claim
that every research or scale milestone in the design proposal has passed.

## Architecture

`kernel.py` owns IDs, custody, material accounting, positions, memory tools,
perception, and typed changes. `engine.py` schedules and executes events.
`inference.py` is the provider boundary. `store.py` commits state/history, raw model
calls and backups. The browser consumes saved state, never informs inhabitants.

Unknown physical verbs use `world_action.description`. The referee must express
effects using validated generic operations. Descriptive attributes cannot edit
protected engine fields or install behavior/code. Effects are validated initially
and committed at completion. Referenced changes trigger re-adjudication. A failed
operation rolls back the whole candidate.

Every event has a compressed full state snapshot and chained hashes. This differs
from the proposal's eventual compact-delta ledger with occasional snapshots. It
simplifies exact replay but needs storage/CPU benchmarks before scale. Hashes detect
accidental corruption, not malicious complete rewriting by a database owner.

Private Markdown is a virtual filesystem inside SQLite. Agents control names and
organization; revisions are retained, including deletions. Limits are 256 files and
1 MiB per agent. Retrieval is substring-based with bounded excerpts. There is no
automatic canonical memory, summarizer, or inherited private context.

Model calls commit before the world effects they support. A crash can leave orphan
calls and cause a pending activation to call again after restart; committed physical
effects cannot be duplicated. Those calls remain in the audit record. Playback never
calls a model. Runtime source hashes are recorded; backend/weights provenance beyond
served aliases must be supplied by the operator.

## Time and failures

Events order by simulated time, priority, and stable insertion ID. Fractional seconds
are supported. Activities overlap in world time; inference requests are serial.
Sleep and waiting schedule wakeups instead of repeatedly invoking the model.

Persistent transport/format failure pauses with the original pending event intact.
Headless mode exits with diagnostics; with the viewer, controls remain available.
Malformed intents receive recorded feedback; invalid referee changes receive bounded
repair attempts then conservative failure. Every activation, including memory tools,
costs simulated time. Output token limits and retries bound individual calls; full
run token quotas and a throughput dashboard are future work.

## Current simulation abstractions

- Terrain is a lazy height field with sparse excavation. The canvas draws surface
  wireframes, not full volumetric meshes. Objects use simple markers.
- Walking checks straight segments. Obstacles require an agent-chosen waypoint;
  automatic A*, swimming, climbing and movement interpolation are future work.
- Natural anchors have finite mass and coarse daily renewal. Searches cannot refill
  them. Transformations conserve total mass, not chemical/elemental composition.
- Objects support attributes, delayed decay, and later referee interactions. Fire
  spread, continuous fuel burn, fluids, gravity, autonomous machines and structural
  collapse are not automatically inferred from prose. Timed effects currently cover
  conditions and decay. This is a limit to open-ended physical dynamics.
- Perception enforces distance, light/weather and terrain occlusion. It exposes
  coarse attributes of visible entities; nuanced inference/deception and hearing
  occlusion need refinement. Referees never receive private notes.
- Animals have seeded inexpensive movement. Predation, animal reproduction,
  seasons and ecosystem balance are not yet modeled.
- Needs and disease severity are coarse research variables, not medical models.
  Nutrition/hydration exhaustion, severe conditions, and aging can cause death.
  More calibrated hazards and environmental injury need evaluation.
- Reproduction requires nearby adults, compatible biological roles and independent
  mutual consent. Consent expires after a day and is consumed per attempt. A simple
  25% chance schedules 280-day gestation. Fertility cycles, genetics and miscarriage
  are not implemented.
- Newborns have independent empty files and no model activations before age three.
  Care reduces dependent needs at a cost to a nourished caregiver. Later children
  receive their age but share the adult model contract. Developmental language,
  cognition and carrying dependent children require refinement.
- Population limits pause a pending birth instead of discarding it. Resume with a
  higher `--max-population`. Large populations are not benchmarked.

## Model and viewer boundaries

The service must expose Chat Completions with system/user messages, temperature,
maximum output tokens and optional JSON response mode. Native tool calling is not
required. Configure a model with enough context capacity and an appropriate chat
template. Exact prompt/response envelopes, usage if supplied, settings, latency and
errors are retained. Secret authorization headers are omitted. Reported deliberation
is model-emitted text, not a claim to expose hidden internal reasoning.

One endpoint serves all roles, optionally with a separate referee model name.
Per-role servers, per-agent model assignments, capability probing, batching and
parallel inference are later extensions. The UI polls committed state each second;
SSE/WebSockets can be added without changing world semantics.

## Remaining research and scale work

Real-model benchmarks should measure novel attempts, repeated depletion, grounded
invention, memory retrieval, intergenerational knowledge transfer, survival, causal
defects, and token/latency costs. Scripted test outcomes cannot establish emergence.

Automatic episode indexing, experiment-branching UI, an optional critic model,
embedding retrieval, richer metrics and multi-day HPC evaluation remain from the
larger proposal. Whole-run replay and consistent checkpoint continuation work now.
