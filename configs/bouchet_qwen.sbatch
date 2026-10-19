#!/usr/bin/env bash
#SBATCH --job-name=eden-qwen7b
#SBATCH --account=pi_sk2433
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300
# Submit from a staged Eden checkout; logs go to that checkout unless overridden.
set -euo pipefail
EDEN_PROJECT_DIR="${EDEN_PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
EDEN_ARCHIVE="${EDEN_ARCHIVE:?Set a durable, unique archive directory}"
EDEN_PYTHON="${EDEN_PYTHON:?Set the existing Transformers environment Python}"
EDEN_MODEL_PATH="${EDEN_MODEL_PATH:?Set the cached instruction-model snapshot path}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
cd "$EDEN_PROJECT_DIR"
if [[ -n "${EDEN_VALIDATION_REPORT:-}" ]]; then
  "$EDEN_PYTHON" -c 'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r.get("complete") and r.get("passed") else 2)' "$EDEN_VALIDATION_REPORT"
fi
mkdir -p "$EDEN_ARCHIVE"
exec 9>"$EDEN_ARCHIVE/.job.lock"
flock -n 9 || { echo "Another job owns this experiment" >&2; exit 1; }
if "$EDEN_PYTHON" - "$EDEN_ARCHIVE/progress.json" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
done = path.exists() and json.loads(path.read_text()).get("status") in {"target_reached", "extinct"}
sys.exit(0 if done else 1)
PY
then
  echo "Experiment already reached its target or ended in extinction; no model load needed."
  exit 0
fi
EDEN_WORK="$(mktemp -d "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/eden-${SLURM_JOB_ID}.XXXXXXXX")"
printf '%s\n' "$EDEN_WORK" > "$EDEN_ARCHIVE/work-directory.txt"
hostname > "$EDEN_ARCHIVE/node.txt"
nvidia-smi > "$EDEN_ARCHIVE/gpu.txt"
if [[ -f "$EDEN_ARCHIVE/latest.sqlite" ]]; then
  mkdir -p "$EDEN_WORK/run"
  cp "$EDEN_ARCHIVE/latest.sqlite" "$EDEN_WORK/run/world.sqlite"
fi
EDEN_SERVER_PID=""
EDEN_TRIAL_PID=""
finish() {
  trap - EXIT USR1 TERM INT
  if [[ -n "$EDEN_TRIAL_PID" ]]; then
    kill -TERM "$EDEN_TRIAL_PID" 2>/dev/null || true
    wait "$EDEN_TRIAL_PID" || true
  fi
  if [[ -n "$EDEN_SERVER_PID" ]]; then
    kill -TERM "$EDEN_SERVER_PID" 2>/dev/null || true
    wait "$EDEN_SERVER_PID" || true
  fi
}
trap finish EXIT
trap 'exit 0' USR1 TERM INT
# A random loopback port prevents collisions between jobs sharing a node.
EDEN_PORT="$($EDEN_PYTHON -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')"
"$EDEN_PYTHON" -u -m eden.hf_server --model-path "$EDEN_MODEL_PATH" --model-name qwen2.5-7b-instruct --port "$EDEN_PORT" > "$EDEN_ARCHIVE/server-${SLURM_JOB_ID}.log" 2>&1 &
EDEN_SERVER_PID=$!
EDEN_READY=0
for ((i=0; i<180; i++)); do
  if ! kill -0 "$EDEN_SERVER_PID" 2>/dev/null; then
    tail -40 "$EDEN_ARCHIVE/server-${SLURM_JOB_ID}.log"
    exit 1
  fi
  if "$EDEN_PYTHON" -c 'import sys,urllib.request; print(urllib.request.urlopen("http://127.0.0.1:"+sys.argv[1]+"/health",timeout=2).read().decode())' "$EDEN_PORT" > "$EDEN_ARCHIVE/backend.json" 2>/dev/null; then EDEN_READY=1; break; fi
  sleep 5
done
if [[ "$EDEN_READY" != 1 ]]; then echo "Model did not become ready" >&2; exit 1; fi
"$EDEN_PYTHON" - "$EDEN_PORT" "$EDEN_ARCHIVE/config.json" <<'PY'
import json, sys
from eden.config import Config
config = Config(model="qwen2.5-7b-instruct", endpoint=f"http://127.0.0.1:{sys.argv[1]}/v1",
                response_format="none", max_tokens=1400, timeout=180, retries=2)
with open(sys.argv[2], "w") as file:
    json.dump(config.to_dict(), file, indent=2)
PY
if [[ "${EDEN_REFEREE_EVAL:-0}" == 1 ]]; then
  "$EDEN_PYTHON" -u -m eden.referee_eval --config "$EDEN_ARCHIVE/config.json" \
    --work "$EDEN_WORK/eval" --output "$EDEN_ARCHIVE/validation.json" --repeats "${EDEN_EVAL_REPEATS:-10}"
  exit $?
fi
# Restored runs retain their recorded endpoint; record the fresh server port if needed.
if [[ -f "$EDEN_WORK/run/world.sqlite" ]]; then
  "$EDEN_PYTHON" - "$EDEN_WORK/run" "$EDEN_PORT" <<'PY'
import sys
from eden.cli import writer_lock
from eden.engine import Engine
with writer_lock(sys.argv[1]):
    engine = Engine(sys.argv[1])
    engine.configure(endpoint=f"http://127.0.0.1:{sys.argv[2]}/v1")
    engine.store.close()
PY
fi
if [[ "${EDEN_DIAGNOSE_COMMUNICATION:-0}" == 1 ]]; then
  "$EDEN_PYTHON" -u -m eden.diagnose --snapshot "$EDEN_WORK/run/world.sqlite" \
    --config "$EDEN_ARCHIVE/config.json" --work "$EDEN_WORK/diagnostics" \
    --output "$EDEN_ARCHIVE/communication-diagnostic-${SLURM_JOB_ID}.json"
fi
EDEN_TRIAL_ARGS=()
if [[ -n "${EDEN_INTERVENTION_NOTE:-}" ]]; then
  EDEN_TRIAL_ARGS+=(--intervention-note "$EDEN_INTERVENTION_NOTE")
fi
"$EDEN_PYTHON" -u -m eden.trial --run "$EDEN_WORK/run" --archive "$EDEN_ARCHIVE" --config "$EDEN_ARCHIVE/config.json" --until "${EDEN_UNTIL:-259200}" --max-events "${EDEN_MAX_EVENTS:-30000}" --max-wall-seconds "${EDEN_MAX_WALL_SECONDS:-0}" "${EDEN_TRIAL_ARGS[@]}" &
EDEN_TRIAL_PID=$!
wait "$EDEN_TRIAL_PID"
