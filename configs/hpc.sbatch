#!/usr/bin/env bash
#SBATCH --job-name=eden
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --signal=B:USR1@600
#SBATCH --output=eden-%j.log
# Add cluster account/partition directives. Model inference is served separately.
# Required: EDEN_PROJECT_DIR, EDEN_ARCHIVE, EDEN_MODEL, EDEN_ENDPOINT
# Optional: EDEN_API_KEY, EDEN_PYTHON, EDEN_CHECKPOINT_SECONDS
set -euo pipefail
: "${EDEN_PROJECT_DIR:?Set the project checkout path}"
: "${EDEN_ARCHIVE:?Set the durable run archive path}"
: "${EDEN_MODEL:?Set the served model name}"
: "${EDEN_ENDPOINT:?Set the inference API base URL}"
: "${SLURM_TMPDIR:?Node-local scratch is required}"
EDEN_PYTHON="${EDEN_PYTHON:-python3}"
EDEN_CHECKPOINT_SECONDS="${EDEN_CHECKPOINT_SECONDS:-300}"
cd "$EDEN_PROJECT_DIR"
mkdir -p "$EDEN_ARCHIVE"
exec 9>"$EDEN_ARCHIVE/.archive.lock"
flock -n 9 || { echo "Another job owns this archive" >&2; exit 1; }
EDEN_WORK_DIR="$(mktemp -d "$SLURM_TMPDIR/eden.XXXXXXXX")"
if [[ -f "$EDEN_ARCHIVE/latest.sqlite" ]]; then
  cp "$EDEN_ARCHIVE/latest.sqlite" "$EDEN_WORK_DIR/world.sqlite"
fi
EDEN_PID=""
EDEN_TIMER_PID=""
checkpoint() {
  if [[ -f "$EDEN_WORK_DIR/world.sqlite" ]]; then
    EDEN_SNAPSHOT="$EDEN_ARCHIVE/checkpoint-${SLURM_JOB_ID:-local}-$(date +%s)-$RANDOM.sqlite"
    "$EDEN_PYTHON" -m eden checkpoint --run "$EDEN_WORK_DIR" --output "$EDEN_SNAPSHOT"
    EDEN_LATEST_TEMP="$EDEN_ARCHIVE/.latest-${SLURM_JOB_ID:-local}.sqlite"
    cp "$EDEN_SNAPSHOT" "$EDEN_LATEST_TEMP"
    mv "$EDEN_LATEST_TEMP" "$EDEN_ARCHIVE/latest.sqlite"
  fi
}
finish() {
  trap - EXIT USR1 TERM INT
  if [[ -n "$EDEN_TIMER_PID" ]]; then kill "$EDEN_TIMER_PID" 2>/dev/null || true; fi
  if [[ -n "$EDEN_PID" ]]; then
    kill -TERM "$EDEN_PID" 2>/dev/null || true
    wait "$EDEN_PID" || true
  fi
  checkpoint
}
trap finish EXIT
trap 'exit 0' USR1 TERM INT
"$EDEN_PYTHON" -m eden run --run "$EDEN_WORK_DIR" --model "$EDEN_MODEL" --endpoint "$EDEN_ENDPOINT" &
EDEN_PID=$!
while kill -0 "$EDEN_PID" 2>/dev/null; do
  sleep "$EDEN_CHECKPOINT_SECONDS" &
  EDEN_TIMER_PID=$!
  wait "$EDEN_TIMER_PID" || true
  EDEN_TIMER_PID=""
  if kill -0 "$EDEN_PID" 2>/dev/null; then checkpoint; fi
done
wait "$EDEN_PID"
