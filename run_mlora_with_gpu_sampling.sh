#!/usr/bin/env bash
# Run mLoRA training on a Delta interactive node, stream live GPU util to console,
# log *only* training output to file, and report mean GPU compute utilization.
set -euo pipefail

########## --- CONFIG (edit these if needed) --- ##########
TRAIN_PY="/projects/beis/akanodia/CS-598-FAL-Project/mlora_pp_train.py"
CONFIG="/projects/beis/akanodia/CS-598-FAL-Project/demo/lora/qwen_job_1_2.yaml"
BASE_MODEL="Qwen/Qwen3-1.7B"
MODEL_TYPE="qwen"

# Dist config for this interactive node
NPROC=2                 # 2 GPUs on this node
MASTER_ADDR="127.0.0.1"
MASTER_PORT="${MASTER_PORT:-12345}"

# GPU sampling config
SAMPLE_MS=50            # sample every 50 ms for live view + averaging
AVG_INTERVAL=10         # print rolling mean to console every 10 seconds

# Logs (match SLURM directory)
LOG_ROOT="/projects/beis/akanodia/CS-598-FAL-Project/logs"
mkdir -p "$LOG_ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$LOG_ROOT/qwen_job_1_2_${STAMP}.out"
###########################################################

python --version || true

# Decide how to launch torch distributed
if command -v torchrun >/dev/null 2>&1; then
  TORCHRUN=(torchrun)
else
  TORCHRUN=(python -m torch.distributed.run)
fi

# Start GPU util sampler: print *to terminal* and also tee into a temp CSV for averaging
TMP_UTIL="$(mktemp "/tmp/gpu_util_${STAMP}_XXXX.csv")"
echo "[info] Sampling GPU util every ${SAMPLE_MS} ms"
echo "[info] Training output will be logged to: $RUN_LOG"
# CSV with header + units so the terminal looks nice; we'll parse numerics from column 3 later.
nvidia-smi --query-gpu=timestamp,index,utilization.gpu --format=csv -lms "$SAMPLE_MS" | tee "$TMP_UTIL" &
SAMPLE_PID=$!

# Rolling-average printer (every AVG_INTERVAL seconds)
ROLL_PID=""
{
  while sleep "$AVG_INTERVAL"; do
    if [[ -s "$TMP_UTIL" ]]; then
      # mean across all GPUs and all samples so far (skip header)
      MEAN_SO_FAR="$(awk -F, 'NR>1 {g=$3+0; sum+=g; n++} END { if (n>0) printf("%.2f", sum/n) }' "$TMP_UTIL")"
      if [[ -n "${MEAN_SO_FAR:-}" ]]; then
        echo "[rolling-mean @ +${SECONDS}s] GPU compute utilization ≈ ${MEAN_SO_FAR}%"
      fi
    fi
  done
} &
ROLL_PID=$!

cleanup() {
  # stop background helpers
  kill "$ROLL_PID" 2>/dev/null || true
  kill "$SAMPLE_PID" 2>/dev/null || true
}
trap cleanup EXIT

export MASTER_ADDR="$MASTER_ADDR"
export MASTER_PORT="$MASTER_PORT"

# ---- Wrapper: reads LOCAL_RANK/RANK set by torchrun and forwards correct args ----
WRAP="$(mktemp "/tmp/mlora_launch_wrapper_${STAMP}_XXXX.py")"
cat > "$WRAP" <<'PY'
import os, sys

TRAIN_PY = os.environ["MLORA_TRAIN_PY"]
BASE_MODEL = os.environ["MLORA_BASE_MODEL"]
MODEL_TYPE = os.environ["MLORA_MODEL_TYPE"]
CONFIG = os.environ["MLORA_CONFIG"]

local_rank = os.environ.get("LOCAL_RANK", "0")
rank = os.environ.get("RANK", "0")

cmd = [
    sys.executable, TRAIN_PY,
    "--base_model", BASE_MODEL,
    "--model_type", MODEL_TYPE,
    "--config", CONFIG,
    "--pipeline",
    "--device", f"cuda:{local_rank}",
    "--rank", rank,
    "--nodes", "2",
    "--recompute",
    "--precision", "fp32",
]

print(f"[wrapper] RANK={rank} LOCAL_RANK={local_rank} -> device=cuda:{local_rank}")
os.execvp(cmd[0], cmd)
PY

export MLORA_TRAIN_PY="$TRAIN_PY"
export MLORA_BASE_MODEL="$BASE_MODEL"
export MLORA_MODEL_TYPE="$MODEL_TYPE"
export MLORA_CONFIG="$CONFIG"

echo "[info] Launching training with ${TORCHRUN[*]} (nproc_per_node=$NPROC) using wrapper: $WRAP"
# IMPORTANT: send *all* training output to the log file ONLY (not to terminal)
# Create the log file up-front so it definitely exists.
: > "$RUN_LOG"
"${TORCHRUN[@]}" --nproc_per_node="$NPROC" --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" "$WRAP" >>"$RUN_LOG" 2>&1

# Stop helpers
cleanup
trap - EXIT

# Final mean across all GPUs and all samples (skip header, parse numeric)
if [[ -s "$TMP_UTIL" ]]; then
  FINAL_MEAN="$(awk -F, 'NR>1 {g=$3+0; sum+=g; n++} END { if (n>0) printf("%.2f", sum/n); else print "" }' "$TMP_UTIL")"
else
  FINAL_MEAN=""
fi

# Clean temp files
rm -f "$TMP_UTIL" "$WRAP"

if [[ -n "$FINAL_MEAN" ]]; then
  echo "[final] Mean GPU compute utilization: ${FINAL_MEAN}%"
else
  echo "[final] Mean GPU compute utilization: n/a (no samples)"
fi

echo "[done] Training log saved to: $RUN_LOG"