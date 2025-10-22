#!/usr/bin/env bash
# Run mLoRA training on a Delta interactive node, stream live GPU util to console,
# log *only* training output to file, and report mean GPU compute & memory utilization.
set -euo pipefail

########## --- CONFIG (edit these if needed) --- ##########
TRAIN_PY="/projects/beis/akanodia/CS-598-FAL-Project/mlora_pp_train.py"
CONFIG="/projects/beis/akanodia/CS-598-FAL-Project/demo/lora/llama_job1.yaml"
BASE_MODEL="meta-llama/Llama-3.1-8B"
MODEL_TYPE="llama"

# Dist config for this interactive node
NPROC=2                 # 2 GPUs on this node
MASTER_ADDR="127.0.0.1"
MASTER_PORT="${MASTER_PORT:-12345}"

# GPU sampling config
SAMPLE_MS=50            # sample every 50 ms for live view + averaging
AVG_INTERVAL=10         # print rolling mean to console every 10 seconds
FILTER_MIN=30           # only average compute util samples strictly > 30%

# Logs (match SLURM directory)
LOG_ROOT="/projects/beis/dsaha1/CS-598-FAL-Project/logs"
mkdir -p "$LOG_ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$LOG_ROOT/new_llama_job_1_${STAMP}.out"
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
echo "[info] Sampling GPU util every ${SAMPLE_MS} ms (compute-avg only values > ${FILTER_MIN}%)"
echo "[info] Training output will be logged to: $RUN_LOG"
# CSV with headers; columns will be:
# 1: timestamp, 2: index, 3: utilization.gpu [%], 4: memory.used [MiB], 5: memory.total [MiB]
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total \
  --format=csv -lms "$SAMPLE_MS" | tee "$TMP_UTIL" &
SAMPLE_PID=$!

# Rolling-average printer (every AVG_INTERVAL seconds)
ROLL_PID=""
{
  while sleep "$AVG_INTERVAL"; do
    if [[ -s "$TMP_UTIL" ]]; then
      # Compute rolling mean for compute util (> FILTER_MIN) and memory util (used/total %)
      read -r ROLL_COMP ROLL_MEM < <(
        awk -F, -v thr="$FILTER_MIN" '
          NR>1 {
            # $3 is "utilization.gpu [%]"
            g=$3+0
            if (g>thr) {gsum+=g; gn++}
            # $4 is "memory.used [MiB]", $5 is "memory.total [MiB]"
            mu=$4+0; mt=$5+0
            if (mt>0) { msum += (100.0 * mu / mt); mn++ }
          }
          END {
            if (gn>0) printf("%.2f ", gsum/gn); else printf("NA ");
            if (mn>0) printf("%.2f\n", msum/mn); else printf("NA\n");
          }
        ' "$TMP_UTIL"
      )

      MSG="[rolling-mean @ +${SECONDS}s]"
      if [[ "$ROLL_COMP" != "NA" ]]; then
        MSG+=" compute(>${FILTER_MIN}%)=${ROLL_COMP}%"
      else
        MSG+=" compute(>${FILTER_MIN}%)=n/a"
      fi
      if [[ "$ROLL_MEM" != "NA" ]]; then
        MSG+=" | memory=%${ROLL_MEM}"
      else
        MSG+=" | memory=n/a"
      fi
      echo "$MSG"
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
: > "$RUN_LOG"   # ensure the log file exists
"${TORCHRUN[@]}" --nproc_per_node="$NPROC" --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" "$WRAP" >>"$RUN_LOG" 2>&1

# Stop helpers
cleanup
trap - EXIT

# Final means across all GPUs and all samples
FINAL_COMP=""; FINAL_MEM=""
if [[ -s "$TMP_UTIL" ]]; then
  read -r FINAL_COMP FINAL_MEM < <(
    awk -F, -v thr="$FILTER_MIN" '
      NR>1 {
        g=$3+0; if (g>thr) {gsum+=g; gn++}
        mu=$4+0; mt=$5+0; if (mt>0) {msum += (100.0 * mu / mt); mn++}
      }
      END {
        if (gn>0) printf("%.2f ", gsum/gn); else printf(" ");
        if (mn>0) printf("%.2f\n", msum/mn); else printf("\n");
      }
    ' "$TMP_UTIL"
  )
fi

# Clean temp files
rm -f "$TMP_UTIL" "$WRAP"

# Print final summary
if [[ -n "${FINAL_COMP// }" ]]; then
  echo "[final] Mean GPU compute utilization (> ${FILTER_MIN}%): ${FINAL_COMP}%"
else
  echo "[final] Mean GPU compute utilization (> ${FILTER_MIN}%): n/a (no qualifying samples)"
fi

if [[ -n "${FINAL_MEM// }" ]]; then
  echo "[final] Mean GPU memory utilization: ${FINAL_MEM}%"
else
  echo "[final] Mean GPU memory utilization: n/a"
fi

echo "[done] Training log saved to: $RUN_LOG"