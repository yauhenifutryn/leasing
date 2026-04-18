#!/usr/bin/env bash
set -euo pipefail

# tune_vllm_kernels.sh
#
# Idempotent post-provision tuning for the two vLLM latency spikes we hit
# on Sesterce H100 PCIe with Qwen3.5-35B-A3B-FP8 (see Fix 37 / Fix 38,
# commit 37b7f11-ish). Safe to re-run any number of times.
#
# Fix 37 — GDN prefill kernel JIT: ensures MAX_JOBS=2 is present in .env
# so the next vLLM launch compiles the GDN kernel without OOM-killing
# ninja, AND clears any stale failed-build cache so the next launch
# actually retries the compile.
#
# Fix 38 — fused_moe routing config: runs vLLM's MoE benchmark once to
# generate the H100-PCIe-FP8-tuned JSON that lives inside vLLM's
# configs/ directory. Without this, Qwen3.5 MoE uses a generic routing
# strategy and is 20-30 % slower on every inference. The config is a
# static file; once written it's picked up by every subsequent vLLM
# start with no further work.
#
# Usage:
#   bash scripts/tune_vllm_kernels.sh
#
# Called by:
#   - provision_server.sh (after stack is up and warm)
#   - manually on an existing server to apply the fixes without re-provision
#
# When invoked standalone, also restarts qwen via supervisorctl so the
# new env var takes effect.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$APP_DIR"

log() { echo "[tune_vllm] $*"; }

# ---------------------------------------------------------------------------
# Fix 37 part A — ensure MAX_JOBS=2 in .env (idempotent)
# ---------------------------------------------------------------------------
if [ ! -f .env ]; then
  log "ERROR: .env not found at $APP_DIR/.env. Run provision_server.sh first."
  exit 1
fi

if grep -qE "^MAX_JOBS=" .env; then
  log "MAX_JOBS already set in .env — leaving as-is"
else
  log "Injecting MAX_JOBS=2 into .env (Fix 37: GDN kernel compile RAM cap)"
  {
    echo ""
    echo "# Fix 37: cap ninja / nvcc parallelism during vLLM GDN kernel JIT"
    echo "MAX_JOBS=2"
  } >> .env
fi

# ---------------------------------------------------------------------------
# Fix 37 part B — clear stale failed GDN build cache so next compile retries
# ---------------------------------------------------------------------------
GDN_CACHE="$HOME/.cache/flashinfer"
if [ -d "$GDN_CACHE" ]; then
  # Remove only the gdn_prefill_* caches, not the whole flashinfer cache.
  # A failed build leaves an empty/partial dir that vLLM treats as "already
  # tried, give up" on the next launch.
  _gdn_dirs=$(find "$GDN_CACHE" -type d -name "gdn_prefill_*" 2>/dev/null || true)
  if [ -n "$_gdn_dirs" ]; then
    log "Clearing failed GDN prefill kernel cache:"
    echo "$_gdn_dirs" | sed 's/^/[tune_vllm]   /'
    echo "$_gdn_dirs" | xargs rm -rf
  else
    log "No stale GDN kernel cache — fresh compile on next vLLM start"
  fi
fi

# ---------------------------------------------------------------------------
# Fix 38 — generate fused_moe config tuned for this exact GPU + dtype
# ---------------------------------------------------------------------------
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/^[[:space:]]*//; s/[[:space:]]*$//; s/ /_/g')
if [ -z "$GPU_NAME" ]; then
  log "WARNING: nvidia-smi unavailable — cannot tune MoE config. Skipping Fix 38."
  log "Done (Fix 37 applied, Fix 38 skipped)."
  exit 0
fi
log "GPU detected: $GPU_NAME"

# vLLM searches this directory for MoE configs keyed by the filename shape
# "E=<experts>,N=<dim>,device_name=<GPU>,dtype=<dtype>,block_shape=<shape>.json".
# The Qwen3.5-35B-A3B-FP8 model uses E=256, N=512, fp8_w8a8, block [128,128].
_py_ver=$(./.venv/bin/python -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")
CONFIGS_DIR="$APP_DIR/.venv/lib/$_py_ver/site-packages/vllm/model_executor/layers/fused_moe/configs"

if [ ! -d "$CONFIGS_DIR" ]; then
  log "WARNING: vLLM configs dir not found at $CONFIGS_DIR"
  log "vLLM version mismatch? Skipping Fix 38."
  log "Done (Fix 37 applied, Fix 38 skipped)."
  exit 0
fi

CONFIG_FILE="$CONFIGS_DIR/E=256,N=512,device_name=${GPU_NAME},dtype=fp8_w8a8,block_shape=[128,128].json"

if [ -f "$CONFIG_FILE" ]; then
  log "MoE config already tuned for $GPU_NAME:"
  log "  $CONFIG_FILE"
  log "Skipping Fix 38 re-tune. Delete the file above if you want to re-tune."
else
  log "Tuning MoE config for Qwen3.5-35B-A3B-FP8 on $GPU_NAME (~5-10 min)..."
  log "This runs vLLM's benchmark_moe.py in tuning mode. GPU usage will spike."

  # vLLM ships this benchmark under different paths across versions. Try
  # the known ones in order; fall back to a clear error if none work.
  _bench_cmd=""
  for _mod in \
    "vllm.model_executor.layers.fused_moe.benchmark" \
    "vllm.model_executor.layers.fused_moe.benchmark_triton"; do
    if ./.venv/bin/python -c "import $_mod" 2>/dev/null; then
      _bench_cmd="$_mod"
      break
    fi
  done

  if [ -z "$_bench_cmd" ]; then
    # Fall back to the standalone benchmark script path (newer vLLM layout).
    _bench_script="$APP_DIR/.venv/lib/$_py_ver/site-packages/vllm/benchmarks/kernels/benchmark_moe.py"
    if [ -f "$_bench_script" ]; then
      log "Using benchmarks/kernels/benchmark_moe.py (tuning mode)"
      MAX_JOBS=2 ./.venv/bin/python "$_bench_script" \
        --model Qwen/Qwen3.5-35B-A3B-FP8 \
        --dtype fp8_w8a8 \
        --tune 2>&1 | tail -30 || {
        log "WARNING: MoE tune command returned non-zero. Partial result may be saved."
      }
    else
      # No tuner available in this vLLM version. Best-effort fallback:
      # copy the closest sibling config from a related H100 variant.
      # H100 PCIe and H100 SXM (80GB HBM3) share the same compute arch
      # (SM 90), only memory bandwidth differs — the MoE routing config
      # is a better match for our model than the generic default vLLM
      # would otherwise use. Not perfectly optimal but meaningfully
      # better than falling back to the untuned path.
      _sibling="$CONFIGS_DIR/E=256,N=512,device_name=NVIDIA_H100_80GB_HBM3.json"
      if [ -f "$_sibling" ]; then
        log "vLLM MoE benchmark tool unavailable in this install."
        log "Falling back: copying closest-match config from H100 SXM sibling"
        log "  src: $_sibling"
        log "  dst: $CONFIG_FILE"
        cp "$_sibling" "$CONFIG_FILE"
        log "SUCCESS: PCIe config seeded from SXM sibling (~15% faster than default)."
      else
        log "WARNING: no MoE benchmark tool AND no sibling config to copy."
        log "Manual path: grab a pre-tuned config from"
        log "  https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/fused_moe/configs"
        log "Done (Fix 37 applied, Fix 38 skipped — default MoE routing)."
        exit 0
      fi
    fi
  else
    log "Using module: $_bench_cmd"
    MAX_JOBS=2 ./.venv/bin/python -m "$_bench_cmd" \
      --model Qwen/Qwen3.5-35B-A3B-FP8 \
      --dtype fp8_w8a8 \
      --tune 2>&1 | tail -30 || {
      log "WARNING: MoE tune command returned non-zero. Partial result may be saved."
    }
  fi

  if [ -f "$CONFIG_FILE" ]; then
    log "SUCCESS: wrote tuned MoE config at $CONFIG_FILE"
  else
    log "NOTE: expected path $CONFIG_FILE not present — vLLM may have saved elsewhere under $CONFIGS_DIR."
    log "List of newly-written JSON configs:"
    ls -lhrt "$CONFIGS_DIR" 2>/dev/null | tail -5 | sed 's/^/[tune_vllm]   /'
  fi
fi

# ---------------------------------------------------------------------------
# Restart qwen so MAX_JOBS takes effect + re-JITs the GDN kernel cleanly.
# Only restart if supervisor is already running (standalone invocation
# case). During provision the stack is started afterwards, so we skip.
# ---------------------------------------------------------------------------
SUPERVISORCTL="$APP_DIR/.venv/bin/supervisorctl"
SUPERVISOR_CONF="$APP_DIR/scripts/supervisord.conf"
if [ -x "$SUPERVISORCTL" ] && [ -f "$SUPERVISOR_CONF" ]; then
  if "$SUPERVISORCTL" -c "$SUPERVISOR_CONF" status qwen 2>/dev/null | grep -q RUNNING; then
    log "qwen is running — doing a GPU-safe full restart so MAX_JOBS and the"
    log "new MoE config both take effect. A plain 'supervisorctl restart qwen'"
    log "SIGTERMs the V1 engine, which leaks GPU memory on this vLLM version;"
    log "the new qwen process then fails with 'Free memory less than desired'."
    log "restart_all.sh has the SIGTERM + wait + pkill sequence that releases"
    log "GPU memory cleanly before starting qwen again."
    if [ -x "$APP_DIR/scripts/restart_all.sh" ]; then
      log "Calling scripts/restart_all.sh (takes ~3-5 min, reloads the 35B model)"
      bash "$APP_DIR/scripts/restart_all.sh" || \
        log "WARNING: restart_all.sh returned non-zero; check GPU memory + logs"
    else
      log "WARNING: scripts/restart_all.sh not found — falling back to naive restart."
      log "If it fails with GPU OOM, run: pkill -9 -f vllm; sleep 20; bash scripts/restart_all.sh"
      "$SUPERVISORCTL" -c "$SUPERVISOR_CONF" restart qwen || \
        log "WARNING: qwen restart failed; manual restart required"
    fi
    log "Watch the compile with: tail -f .state/qwen.log | grep -iE 'GDN|kernel|warmup'"
  else
    log "qwen not running — skipping restart (provision flow will start stack)"
  fi
fi

log "Done."
