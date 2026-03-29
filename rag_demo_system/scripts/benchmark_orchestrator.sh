#!/usr/bin/env bash
# benchmark_orchestrator.sh
# Sequential 4-step benchmark matrix runner.
#
# Steps:
#   1. RAG comparison: baseline (our_rag) vs dify_rag
#   2. Brain comparison: winning RAG + Qwen3-30B vs Qwen3.5-35B
#   3. Omni hybrid: best split pipeline vs omni_hybrid
#   4. (conditional) STT/TTS matrix: qwen3_tts, qwen3_asr, voxtral
#
# Usage:
#   SERVER_IP=<ip> bash benchmark_orchestrator.sh
#
# Required env vars:
#   SERVER_IP  - Remote server IP for scp commands (auto-detected if unset)
#
# Per D-08 through D-13 (research decisions).

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$APP_DIR/results"
FIXTURE="$APP_DIR/fixtures/bench_questions_ru.jsonl"
SUPERVISORCTL="$APP_DIR/.venv/bin/supervisorctl -c $APP_DIR/scripts/supervisord.conf"
BENCHMARK_RUNNER="$APP_DIR/.venv/bin/python $APP_DIR/scripts/benchmark_runner.py"
BENCHMARK_COMPARE="$APP_DIR/.venv/bin/python $APP_DIR/scripts/benchmark_compare.py"
SMOKE_TEST="$SCRIPT_DIR/smoke_test.sh"

WS_URL="ws://localhost:8000/ws/voice"
BACKEND_URL="http://localhost:8000"
VLLM_PORT=8787  # Avoid 8001 which RunPod reserves internally
VRAM_LIMIT_GB=75
HEALTH_TIMEOUT=300
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Auto-detect server IP if not provided
SERVER_IP="${SERVER_IP:-$(curl -s ifconfig.me 2>/dev/null || hostname -I | awk '{print $1}')}"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

log() { echo "[orch][$(date +%H:%M:%S)] $*"; }

# wait_healthy <url> [timeout_seconds]
# Poll URL at 10-second intervals until it returns HTTP 200 or timeout.
# Per Pitfall 6: default 300-second timeout for brain model loads.
wait_healthy() {
  local url="$1"
  local max_wait="${2:-$HEALTH_TIMEOUT}"
  local elapsed=0
  log "Waiting for $url (timeout: ${max_wait}s)"
  until curl -fsS "$url" >/dev/null 2>&1; do
    sleep 10
    elapsed=$((elapsed + 10))
    if [ "$elapsed" -ge "$max_wait" ]; then
      log "ERROR: $url did not become healthy within ${max_wait}s"
      exit 1
    fi
    log "  ... ${elapsed}s elapsed"
  done
  log "$url healthy (${elapsed}s)"
}

# check_vram <context>
# Report VRAM usage and abort if used >= VRAM_LIMIT_GB.
# Per research Pattern 4.
check_vram() {
  local context="$1"
  local used_mib
  used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  local used_gb=$((used_mib / 1024))
  local total_mib
  total_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
  log "VRAM [$context]: used=${used_gb}GB (${used_mib}MiB) / total=$((total_mib / 1024))GB"
  if [ "$used_gb" -ge "$VRAM_LIMIT_GB" ]; then
    log "ERROR: VRAM used (${used_gb}GB) >= limit (${VRAM_LIMIT_GB}GB)"
    exit 1
  fi
}

# swap_model <stop_program> <start_program> <health_url>
# Stops one supervisord program, verifies VRAM freed, starts another.
# Per D-09: stop -> VRAM check -> start -> wait_healthy -> VRAM check.
swap_model() {
  local stop_program="$1"
  local start_program="$2"
  local health_url="$3"

  log "Stopping $stop_program"
  $SUPERVISORCTL stop "$stop_program" 2>/dev/null || true
  sleep 5

  log "Verifying VRAM freed after stop"
  check_vram "post-stop"

  log "Starting $start_program"
  $SUPERVISORCTL start "$start_program"
  wait_healthy "$health_url" "$HEALTH_TIMEOUT"
  check_vram "post-start"
}

# run_benchmark <profile>
# Loads the profile env overlay, invokes benchmark_runner.py, and echoes
# the output file path to stdout.
# Per Pitfall 7: explicit --ws-url and --backend-url flags.
run_benchmark() {
  local profile="$1"
  local output_file="$RESULTS_DIR/bench_${profile}_${TIMESTAMP}.jsonl"

  log "Running benchmark: profile=$profile output=$output_file"

  # Load profile env vars for smoke test / sidecar awareness
  export BENCH_PROFILE="$profile"
  if [ -f "$APP_DIR/.env.bench.$profile" ]; then
    set -a
    source "$APP_DIR/.env.bench.$profile"
    set +a
  fi

  $BENCHMARK_RUNNER \
    --fixture "$FIXTURE" \
    --profile "$profile" \
    --output "$output_file" \
    --ws-url "$WS_URL" \
    --backend-url "$BACKEND_URL" \
    --timeout 30 \
    --warmup 3

  log "Benchmark complete: $output_file"
  echo "$output_file"
}

# run_comparison <file_a> <file_b> <label>
# Invokes benchmark_compare.py and tees output to both stdout and a file.
# Per D-11.
run_comparison() {
  local file_a="$1"
  local file_b="$2"
  local label="$3"

  log "=== Comparison: $label ==="
  $BENCHMARK_COMPARE "$file_a" "$file_b" | tee "$RESULTS_DIR/compare_${label}_${TIMESTAMP}.md"
  log "Comparison saved: $RESULTS_DIR/compare_${label}_${TIMESTAMP}.md"
}

# print_scp <label> <file1> [file2 ...]
# Prints a ready-to-paste scp command for the given result files.
# Per D-13; SERVER_IP auto-detected via Pitfall 9 pattern.
print_scp() {
  local label="$1"
  shift
  log "--- SCP command ($label) ---"
  local paths=""
  for f in "$@"; do
    paths="$paths root@${SERVER_IP}:${f}"
  done
  echo "scp $paths ./"
}

# ---------------------------------------------------------------------------
# Main orchestration flow
# ---------------------------------------------------------------------------

main() {
  mkdir -p "$RESULTS_DIR"

  log "============================================="
  log "  Benchmark Orchestrator"
  log "  Server: $SERVER_IP"
  log "  Timestamp: $TIMESTAMP"
  log "============================================="

  # --- Pre-flight: smoke test with baseline profile ---
  log "=== Pre-flight: Smoke test (baseline) ==="
  BENCH_PROFILE=baseline bash "$SMOKE_TEST"

  # ===========================================================================
  # STEP 1: Baseline RAG benchmark
  # ===========================================================================
  log "============================================="
  log "  STEP 1: Baseline RAG benchmark (our_rag)"
  log "  Dify RAG comparison skipped (no DIFY_API_KEY configured)"
  log "============================================="

  RESULT_BASELINE=$(run_benchmark "baseline")
  print_scp "step1" "$RESULT_BASELINE"

  WINNING_RAG="baseline"
  log "RAG winner: baseline (our_rag) -- Dify comparison skipped"

  # BEST_SPLIT_RESULT tracks the best split pipeline result seen so far.
  # It starts as the baseline and is updated when a better stack wins.
  # Step 3 Omni comparison uses this variable as the left side.
  BEST_SPLIT_RESULT="$RESULT_BASELINE"

  # RESULT_FOR_BRAIN_COMPARE is the "left side" of the Step 2 comparison.
  # If dify_rag won Step 1, the brain comparison must use dify_rag as its
  # baseline to avoid attributing the RAG advantage to the brain upgrade.
  RESULT_FOR_BRAIN_COMPARE="$RESULT_BASELINE"
  if [ "$WINNING_RAG" = "dify_rag" ]; then
    log "dify_rag won Step 1 -- using dify_rag result as brain comparison baseline"
    RESULT_FOR_BRAIN_COMPARE="$RESULT_DIFY"
    BEST_SPLIT_RESULT="$RESULT_DIFY"
  fi

  # ===========================================================================
  # STEP 2: Brain Comparison (Qwen3-30B vs Qwen3.5-35B)
  # ===========================================================================
  log "============================================="
  log "  STEP 2: Brain Comparison"
  log "  baseline (Qwen3-30B) vs brain_upgrade (Qwen3.5-35B)"
  log "  Using winning RAG: $WINNING_RAG"
  log "============================================="

  # Stop current vLLM instance and wait for VRAM to free before loading the
  # larger Qwen3.5-35B-A3B model. Patch .env to point to the new model path.
  log "Restarting vLLM with Qwen3.5-35B-A3B for brain_upgrade"
  $SUPERVISORCTL stop qwen 2>/dev/null || true
  sleep 10
  check_vram "pre-brain-upgrade"

  # Patch .env to switch model path for vLLM startup
  sed -i.bak 's|Qwen/Qwen3-30B-A3B|Qwen/Qwen3.5-35B-A3B|g' "$APP_DIR/.env"
  $SUPERVISORCTL start qwen
  wait_healthy "http://127.0.0.1:$VLLM_PORT/health" "$HEALTH_TIMEOUT"
  check_vram "post-brain-upgrade"

  RESULT_BRAIN_UPGRADE=$(run_benchmark "brain_upgrade")
  run_comparison "$RESULT_FOR_BRAIN_COMPARE" "$RESULT_BRAIN_UPGRADE" "brain"
  print_scp "step2" "$RESULT_BRAIN_UPGRADE"

  # Restore baseline model for subsequent steps
  log "Restoring vLLM to Qwen3-30B-A3B"
  $SUPERVISORCTL stop qwen 2>/dev/null || true
  sleep 10
  mv "$APP_DIR/.env.bak" "$APP_DIR/.env" 2>/dev/null || true
  $SUPERVISORCTL start qwen
  wait_healthy "http://127.0.0.1:$VLLM_PORT/health" "$HEALTH_TIMEOUT"

  # Prompt the user to declare the brain winner.
  log ""
  log "Review the brain comparison table above."
  log "Type 'baseline' if the original brain won, or 'brain_upgrade' if Qwen3.5-35B won."
  read -rp "[orch] Brain winner (baseline/brain_upgrade): " WINNING_BRAIN
  WINNING_BRAIN="${WINNING_BRAIN:-baseline}"
  log "Brain winner selected: $WINNING_BRAIN"

  # Update BEST_SPLIT_RESULT if the upgraded brain outperformed the current best
  if [ "$WINNING_BRAIN" = "brain_upgrade" ]; then
    BEST_SPLIT_RESULT="$RESULT_BRAIN_UPGRADE"
    log "brain_upgrade won Step 2 -- BEST_SPLIT_RESULT updated"
  fi

  # ===========================================================================
  # STEP 3: Omni Hybrid vs Best Split Pipeline
  # ===========================================================================
  log "============================================="
  log "  STEP 3: Omni Hybrid vs Best Split Pipeline"
  log "  Using winning RAG: $WINNING_RAG, winning brain: $WINNING_BRAIN"
  log "  Best split result: $BEST_SPLIT_RESULT"
  log "============================================="

  # swap_model handles the full stop -> VRAM check -> start sequence.
  # Stops qwen (freeing GPU RAM) then starts qwen3_omni.
  swap_model "qwen" "qwen3_omni" "http://127.0.0.1:8002/health"

  RESULT_OMNI=$(run_benchmark "omni_hybrid")

  # Compare Omni against the best split pipeline result from Steps 1-2,
  # not against the initial baseline.
  run_comparison "$BEST_SPLIT_RESULT" "$RESULT_OMNI" "omni_vs_split"
  print_scp "step3" "$RESULT_OMNI"

  # Per D-10: pause after Step 3 so the user can review Omni results before
  # deciding whether the STT/TTS matrix is worth running.
  log "============================================="
  log "  STEP 3 COMPLETE: Omni vs Split Pipeline"
  log "============================================="
  log ""
  log "Review the comparison table above."
  log "Type 'continue' to run Step 4 (STT/TTS matrix)"
  log "Type 'skip' to finish"
  log ""
  read -r USER_DECISION
  USER_DECISION="${USER_DECISION:-skip}"

  # ===========================================================================
  # STEP 4: STT/TTS Provider Matrix (conditional on user choice)
  # ===========================================================================
  if [ "$USER_DECISION" = "continue" ]; then
    log "============================================="
    log "  STEP 4: STT/TTS Provider Matrix"
    log "============================================="

    # Stop Omni and restart baseline vLLM brain before running STT/TTS tests
    log "Stopping Omni, restarting vLLM baseline brain"
    $SUPERVISORCTL stop qwen3_omni 2>/dev/null || true
    sleep 10
    $SUPERVISORCTL start qwen
    wait_healthy "http://127.0.0.1:$VLLM_PORT/health" "$HEALTH_TIMEOUT"
    check_vram "post-omni-stop"

    # --- Qwen3-TTS benchmark ---
    log "--- Qwen3-TTS benchmark ---"
    $SUPERVISORCTL start qwen3_tts 2>/dev/null || true
    wait_healthy "http://127.0.0.1:50003/health" 120
    RESULT_QWEN3_TTS=$(run_benchmark "qwen3_tts")
    run_comparison "$RESULT_BASELINE" "$RESULT_QWEN3_TTS" "qwen3_tts"
    $SUPERVISORCTL stop qwen3_tts 2>/dev/null || true

    # --- Qwen3-ASR benchmark ---
    log "--- Qwen3-ASR benchmark ---"
    $SUPERVISORCTL start qwen3_asr 2>/dev/null || true
    wait_healthy "http://127.0.0.1:50004/health" 120
    RESULT_QWEN3_ASR=$(run_benchmark "qwen3_asr")
    run_comparison "$RESULT_BASELINE" "$RESULT_QWEN3_ASR" "qwen3_asr"
    $SUPERVISORCTL stop qwen3_asr 2>/dev/null || true

    # --- Voxtral benchmark ---
    log "--- Voxtral benchmark ---"
    $SUPERVISORCTL start voxtral 2>/dev/null || true
    wait_healthy "http://127.0.0.1:50005/health" 120
    RESULT_VOXTRAL=$(run_benchmark "voxtral")
    run_comparison "$RESULT_BASELINE" "$RESULT_VOXTRAL" "voxtral"
    $SUPERVISORCTL stop voxtral 2>/dev/null || true

    print_scp "step4" "$RESULT_QWEN3_TTS" "$RESULT_QWEN3_ASR" "$RESULT_VOXTRAL"
  else
    log "Step 4 skipped by user"
    # Stop Omni and restore baseline for clean stack state
    $SUPERVISORCTL stop qwen3_omni 2>/dev/null || true
    sleep 5
    $SUPERVISORCTL start qwen
    wait_healthy "http://127.0.0.1:$VLLM_PORT/health" "$HEALTH_TIMEOUT" || true
  fi

  # ---------------------------------------------------------------------------
  # Summary
  # ---------------------------------------------------------------------------
  log "============================================="
  log "  BENCHMARK COMPLETE"
  log "============================================="
  log "Results directory: $RESULTS_DIR"
  ls -la "$RESULTS_DIR"/bench_*_"${TIMESTAMP}".jsonl 2>/dev/null || true
  ls -la "$RESULTS_DIR"/compare_*_"${TIMESTAMP}".md 2>/dev/null || true
  log ""
  log "Transfer all results:"
  echo "scp root@${SERVER_IP}:${RESULTS_DIR}/*_${TIMESTAMP}* ./"
}

main "$@"
