# Server Deployment Playbook

Step-by-step guide for deploying the voice AI benchmark stack to a GPU server, running the benchmark matrix, and doing manual voice quality testing.

The provisioning script auto-detects its environment and adapts:
- **VM providers** (TensorDock, Lambda, your own physical server): installs Docker, NVIDIA drivers if needed, runs Qdrant in a Docker container (production-grade, auto-restarts)
- **Container providers** (RunPod, Vast.ai): skips Docker/driver install, downloads Qdrant as a binary (same engine, same port, same results)

Both paths produce identical benchmark output.

---

## Part 1: GPU Server Setup

### 1.1 Choose a Provider

| Provider | Type | A100 80GB price | Docker available | Notes |
|----------|------|-----------------|------------------|-------|
| **RunPod** (recommended) | Container | ~$1.20-1.50/hr | No (auto-detected, Qdrant runs as binary) | Simple dashboard, reliable SSH, good availability |
| TensorDock | VM | ~$1.18/hr | Yes (Qdrant runs in Docker) | Real VM, but availability issues |
| Lambda Cloud | VM | ~$1.29/hr | Yes | Often sold out |
| Vast.ai | Container | ~$0.80-1.20/hr | No (auto-detected) | Cheapest, variable host quality |
| Your own server | VM | N/A | Yes | Full control, use for production after benchmarks |

### 1.2 Create the Instance

**Minimum specs:**
- **GPU:** 1x A100 80GB (SXM preferred over PCIe for higher memory bandwidth)
- **CPU:** 16 vCPUs recommended (8 minimum; runs ~8 concurrent processes)
- **RAM:** 64GB+ (128GB recommended)
- **Disk/Volume:** 500GB+ (models are ~200GB, venvs ~30GB, repo + results need space)
- **Ports:** expose internal port 8000 (for browser-based voice testing later)

**RunPod setup:**
1. Create a GPU Pod (not Serverless)
2. Select A100 SXM 80GB
3. Template: `Ubuntu 22.04` (`runpod/base:1.0.3-ubuntu2204`)
4. Container disk: 20GB (default)
5. Volume disk: **500GB** (increase from default)
6. Enable SSH terminal access
7. Add your SSH key in RunPod settings (Settings > SSH Keys)

**TensorDock setup:**
1. Deploy a VM: A100 80GB, 16 vCPU, 128GB RAM, 600GB disk
2. OS: Ubuntu 24 ML Everything (pre-installed NVIDIA drivers, skip driver step)
3. Request port 8000 in port forwarding

### 1.3 SSH Access

Always SSH from your Mac terminal (not the provider's web terminal). You need copy-paste, multiple tabs, and scp.

```bash
# Generate SSH key (once, on your Mac)
ssh-keygen -t ed25519 -C "gpu-server" -f ~/.ssh/id_ed25519_gpu -N ""
cat ~/.ssh/id_ed25519_gpu.pub
# Paste the output into your provider's SSH key settings

# RunPod (check pod dashboard for exact command)
ssh <pod-id>@ssh.runpod.io -i ~/.ssh/id_ed25519_gpu

# TensorDock / Lambda / other VM providers
ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519_gpu
```

### 1.4 Get a HuggingFace Token

You need an HF token with access to gated models (Qwen, Mistral):

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a token with "Read" access
3. Accept the license on each model page (visit each and click "Agree"):
   - `Qwen/Qwen3-30B-A3B`
   - `Qwen/Qwen3.5-35B-A3B`
   - `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
   - `Qwen/Qwen3-ASR-1.7B`
   - `Qwen/Qwen3-Omni-30B-A3B-Instruct`
   - `mistralai/Voxtral-Mini-4B-Realtime-2602`
   - `FunAudioLLM/SenseVoiceSmall`

---

## Part 2: Provisioning (One Command)

SSH into the server and run:

```bash
export HF_TOKEN=hf_YOUR_TOKEN_HERE
cd /workspace
git clone --branch claude/qwen-voice-next https://github.com/yauhenifutryn/leasing.git
cd leasing
bash rag_demo_system/scripts/provision_server.sh
```

**What this does (9 steps, fully automated):**

| Step | What happens | Time estimate |
|------|-------------|---------------|
| 1 | Installs apt packages (git, curl, python3, jq; Docker only on VMs) | 1-2 min |
| 2 | Checks nvidia-smi; on VMs installs driver if missing (exits with "REBOOT REQUIRED"); in containers just verifies GPU is visible | 0-5 min |
| 3 | Clones the repo (or pulls if already exists) | 1 min |
| 4-5 | Creates 6 isolated Python venvs (backend, voice-oss, qwen3-tts, qwen3-asr, voxtral, qwen3-omni) | 10-15 min |
| 6 | Downloads 7 HuggingFace models (~200GB total; resume-safe) | 30-90 min |
| 7 | Starts Qdrant: Docker container on VMs, binary download on containers | 30 sec |
| 8 | Generates .env with all service URLs and vLLM config | instant |
| 9 | Starts the supervisor stack (backend + vLLM + sidecars) | 2-5 min |

**If the script exits with "REBOOT REQUIRED" (VMs only, never happens on RunPod/Vast.ai):**

```bash
sudo reboot
# Wait 30 seconds, then SSH back in
export HF_TOKEN=hf_YOUR_TOKEN_HERE
cd /workspace/leasing
bash rag_demo_system/scripts/provision_server.sh
```

The script is idempotent; it skips what is already done.

**If model download gets interrupted:** just re-run the script. `huggingface-cli download` resumes automatically.

### 2.1 Verify Everything Works

After provisioning completes, run the smoke test:

```bash
cd /workspace/leasing/rag_demo_system
bash scripts/smoke_test.sh
```

This checks: UI, backend health, Qdrant indexing, chat stream, sidecar health, vLLM readiness, and VRAM. You should see `=== Smoke test PASSED ===`.

---

## Part 3: Running the Benchmark Matrix

```bash
cd /workspace/leasing/rag_demo_system
bash scripts/benchmark_orchestrator.sh
```

**What happens step by step:**

### Step 1: RAG Comparison (~20-30 min)
- Runs 85 questions through `our_rag` (baseline)
- Runs 85 questions through `dify_rag`
- Prints a side-by-side comparison table (latency + quality metrics)
- **Asks you:** "RAG winner (baseline/dify_rag):" -- type your answer based on the table

### Step 2: Brain Comparison (~20-30 min)
- Stops vLLM, swaps to Qwen3.5-35B-A3B, restarts vLLM
- Runs 85 questions with the upgraded brain
- Prints comparison table vs the winning RAG result
- Restores the original brain model
- **Asks you:** "Brain winner (baseline/brain_upgrade):" -- type your answer

### Step 3: Omni Hybrid (~15-20 min)
- Stops vLLM, starts Qwen3-Omni-Instruct sidecar (different model entirely)
- Runs 85 questions through the Omni hybrid path
- Prints comparison: Omni vs best split pipeline result from Steps 1-2
- **Asks you:** "continue" (run Step 4) or "skip" (finish)

### Step 4: STT/TTS Matrix (optional, ~30-40 min)
- Only runs if you typed "continue" after Step 3
- Tests Qwen3-TTS, Qwen3-ASR, Voxtral providers one at a time
- Each provider: start sidecar, run 85 questions, compare vs baseline, stop sidecar

### After Each Step
- The orchestrator prints an `scp` command you can paste on your Mac to copy results locally
- Comparison tables are saved as markdown files in `rag_demo_system/results/`

### 3.1 Copy Results to Your Mac

After the orchestrator finishes, it prints a final scp command. On your Mac:

```bash
# RunPod (use their proxy)
scp -i ~/.ssh/id_ed25519_gpu <pod-id>@ssh.runpod.io:/workspace/leasing/rag_demo_system/results/* ./benchmark_results/

# VM providers (direct)
scp -P <PORT> -i ~/.ssh/id_ed25519_gpu root@<IP>:/workspace/leasing/rag_demo_system/results/* ./benchmark_results/
```

---

## Part 4: Manual Voice Quality Testing

The benchmark measures latency and keyword accuracy automatically, but it cannot judge how the voice actually sounds. You need to test this yourself by talking to the system through the browser.

### 4.1 Access the Web UI

The UI runs on port 8000.

**RunPod:** If you exposed port 8000 during setup, RunPod assigns an external URL. Check the pod dashboard under "Connect" for the HTTP URL on port 8000. Open it directly in your browser.

**VM providers:** Set up an SSH tunnel from your Mac:

```bash
ssh -L 8000:localhost:8000 -p <PORT> -i ~/.ssh/id_ed25519_gpu root@<IP> -N
```

Then open `http://localhost:8000` in your browser.

### 4.2 Voice Testing Checklist

Test with the winning configuration from the benchmark. Here is a smart order:

**A. Baseline voice quality (5 min)**

1. Open the UI, select `our_rag` backend, default brain, default STT/TTS
2. Ask 3-4 questions in Russian using your microphone:
   - Short factual: "Какой минимальный аванс по лизингу?"
   - Long factual: "Расскажите подробно о требованиях к лизингу грузового транспорта"
   - Out-of-scope: "Какая погода в Минске?"
3. Listen for: response speed, pronunciation clarity, Russian accent quality, domain term accuracy (лизингополучатель, аванс, 84 месяца)

**B. Winning configuration from benchmarks (5 min)**

4. Switch to the winning RAG, brain, and provider combination from the benchmark results
5. Ask the same 3-4 questions
6. Compare against baseline: faster? clearer? better domain terms?

**C. Omni hybrid (5 min)**

7. Select "Qwen3-Omni" as voice provider
8. Ask the same questions
9. Note: Omni speaks its own audio directly (no separate TTS). Compare naturalness vs the split pipeline

**D. Out-of-scope handling (2 min)**

10. Ask 2-3 questions completely unrelated to leasing: sports, weather, cooking
11. The system should refuse or say it cannot help (strict grounding)

**E. Edge cases (3 min)**

12. Speak very quickly
13. Speak with background noise (play music from your phone)
14. Ask a question mid-sentence, then change your mind and ask something else
15. Stay silent for 10 seconds, then speak

### 4.3 What to Write Down

For each test, note:

| Test | Config | Response time (fast/ok/slow) | Voice quality (clear/ok/muddy) | Domain accuracy (correct/partial/wrong) | Notes |
|------|--------|-----|------|------|-------|
| Short factual | baseline | | | | |
| Short factual | winner | | | | |
| Short factual | omni | | | | |

This is your subjective voice quality matrix. Combined with the benchmark JSONL numbers, you have the full picture to decide what to merge to main.

### 4.4 Switching Configurations on the Server

If you need to swap models between voice tests (e.g. switch from baseline brain to upgraded brain):

```bash
cd /workspace/leasing/rag_demo_system

# Stop current brain, load upgraded brain
.venv/bin/supervisorctl -c scripts/supervisord.conf stop qwen
# Edit .env to change RAG_LLM_MODEL to Qwen/Qwen3.5-35B-A3B
sed -i 's|Qwen/Qwen3-30B-A3B|Qwen/Qwen3.5-35B-A3B|' .env
.venv/bin/supervisorctl -c scripts/supervisord.conf start qwen

# Wait for vLLM to load (check health)
watch -n5 'curl -s http://localhost:8001/health && echo OK || echo LOADING'
# Once it shows OK, refresh the browser and test
```

To switch to Omni mode:

```bash
.venv/bin/supervisorctl -c scripts/supervisord.conf stop qwen
sleep 10
.venv/bin/supervisorctl -c scripts/supervisord.conf start qwen3_omni
watch -n5 'curl -s http://localhost:8002/health && echo OK || echo LOADING'
# Select "Qwen3-Omni" in the UI dropdown
```

---

## Part 5: After Testing

### 5.1 Decide What Won

Based on benchmark numbers + voice quality impressions:

1. **Best RAG backend:** our_rag or dify_rag?
2. **Best brain model:** Qwen3-30B or Qwen3.5-35B?
3. **Is Omni viable?** Better/worse/comparable to split pipeline?
4. **If Omni lost:** which STT/TTS combination was best?

### 5.2 Shut Down the Server

Providers charge by the hour. When done:

```bash
# On the server: stop the stack
cd /workspace/leasing/rag_demo_system
bash scripts/stack.sh down
```

Then go to your provider's dashboard and stop/destroy the instance.

### 5.3 Branch Strategy (Do Not Merge Experiment to Main)

The benchmarking system is an experiment, not a product feature. Do not merge the full `claude/qwen-voice-next` branch to main. Instead, use a three-branch model:

**Branch 1: `main`**
The production product. Untouched until you deliberately bring proven pieces in.

**Branch 2: `claude/qwen-voice-next`**
Stays as-is permanently. The full experiment with all 5 phases, benchmarking infrastructure, all adapters (winners and losers), provisioning scripts, orchestrator, fixture questions. This is your reference. You can always come back to re-run benchmarks or test new models later.

**Branch 3: New branch off main (e.g. `feature/voice-pipeline`)**
Created after benchmarks. Cherry-pick only the winning components. Clean, no experimental baggage.

### 5.4 What to Cherry-Pick (After Benchmarks)

Based on which combination won, cherry-pick only these into branch 3:

| Component | Cherry-pick if... |
|-----------|-------------------|
| Winning TTS adapter + sidecar (e.g. qwen3_tts_sidecar.py) | That TTS provider won the benchmark |
| Winning STT adapter + sidecar | That STT provider won |
| Brain model routing code (ChatRequest.brain_model field) | You want switchable brain models in production |
| Timing instrumentation (voice_session.py timestamps, structured JSON logs) | You want latency monitoring in production |
| Voice session dataclass (VoiceSession with stack_id) | Always; it is the foundation |
| UI selectors (brain model, STT, TTS dropdowns) | You want runtime switching in the production UI |
| Omni hybrid adapter + sidecar | Omni proved viable in benchmarks |

**Do NOT cherry-pick:**
- Benchmark runner, comparison scripts, fixture questions (test harness only)
- 7 env profiles for A/B testing (benchmarking only)
- Provisioning script, orchestrator (server deployment tooling)
- Adapters that lost the benchmark (dead code)
- `.planning/` directory (internal planning artifacts)

### 5.5 Cherry-Pick Workflow

```bash
# 1. Create the clean branch off main
git checkout main
git checkout -b feature/voice-pipeline

# 2. Identify the commits you need from the experiment branch
git log --oneline claude/qwen-voice-next -- rag_demo_system/backend/voice_session.py
git log --oneline claude/qwen-voice-next -- rag_demo_system/backend/voice_adapters.py
# ... etc for each winning component

# 3. Cherry-pick them (use -n to stage without committing, then review)
git cherry-pick -n <commit-hash>
# Review staged changes, remove anything experimental
git commit -m "feat: add winning voice pipeline components from benchmark"

# 4. Test on main before merging
```

---

*Generated from Phase 5: Server Deployment and Benchmarks*
*Branch: `claude/qwen-voice-next`*
