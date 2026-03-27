# Server Deployment Playbook

Step-by-step guide for deploying the voice AI benchmark stack to a TensorDock A100 80GB VM, running the benchmark matrix, and doing manual voice quality testing.

---

## Part 1: TensorDock VM Setup

### 1.1 Create the VM

1. Go to [tensordock.com](https://tensordock.com) and log in
2. Deploy a new VM:
   - **GPU:** 1x A100 80GB
   - **CPU:** 16 vCPUs recommended (8 minimum; runs ~8 concurrent processes: vLLM, backend, sidecars, Qdrant, benchmark runner)
   - **RAM:** 64GB+ (128GB recommended for comfort)
   - **Disk:** 512GB+ (models alone are ~200GB)
   - **OS:** Ubuntu 22.04
   - **Region:** pick what's available

3. After creation, note the **IP address** and **SSH port** from the dashboard

### 1.2 SSH Access

You have two options:

**Option A: SSH from your Mac (recommended)**

```bash
ssh root@<IP> -p <PORT>
```

This is better because you can copy-paste commands, run multiple terminal tabs, and scp files back easily.

**Option B: TensorDock web terminal**

Works in a pinch but no copy-paste support and no scp. Use only if SSH is blocked.

### 1.3 Get a HuggingFace Token

You need an HF token with access to gated models (Qwen, Mistral):

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a token with "Read" access
3. Accept the license on each model page (visit each and click "Agree"):
   - `Qwen/Qwen3-30B-A3B`
   - `Qwen/Qwen3.5-35B-A3B`
   - `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
   - `Qwen/Qwen3-ASR-1.7B`
   - `Qwen/Qwen3-Omni-30B-A3B`
   - `mistralai/Voxtral-Mini-4B-Realtime-2602`
   - `FunAudioLLM/SenseVoiceSmall`

---

## Part 2: Provisioning (One Command)

SSH into the server and run:

```bash
export HF_TOKEN=hf_YOUR_TOKEN_HERE
curl -fsSL https://raw.githubusercontent.com/yauhenifutryn/leasing/claude/qwen-voice-next/rag_demo_system/scripts/provision_server.sh | bash
```

Or if you prefer to clone first:

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
| 1 | Installs apt packages (git, curl, python3, docker, jq) | 1-2 min |
| 2 | Checks nvidia-smi; installs GPU driver if missing (exits with "REBOOT REQUIRED" if driver was installed; re-run after reboot) | 0-5 min |
| 3 | Clones the repo (or pulls if already exists) | 1 min |
| 4-5 | Creates 6 isolated Python venvs (backend, voice-oss, qwen3-tts, qwen3-asr, voxtral, qwen3-omni) | 10-15 min |
| 6 | Downloads 7 HuggingFace models (~200GB total; resume-safe) | 30-90 min |
| 7 | Starts Qdrant vector DB in Docker | 30 sec |
| 8 | Generates .env with all service URLs and vLLM config | instant |
| 9 | Starts the supervisor stack (backend + vLLM + sidecars) | 2-5 min |

**If the script exits with "REBOOT REQUIRED":**

```bash
sudo reboot
# Wait 30 seconds, then SSH back in
export HF_TOKEN=hf_YOUR_TOKEN_HERE
cd /workspace/leasing
bash rag_demo_system/scripts/provision_server.sh
```

The script is idempotent; it skips what's already done.

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
- Stops vLLM, starts Qwen3-Omni sidecar (different model entirely)
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
scp -P <PORT> root@<IP>:/workspace/leasing/rag_demo_system/results/* ./benchmark_results/
```

---

## Part 4: Manual Voice Quality Testing

The benchmark measures latency and keyword accuracy automatically, but it cannot judge how the voice actually sounds. You need to test this yourself by talking to the system through the browser.

### 4.1 Access the Web UI

The UI runs on port 8000. To access it from your Mac, set up an SSH tunnel:

```bash
ssh -L 8000:localhost:8000 -p <PORT> root@<IP> -N
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

### 5.2 Shut Down the VM

TensorDock charges by the hour. When done:

```bash
# On the server: stop the stack
cd /workspace/leasing/rag_demo_system
bash scripts/stack.sh down
sudo docker stop qdrant
```

Then go to TensorDock dashboard and stop/destroy the VM.

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
