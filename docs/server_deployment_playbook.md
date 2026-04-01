# Server Deployment Playbook

Step-by-step guide for deploying the Micro Leasing voice assistant to a GPU server.

The provisioning script auto-detects its environment and adapts:
- **Bare metal / KVM VMs** (Jarvis Labs, Hetzner, client servers): installs build tools, CUDA toolkit, Docker, loads nvidia-uvm
- **Container providers** (Vast.ai, RunPod): skips driver/Docker install, uses existing CUDA, downloads Qdrant as binary

---

## Part 1: GPU Server Setup

### 1.1 Choose a Provider

| Provider | Type | H100 80GB price | Notes |
|----------|------|-----------------|-------|
| **Client bare metal** | Physical | N/A | Full control, no virtualization overhead |
| Jarvis Labs | KVM VM | ~$0.20/hr (reserved) | Budget-friendly, needs UVM fix (automated) |
| Vast.ai | Docker container | ~$2.00/hr | Variable host quality, quick setup |
| RunPod | Docker container | ~$2.39/hr | Reliable, good dashboard |
| Sesterce | Bare metal | ~$2.19/hr | True bare metal, $25 min deposit |
| Lambda Labs | Bare metal | ~$2.49/hr | Often sold out |
| Azure NCads H100 v5 | Hyper-V VM | ~$8/hr | Enterprise, 40-core quota needed |

### 1.2 Hardware Requirements

| Spec | Minimum | Recommended |
|------|---------|-------------|
| GPU | 1x H100 80GB (PCIe or SXM) | 1x H100 80GB+ |
| CPU | 16 vCPUs | 24+ vCPUs (Intel Xeon / AMD EPYC) |
| RAM | 64 GB | 128-196 GB |
| Disk | 200 GB | 500+ GB |
| OS | Ubuntu 22.04 LTS | Ubuntu 22.04/24.04 LTS |
| NVIDIA Driver | 550+ | Latest stable |

A100 80GB also works at reduced vLLM utilization (0.55 instead of 0.60+).

### 1.3 VRAM Budget

| Component | VRAM |
|-----------|------|
| vLLM (Qwen3.5-35B-A3B-FP8, 0.55 util) | ~45 GB |
| Whisper STT (large-v3) | ~3.7 GB |
| Embedding (e5-large) | ~3.3 GB |
| Reranker (mmarco) | ~3.3 GB |
| Silero TTS + VAD | 0 GB (CPU) |
| **Total GPU** | **~55 GB** |
| **Headroom (H100 80GB)** | **~25 GB** |

### 1.4 Get Required Tokens

**HuggingFace token** (required for model downloads):
1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a token with "Read" access
3. Accept the license on the model page: [Qwen/Qwen3.5-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8)

**ngrok token** (optional, for public access):
1. Sign up at [ngrok.com](https://ngrok.com)
2. Get your auth token from [dashboard.ngrok.com/get-started/your-authtoken](https://dashboard.ngrok.com/get-started/your-authtoken)

---

## Part 2: Provisioning (One Command)

SSH into the server and run:

```bash
git clone --branch feature/voice-pipeline https://github.com/yauhenifutryn/leasing.git
cd leasing/rag_demo_system
HF_TOKEN=hf_YOUR_TOKEN NGROK_AUTHTOKEN=YOUR_NGROK_TOKEN bash scripts/provision_server.sh
```

### What This Does (9 Steps, Fully Automated)

| Step | What happens | Time |
|------|-------------|------|
| 1 | System packages (build-essential, cmake, ninja, ffmpeg, Docker) | 1-2 min |
| 2 | NVIDIA driver check, nvidia-uvm loading, CUDA toolkit verification | 30 sec |
| 3 | Clone or update the repository | 30 sec |
| 4-5 | Create 2 Python venvs (backend+vLLM, voice-oss) | 10-15 min |
| 6 | Download HuggingFace models (~50 GB) + Silero VAD | 10-30 min |
| 7 | Start Qdrant (Docker on VMs, binary in containers) | 30 sec |
| 8 | Auto-generate .env (GPU detection, CUDA lib paths, service commands) | instant |
| 9 | Start supervisor stack (backend + vLLM + Whisper + Silero TTS) | 2-5 min |

The script is **idempotent**: safe to re-run. Skips completed steps.

### First-Time Boot Sequence

After provisioning completes, vLLM needs ~2 minutes to:
1. Load model weights (14 shards, ~12 sec)
2. JIT-compile flashinfer GDN attention kernels (~30 sec, first run only)
3. Capture CUDA graphs for prefill and decode (~60 sec)

Then run the smoke test:

```bash
sleep 120 && bash scripts/smoke_test.sh
```

### Expose to the Internet

```bash
ngrok http 8000
```

Opens a public HTTPS URL for browser access to the voice assistant.

---

## Part 3: After Instance Restart

```bash
cd ~/leasing/rag_demo_system
bash scripts/restart_all.sh
bash scripts/smoke_test.sh
```

### Workflow Summary

| Situation | Command |
|-----------|---------|
| First time setup | `HF_TOKEN=... bash scripts/provision_server.sh` |
| After instance restart | `bash scripts/restart_all.sh` |
| After git pull with changes | `bash scripts/restart_all.sh` |
| Verify everything works | `bash scripts/smoke_test.sh` |
| Something is broken | `bash scripts/doctor.sh` |
| CUDA issues on VM/KVM | `bash scripts/fix_cuda_and_verify.sh` |
| Capture system info | `bash scripts/system_snapshot.sh before` |

---

## Part 4: Troubleshooting

### Common Issues

| Error | Cause | Fix |
|-------|-------|-----|
| `torch.cuda.is_available()` returns False | nvidia-uvm module not loaded (KVM VMs) | `bash scripts/fix_cuda_and_verify.sh` (automated in provision) |
| `Could not find nvcc` | CUDA toolkit missing | `sudo apt install cuda-nvcc-12-4` (automated in provision) |
| `No such file or directory: 'ninja'` | ninja build tool missing | `sudo apt install ninja-build && pip install ninja` (automated in provision) |
| `cudaErrorUnsupportedPtxVersion` | Whisper using wrong CUDA libs | Re-run provision to regenerate .env with correct paths |
| vLLM EXITED after start | flashinfer JIT needs nvcc + ninja + CUDA_HOME | Re-run provision (installs all build deps) |
| `LLM не настроен` in chat | vLLM still loading or crashed | Check: `tail -20 .state/qwen.err.log` |
| Model download interrupted | Network timeout | Re-run provision; downloads resume automatically |

### Diagnostic Commands

```bash
# Check all service status
.venv/bin/supervisorctl -c scripts/supervisord.conf status

# Watch vLLM loading progress
tail -f .state/qwen.err.log

# Check GPU memory
nvidia-smi

# Test individual services
curl -s http://localhost:8787/health      # vLLM
curl -s http://localhost:50002/health     # Whisper
curl -s http://localhost:50006/health     # Silero TTS
curl -s http://localhost:6333/healthz     # Qdrant
curl -s http://localhost:8000/api/health  # Backend
```

---

## Part 5: Client Bare Metal Server Setup

For the client's physical server, the setup is identical. Hardware spec:

| Component | Recommended |
|-----------|-------------|
| GPU | 1x NVIDIA H100 80GB PCIe or SXM |
| CPU | Intel Xeon Platinum 8468 or AMD EPYC 9554 (24+ cores) |
| RAM | 196 GB DDR5 |
| Storage | 1 TB NVMe SSD |
| OS | Ubuntu 22.04.5 LTS |
| Network | 1 Gbps+ for model downloads, then minimal bandwidth |

### Setup Steps

1. Install Ubuntu 22.04/24.04 LTS
2. Install NVIDIA driver: `sudo ubuntu-drivers install && sudo reboot`
3. Clone repo and run provisioning:

```bash
git clone --branch feature/voice-pipeline https://github.com/yauhenifutryn/leasing.git
cd leasing/rag_demo_system
bash scripts/fix_cuda_and_verify.sh          # verify CUDA (should PASS on bare metal)
HF_TOKEN=hf_... bash scripts/provision_server.sh
sleep 120 && bash scripts/smoke_test.sh
```

4. For persistent access, set up a reverse proxy (nginx) or ngrok with a reserved domain.

### Validated On

| Platform | GPU | OS | Status |
|----------|-----|-----|--------|
| Jarvis Labs | H100 80GB HBM3 (KVM) | Ubuntu 22.04.5 | Working (UVM fix applied automatically) |
| Vast.ai | H100 80GB (Docker) | Ubuntu 22.04 | Working |

---

## Appendix: Platform-Specific Notes

### Vast.ai / RunPod (Docker containers)
- Port 8001 is reserved by the platform; vLLM uses 8787 instead
- Docker not available inside containers; Qdrant runs as a binary
- CUDA toolkit pre-installed in devel images
- `/workspace` is the default data volume

### Jarvis Labs (KVM VMs)
- nvidia-uvm module must be loaded (automated by provision script)
- Username: `cloud`, home dir: `/home/cloud`
- Workspace auto-detects to `$HOME` (no `/workspace`)

### Azure NCads H100 v5
- Hyper-V VM with discrete device assignment (not bare metal)
- Smallest H100: NC24ads_H100_v5 (1x H100, 24 vCPUs, 220 GB RAM)
- 40-core quota required; request via Azure support
- Enterprise-grade but expensive (~$8/hr pay-as-you-go)
