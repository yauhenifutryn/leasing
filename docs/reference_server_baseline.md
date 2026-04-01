# Reference Server Baseline

Clean system state captured before provisioning on Jarvis Labs H100 instance.
Date: 2026-04-01

This represents the minimum starting point for deployment. The `provision_server.sh`
script installs everything else on top of this.

## Hardware

| Component | Value |
|-----------|-------|
| GPU | NVIDIA H100 80GB HBM3 |
| GPU Memory | 81559 MiB |
| GPU Driver | 550.163.01 |
| GPU PCI Bus | 00000000:8D:00.0 |
| CPU | Intel Xeon Platinum 8468 |
| RAM | 196 GiB |
| Disk | 621 GB (605 GB free) |

## Software (Pre-installed)

| Component | Value |
|-----------|-------|
| OS | Ubuntu 22.04.5 LTS (Jammy Jellyfish) |
| Kernel | Linux (uname output not captured) |
| Python | 3.10.12 |
| CUDA Toolkit | 12.4 (V12.4.131) |
| Docker | 28.1.1 |
| Packages | 1161 apt packages |
| Virtualization | KVM |

## What Provision Adds

The `provision_server.sh` script installs on top of this baseline:

- Build tools: build-essential, cmake, ninja-build, ccache, pkg-config
- CUDA toolkit 12.6+ (for flashinfer JIT, from NVIDIA apt repo)
- Python 3.12 (if system Python < 3.12)
- python3-dev, ffmpeg, libssl-dev, libnuma-dev
- ngrok (optional, for public URL)
- 2 Python venvs with all pip packages
- HuggingFace models (~50 GB)
- Qdrant vector database (Docker container)

## Client Server Requirements

For a bare metal server, the client needs:

1. **Hardware**: Match or exceed the specs above
2. **OS**: Ubuntu 22.04 or 24.04 LTS (fresh install)
3. **NVIDIA Driver**: Install via `sudo ubuntu-drivers install && sudo reboot`
4. **Everything else**: Handled by `provision_server.sh`
