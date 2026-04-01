#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# fix_cuda_and_verify.sh
#
# Run this FIRST on any new GPU instance (Jarvis Labs, Vast.ai, Sesterce, etc.)
# before provisioning. Diagnoses and fixes common CUDA initialization failures,
# especially on KVM-virtualized GPU instances where nvidia-smi works but
# torch.cuda.is_available() returns False.
#
# Usage:
#   bash fix_cuda_and_verify.sh
#
# Exit codes:
#   0 = CUDA working, safe to proceed with provision_server.sh
#   1 = CUDA broken, cannot be fixed from inside the VM
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}[PASS]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
info() { echo -e "      $*"; }

FIXES_APPLIED=0
FATAL=0

echo "================================================================="
echo "  CUDA Diagnostic and Fix Script"
echo "  $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
echo "================================================================="
echo ""

# -------------------------------------------------------------------------
# 1. Check nvidia-smi
# -------------------------------------------------------------------------
echo "--- Step 1: NVIDIA Driver ---"
if nvidia-smi &>/dev/null; then
  DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
  pass "nvidia-smi OK: $GPU_NAME ($GPU_MEM), driver $DRIVER_VER"
else
  fail "nvidia-smi not found or failed"
  info "NVIDIA driver is not installed or GPU is not visible."
  info "If this is a cloud instance, select a GPU-enabled template."
  info "If bare metal, install drivers: sudo ubuntu-drivers install && reboot"
  exit 1
fi

# -------------------------------------------------------------------------
# 2. Check and load nvidia-uvm kernel module
# -------------------------------------------------------------------------
echo ""
echo "--- Step 2: nvidia-uvm Kernel Module ---"
if lsmod | grep -q nvidia_uvm; then
  pass "nvidia-uvm module already loaded"
else
  warn "nvidia-uvm module NOT loaded (this is the #1 cause of CUDA failures on VMs)"
  info "Attempting to load nvidia-uvm..."
  if sudo modprobe nvidia-uvm 2>/dev/null; then
    pass "nvidia-uvm loaded successfully"
    FIXES_APPLIED=$((FIXES_APPLIED + 1))
  else
    fail "Cannot load nvidia-uvm module"
    info "This usually means the kernel module is not installed or the kernel"
    info "does not match the driver version. Try reinstalling NVIDIA drivers."
    FATAL=1
  fi
fi

# -------------------------------------------------------------------------
# 3. Check /dev/nvidia* device nodes
# -------------------------------------------------------------------------
echo ""
echo "--- Step 3: Device Nodes ---"
MISSING_DEVS=0

for dev in /dev/nvidia0 /dev/nvidiactl; do
  if [ -e "$dev" ]; then
    pass "$dev exists"
  else
    fail "$dev MISSING"
    MISSING_DEVS=1
  fi
done

for dev in /dev/nvidia-uvm /dev/nvidia-uvm-tools; do
  if [ -e "$dev" ]; then
    pass "$dev exists"
  else
    warn "$dev MISSING"
    MISSING_DEVS=1
  fi
done

if [ "$MISSING_DEVS" -eq 1 ]; then
  info "Attempting to create device nodes with nvidia-modprobe..."
  if command -v nvidia-modprobe &>/dev/null; then
    sudo nvidia-modprobe -u -c=0
    FIXES_APPLIED=$((FIXES_APPLIED + 1))
    # Re-check
    for dev in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm; do
      if [ -e "$dev" ]; then
        pass "$dev now exists"
      else
        fail "$dev still missing after nvidia-modprobe"
        FATAL=1
      fi
    done
  else
    info "nvidia-modprobe not found. Installing..."
    if sudo apt-get install -y nvidia-utils-$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1) 2>/dev/null; then
      sudo nvidia-modprobe -u -c=0
      FIXES_APPLIED=$((FIXES_APPLIED + 1))
    else
      warn "Could not install nvidia-modprobe. Trying manual device node creation..."
      # Manual fallback: create device nodes from kernel major numbers
      if [ -e /proc/driver/nvidia/params ]; then
        NVIDIA_MAJOR=$(grep -oP 'DeviceFileUID=\K\d+' /proc/driver/nvidia/params 2>/dev/null || echo "")
      fi
      fail "Cannot create device nodes automatically"
      FATAL=1
    fi
  fi
fi

# -------------------------------------------------------------------------
# 4. Check environment variables
# -------------------------------------------------------------------------
echo ""
echo "--- Step 4: Environment Variables ---"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  if [ "$CUDA_VISIBLE_DEVICES" = "" ]; then
    fail "CUDA_VISIBLE_DEVICES is set to empty string (hides all GPUs)"
    info "Fix: unset CUDA_VISIBLE_DEVICES"
    FATAL=1
  else
    pass "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  fi
else
  pass "CUDA_VISIBLE_DEVICES not set (all GPUs visible)"
fi

if [ -n "${LD_LIBRARY_PATH:-}" ]; then
  info "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
else
  pass "LD_LIBRARY_PATH not set (system defaults)"
fi

# -------------------------------------------------------------------------
# 5. Check for Xid errors (GPU hardware issues)
# -------------------------------------------------------------------------
echo ""
echo "--- Step 5: GPU Health (Xid Errors) ---"
XID_COUNT=$(dmesg 2>/dev/null | grep -ci "xid" || echo "0")
if [ "$XID_COUNT" -gt 0 ]; then
  warn "$XID_COUNT Xid error(s) in kernel log"
  dmesg 2>/dev/null | grep -i "xid" | tail -5 | while read -r line; do
    info "  $line"
  done
else
  pass "No Xid errors in kernel log"
fi

# -------------------------------------------------------------------------
# 6. Check ECC errors
# -------------------------------------------------------------------------
echo ""
echo "--- Step 6: ECC Status ---"
ECC_UNCORR=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.aggregate.total --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
if [ "$ECC_UNCORR" = "N/A" ] || [ "${ECC_UNCORR:-0}" -eq 0 ] 2>/dev/null; then
  pass "No uncorrectable ECC errors"
else
  warn "$ECC_UNCORR uncorrectable ECC error(s) detected"
  info "GPU may need a reset: sudo nvidia-smi -r"
fi

# -------------------------------------------------------------------------
# 7. Detect virtualization type
# -------------------------------------------------------------------------
echo ""
echo "--- Step 7: Virtualization Detection ---"
VIRT_TYPE="unknown"
if command -v systemd-detect-virt &>/dev/null; then
  VIRT_TYPE=$(systemd-detect-virt 2>/dev/null || echo "none")
elif [ -f /sys/class/dmi/id/sys_vendor ]; then
  VENDOR=$(cat /sys/class/dmi/id/sys_vendor 2>/dev/null || echo "")
  case "$VENDOR" in
    *QEMU*|*KVM*) VIRT_TYPE="kvm" ;;
    *Xen*) VIRT_TYPE="xen" ;;
    *Microsoft*) VIRT_TYPE="microsoft" ;;
    *VMware*) VIRT_TYPE="vmware" ;;
    *) VIRT_TYPE="bare-metal-or-unknown" ;;
  esac
fi

if [ "$VIRT_TYPE" = "none" ] || [ "$VIRT_TYPE" = "bare-metal-or-unknown" ]; then
  pass "Bare metal detected (no virtualization)"
elif [ "$VIRT_TYPE" = "kvm" ]; then
  warn "KVM virtualization detected"
  info "This is a virtual machine, not bare metal."
  info "CUDA issues are common with KVM GPU passthrough."
  info "If CUDA still fails after fixes, the problem is in the host VM config."
else
  warn "Virtualization detected: $VIRT_TYPE"
fi

# -------------------------------------------------------------------------
# 8. Test CUDA with Python (the actual test)
# -------------------------------------------------------------------------
echo ""
echo "--- Step 8: CUDA Runtime Test ---"

# Find a Python with torch installed, or use system python
PYTHON_BIN=""
for candidate in \
  /workspace/leasing/rag_demo_system/.venv/bin/python \
  /workspace/leasing/rag_demo_system/.venv-voice-oss/bin/python \
  $(which python3 2>/dev/null) \
  $(which python 2>/dev/null); do
  if [ -n "$candidate" ] && [ -x "$candidate" ]; then
    if "$candidate" -c "import torch" 2>/dev/null; then
      PYTHON_BIN="$candidate"
      break
    fi
  fi
done

if [ -z "$PYTHON_BIN" ]; then
  warn "No Python with PyTorch found. Installing torch for testing..."
  if command -v pip3 &>/dev/null; then
    pip3 install torch --quiet 2>/dev/null
    PYTHON_BIN=$(which python3)
  else
    fail "Cannot test CUDA: no Python with torch available"
    info "Run provision_server.sh first to install venvs, then re-run this script."
    FATAL=1
  fi
fi

if [ -n "$PYTHON_BIN" ]; then
  CUDA_TEST=$("$PYTHON_BIN" -c "
import torch
import sys

cuda_ok = torch.cuda.is_available()
print(f'torch_version={torch.__version__}')
print(f'cuda_compiled={torch.version.cuda}')
print(f'cuda_available={cuda_ok}')

if cuda_ok:
    print(f'device_name={torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'device_memory_gb={props.total_mem / 1e9:.1f}')
    # Quick allocation test
    try:
        x = torch.randn(1000, 1000, device='cuda')
        y = x @ x
        del x, y
        torch.cuda.empty_cache()
        print('alloc_test=pass')
    except Exception as e:
        print(f'alloc_test=fail:{e}')
else:
    # Try to get more info about why CUDA failed
    try:
        torch.cuda.init()
    except Exception as e:
        print(f'init_error={e}')
    print(f'device_count={torch.cuda.device_count()}')
" 2>&1)

  echo "$CUDA_TEST" | while IFS='=' read -r key val; do
    case "$key" in
      torch_version) info "PyTorch version: $val" ;;
      cuda_compiled) info "CUDA compiled version: $val" ;;
      cuda_available)
        if [ "$val" = "True" ]; then
          pass "torch.cuda.is_available() = True"
        else
          fail "torch.cuda.is_available() = False"
        fi
        ;;
      device_name) pass "GPU device: $val" ;;
      device_memory_gb) info "GPU memory: ${val} GB" ;;
      alloc_test)
        if [ "$val" = "pass" ]; then
          pass "CUDA tensor allocation test passed"
        else
          fail "CUDA tensor allocation failed: $val"
        fi
        ;;
      init_error) fail "CUDA init error: $val" ;;
      device_count) info "CUDA device count: $val" ;;
    esac
  done

  # Check if CUDA actually worked
  if echo "$CUDA_TEST" | grep -q "cuda_available=True" && echo "$CUDA_TEST" | grep -q "alloc_test=pass"; then
    CUDA_WORKS=1
  else
    CUDA_WORKS=0
  fi
else
  CUDA_WORKS=0
fi

# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------
echo ""
echo "================================================================="
echo "  SUMMARY"
echo "================================================================="
echo ""

if [ "$FIXES_APPLIED" -gt 0 ]; then
  info "$FIXES_APPLIED fix(es) applied during this run"
fi

if [ "${CUDA_WORKS:-0}" -eq 1 ]; then
  pass "CUDA is working. Safe to run provision_server.sh"
  echo ""
  info "Next steps:"
  info "  HF_TOKEN=hf_... bash rag_demo_system/scripts/provision_server.sh"
  exit 0
elif [ "$FATAL" -eq 1 ]; then
  fail "CUDA is broken and cannot be fixed from inside this VM"
  echo ""
  info "Likely causes:"
  info "  1. KVM host has incomplete GPU passthrough (IOMMU/BAR config)"
  info "  2. cgroup restrictions blocking CUDA device nodes"
  info "  3. Driver version mismatch between host and guest"
  echo ""
  info "Recommended actions:"
  info "  - Contact Jarvis Labs support about CUDA initialization failures"
  info "  - Try a different instance (hardware may vary)"
  info "  - Switch to a bare-metal provider (Sesterce, Lambda Labs)"
  exit 1
else
  warn "CUDA status unclear. Fixes were applied; re-run this script to verify."
  info "  bash fix_cuda_and_verify.sh"
  exit 1
fi
