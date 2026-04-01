#!/usr/bin/env bash
# fix_cuda_vm.sh - Diagnose and fix CUDA Error 802 on KVM/QEMU VMs
# Run this BEFORE provision_server.sh on any new VM where CUDA fails.
set -euo pipefail

echo "=== CUDA VM Fix Script ==="
echo ""

# Step 0: Basic checks
echo "--- Step 0: Basic checks ---"
nvidia-smi -L 2>/dev/null || { echo "FATAL: nvidia-smi not found. No GPU driver."; exit 1; }
echo "GPU found: $(nvidia-smi -L)"
echo "Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
echo ""

# Step 1: Enable GPU persistence mode
echo "--- Step 1: GPU persistence mode ---"
sudo nvidia-smi -pm 1 2>/dev/null && echo "Persistence mode: ON" || echo "WARNING: could not enable persistence mode"
echo ""

# Step 2: Fix KASLR/HMM conflict (most common cause on KVM VMs)
echo "--- Step 2: Disable HMM in nvidia_uvm (KASLR fix) ---"
echo 'options nvidia_uvm uvm_disable_hmm=1' | sudo tee /etc/modprobe.d/nvidia-uvm-hmm.conf
sudo rmmod nvidia_uvm 2>/dev/null || true
sudo modprobe nvidia_uvm
echo "nvidia_uvm reloaded with HMM disabled"
echo ""

# Step 3: Set environment variables that help on VMs
echo "--- Step 3: Environment variables ---"
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
echo "PYTORCH_NVML_BASED_CUDA_CHECK=1"
echo "CUDA_DEVICE_ORDER=PCI_BUS_ID"

# Make them permanent
sudo tee /etc/profile.d/cuda-vm-fix.sh > /dev/null << 'EOF'
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
EOF
echo "Saved to /etc/profile.d/cuda-vm-fix.sh"
echo ""

# Step 4: Ensure nvidia-persistenced is running
echo "--- Step 4: nvidia-persistenced ---"
if command -v nvidia-persistenced &>/dev/null; then
    sudo nvidia-persistenced --persistence-mode 2>/dev/null || true
    echo "nvidia-persistenced: running"
else
    echo "nvidia-persistenced: not installed (optional)"
fi
echo ""

# Step 5: Remove any system CUDA that conflicts with pip CUDA
echo "--- Step 5: Clean system CUDA ---"
for cuda_dir in /usr/local/cuda /usr/local/cuda-*; do
    if [ -d "$cuda_dir" ] && [[ ! "$cuda_dir" == *.disabled ]]; then
        echo "Moving $cuda_dir -> ${cuda_dir}.disabled"
        sudo mv "$cuda_dir" "${cuda_dir}.disabled" 2>/dev/null || true
    fi
done
sudo rm -f /etc/profile.d/cuda.sh 2>/dev/null || true
unset LD_LIBRARY_PATH 2>/dev/null || true
sudo ldconfig 2>/dev/null || true
echo "System CUDA cleaned"
echo ""

# Step 6: Quick CUDA test with raw ctypes (no PyTorch needed)
echo "--- Step 6: Raw CUDA test ---"
python3 -c "
import ctypes
try:
    cuda = ctypes.CDLL('libcuda.so.1')
    r = cuda.cuInit(0)
    print(f'cuInit: {r} (0=success)')
    count = ctypes.c_int()
    cuda.cuDeviceGetCount(ctypes.byref(count))
    print(f'Devices: {count.value}')
    if r == 0 and count.value > 0:
        name = ctypes.create_string_buffer(256)
        cuda.cuDeviceGetName(name, 256, 0)
        print(f'GPU 0: {name.value.decode()}')
        print('CUDA DRIVER: OK')
    else:
        print('CUDA DRIVER: FAILED')
except Exception as e:
    print(f'CUDA DRIVER: FAILED ({e})')
" 2>&1
echo ""

# Step 7: If we have a venv with torch, test that too
echo "--- Step 7: PyTorch CUDA test ---"
VENV_PYTHON="$(dirname "$0")/../.venv/bin/python"
if [ -x "$VENV_PYTHON" ]; then
    PYTORCH_NVML_BASED_CUDA_CHECK=1 CUDA_DEVICE_ORDER=PCI_BUS_ID \
    "$VENV_PYTHON" -c "
import torch
print(f'torch: {torch.__version__}')
print(f'CUDA compiled: {torch.version.cuda}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    t = torch.zeros(1).cuda()
    print(f'Tensor on GPU: {t.device}')
    print('PYTORCH CUDA: OK')
else:
    print('PYTORCH CUDA: FAILED')
" 2>&1
else
    echo "No venv found, skipping PyTorch test. Run provision first."
fi

echo ""
echo "=== Done ==="
echo "If CUDA DRIVER shows OK but PYTORCH shows FAILED:"
echo "  Try: sudo reboot  (then re-run this script)"
echo "If both show FAILED:"
echo "  This VM's GPU passthrough is incompatible with CUDA."
echo "  Use Vast.ai (container) or true bare-metal instead."
