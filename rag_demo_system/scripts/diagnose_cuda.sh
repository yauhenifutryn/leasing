#!/usr/bin/env bash
set -euo pipefail
APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== CUDA Diagnostic ==="

"$APP_DIR/.venv/bin/python" << 'PYEOF'
import ctypes, os, subprocess

print("--- 1. Driver API (cuInit) ---")
try:
    cuda_driver = ctypes.CDLL("libcuda.so.1")
    r = cuda_driver.cuInit(0)
    print(f"cuInit: {r} (0=success)")
    count = ctypes.c_int()
    cuda_driver.cuDeviceGetCount(ctypes.byref(count))
    print(f"cuDeviceGetCount: {count.value}")
except Exception as e:
    print(f"FAILED: {e}")

print("\n--- 2. Runtime API (cudaGetDeviceCount) ---")
try:
    for lib_name in ["libcudart.so.12", "libcudart.so"]:
        try:
            cuda_rt = ctypes.CDLL(lib_name)
            count = ctypes.c_int()
            err = cuda_rt.cudaGetDeviceCount(ctypes.byref(count))
            print(f"{lib_name}: err={err}, count={count.value}")
            break
        except OSError:
            print(f"{lib_name}: not found")
except Exception as e:
    print(f"FAILED: {e}")

print("\n--- 3. NVML (what PyTorch uses internally) ---")
try:
    nvml = ctypes.CDLL("libnvidia-ml.so.1")
    r = nvml.nvmlInit_v2()
    print(f"nvmlInit: {r} (0=success)")
    count = ctypes.c_uint()
    r = nvml.nvmlDeviceGetCount_v2(ctypes.byref(count))
    print(f"nvmlDeviceGetCount: err={r}, count={count.value}")
    if count.value > 0:
        handle = ctypes.c_void_p()
        r = nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle))
        name = ctypes.create_string_buffer(256)
        nvml.nvmlDeviceGetName(handle, name, 256)
        print(f"Device 0: {name.value.decode()}")
    nvml.nvmlShutdown()
except Exception as e:
    print(f"FAILED: {e}")

print("\n--- 4. PyTorch CUDA ---")
try:
    import torch
    print(f"torch version: {torch.__version__}")
    print(f"torch cuda compiled: {torch.version.cuda}")
    print(f"torch cuda available: {torch.cuda.is_available()}")
    print(f"torch device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"torch device name: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"FAILED: {e}")

print("\n--- 5. Environment ---")
for var in ["CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "CUDA_HOME", "LD_LIBRARY_PATH", "LD_PRELOAD"]:
    print(f"{var}={os.environ.get(var, '(not set)')}")

print("\n--- 6. Loaded CUDA libs (from /proc) ---")
pid = os.getpid()
try:
    maps = open(f"/proc/{pid}/maps").read()
    for lib in ["libcuda", "libcudart", "libnvidia-ml", "libnvrtc"]:
        matches = [l.split()[-1] for l in maps.splitlines() if lib in l]
        unique = sorted(set(matches))
        if unique:
            print(f"{lib}: {', '.join(unique)}")
except Exception as e:
    print(f"FAILED: {e}")

print("\n--- 7. nvidia-smi GPU list ---")
try:
    out = subprocess.check_output(["nvidia-smi", "-L"]).decode().strip()
    print(out)
except Exception as e:
    print(f"FAILED: {e}")

print("\n=== END ===")
PYEOF
