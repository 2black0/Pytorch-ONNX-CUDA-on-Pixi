#!/bin/bash

# Ensure the script is run within a pixi environment
if [[ -z "$PIXI_PROJECT_ROOT" ]] && [[ -z "$CONDA_PREFIX" ]]; then
    echo "⚠️  WARNING: This script must be run via 'pixi run check'"
    echo "   Or run 'pixi shell' first."
    exit 1
fi

echo "======================================================="
echo "🔍 SMART ENVIRONMENT CHECKER (Deep Scan)"
echo "======================================================="

python - <<'EOF'
import sys
import importlib
import importlib.metadata
import importlib.util
import os
import subprocess
import re
import shutil

# --- CONFIGURATION ---

CONDA_ALIAS_MAP = {
    "pytorch-gpu": "torch",
    "pytorch-cpu": "torch",
    "torchvision": "torchvision",
    "onnxruntime-gpu": "onnxruntime",
    "opencv-contrib-python": "cv2",
    "opencv-python": "cv2",
    "opencv": "cv2",
    "pillow": "PIL",
    "scikit-learn": "sklearn",
    "python-dateutil": "dateutil",
    "pyqt6": "PyQt6"
}

IGNORE_LIST = {
    "python", "pip", "wheel", "setuptools", 
    "cuda", "cuda-version", "c-compiler", "cxx-compiler",
    "make", "cmake"
}

# --- HELPERS ---
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
MAGENTA = '\033[95m'
RESET = '\033[0m'
HOME_DIR = os.path.expanduser("~")

def shorten_path(path_str):
    if path_str and path_str.startswith(HOME_DIR):
        return path_str.replace(HOME_DIR, "~")
    return str(path_str)

def get_metadata_maps():
    dist_to_import = {}
    import_to_dists = {}
    try:
        import_to_dists = importlib.metadata.packages_distributions()
        for import_name, dist_names in import_to_dists.items():
            for dist in dist_names:
                dist_lower = dist.lower()
                if dist_lower not in dist_to_import:
                    dist_to_import[dist_lower] = import_name
    except: pass
    return dist_to_import, import_to_dists

def resolve_import_name(pkg_name, metadata_map):
    pkg_lower = pkg_name.lower()
    if pkg_lower in CONDA_ALIAS_MAP:
        return CONDA_ALIAS_MAP[pkg_lower]
    if pkg_lower in metadata_map:
        return metadata_map[pkg_lower]
    return pkg_name.replace("-", "_")

def strip_toml_comment(line):
    in_single = False
    in_double = False
    escape = False
    for i, ch in enumerate(line):
        if ch == "\\" and in_double and not escape:
            escape = True
            continue
        if ch == '"' and not in_single and not escape:
            in_double = not in_double
        elif ch == "'" and not in_double:
            in_single = not in_single
        elif ch == "#" and not in_single and not in_double:
            return line[:i]
        escape = False
    return line

def parse_pixi_toml():
    dependencies = set()
    if not os.path.exists("pixi.toml"):
        return []
    with open("pixi.toml", "r") as f:
        lines = f.readlines()
    section_regex = re.compile(r"^\[.*dependencies.*\]")
    in_dep_section = False
    for line in lines:
        raw = strip_toml_comment(line).strip()
        if not raw:
            continue
        if section_regex.match(raw):
            in_dep_section = True
            continue
        elif raw.startswith("["):
            in_dep_section = False
            continue
        if in_dep_section and "=" in raw:
            parts = raw.split("=", 1)
            pkg_name = parts[0].strip().strip('"').strip("'")
            if pkg_name.lower() not in IGNORE_LIST:
                dependencies.add(pkg_name)
    return sorted(list(dependencies))

# --- ROBUST SYSTEM INFO HELPERS ---
def get_nvidia_smi_path():
    # 1. Cek PATH environment normal
    path = shutil.which("nvidia-smi")
    if path: return path
    # 2. Cek Hardcoded Path (Ubuntu Standard)
    if os.path.exists("/usr/bin/nvidia-smi"): return "/usr/bin/nvidia-smi"
    if os.path.exists("/bin/nvidia-smi"): return "/bin/nvidia-smi"
    return None

def get_gpu_info_sys():
    smi = get_nvidia_smi_path()
    
    # Init default values
    driver, cuda_ver, name, vram = "N/A", "N/A", "Unknown", "N/A"
    
    # METHOD 1: Try nvidia-smi
    if smi:
        try:
            out = subprocess.check_output(
                [smi, "--query-gpu=driver_version,cuda_version,name,memory.total", "--format=csv,noheader"], 
                encoding="utf-8"
            ).strip()
            if "\n" in out: out = out.split("\n")[0]
            parts = out.split(", ")
            if len(parts) >= 4:
                driver, cuda_ver, name, vram = parts[0], parts[1], parts[2], parts[3]
        except: pass

    # METHOD 2: Try /proc for driver version (even when nvidia-smi fails)
    if driver == "N/A":
        try:
            with open("/proc/driver/nvidia/version", "r") as f:
                data = f.read()
            m = re.search(r"Kernel Module\s+([0-9.]+)", data)
            if not m:
                m = re.search(r"NVRM version:\s+.*?\s([0-9.]+)\s", data)
            if m:
                driver = m.group(1)
        except: pass

    # METHOD 3: Fallback to PyTorch if nvidia-smi failed but Torch works
    # (Ini trik agar tidak muncul "Unknown" atau "No GPU" padahal Torch bisa baca)
    if name == "Unknown":
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                vram = "Detected via Torch"
                if cuda_ver == "N/A": cuda_ver = torch.version.cuda
        except: pass

    return driver, cuda_ver, name, vram

def get_nvcc_ver():
    # Cek nvcc
    nvcc = shutil.which("nvcc")
    if not nvcc:
        if os.path.exists("/usr/local/cuda/bin/nvcc"): nvcc = "/usr/local/cuda/bin/nvcc"
        elif os.path.exists("/usr/bin/nvcc"): nvcc = "/usr/bin/nvcc"
    
    if not nvcc: return "Not Found"
    
    try:
        out = subprocess.check_output([nvcc, "--version"], encoding="utf-8")
        match = re.search(r"release ([0-9\.]+)", out)
        return match.group(1) if match else "Unknown"
    except: return "Error"

def fmt_fix(value, fix_when=None):
    if fix_when is None:
        fix_when = {"N/A", "Not Found", "Unknown", "Error"}
    if value in fix_when:
        return f"{value} << Fix"
    return value

def get_cudnn_version():
    try:
        import ctypes
        lib = ctypes.CDLL("libcudnn.so")
        lib.cudnnGetVersion.restype = ctypes.c_size_t
        ver = int(lib.cudnnGetVersion())
        if ver >= 10000:
            major = ver // 10000
            minor = (ver % 10000) // 1000
            patch = (ver % 1000) // 100
            build = ver % 100
            return f"{major}.{minor}.{patch}.{build}"
        major = ver // 1000
        minor = (ver % 1000) // 100
        patch = ver % 100
        return f"{major}.{minor}.{patch}"
    except Exception:
        return None

def safe_opencv_cuda_device_count(cv2):
    try:
        return cv2.cuda.getCudaEnabledDeviceCount()
    except Exception:
        return 0

def get_module_path(import_name):
    try:
        spec = importlib.util.find_spec(import_name)
    except Exception:
        return None
    if spec is None:
        return None
    if spec.origin in (None, "built-in", "frozen"):
        if spec.submodule_search_locations:
            return str(list(spec.submodule_search_locations)[0])
        return "Built-in"
    return spec.origin

def get_dist_version(pkg_name, import_name, import_to_dists):
    for dist in import_to_dists.get(import_name, []):
        try:
            return importlib.metadata.version(dist)
        except Exception:
            pass
    try:
        return importlib.metadata.version(pkg_name)
    except Exception:
        return "N/A"

def get_import_version(import_name):
    try:
        mod = importlib.import_module(import_name)
        if hasattr(mod, "__version__"):
            return mod.__version__
        if hasattr(mod, "VERSION"):
            return mod.VERSION
    except Exception:
        return "N/A"
    return "N/A"

# --- PART 1: PACKAGE INVENTORY ---

DIST_TO_IMPORT, IMPORT_TO_DISTS = get_metadata_maps()
TOML_PKGS = parse_pixi_toml()

if "opencv" not in TOML_PKGS:
    TOML_PKGS.append("opencv")
    TOML_PKGS.sort()

print(f"{'PACKAGE':<25} | {'IMPORT NAME':<15} | {'VERSION':<15} | {'PATH'}")
print("-" * 110)
print(f"{'python':<25} | {'sys':<15} | {sys.version.split()[0]:<15} | {shorten_path(sys.executable)}")

for pkg_name in TOML_PKGS:
    import_name = resolve_import_name(pkg_name, DIST_TO_IMPORT)
    raw_path = get_module_path(import_name)
    if not raw_path:
        print(f"{pkg_name:<25} | {import_name:<15} | {RED}{'NOT FOUND':<15}{RESET} | {RED}-{RESET}")
        continue

    version = get_dist_version(pkg_name, import_name, IMPORT_TO_DISTS)
    if version == "N/A" and import_name == "cv2":
        version = get_import_version(import_name)
    if raw_path == "Built-in":
        path = "Built-in"
    else:
        path = shorten_path(raw_path)
        if ".pixi" in raw_path or "conda" in raw_path or sys.prefix in raw_path:
            path = f"{GREEN}{path}{RESET}"
        else:
            path = f"{YELLOW}{path} (System Link){RESET}"

    print(f"{pkg_name:<25} | {import_name:<15} | {GREEN}{version:<15}{RESET} | {path}")

print("-" * 110)

# --- PART 2: SYSTEM CHECKS ---
# Gather Global Info First
driver_sys, cuda_sys, gpu_name_sys, vram_sys = get_gpu_info_sys()

print("")

# ==========================================
# 🖥️ GPU SYSTEM CHECK (GLOBAL)
# ==========================================
print("🖥️ GPU SYSTEM CHECK")
nvcc_ver = get_nvcc_ver()
if cuda_sys in {"N/A", "Not Found", "Unknown"} and nvcc_ver not in {"Not Found", "Error", "Unknown"}:
    cuda_sys = nvcc_ver

cudnn_sys = "Not Found"
try:
    ld_out = subprocess.check_output("ldconfig -p | grep libcudnn.so", shell=True, encoding="utf-8")
    if "libcudnn.so" in ld_out:
        cudnn_ver = get_cudnn_version()
        cudnn_sys = cudnn_ver if cudnn_ver else "Detected (System Lib)"
except: pass

gpu_model = gpu_name_sys if gpu_name_sys != "Unknown" else "N/A"
gpu_model_display = fmt_fix(gpu_model, {"N/A", "Unknown"})
driver_display = driver_sys
cuda_display = fmt_fix(cuda_sys)
nvcc_display = fmt_fix(nvcc_ver)
cudnn_display = cudnn_sys

print(f"  ├─ GPU Model         : {gpu_model_display}")
print(f"  ├─ NVIDIA Driver     : {driver_display}")
print(f"  ├─ System CUDA       : {cuda_display}")
print(f"  ├─ NVCC Version      : {nvcc_display}")
print(f"  └─ System cuDNN      : {cudnn_display}")

print("")

# ==========================================
# 📘 OPENCV CHECK
# ==========================================
print("📘 OPENCV CHECK")
try:
    import cv2
    import numpy as np
    
    # Info
    build_info = cv2.getBuildInformation()
    has_contrib = "YES" if "opencv_contrib" in build_info else "NO"
    
    def get_info(pattern, default="N/A"):
        match = re.search(pattern, build_info, re.IGNORECASE)
        return match.group(1).strip() if match else default

    nonfree = "YES" if re.search(r"Non-free algorithms:\s+YES", build_info, re.IGNORECASE) else "NO"

    print(f"  ├─ Version           : {cv2.__version__}")
    print(f"  ├─ Build Type        : Custom")
    print(f"  ├─ Contrib / NonFree : {has_contrib} / {nonfree}")

    dev_count = safe_opencv_cuda_device_count(cv2)
    if dev_count > 0:
        print(f"  ├─ GPU Device        : Supported ✅")

        # Test
        try:
            host_mat = np.random.random((256, 256)).astype(np.float32)
            gpu_mat = cv2.cuda_GpuMat()
            gpu_mat.upload(host_mat)
            ret, gpu_thresh = cv2.cuda.threshold(gpu_mat, 0.5, 1.0, cv2.THRESH_BINARY)
            result = gpu_thresh.download()
            if result.shape == (256, 256):
                print("  └─ OpenCV Test       : PASS (Memory Upload -> Threshold -> Download)")
            else:
                print("  └─ OpenCV Test       : FAIL (Shape Mismatch)")
        except Exception as e:
            print(f"  └─ OpenCV Test       : FAIL ({e})")
    else:
        print("  ├─ GPU Device        : Not Supported")
        print("  └─ OpenCV Test       : SKIPPED (No CUDA device)")

except ImportError:
    print("")
    print("  └─ Status            : Not Installed")


# ==========================================
# 🔥 PYTORCH CHECK
# ==========================================
print("\n🔥 PYTORCH CHECK")
try:
    import torch
    print(f"  ├─ Version           : {torch.__version__}")
    
    if torch.cuda.is_available():
        gpu_name_torch = torch.cuda.get_device_name(0)
        print(f"  ├─ GPU Device        : Supported ✅")
        
        # Test
        try:
            a = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).cuda()
            b = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).cuda()
            c = torch.matmul(a, b)
            print("  └─ PyTorch Test      : PASS (Tensor Creation + MatMul on GPU)")
        except Exception as e:
            print(f"  └─ PyTorch Test      : FAIL ({e})")
    else:
        print("  ├─ GPU Device        : Not Supported")
        print("  └─ PyTorch Test      : SKIPPED (CPU Only)")

except ImportError:
    print("")
    print("  └─ Status            : Not Installed")


# ==========================================
# 🚀 ONNX RUNTIME CHECK
# ==========================================
print("\n🚀 ONNX RUNTIME CHECK")
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    
    print(f"  ├─ Version           : {ort.__version__}")
    print(f"  ├─ Providers         : {providers}")
    
    if 'CUDAExecutionProvider' in providers:
        print("  ├─ GPU Device        : Supported ✅")
        try:
            # Smoke test (Provider registration)
            print("  └─ ONNX Test         : PASS (CUDAExecutionProvider is registered)")
        except Exception as e:
            print(f"  └─ ONNX Test         : FAIL ({e})")
    else:
        print("  ├─ GPU Device        : Not Supported")
        print("  └─ ONNX Test         : FAIL")

except ImportError:
    print("")
    print("  └─ Status            : Not Installed")

EOF
echo ""
echo "======================================================="
