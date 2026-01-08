#!/bin/bash

# Ensure the script is run within a pixi environment
if [[ -z "$PIXI_PROJECT_ROOT" ]] && [[ -z "$CONDA_PREFIX" ]]; then
    echo "⚠️  WARNING: This script must be run via 'pixi run bash check.sh'"
    echo "   Or run 'pixi shell' first."
    exit 1
fi

echo "======================================================="
echo "🔍 SMART ENVIRONMENT CHECKER (Deep Scan)"
echo "======================================================="

python - <<EOF
import sys
import importlib
import importlib.metadata
import os
import subprocess
import re

# --- CONFIGURATION ---

# 1. TRANSLATION LAYER (Map Conda Package Names -> Python Import Names)
#    This is necessary because Python usually has no record of Conda-specific 
#    names like 'pytorch-gpu' or 'opencv-contrib-python'.
CONDA_ALIAS_MAP = {
    "pytorch-gpu": "torch",
    "pytorch-cpu": "torch",
    "torchvision": "torchvision",
    "onnxruntime-gpu": "onnxruntime",
    "opencv-contrib-python": "cv2",
    "opencv-python": "cv2",
    "pillow": "PIL",
    "scikit-learn": "sklearn",
    "python-dateutil": "dateutil",
    "pyqt6": "PyQt6"
}

# 2. Ignore list (System tools / Non-python libraries)
IGNORE_LIST = {
    "python", "pip", "wheel", "setuptools", 
    "cuda", "cuda-version", "c-compiler", "cxx-compiler",
    "make", "cmake"
}

# --- HELPERS ---
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'
HOME_DIR = os.path.expanduser("~")

def shorten_path(path_str):
    if path_str and path_str.startswith(HOME_DIR):
        return path_str.replace(HOME_DIR, "~")
    return str(path_str)

def get_metadata_map():
    """
    Builds a dynamic map from all installed Python packages.
    Returns: {'distribution-name': 'ImportName'}
    """
    mapping = {}
    try:
        # packages_distributions returns: {'ImportName': ['DistName1', 'DistName2']}
        dists = importlib.metadata.packages_distributions()
        
        # Invert it to: {'distname': 'ImportName'}
        for import_name, dist_names in dists.items():
            for dist in dist_names:
                dist_lower = dist.lower()
                # If multiple imports exist, prioritize the one that matches logic
                # (e.g. prefer 'cv2' over 'opencv_contrib_python' if both exist as imports)
                if dist_lower not in mapping:
                    mapping[dist_lower] = import_name
                else:
                    # Heuristic: Prefer imports without underscores if possible
                    if "_" not in import_name and "_" in mapping[dist_lower]:
                        mapping[dist_lower] = import_name
    except Exception as e:
        pass
    return mapping

def resolve_import_name(pkg_name, metadata_map):
    """Guesses the Python import name based on priority."""
    pkg_lower = pkg_name.lower()
    
    # 1. Check Translation Layer (Most accurate for Conda-specific names)
    if pkg_lower in CONDA_ALIAS_MAP:
        return CONDA_ALIAS_MAP[pkg_lower]
    
    # 2. Check Metadata Map (Auto-detected from environment)
    if pkg_lower in metadata_map:
        return metadata_map[pkg_lower]
        
    # 3. Fallback Heuristics (Replace hyphens with underscores)
    return pkg_name.replace("-", "_")

def parse_pixi_toml():
    """Parses pixi.toml to get the list of requested dependencies."""
    dependencies = set()
    if not os.path.exists("pixi.toml"):
        print(f"{RED}pixi.toml not found!{RESET}")
        return []

    with open("pixi.toml", "r") as f:
        lines = f.readlines()

    # Regex to catch any dependency section header
    section_regex = re.compile(r"^\[.*dependencies.*\]")
    in_dep_section = False
    
    for line in lines:
        line = line.strip()
        if section_regex.match(line):
            in_dep_section = True
            continue
        elif line.startswith("["):
            in_dep_section = False
            continue

        if in_dep_section and "=" in line:
            parts = line.split("=")
            pkg_name = parts[0].strip().strip('"').strip("'")
            if pkg_name.lower() not in IGNORE_LIST:
                dependencies.add(pkg_name)

    return sorted(list(dependencies))

# --- MAIN EXECUTION ---

# 1. Setup Environment Knowledge
METADATA_MAP = get_metadata_map()
TOML_PKGS = parse_pixi_toml()

print(f"{'PACKAGE (toml)':<25} | {'IMPORT NAME':<15} | {'VERSION':<15} | {'PATH'}")
print("-" * 110)

# 2. Check Python System
print(f"{'python':<25} | {'sys':<15} | {sys.version.split()[0]:<15} | {shorten_path(sys.executable)}")

# 3. Iterate through Packages
for pkg_name in TOML_PKGS:
    
    # Determine import name
    import_name = resolve_import_name(pkg_name, METADATA_MAP)
    
    try:
        mod = importlib.import_module(import_name)
        
        # Get Version
        version = "N/A"
        if hasattr(mod, "__version__"):
            version = mod.__version__
        elif hasattr(mod, "VERSION"):
            version = mod.VERSION
        
        # Fix for PyQt6 (version is hidden inside QtCore)
        if import_name == "PyQt6" and version == "N/A":
            try:
                from PyQt6.QtCore import PYQT_VERSION_STR
                version = PYQT_VERSION_STR
            except: pass

        # Get Path & Validate
        path = "Built-in"
        if hasattr(mod, "__file__") and mod.__file__:
            raw_path = mod.__file__
            path = shorten_path(raw_path)
            
            # Validate Path:
            # Green = Inside .pixi or conda environment
            # Yellow = Outside environment (e.g., ~/.local/lib via pip user install)
            if ".pixi" in raw_path or "conda" in raw_path or sys.prefix in raw_path:
                path = f"{GREEN}{path}{RESET}"
            else:
                path = f"{YELLOW}{path} (System Leak?){RESET}"

        print(f"{pkg_name:<25} | {import_name:<15} | {GREEN}{version:<15}{RESET} | {path}")

    except ImportError:
        # If import fails, it means the map is wrong or package is missing
        print(f"{pkg_name:<25} | {import_name:<15} | {RED}{'NOT FOUND':<15}{RESET} | {RED}-{RESET}")

print("-" * 110)

# --- GPU CHECKS ---
print("\n🔍 GPU ACCELERATION CHECK")

# 1. NVIDIA Driver
try:
    driver_ver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], 
        encoding="utf-8"
    ).strip()
    print(f"NVIDIA Driver          : {GREEN}{driver_ver}{RESET}")
except:
    print(f"NVIDIA Driver          : {RED}nvidia-smi not found{RESET}")

# 2. PyTorch GPU
try:
    import torch
    cuda_avail = torch.cuda.is_available()
    print(f"PyTorch CUDA Available : {GREEN if cuda_avail else RED}{cuda_avail}{RESET}")
    
    if cuda_avail:
        print(f"CUDA Version (Torch)   : {torch.version.cuda}")
        print(f"Device Name            : {torch.cuda.get_device_name(0)}")
        try:
            torch.tensor([1.0]).cuda()
            print(f"Tensor Test            : {GREEN}Success{RESET}")
        except Exception as e:
            print(f"Tensor Test            : {RED}Failed ({e}){RESET}")
    else:
        # Check if user specifically asked for GPU but didn't get it
        if "pytorch-gpu" in TOML_PKGS:
             print(f"{YELLOW}Warning: 'pytorch-gpu' requested, but Torch is running on CPU.{RESET}")

except ImportError: pass

# 3. ONNX Runtime GPU
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    gpu_support = 'CUDAExecutionProvider' in providers
    color = GREEN if gpu_support else RED
    print(f"ONNX GPU Support       : {color}{'Yes' if gpu_support else 'No'}{RESET} (Providers: {providers})")
except ImportError: pass

EOF
echo ""
echo "======================================================="