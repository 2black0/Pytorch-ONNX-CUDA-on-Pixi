# Pytorch-ONNX-CUDA-on-Pixi 🚀

A robust, reproducible development environment template using **Pixi**. This setup ensures **PyTorch** and **ONNX Runtime** are correctly linked with **CUDA** on Linux, solving common dependency hell issues (like mismatched `libstdc++` or missing CUDA providers).

## 📂 Project Structure

```text
.
├── pixi.toml            # Project configuration and dependencies
├── pixi.lock            # Lockfile for reproducible environments
├── README.md            # Project documentation
└── script/              # Verification scripts
    ├── check.sh         # Smart environment checking script
    ├── test_ort_cuda.py # ONNX Runtime CUDA inference test
    └── test_torch_cuda.py # PyTorch CUDA computation test

```

## 🛠 Prerequisites

Before running this project, ensure you have the following installed:

* **Linux OS** (Tested on x86_64 Ubuntu/Debian based systems).
* **NVIDIA Drivers** (Recommended version 535+ for CUDA 12).
* **[Pixi Package Manager](https://pixi.sh/)**.

## 📦 Installation

1. **Clone this repository:**
```bash
git clone https://github.com/2black0/Pytorch-ONNX-CUDA-on-Pixi.git
cd Pytorch-ONNX-CUDA-on-Pixi
```


2. **Install dependencies:**
Pixi will automatically set up the environment including CUDA libraries, Python, and drivers.
```bash
pixi install
```



## ✅ Verification & Usage

This project includes three verification scripts to ensure your environment is fully hardware-accelerated.

### 1. Smart Environment Check

Run the comprehensive checker script to validate versions, paths, and GPU visibility. This script automatically detects libraries defined in `pixi.toml`.

```bash
pixi run bash ./script/check.sh
```

**Expected Output:**

```text
=======================================================
🔍 SMART ENVIRONMENT CHECKER (Deep Scan)
=======================================================
PACKAGE (toml)            | IMPORT NAME     | VERSION         | PATH
--------------------------------------------------------------------------------------------------------------
python                    | sys             | 3.10.19         | ~/.pixi/envs/default/bin/python
evo                       | evo             | v1.34.1         | ~/.pixi/envs/default/lib/python3.10/site-packages/evo/__init__.py
kornia                    | kornia          | 0.8.2           | ~/.pixi/envs/default/lib/python3.10/site-packages/kornia/__init__.py
matplotlib                | matplotlib      | 3.10.8          | ~/.pixi/envs/default/lib/python3.10/site-packages/matplotlib/__init__.py
numpy                     | numpy           | 2.2.6           | ~/.pixi/envs/default/lib/python3.10/site-packages/numpy/__init__.py
onnx                      | onnx            | 1.20.0          | ~/.pixi/envs/default/lib/python3.10/site-packages/onnx/__init__.py
onnxruntime               | onnxruntime     | 1.23.2          | ~/.pixi/envs/default/lib/python3.10/site-packages/onnxruntime/__init__.py
onnxruntime-gpu           | onnxruntime     | 1.23.2          | ~/.pixi/envs/default/lib/python3.10/site-packages/onnxruntime/__init__.py
opencv-contrib-python     | cv2             | 4.12.0          | ~/.pixi/envs/default/lib/python3.10/site-packages/cv2/__init__.py
pillow                    | PIL             | 12.1.0          | ~/.pixi/envs/default/lib/python3.10/site-packages/PIL/__init__.py
pyqt6                     | PyQt6           | 6.8.1           | ~/.pixi/envs/default/lib/python3.10/site-packages/PyQt6/__init__.py
pytorch-cpu               | torch           | 2.9.1           | ~/.pixi/envs/default/lib/python3.10/site-packages/torch/__init__.py
pytorch-gpu               | torch           | 2.9.1           | ~/.pixi/envs/default/lib/python3.10/site-packages/torch/__init__.py
scipy                     | scipy           | 1.15.2          | ~/.pixi/envs/default/lib/python3.10/site-packages/scipy/__init__.py
torchvision               | torchvision     | 0.24.1          | ~/.pixi/envs/default/lib/python3.10/site-packages/torchvision/__init__.py
tqdm                      | tqdm            | 4.67.1          | ~/.pixi/envs/default/lib/python3.10/site-packages/tqdm/__init__.py
--------------------------------------------------------------------------------------------------------------

🔍 GPU ACCELERATION CHECK
NVIDIA Driver          : 580.95.05
PyTorch CUDA Available : True
CUDA Version (Torch)   : 12.9
Device Name            : NVIDIA GeForce RTX 4060 Ti
Tensor Test            : Success
ONNX GPU Support       : Yes (Providers: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])

=======================================================
```

### 2. PyTorch CUDA Test

Performs matrix multiplication on the GPU to verify cuDNN and memory allocation.

```bash
pixi run python ./script/test_torch_cuda.py
```

**Expected Output:**

```text
==================================================
🧪 PYTORCH CUDA TEST
==================================================
Python Version  : 3.10.19
PyTorch Version : 2.9.1
✅ CUDA Available : Yes
🔢 CUDA Version   : 12.9
📦 cuDNN Version  : 91002
🖥️  GPU Device     : NVIDIA GeForce RTX 4060 Ti
--------------------------------------------------

🔄 Running computation test...
   - Moving tensors to GPU...
   - Performing Matrix Multiplication on GPU...
   - Result is on GPU. Syncing...
✅ PyTorch CUDA Computation PASSED!
```

### 3. ONNX Runtime CUDA Test

Creates a dummy ONNX model (IR v9), saves it, and runs inference using the `CUDAExecutionProvider`.

```bash
pixi run python ./script/test_ort_cuda.py
```

**Expected Output:**

```text
==================================================
🧪 ONNX RUNTIME CUDA TEST
==================================================
Python Version       : 3.10.19
ONNX Runtime Version : 1.23.2
ONNX Version         : 1.20.0
Numpy Version        : 2.2.6
--------------------------------------------------
Providers Available  : ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
✅ CUDA Provider detected.

🔄 Preparing Inference Test...
   - Created dummy model at: temp_test_model.onnx (IR Version: 9)
   - Starting InferenceSession with CUDAExecutionProvider...
   - Running Inference...
✅ ONNX Inference Result Match! (Computation Correct)
✅ ONNX Runtime GPU is working correctly!
   - Cleanup: Removed temp model file.
```

## 🧩 Key Dependencies

This project relies on the following key libraries (managed via `conda-forge` and `pypi`):

* **Python**: 3.10.*
* **PyTorch**: 2.9.1 (CUDA 12.9)
* **ONNX Runtime GPU**: 1.23.2
* **CUDA Toolkit**: 12.*
* **EVO**: For trajectory evaluation.
* **Kornia, OpenCV, Pillow, Matplotlib**: For image processing and visualization.

## 📝 Configuration Note

To ensure system isolation and proper library linking, `pixi.toml` is configured with:

```toml
[activation]
env = { PYTHONNOUSERSITE = "1", LD_LIBRARY_PATH = "$CONDA_PREFIX/lib" }
```

This prevents Python from loading packages from `~/.local/lib` and forces the system to prioritize Pixi's C++ libraries (fixing common `CXXABI` errors).