# Pytorch-ONNX-CUDA-on-Pixi 🚀

Reproducible CUDA-enabled development environment using **Pixi** with PyTorch, ONNX Runtime, and optimization solvers for robotics/SLAM applications.

## 🎯 What's Included

- **Deep Learning**: PyTorch + ONNX Runtime (CUDA accelerated)
- **Computer Vision**: OpenCV (CUDA accelerated), Kornia
- **Optimization Solvers**: GTSAM, g2o, Ceres, Theseus, PyPose
- **Visualization**: Matplotlib, Rerun
- **Trajectory Tools**: EVO

## 🛠 Prerequisites

- **Linux** (Ubuntu 22.04+ recommended)
- **NVIDIA Drivers** (580+ for CUDA 12.9)
- **[Pixi](https://pixi.sh/)** package manager

## 📦 Clone

```bash
# Clone repository
git clone https://github.com/2black0/Pytorch-ONNX-CUDA-on-Pixi.git
cd Pytorch-ONNX-CUDA-on-Pixi
```

## 🧰 OpenCV Setup Options

```bash
# Optional: Link system OpenCV with CUDA support, you must have compiled it beforehand
pixi run link-opencv

# if not linking OpenCV, open comment on 'opencv-contrib-python' in pixi.toml and run:
pixi install
```

## ✅ Verify installation

```bash
pixi run check
```

with OpenCV CUDA supported will produce output similar to:

```bash
✨ Pixi task (check in default): bash scripts/check.sh
=======================================================
🔍 SMART ENVIRONMENT CHECKER (Deep Scan)
=======================================================
PACKAGE                   | IMPORT NAME     | VERSION         | PATH
--------------------------------------------------------------------------------------------------------------
python                    | sys             | 3.10.19         | ~/Documents/GitHub/Pytorch-ONNX-CUDA-on-Pixi/.pixi/envs/default/bin/python
evo                       | evo             | 1.34.2          | ~/Documents/GitHub/Pytorch-ONNX-CUDA-on-Pixi/.pixi/envs/default/lib/python3.10/site-packages/evo/__init__.py
g2o-python                | g2o             | 0.0.12          | ~/Documents/GitHub/Pytorch-ONNX-CUDA-on-Pixi/.pixi/envs/default/lib/python3.10/site-packages/g2o/__init__.py
gtsam                     | gtsam           | 4.2a9           | ~/Documents/GitHub/Pytorch-ONNX-CUDA-on-Pixi/.pixi/envs/default/lib/python3.10/site-packages/gtsam/__init__.py
kornia                    | kornia          | 0.8.2           | ~/Documents/GitHub/Pytorch-ONNX-CUDA-on-Pixi/.pixi/envs/default/lib/python3.10/site-packages/kornia/__init__.py
matplotlib                | matplotlib      | 3.10.8          | ~/Documents/GitHub/Pytorch-ONNX-CUDA-on-Pixi/.pixi/envs/default/lib/python3.10/site-packages/matplotlib/__init__.py
numpy                     | numpy           | 2.2.6           | ~/Documents/GitHub/Pytorch-ONNX-CUDA-on-Pixi/.pixi/envs/default/lib/python3.10/site-packages/numpy/__init__.py
onnx                      | onnx            | 1.20.0          | ~/Documents/GitHub/Pytorch-ONNX-CUDA-on-Pixi/.pixi/envs/default/lib/python3.10/site-packages/onnx/__init__.py
onnxruntime               | onnxruntime     | 1.23.2          | ~/Documents/GitHub/Pytorch-ONNX-CUDA-on-Pixi/.pixi/envs/default/lib/python3.10/site-packages/onnxruntime/__init__.py
onnxruntime-gpu           | onnxruntime     | 1.23.2          | ~/Documents/GitHub/Pytorch-ONNX-CUDA-on-Pixi/.pixi/envs/default/lib/python3.10/site-packages/onnxruntime/__init__.py
opencv                    | cv2             | 4.13.0          | ~/Documents/GitHub/Pytorch-ONNX-CUDA-on-Pixi/.pixi/envs/default/lib/python3.10/site-packages/cv2/__init__.py
pillow                    | PIL             | 12.1.0          | ~/Documents/GitHub/Pytorch-ONNX-CUDA-on-Pixi/.pixi/envs/default/lib/python3.10/site-packages/PIL/__init__.py
pyceres                   | pyceres         | 2.6             | ~/Documents/GitHub/Pytorch-ONNX-CUDA-on-Pixi/.pixi/envs/default/lib/python3.10/site-packages/pyceres.cpython-310-x86_64-linux-gnu.so
pypose                    | pypose          | 0.7.5           | ~/Documents/GitHub/Pytorch-ONNX-CUDA-on-Pixi/.pixi/envs/default/lib/python3.10/site-packages/pypose/__init__.py
pyqt6                     | PyQt6           | 13.10.0         | ~/Documents/GitHub/Pytorch-ONNX-CUDA-on-Pixi/.pixi/envs/default/lib/python3.10/site-packages/PyQt6/__init__.py
pytorch-cpu               | torch           | 2.9.1           | ~/Documents/GitHub/Pytorch-ONNX-CUDA-on-Pixi/.pixi/envs/default/lib/python3.10/site-packages/torch/__init__.py
pytorch-gpu               | torch           | 2.9.1           | ~/Documents/GitHub/Pytorch-ONNX-CUDA-on-Pixi/.pixi/envs/default/lib/python3.10/site-packages/torch/__init__.py
rerun                     | rerun           | 1.0.30          | ~/Documents/GitHub/Pytorch-ONNX-CUDA-on-Pixi/.pixi/envs/default/lib/python3.10/site-packages/rerun/__init__.py
scipy                     | scipy           | 1.15.2          | ~/Documents/GitHub/Pytorch-ONNX-CUDA-on-Pixi/.pixi/envs/default/lib/python3.10/site-packages/scipy/__init__.py
theseus-ai                | theseus         | 0.2.3           | ~/Documents/GitHub/Pytorch-ONNX-CUDA-on-Pixi/.pixi/envs/default/lib/python3.10/site-packages/theseus/__init__.py
torchvision               | torchvision     | 0.24.1          | ~/Documents/GitHub/Pytorch-ONNX-CUDA-on-Pixi/.pixi/envs/default/lib/python3.10/site-packages/torchvision/__init__.py
tqdm                      | tqdm            | 4.67.1          | ~/Documents/GitHub/Pytorch-ONNX-CUDA-on-Pixi/.pixi/envs/default/lib/python3.10/site-packages/tqdm/__init__.py
triton                    | triton          | 3.5.1           | ~/Documents/GitHub/Pytorch-ONNX-CUDA-on-Pixi/.pixi/envs/default/lib/python3.10/site-packages/triton/__init__.py
--------------------------------------------------------------------------------------------------------------

🖥️ GPU SYSTEM CHECK
  ├─ GPU Model         : NVIDIA GeForce RTX 4060 Ti
  ├─ NVIDIA Driver     : 580.95.05
  ├─ System CUDA       : 12.9
  ├─ NVCC Version      : 12.9
  └─ System cuDNN      : 9.1.0.2

📘 OPENCV CHECK
  ├─ Version           : 4.13.0
  ├─ Build Type        : Custom
  ├─ Contrib / NonFree : YES / YES
  ├─ GPU Device        : Supported ✅
  └─ OpenCV Test       : PASS (Memory Upload -> Threshold -> Download)

🔥 PYTORCH CHECK
  ├─ Version           : 2.9.1
  ├─ GPU Device        : Supported ✅
  └─ PyTorch Test      : PASS (Tensor Creation + MatMul on GPU)

🚀 ONNX RUNTIME CHECK
  ├─ Version           : 1.23.2
  ├─ Providers         : ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
  ├─ GPU Device        : Supported ✅
  └─ ONNX Test         : PASS (CUDAExecutionProvider is registered)

=======================================================
```

## 📝 Notes

- Environment isolation via `PYTHONNOUSERSITE=1` prevents conflicts with system packages
- `LD_LIBRARY_PATH` ensures correct C++ library linking (fixes CXXABI errors)
- GPU environment enabled by default, CPU-only available via `pixi shell -e cpu`

## 📄 License

MIT License - See repository for details
