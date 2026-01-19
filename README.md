# Pytorch-ONNX-CUDA-on-Pixi 🚀

Reproducible CUDA-enabled development environment using **Pixi** with PyTorch, ONNX Runtime, and optimization solvers for robotics/SLAM applications.

## 🎯 What's Included

- **Deep Learning**: PyTorch + ONNX Runtime (CUDA accelerated)
- **Computer Vision**: OpenCV, Kornia
- **Optimization Solvers**: GTSAM, g2o, Ceres, Theseus, PyPose
- **Visualization**: Matplotlib, Rerun
- **Trajectory Tools**: EVO

## 🛠 Prerequisites

- **Linux** (Ubuntu/Debian x86_64)
- **NVIDIA Drivers** (535+ for CUDA 12)
- **[Pixi](https://pixi.sh/)** package manager

## 📦 Quick Start

```bash
# Clone repository
git clone https://github.com/2black0/Pytorch-ONNX-CUDA-on-Pixi.git
cd Pytorch-ONNX-CUDA-on-Pixi

# Install all dependencies
pixi install

# Verify installation
pixi run check
```



## 📊 Optimization Solvers

This environment includes multiple state-of-the-art optimization libraries for robotics and SLAM:

| Solver | Type | Device | Best For |
|--------|------|--------|----------|
| **GTSAM** | Factor Graph | CPU | General SLAM, sensor fusion |
| **g2o** | Graph Optimization | CPU | Pose graph optimization |
| **Ceres** | Non-linear LS | CPU | Bundle adjustment, calibration |
| **Theseus** | Differentiable | GPU | Learning-based optimization |
| **PyPose** | Lie Group | GPU | Differentiable robotics |

**Benchmark solvers:**
```bash
pixi run python script/benchmark_solver.py
```

## ✅ Testing & Verification

### Environment Check
```bash
pixi run check
```

Validates all packages, versions, and GPU acceleration.

### PyTorch CUDA
```bash
pixi run test-torch
```

### ONNX Runtime CUDA
```bash
pixi run test-onnx
```

### OpenCV CUDA (Optional)

For GPU-accelerated OpenCV:

1. Compile OpenCV with CUDA support (Python 3.10)
2. Link to Pixi environment:
```bash
pixi run link-opencv
```

3. Benchmark CPU vs GPU:
```bash
pixi run python script/opencv_cpu_cuda_benchmark.py
```

## 🔧 Project Structure

```
.
├── pixi.toml                         # Dependencies & configuration
├── script/
│   ├── benchmark_solver.py           # Compare optimization solvers
│   ├── check.sh                      # Environment checker
│   ├── test_torch_cuda.py            # PyTorch GPU test
│   ├── test_ort_cuda.py              # ONNX Runtime GPU test
│   ├── link_opencv.sh                # Link system OpenCV (CUDA)
│   └── opencv_cpu_cuda_benchmark.py  # OpenCV CPU/GPU benchmark
└── README.md
```

## 🧩 Key Dependencies

- **Python**: 3.10
- **PyTorch**: 2.9.1 (CUDA 12.9)
- **ONNX Runtime GPU**: 1.23.2
- **CUDA Toolkit**: 12.x
- **Optimization**: GTSAM, g2o, Ceres, Theseus, PyPose
- **Vision**: OpenCV, Kornia, Pillow
- **Tools**: EVO, Matplotlib, Rerun, SciPy

## 📝 Notes

- Environment isolation via `PYTHONNOUSERSITE=1` prevents conflicts with system packages
- `LD_LIBRARY_PATH` ensures correct C++ library linking (fixes CXXABI errors)
- GPU environment enabled by default, CPU-only available via `pixi shell -e cpu`

## 📄 License

MIT License - See repository for details
