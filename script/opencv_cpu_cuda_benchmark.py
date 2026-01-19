import cv2
import numpy as np
import time

def benchmark_cuda_v2():
    print(f"OpenCV Version: {cv2.__version__}")
    print(f"Device: {cv2.cuda.printCudaDeviceInfo(0)}")

    # 1. Data Preparation (4K Image)
    shape = (2160, 3840, 3)
    print(f"\nPreparing image {shape} (~25MB)...")
    img_cpu = np.random.randint(0, 256, shape, dtype=np.uint8)
    
    # ---------------------------------------------------------
    # WARM-UP GPU (Very Important!)
    # Execute dummy to exclude driver initialization from timing
    # ---------------------------------------------------------
    print("Performing GPU Warm-up...", end="", flush=True)
    dummy_mat = cv2.cuda_GpuMat()
    dummy_mat.upload(np.zeros((100,100,3), dtype=np.uint8))
    cv2.cuda.createGaussianFilter(cv2.CV_8UC3, cv2.CV_8UC3, (3, 3), 0).apply(dummy_mat)
    print(" Done.\n")

    iterations = 50  # Repeat 50 times for stable average

    # ---------------------------------------------------------
    # TEST 1: CPU BENCHMARK
    # ---------------------------------------------------------
    start_cpu = time.time()
    for _ in range(iterations):
        cv2.GaussianBlur(img_cpu, (15, 15), 0)
    end_cpu = time.time()
    avg_cpu = (end_cpu - start_cpu) / iterations
    print(f"CPU Average Time: {avg_cpu*1000:.2f} ms")

    # ---------------------------------------------------------
    # TEST 2: GPU BENCHMARK (Including Upload/Download)
    # Scenario: Read camera frame -> GPU -> CPU
    # ---------------------------------------------------------
    start_gpu_full = time.time()
    gpu_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC3, cv2.CV_8UC3, (15, 15), 0)
    
    for _ in range(iterations):
        # Upload
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(img_cpu)
        # Process
        gpu_result = gpu_filter.apply(gpu_frame)
        # Download
        res = gpu_result.download()
        
    end_gpu_full = time.time()
    avg_gpu_full = (end_gpu_full - start_gpu_full) / iterations
    print(f"GPU (Total) Time: {avg_gpu_full*1000:.2f} ms (Upload + Process + Download)")

    # ---------------------------------------------------------
    # TEST 3: GPU BENCHMARK (Pure Compute)
    # Scenario: SLAM Pipeline (Data stays on GPU for next process)
    # ---------------------------------------------------------
    # Upload once outside loop
    gpu_frame_resident = cv2.cuda_GpuMat()
    gpu_frame_resident.upload(img_cpu)
    
    start_gpu_pure = time.time()
    for _ in range(iterations):
        # Result stays in GPU memory (without download)
        gpu_result_resident = gpu_filter.apply(gpu_frame_resident)
        
    end_gpu_pure = time.time()
    avg_gpu_pure = (end_gpu_pure - start_gpu_pure) / iterations
    print(f"GPU (Compute) Time: {avg_gpu_pure*1000:.2f} ms (Without data transfer)")

    # ---------------------------------------------------------
    # CONCLUSION
    # ---------------------------------------------------------
    print("\n--- ANALYSIS ---")
    print(f"CPU Speed    : {1/avg_cpu:.1f} FPS")
    print(f"GPU (Pipeline): {1/avg_gpu_full:.1f} FPS (Bottleneck at PCIe)")
    print(f"GPU (Compute) : {1/avg_gpu_pure:.1f} FPS (True Potential)")
    
    speedup_pure = avg_cpu / avg_gpu_pure
    print(f"\n🚀 Potential Speedup for Complex Algorithms (Pure Compute): {speedup_pure:.2f}x")

if __name__ == "__main__":
    benchmark_cuda_v2()
