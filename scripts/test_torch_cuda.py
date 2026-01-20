import torch
import sys

def test_pytorch():
    print("=" * 50)
    print("🧪 PYTORCH CUDA TEST")
    print("=" * 50)
    
    # Library Versions
    print(f"Python Version  : {sys.version.split()[0]}")
    print(f"PyTorch Version : {torch.__version__}")
    
    # 1. Check CUDA Availability
    if not torch.cuda.is_available():
        print("❌ CUDA is NOT available. Check your driver or install 'pytorch-gpu'.")
        sys.exit(1)
        
    print(f"✅ CUDA Available : Yes")
    print(f"🔢 CUDA Version   : {torch.version.cuda}")
    print(f"📦 cuDNN Version  : {torch.backends.cudnn.version()}")
    
    # 2. Check Device
    device_id = 0
    device_name = torch.cuda.get_device_name(device_id)
    print(f"🖥️  GPU Device     : {device_name}")
    print("-" * 50)
    
    # 3. Test Memory Allocation & Computation
    try:
        print("\n🔄 Running computation test...")
        
        # Create tensor on CPU
        x = torch.randn(5000, 5000)
        y = torch.randn(5000, 5000)
        
        print(f"   - Moving tensors to GPU...")
        device = torch.device("cuda")
        x_gpu = x.to(device)
        y_gpu = y.to(device)
        
        print(f"   - Performing Matrix Multiplication on GPU...")
        z_gpu = torch.matmul(x_gpu, y_gpu)
        
        # Ensure output is on GPU
        if z_gpu.is_cuda:
            print("   - Result is on GPU. Syncing...")
            torch.cuda.synchronize() # Wait for completion
            print("✅ PyTorch CUDA Computation PASSED!")
        else:
            print("❌ Computation output is NOT on GPU.")
            
    except Exception as e:
        print(f"❌ Computation FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_pytorch()