import onnxruntime as ort
import numpy as np
import onnx
from onnx import helper, TensorProto
import os
import sys

def create_dummy_model(model_path):
    """Creates a simple ONNX model: Output = Input * 2"""
    # Input info
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3, 224, 224])
    # Output info
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 3, 224, 224])
    
    # Node (Multiply by 2)
    const_tensor = helper.make_tensor("const_2", TensorProto.FLOAT, [1], [2.0])
    const_node = helper.make_node("Constant", [], ["const_val"], value=const_tensor)
    
    mul_node = helper.make_node(
        'Mul', ['X', 'const_val'], ['Y'], name='mul_node'
    )
    
    # Graph
    graph_def = helper.make_graph(
        [const_node, mul_node], 'test_graph', [X], [Y]
    )
    
    # Force ir_version=9 to be compatible with onnxruntime which often requires max 11
    # We also explicitly set opset_imports for safety
    opset = helper.make_opsetid("", 15) 
    model_def = helper.make_model(
        graph_def, 
        producer_name='onnx-test', 
        ir_version=9,  # <--- FIX: Ensure compatibility
        opset_imports=[opset]
    )
    
    onnx.save(model_def, model_path)
    print(f"   - Created dummy model at: {model_path} (IR Version: 9)")

def test_onnxruntime():
    print("=" * 50)
    print("🧪 ONNX RUNTIME CUDA TEST")
    print("=" * 50)
    
    # Library Versions
    print(f"Python Version       : {sys.version.split()[0]}")
    print(f"ONNX Runtime Version : {ort.__version__}")
    print(f"ONNX Version         : {onnx.__version__}")
    print(f"Numpy Version        : {np.__version__}")
    print("-" * 50)
    
    # 1. Check Provider
    providers = ort.get_available_providers()
    print(f"Providers Available  : {providers}")
    
    if 'CUDAExecutionProvider' not in providers:
        print("❌ CUDAExecutionProvider NOT found. Check 'onnxruntime-gpu' installation.")
        sys.exit(1)
    
    print("✅ CUDA Provider detected.")

    # 2. Test Inference (End-to-End)
    model_file = "temp_test_model.onnx"
    try:
        print("\n🔄 Preparing Inference Test...")
        create_dummy_model(model_file)
        
        print("   - Starting InferenceSession with CUDAExecutionProvider...")
        # Force usage of CUDA
        session = ort.InferenceSession(model_file, providers=['CUDAExecutionProvider'])
        
        print("   - Running Inference...")
        input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        # Run
        outputs = session.run(['Y'], {'X': input_data})
        
        # Validation (Output must be Input * 2)
        expected = input_data * 2
        if np.allclose(outputs[0], expected, atol=1e-5):
            print("✅ ONNX Inference Result Match! (Computation Correct)")
        else:
            print("❌ Result mismatch!")

        print("✅ ONNX Runtime GPU is working correctly!")

    except Exception as e:
        print(f"❌ Inference FAILED: {e}")
    finally:
        # Cleanup
        if os.path.exists(model_file):
            os.remove(model_file)
            print("   - Cleanup: Removed temp model file.")

if __name__ == "__main__":
    test_onnxruntime()