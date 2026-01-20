import numpy as np
import time
import gtsam

def test_gtsam_standalone():
    print("="*40)
    print("🧪 TEST GTSAM (CPU)")
    print("="*40)

    TARGET = np.array([1.0, 2.0, 3.0])
    START  = np.array([0.0, 0.0, 0.0])

    print(f"Goal: {START} -> {TARGET}")

    # 1. Setup Graph
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()
    
    # Key 'p' index 0
    key = gtsam.symbol('p', 0)
    
    # Noise Model (Simpangan baku = 0.1)
    noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
    
    # Prior Factor (Menarik titik ke Target)
    graph.add(gtsam.PriorFactorPoint3(key, gtsam.Point3(*TARGET), noise))
    initial_estimate.insert(key, gtsam.Point3(*START))

    # 2. Optimize
    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosityLM("SILENT")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)

    t0 = time.perf_counter()
    result = optimizer.optimize()
    t1 = time.perf_counter()

    final_pos = result.atPoint3(key)
    
    print(f"✅ Result: {final_pos}")
    print(f"⏱️ Time  : {(t1-t0)*1000:.4f} ms")

if __name__ == "__main__":
    test_gtsam_standalone()