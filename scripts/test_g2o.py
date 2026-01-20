import numpy as np
import g2o
import time

def test_g2o_robust():
    print("="*50)
    print("🧪 TEST G2O (CPU) - ROBUST SE3 MODE")
    print("="*50)

    # 1. SETUP OPTIMIZER
    # Menggunakan BlockSolverSE3 (6-DOF) yang paling standar
    optimizer = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
    algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer.set_algorithm(algorithm)

    # 2. DEFINISI MASALAH: GESER ROBOT 1 METER KE KANAN
    # Kita ingin Robot bergerak dari (0,0,0) ke (1,0,0)
    
    # --- Node 0: Titik Awal (FIXED / JANGKAR) ---
    # Ini trik untuk menggantikan 'Prior Factor' yang sering error di Python wrapper.
    # Kita buat Node 0 diam (set_fixed=True) di posisi (0,0,0).
    v0 = g2o.VertexSE3()
    v0.set_id(0)
    v0.set_estimate(g2o.Isometry3d(np.identity(4))) # Posisi 0,0,0
    v0.set_fixed(True) # <--- INI KUNCINYA
    optimizer.add_vertex(v0)

    # --- Node 1: Titik Target (VARIABLE) ---
    # Tebakan awal kita salah (kita taruh di 0,0,0 juga)
    v1 = g2o.VertexSE3()
    v1.set_id(1)
    v1.set_estimate(g2o.Isometry3d(np.identity(4))) 
    v1.set_fixed(False) # Ini yang mau kita optimasi
    optimizer.add_vertex(v1)

    # --- Edge: Relasi Node 0 ke Node 1 ---
    # Kita bilang ke optimizer: "Jarak Node 0 ke Node 1 HARUSNYA [1, 0, 0]"
    edge = g2o.EdgeSE3()
    edge.set_vertex(0, v0)
    edge.set_vertex(1, v1)
    
    # Target Measurement (Transformasi 1 meter di X)
    target_pose = g2o.Isometry3d(np.identity(4))
    target_pose.set_translation(np.array([1.0, 2.0, 3.0])) # Target: x=1, y=2, z=3
    edge.set_measurement(target_pose)
    
    # Information Matrix (Presisi/Bobot)
    edge.set_information(np.identity(6) * 100.0)
    optimizer.add_edge(edge)

    # 3. OPTIMIZE
    print(f"   Start Estimate (Node 1): {v1.estimate().translation()}")
    print("🔄 Optimizing...")
    
    optimizer.initialize_optimization()
    
    t0 = time.perf_counter()
    optimizer.optimize(10) # 10 Iterasi
    t1 = time.perf_counter()

    # 4. HASIL
    res = v1.estimate().translation()
    print(f"   Final Estimate (Node 1): {res}")
    print(f"⏱️ Time: {(t1-t0)*1000:.4f} ms")

    # Verifikasi
    expected = np.array([1.0, 2.0, 3.0])
    if np.allclose(res, expected, atol=1e-3):
        print("✅ SUCCESS: g2o optimization converged!")
    else:
        print("❌ FAILED: Result did not converge.")

if __name__ == "__main__":
    test_g2o_robust()