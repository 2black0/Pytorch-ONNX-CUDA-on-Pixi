import numpy as np
import pyceres
import time

# Definisi Cost Function yang Benar
class DistanceCost(pyceres.CostFunction):
    def __init__(self, target):
        super().__init__()
        # 1 Residual (dimensi 3), 1 Parameter Block (dimensi 3)
        self.set_num_residuals(3)
        self.set_parameter_block_sizes([3])
        self.target = target

    def Evaluate(self, parameters, residuals, jacobians):
        x = parameters[0] 
        
        # 1. Isi Residuals (In-place modification)
        # Residual = x - target
        residuals[0] = x[0] - self.target[0]
        residuals[1] = x[1] - self.target[1]
        residuals[2] = x[2] - self.target[2]
        
        # 2. Isi Jacobian (Perbaikan Disini!)
        if jacobians is not None:
            if jacobians[0] is not None:
                # Matriks Identitas 3x3 (karena turunan x terhadap x adalah 1)
                # Flatten menjadi array 1D (9 elemen)
                J = np.identity(3).flatten()
                
                # PENTING: Gunakan np.copyto untuk menulis ke memori C++
                np.copyto(jacobians[0], J)
                
        return True

def test_pyceres_standalone():
    print("="*40)
    print("🧪 TEST PYCERES (CPU)")
    print("="*40)

    TARGET = np.array([1.0, 2.0, 3.0])
    START  = np.array([0.0, 0.0, 0.0])

    prob = pyceres.Problem()
    
    # Variable harus copy agar contigous di memori
    x_var = np.array(START, copy=True, dtype=np.float64)
    
    cost_func = DistanceCost(TARGET)
    prob.add_residual_block(cost_func, None, [x_var])

    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.DENSE_QR
    options.minimizer_progress_to_stdout = True # Lihat progres solver

    summary = pyceres.SolverSummary()
    
    print("🔄 Optimizing...")
    t0 = time.perf_counter()
    pyceres.solve(options, prob, summary)
    t1 = time.perf_counter()

    print(summary.BriefReport())
    print(f"✅ Result: {x_var}")
    print(f"⏱️ Time  : {(t1-t0)*1000:.4f} ms")

if __name__ == "__main__":
    test_pyceres_standalone()