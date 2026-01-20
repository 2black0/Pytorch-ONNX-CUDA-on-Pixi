import torch
import theseus as th

def test_theseus_optimization():
    print("=" * 50)
    print("🧪 THESEUS-AI OPTIMIZATION TEST (Fixed)")
    print("=" * 50)

    # 1. Cek Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")

    # Gunakan float32 (Standar GPU)
    dtype = torch.float32 

    # 2. Setup Masalah Optimasi
    # Definisi variabel (tanpa data awal dulu, cuma dimensi doang)
    x = th.Vector(1, name="x", dtype=dtype) 
    target = th.Vector(1, name="target", dtype=dtype)
    
    # Cost Function
    cost_function = th.Difference(
        x, 
        target, 
        th.ScaleCostWeight(1.0)
    )

    # 3. Masukkan ke Objective
    objective = th.Objective()
    objective.add(cost_function)

    # 4. Buat Optimizer
    optimizer = th.LevenbergMarquardt(
        objective,
        max_iterations=15,
        step_size=1.0
    )

    # --- PERBAIKAN UTAMA DI SINI ---
    # Bungkus optimizer dengan TheseusLayer
    # Ini yang akan otomatis menghitung batch size dari input
    theseus_layer = th.TheseusLayer(optimizer)
    theseus_layer.to(device) # Pindahkan layer (dan objective) ke GPU

    # 5. Siapkan Data Batch
    # Batch size = 2 (Contoh: Kita mau optimasi 2 kondisi sekaligus)
    inputs = {
        "x": torch.tensor([[0.0], [5.0]], device=device, dtype=dtype),
        "target": torch.tensor([[1.0], [1.0]], device=device, dtype=dtype)
    }

    print("\n🔄 Starting Optimization...")
    print(f"   Initial Values x: {inputs['x'].tolist()}")
    
    # 6. Jalankan Optimasi via Layer
    # Layer.forward() mengembalikan (updated_inputs, info)
    with torch.no_grad():
        updated_inputs, info = theseus_layer.forward(
            input_tensors=inputs,
            optimizer_kwargs={"track_best_solution": True, "verbose": False}
        )

    # 7. Cek Hasil
    # Ambil nilai 'x' yang sudah diupdate dari output layer
    final_x = updated_inputs["x"]
    
    print(f"   Final Values x  : {final_x.tolist()}")
    print(f"   Status          : {info.status}")

    # Verifikasi
    expected = torch.tensor([[1.0], [1.0]], device=device, dtype=dtype)
    if torch.allclose(final_x, expected, atol=1e-3):
        print("\n✅ SUCCESS: Theseus berhasil mengoptimasi nilai ke target!")
    else:
        print("\n❌ FAILED: Hasil tidak konvergen.")

if __name__ == "__main__":
    test_theseus_optimization()