#!/usr/bin/env python3
"""Smoke test for ONNX Runtime on CPU (and GPU when available)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional

import importlib
import traceback

import numpy as np


TARGET_TOL = 1e-5


def _load_ort():
    try:
        return importlib.import_module("onnxruntime")
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("onnxruntime is not installed") from exc


def _create_simple_model(path: Path) -> None:
    try:
        import onnx
        from onnx import helper, TensorProto
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("onnx is required to build the smoke-test model") from exc

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 2])
    W = helper.make_tensor("W", TensorProto.FLOAT, [2, 2], [2.0, 0.0, 0.0, -1.5])
    B = helper.make_tensor("B", TensorProto.FLOAT, [2], [0.5, -0.5])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 2])

    node = helper.make_node("Gemm", inputs=["X", "W", "B"], outputs=["Y"], transB=0)
    graph = helper.make_graph([node], "linear", [X], [Y], initializer=[W, B])
    model = helper.make_model(graph, producer_name="onnxruntime_smoke_test")
    model.ir_version = 7
    if model.opset_import:
        model.opset_import[0].version = 13
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _run_session(session, inputs: np.ndarray) -> np.ndarray:
    ort_inputs = {session.get_inputs()[0].name: inputs}
    outputs = session.run(None, ort_inputs)
    return outputs[0]


def run_smoke_test() -> Dict[str, Optional[float]]:
    ort = _load_ort()

    results: Dict[str, Optional[float]] = {"cpu_error": None, "gpu_error": None}

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "linear.onnx"
        _create_simple_model(model_path)

        inputs = np.array([[0.2, -0.1], [0.4, 0.5]], dtype=np.float32)
        expected = inputs @ np.array([[2.0, 0.0], [0.0, -1.5]], dtype=np.float32) + np.array(
            [0.5, -0.5], dtype=np.float32
        )

        print("[CPU] Running ONNX Runtime inference...")
        sess_options = ort.SessionOptions()
        cpu_session = ort.InferenceSession(model_path.as_posix(), sess_options, providers=["CPUExecutionProvider"])
        cpu_out = _run_session(cpu_session, inputs)
        cpu_error = float(np.max(np.abs(cpu_out - expected)))
        results["cpu_error"] = cpu_error
        print(f"[CPU] Max abs error: {cpu_error:.6e}")

        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            print("[GPU] CUDAExecutionProvider detected; running GPU inference...")
            gpu_session = ort.InferenceSession(
                model_path.as_posix(), sess_options, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            gpu_out = _run_session(gpu_session, inputs)
            gpu_error = float(np.max(np.abs(gpu_out - expected)))
            results["gpu_error"] = gpu_error
            print(f"[GPU] Max abs error: {gpu_error:.6e}")
        else:
            print("[GPU] CUDAExecutionProvider not available; skipping GPU run.")

    return results


def main() -> None:
    try:
        ort = _load_ort()
    except RuntimeError as exc:
        print(f"ONNX Runtime not installed: {exc}")
        sys.exit(1)

    print("=== Environment ===")
    print(f"Python version      : {sys.version.split()[0]}")
    print(f"onnxruntime version : {ort.__version__}")
    print(f"Providers available : {', '.join(ort.get_available_providers())}")

    print("\n=== ONNX Runtime Smoke Test ===")
    print("Objective: ensure a simple linear ONNX model runs on CPU and GPU (if available)")

    try:
        result = run_smoke_test()
    except Exception as exc:
        print(f"Smoke test failed: {exc}")
        traceback.print_exc()
        sys.exit(2)

    print("\n=== Results ===")
    print(f"CPU max abs error : {result['cpu_error']:.6e}")
    if result["gpu_error"] is not None:
        print(f"GPU max abs error : {result['gpu_error']:.6e}")
    else:
        print("GPU max abs error : skipped (provider unavailable)")

    cpu_ok = result["cpu_error"] is not None and result["cpu_error"] < TARGET_TOL
    gpu_ok = True
    if result["gpu_error"] is not None:
        gpu_ok = result["gpu_error"] < TARGET_TOL

    if cpu_ok and gpu_ok:
        print("Status: SUCCESS - ONNX Runtime produced outputs within tolerance.")
        sys.exit(0)

    print("Status: FAILURE - outputs exceeded the allowed tolerance.")
    sys.exit(2)


if __name__ == "__main__":
    main()
