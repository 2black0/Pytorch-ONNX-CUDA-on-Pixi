#!/usr/bin/env python3
"""Smoke test for PyTorch on CPU (and GPU when available)."""

from __future__ import annotations

import sys
import traceback
from typing import Any, Dict, Optional

import importlib

_TORCH: Any | None = None


def _load_torch() -> Any:
    global _TORCH
    if _TORCH is None:
        try:
            _TORCH = importlib.import_module("torch")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment specific
            raise RuntimeError("PyTorch is not installed") from exc
    return _TORCH


TARGET_LOSS = 1e-4


TRUE_W = [[1.5, -0.7, 0.2, 0.8], [-0.3, 1.1, -0.4, 0.5]]
TRUE_B = [0.25, -0.4]


def _train_toy_model(device: Any, steps: int = 300) -> float:
    torch = _load_torch()
    torch.manual_seed(42)
    model = torch.nn.Linear(4, 2).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    inputs = torch.linspace(-1, 1, 64, device=device).view(32, 2)
    features = torch.cat([inputs, inputs**2], dim=1)
    true_w = torch.tensor(TRUE_W, dtype=features.dtype, device=device).t()
    true_b = torch.tensor(TRUE_B, dtype=features.dtype, device=device)
    target = features @ true_w + true_b

    loss = torch.tensor(0.0, device=device)
    for _ in range(steps):
        optimizer.zero_grad()
        pred = model(features)
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        optimizer.step()

    return float(loss.detach().cpu().item())


def run_smoke_test() -> Dict[str, Optional[float]]:
    torch = _load_torch()
    results: Dict[str, Optional[float]] = {"cpu_loss": None, "gpu_loss": None}

    print("[CPU] Running toy training loop...")
    results["cpu_loss"] = _train_toy_model(torch.device("cpu"))
    print(f"[CPU] Final loss: {results['cpu_loss']:.6e}")

    if torch.cuda.is_available():
        print("[GPU] CUDA detected, running the same training loop on GPU...")
        try:
            results["gpu_loss"] = _train_toy_model(torch.device("cuda"))
            print(f"[GPU] Final loss: {results['gpu_loss']:.6e}")
        except Exception as exc:
            print(f"[GPU] Training failed: {exc}")
            traceback.print_exc()
            raise
    else:
        print("[GPU] CUDA not available; skipping GPU portion.")

    return results


def _format_device_summary() -> str:
    try:
        torch = _load_torch()
    except RuntimeError:
        return "unavailable"
    devices = ["CPU"]
    if torch.cuda.is_available():
        devices.append("CUDA")
    return ", ".join(devices)


def main() -> None:
    try:
        torch = _load_torch()
    except RuntimeError as exc:
        print(f"PyTorch not installed: {exc}")
        sys.exit(1)
    print("=== Environment ===")
    print(f"Python version  : {sys.version.split()[0]}")
    print(f"PyTorch version : {torch.__version__}")
    print(f"Devices         : {_format_device_summary()}")

    print("\n=== PyTorch Smoke Test ===")
    print("Objective: ensure a simple model trains on CPU and (optionally) GPU without errors")

    try:
        result = run_smoke_test()
    except Exception:
        sys.exit(2)

    print("\n=== Results ===")
    print(f"CPU loss : {result['cpu_loss']:.6e}")
    if result["gpu_loss"] is not None:
        print(f"GPU loss : {result['gpu_loss']:.6e}")
    else:
        print("GPU loss : skipped (device unavailable)")

    torch = _load_torch()
    cpu_ok = result["cpu_loss"] is not None and result["cpu_loss"] < TARGET_LOSS
    gpu_ok = True
    if torch.cuda.is_available():
        gpu_ok = result["gpu_loss"] is not None and result["gpu_loss"] < TARGET_LOSS

    if cpu_ok and gpu_ok:
        print("Status: SUCCESS - PyTorch training loop converged within tolerance.")
        sys.exit(0)

    print("Status: FAILURE - loss did not reach the expected tolerance.")
    sys.exit(2)


if __name__ == "__main__":
    main()
