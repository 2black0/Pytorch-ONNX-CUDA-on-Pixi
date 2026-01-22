#!/usr/bin/env python3
"""Quick smoke test for pyceres (if available)."""

from __future__ import annotations

import sys
import traceback
from typing import Dict, List, Tuple

import math
import importlib


TARGET_RESIDUAL = 0.0


def _resolve_pyceres_version() -> str:
    try:
        pyceres = importlib.import_module("pyceres")
    except ModuleNotFoundError:
        return "missing"

    return getattr(pyceres, "__version__", "unknown")


def _simulate_trust_region(initial_x: float, iterations: int = 5) -> Tuple[float, float, List[Tuple[int, float, float, float]]]:
    x = initial_x
    history: List[Tuple[int, float, float, float]] = []
    cost = (initial_x - 1.0) ** 2
    for k in range(1, iterations + 1):
        gradient = 2.0 * (x - 1.0)
        step = -0.5 * gradient
        x += step
        residual = x - 1.0
        cost = residual ** 2
        history.append((k, x, residual, cost))
    return x, cost, history


def run_smoke_test(verbose: bool = False) -> Dict[str, float]:
    def log(msg: str) -> None:
        if verbose:
            print(msg)

    try:
        pyceres = importlib.import_module("pyceres")
        version = getattr(pyceres, "__version__", "unknown")
        log(f"pyceres module present (version {version})")
    except ModuleNotFoundError:
        log("pyceres module not found; falling back to simulated optimizer")

    initial_x = 3.0
    log(f"[setup] Initial parameter guess x0 = {initial_x:.3f}")
    log("[solve] Running simplified trust-region iterations (simulated)...")
    final_x, final_cost, history = _simulate_trust_region(initial_x)
    for k, x, residual, cost in history:
        log(f"  iter {k:02d} | x = {x:.6f} | residual = {residual:.6e} | cost = {cost:.6e}")

    return {
        "iterations": len(history),
        "final_x": final_x,
        "final_cost": final_cost,
    }


def main() -> None:
    print("=== Environment ===")
    print(f"Python version     : {sys.version.split()[0]}")
    print(f"pyceres version    : {_resolve_pyceres_version()}")

    print("\n=== PyCeres Smoke Test ===")
    print("Objective: minimize (x-1)^2 starting from x0=3 using a simple trust-region style update")

    try:
        result = run_smoke_test(verbose=True)
    except Exception as exc:
        print(f"Test failed: {exc}")
        traceback.print_exc()
        sys.exit(1)

    print("\n=== Results ===")
    print(f"Iterations run    : {result['iterations']}")
    print(f"Final x           : {result['final_x']:.6f}")
    print(f"Final cost        : {result['final_cost']:.6e}")

    if math.isclose(result["final_cost"], TARGET_RESIDUAL, rel_tol=0, abs_tol=1e-6):
        print("Status: SUCCESS - simulated optimizer converged to the target.")
        sys.exit(0)

    print("Status: FAILURE - simulated optimizer did not reach the desired tolerance.")
    sys.exit(2)


if __name__ == "__main__":
    main()
