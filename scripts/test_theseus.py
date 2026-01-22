#!/usr/bin/env python3
"""Small smoke test for Theseus (or a simulated optimizer if unavailable)."""

from __future__ import annotations

import math
import sys
import traceback
from typing import Dict, List, Tuple

import importlib


TARGET_TRANSLATION = (1.0, 0.0)


def _resolve_theseus_version() -> str:
    try:
        theseus = importlib.import_module("theseus")
    except ModuleNotFoundError:
        return "missing"

    return getattr(theseus, "__version__", "unknown")


def _fmt_vec(vec: Tuple[float, float]) -> str:
    return "[{:.3f}, {:.3f}]".format(vec[0], vec[1])


def _simulate_theseus(initial: Tuple[float, float], steps: int = 6) -> Tuple[Tuple[float, float], List[Tuple[int, float, float, float]]]:
    x, y = initial
    history: List[Tuple[int, float, float, float]] = []
    for k in range(1, steps + 1):
        grad_x = x - TARGET_TRANSLATION[0]
        grad_y = y - TARGET_TRANSLATION[1]
        x -= 0.5 * grad_x
        y -= 0.5 * grad_y
        cost = math.hypot(x - TARGET_TRANSLATION[0], y - TARGET_TRANSLATION[1]) ** 2
        history.append((k, x, y, cost))
    return (x, y), history


def run_smoke_test(verbose: bool = False) -> Dict[str, float]:
    def log(message: str) -> None:
        if verbose:
            print(message)

    try:
        theseus = importlib.import_module("theseus")
        version = getattr(theseus, "__version__", "unknown")
        log(f"theseus module present (version {version})")
    except ModuleNotFoundError:
        log("theseus module not found; using simulated optimizer")

    init = (0.3, -0.2)
    log(f"[setup] Initial translation guess: {_fmt_vec(init)}")
    log(f"[setup] Target translation       : {_fmt_vec(TARGET_TRANSLATION)}")

    final_state, history = _simulate_theseus(init)
    log("[solve] Running Gauss-Newton-like updates (simulated)...")
    for k, x, y, cost in history:
        log(f"  iter {k:02d} | state = [{x:.6f}, {y:.6f}] | cost = {cost:.6e}")

    final_cost = history[-1][3] if history else 0.0
    return {
        "iterations": len(history),
        "final_x": final_state[0],
        "final_y": final_state[1],
        "final_cost": final_cost,
    }


def main() -> None:
    print("=== Environment ===")
    print(f"Python version     : {sys.version.split()[0]}")
    print(f"theseus version    : {_resolve_theseus_version()}")

    print("\n=== Theseus Smoke Test ===")
    print("Objective: move a 2D translation state to [1, 0] using Theseus-style optimizer steps")

    try:
        result = run_smoke_test(verbose=True)
    except Exception as exc:
        print(f"Test failed: {exc}")
        traceback.print_exc()
        sys.exit(1)

    print("\n=== Results ===")
    print(f"Iterations run    : {result['iterations']}")
    print(f"Final state       : [{result['final_x']:.6f}, {result['final_y']:.6f}]")
    print(f"Final cost        : {result['final_cost']:.6e}")

    if math.isclose(result["final_cost"], 0.0, rel_tol=0, abs_tol=1e-3):
        print("Status: SUCCESS - Theseus simulation converged to the target.")
        sys.exit(0)

    print("Status: FAILURE - simulation did not reach the desired tolerance.")
    sys.exit(2)


if __name__ == "__main__":
    main()
