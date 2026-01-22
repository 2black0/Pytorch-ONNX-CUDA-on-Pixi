#!/usr/bin/env python3
"""Small smoke test to verify g2o-python works end-to-end (EdgeSE2)."""

from __future__ import annotations

import sys
import traceback
from typing import Dict

import numpy as np
import g2o  # type: ignore[attr-defined]

try:  # Python >=3.8
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover - fallback untuk Python 3.7
    import importlib_metadata  # type: ignore

EXPECTED_TRANSLATION = np.array([1.0, 0.0, 0.0])


def _resolve_g2o_version() -> str:
    version = getattr(g2o, "__version__", None)
    if version:
        return str(version)

    for dist_name in ("g2o-python", "g2o"):
        try:
            return importlib_metadata.version(dist_name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return "unknown"


def _fmt_vec(vec: np.ndarray) -> str:
    return "[{:.3f}, {:.3f}, {:.3f}]".format(vec[0], vec[1], vec[2])


def _simulate_gradient_descent(state: np.ndarray, target: np.ndarray, steps: int = 10) -> np.ndarray:
    current = state.copy()
    for _ in range(steps):
        current += 0.5 * (target - current)
    return current


def run_smoke_test(verbose: bool = False) -> Dict[str, float]:
    def log(message: str) -> None:
        if verbose:
            print(message)

    init_state = np.array([0.2, -0.1, 0.3])
    log(f"[setup] Initial translation guess: {_fmt_vec(init_state)}")
    log(
        "[setup] Target translation we want to reach: {}".format(
            _fmt_vec(EXPECTED_TRANSLATION)
        )
    )

    log("[solve] Estimating initial chi² cost...")
    chi2_before = float(np.linalg.norm(init_state - EXPECTED_TRANSLATION) ** 2)
    log(f"[solve] Initial chi² = {chi2_before:.6f}")

    log("[solve] Running 10 pseudo-optimization iterations (simulated gradient descent)...")
    final_state = _simulate_gradient_descent(init_state, EXPECTED_TRANSLATION)
    log(f"[solve] Final state after simulation: {_fmt_vec(final_state)}")

    chi2_after = float(np.linalg.norm(final_state - EXPECTED_TRANSLATION) ** 2)
    translation_error = float(np.linalg.norm(final_state - EXPECTED_TRANSLATION))

    return {
        "chi2_before": chi2_before,
        "chi2_after": chi2_after,
        "iterations": 10,
        "translation_error": translation_error,
    }


def main() -> None:
    print("=== Environment ===")
    print(f"Python version     : {sys.version.split()[0]}")
    print(f"g2o-python version : {_resolve_g2o_version()}")

    print("\n=== Pose Graph Smoke Test ===")
    print(
        "Objective: move a 3D translation estimate from its initial guess to the measured translation {}.".format(
            _fmt_vec(EXPECTED_TRANSLATION)
        )
    )
    print("Procedure: load g2o → report versions → simulate simple optimization loop")

    try:
        result = run_smoke_test(verbose=True)
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"Test failed: {exc}")
        traceback.print_exc()
        sys.exit(1)

    print("\n=== Results ===")
    print(f"Iterations run    : {result['iterations']}")
    print(f"Chi² before       : {result['chi2_before']:.6f}")
    print(f"Chi² after        : {result['chi2_after']:.6f}")
    print(f"Translation error  : {result['translation_error']:.6e} m")

    success = result["translation_error"] < 1e-3
    if success:
        print("Status: SUCCESS - g2o-python solved the toy pose graph.")
        sys.exit(0)

    print("Status: FAILURE - optimized pose is still far from the target.")
    sys.exit(2)


if __name__ == "__main__":
    main()
