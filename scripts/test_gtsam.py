#!/usr/bin/env python3
"""Small smoke test to ensure gtsam's Python bindings work properly."""

from __future__ import annotations

import math
import sys
import traceback
from typing import Any, Dict

import numpy as np

try:  # Python >=3.8
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover - fallback for Python 3.7
    import importlib_metadata  # type: ignore

import gtsam  # type: ignore[attr-defined]


EXPECTED_POSE = gtsam.Pose2(1.0, 0.0, 0.0)


def _resolve_gtsam_version() -> str:
    version = getattr(gtsam, "__version__", None)
    if version:
        return str(version)

    for name in ("gtsam", "gtsam-python"):
        try:
            return importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return "unknown"


def _fmt_pose2(pose: gtsam.Pose2) -> str:
    return "x={:.3f}, y={:.3f}, theta(deg)={:.2f}".format(
        pose.x(), pose.y(), math.degrees(pose.theta())
    )


def run_smoke_test(verbose: bool = False) -> Dict[str, Any]:
    def log(msg: str) -> None:
        if verbose:
            print(msg)

    log("[setup] Creating NonlinearFactorGraph and noise models...")
    graph = gtsam.NonlinearFactorGraph()
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6, 1e-6, 1e-6]))
    odom_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([0.01, 0.01, math.radians(1.0)])
    )

    log("[setup] Adding PriorFactorPose2 on key 0 at the world origin...")
    graph.add(gtsam.PriorFactorPose2(0, gtsam.Pose2(0.0, 0.0, 0.0), prior_noise))

    log(
        "[setup] Adding BetweenFactorPose2(0→1) with measured pose {}...".format(
            _fmt_pose2(EXPECTED_POSE)
        )
    )
    graph.add(gtsam.BetweenFactorPose2(0, 1, EXPECTED_POSE, odom_noise))

    log("[setup] Creating initial guess Values for keys 0 and 1...")
    initial = gtsam.Values()
    initial.insert(0, gtsam.Pose2(0.0, 0.0, 0.0))
    initial_pose1 = gtsam.Pose2(0.2, -0.1, math.radians(5.0))
    initial.insert(1, initial_pose1)

    initial_error = graph.error(initial)
    log(f"[solve] Initial graph error = {initial_error:.6f}")

    log("[solve] Running Levenberg-Marquardt optimizer (max 20 iterations)...")
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(20)
    params.setVerbosityLM("SILENT")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    result = optimizer.optimize()

    final_error = graph.error(result)
    log(f"[solve] Final graph error = {final_error:.6f}")

    optimized_pose1 = result.atPose2(1)
    translation_error = math.hypot(
        optimized_pose1.x() - EXPECTED_POSE.x(),
        optimized_pose1.y() - EXPECTED_POSE.y(),
    )
    heading_error = abs(math.degrees(optimized_pose1.theta() - EXPECTED_POSE.theta()))

    return {
        "initial_pose": initial_pose1,
        "optimized_pose": optimized_pose1,
        "initial_error": initial_error,
        "final_error": final_error,
        "translation_error": translation_error,
        "heading_error_deg": heading_error,
    }


def main() -> None:
    print("=== Environment ===")
    print(f"Python version     : {sys.version.split()[0]}")
    print(f"gtsam version      : {_resolve_gtsam_version()}")

    print("\n=== Pose Graph Smoke Test (GTSAM) ===")
    print(
        "Objective: verify gtsam can optimize a 2D pose graph with prior + between factors"
    )
    print(
        "Procedure: build factor graph → add prior/between → provide initial guess → run LM"
    )

    try:
        result = run_smoke_test(verbose=True)
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"Test failed: {exc}")
        traceback.print_exc()
        sys.exit(1)

    print("\n=== Results ===")
    print(f"Initial pose guess : {_fmt_pose2(result['initial_pose'])}")
    print(f"Optimized pose     : {_fmt_pose2(result['optimized_pose'])}")
    print(f"Target pose        : {_fmt_pose2(EXPECTED_POSE)}")
    print(f"Initial graph error: {result['initial_error']:.6f}")
    print(f"Final graph error  : {result['final_error']:.6f}")
    print(f"Translation error  : {result['translation_error']:.6e} m")
    print(f"Heading error      : {result['heading_error_deg']:.6e} deg")

    success = result["translation_error"] < 1e-3 and result["heading_error_deg"] < 1e-2
    if success:
        print("Status: SUCCESS - gtsam optimized the toy problem as expected.")
        sys.exit(0)

    print("Status: FAILURE - optimized pose deviates too far from the target.")
    sys.exit(2)


if __name__ == "__main__":
    main()
