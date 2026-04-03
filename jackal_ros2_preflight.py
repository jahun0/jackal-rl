#!/usr/bin/env python3
"""ROS2/Nav2 preflight checks for Ubuntu execution handoff.

Purpose:
- Convert the existing ROS1-only runtime assumptions into an explicit ROS2 gate.
- Give a single command John can run on Ubuntu to validate required endpoints
  before attempting a full real-backend run.

This script is intentionally dependency-light (stdlib only).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, Set


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


def run_ros2_list(noun: str) -> Set[str]:
    proc = subprocess.run(
        ["ros2", noun, "list"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"`ros2 {noun} list` failed (exit={proc.returncode}): {proc.stderr.strip()}"
        )
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def has_any(candidates: Iterable[str], values: Set[str]) -> bool:
    return any(candidate in values for candidate in candidates)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check ROS2/Nav2 endpoint readiness for jackal-rl real backend."
    )
    parser.add_argument(
        "--strict-odometry-filtered",
        action="store_true",
        help="Require /odometry/filtered specifically (otherwise /odom OR /odometry/filtered is accepted).",
    )
    args = parser.parse_args()

    if shutil.which("ros2") is None:
        print("[FAIL] ros2 CLI not found on PATH. Install/source ROS2 (Ubuntu target).")
        return 2

    try:
        topics = run_ros2_list("topic")
        services = run_ros2_list("service")
        actions = run_ros2_list("action")
    except RuntimeError as exc:
        print(f"[FAIL] {exc}")
        return 2

    checks: list[CheckResult] = []

    checks.append(
        CheckResult(
            name="LaserScan topic",
            passed="/scan" in topics,
            detail="requires /scan",
        )
    )
    checks.append(
        CheckResult(
            name="Map topic",
            passed="/map" in topics,
            detail="requires /map",
        )
    )

    if args.strict_odometry_filtered:
        odom_pass = "/odometry/filtered" in topics
        odom_detail = "requires /odometry/filtered"
    else:
        odom_pass = has_any(("/odom", "/odometry/filtered"), topics)
        odom_detail = "requires /odom OR /odometry/filtered"

    checks.append(CheckResult(name="Odometry topic", passed=odom_pass, detail=odom_detail))

    checks.append(
        CheckResult(
            name="Nav2 NavigateToPose action",
            passed="/navigate_to_pose" in actions,
            detail="requires /navigate_to_pose",
        )
    )

    checks.append(
        CheckResult(
            name="Global costmap clear service",
            passed="/global_costmap/clear_entirely_global_costmap" in services,
            detail="requires /global_costmap/clear_entirely_global_costmap",
        )
    )
    checks.append(
        CheckResult(
            name="Local costmap clear service",
            passed="/local_costmap/clear_entirely_local_costmap" in services,
            detail="requires /local_costmap/clear_entirely_local_costmap",
        )
    )

    optional_cmd_vel = "/cmd_vel" in topics

    print("=== jackal-rl ROS2/Nav2 preflight ===")
    for result in checks:
        marker = "PASS" if result.passed else "FAIL"
        print(f"[{marker}] {result.name} ({result.detail})")

    print(
        "[PASS] cmd_vel topic present (optional sanity)" if optional_cmd_vel
        else "[WARN] /cmd_vel not visible (optional, may be remapped by controller)"
    )

    hard_fail = any(not result.passed for result in checks)
    if hard_fail:
        print("\nPRECHECK_RESULT: FAIL")
        print("Bring up Jackal + Nav2 stack on Ubuntu, then rerun this command.")
        return 1

    print("\nPRECHECK_RESULT: PASS")
    print("ROS2 endpoints look ready for an adapter-backed real-environment smoke.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
