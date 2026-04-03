# ROS2 migration handoff notes (Ubuntu target)

This repo's `test_real_env.py` is ROS1-first (`rospy`, `move_base`, `dynamic_reconfigure`) and currently fails on macOS where ROS1 is absent.

## Why this note exists

The lane should not stop at "`rospy` missing on Mac".
This document captures a minimal ROS2-forward execution contract so Ubuntu execution can proceed with less ambiguity.

Reference blocker (kept unchanged):
- `../../ideas/jackal_real_backend_smoke_status.md`

## ROS1 -> ROS2/Nav2 endpoint mapping

| Current ROS1 assumption in `test_real_env.py` | ROS2/Nav2-oriented target |
|---|---|
| `rospy` node + ROS1 subscribers/publishers | `rclpy` node + ROS2 subscriptions/publishers |
| `move_base` action server (`MoveBaseAction`) | Nav2 `NavigateToPose` action (`/navigate_to_pose`) |
| `/move_base/clear_costmaps` service | `/global_costmap/clear_entirely_global_costmap` and `/local_costmap/clear_entirely_local_costmap` |
| `/odometry/filtered` topic (hard-coded) | prefer `/odom` OR `/odometry/filtered` (platform dependent) |
| `/scan`, `/map`, `/cmd_vel` | keep equivalent topics in ROS2 (allow remaps where needed) |

## Minimal ROS2 readiness gate added in this pass

New script:
- `jackal_ros2_preflight.py`

What it checks via `ros2 ... list`:
1. topics: `/scan`, `/map`, and odometry (`/odom` or `/odometry/filtered`)
2. action: `/navigate_to_pose`
3. services: both Nav2 costmap clear services

Run on Ubuntu after sourcing ROS2 and launching Jackal/Nav2 stack:

```bash
python3 jackal_ros2_preflight.py
# optional strict odom requirement
python3 jackal_ros2_preflight.py --strict-odometry-filtered
```

## Honest scope boundary for this commit

- No claim of full ROS2 runtime migration of `test_real_env.py` yet.
- No local Ubuntu/ROS execution attempted from this Mac.
- This pass only lands ROS2-facing assumptions + a runnable preflight gate to de-risk next Ubuntu run.
