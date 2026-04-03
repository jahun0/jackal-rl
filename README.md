# jackal-rl
JACKAL Reinforcement Learning for Situational Awareness Level Control using Cloud Robotics

## Local MARL bootstrap (OpenClaw)

For the local Jackal MARL bridge/stub work, use the repo-local virtual environment and requirements file instead of touching global Python.

### Bootstrap

```bash
cd /Users/jh-5/.openclaw/workspace/research/repos/jackal-rl
bash jackal_framework_bootstrap.sh
```

This will:
- create `.venv_marl/` if missing
- install the minimal local packages from `requirements.txt`
- rerun `jackal_framework_ingest_check.py`

### Manual setup

```bash
cd /Users/jh-5/.openclaw/workspace/research/repos/jackal-rl
python3 -m venv .venv_marl
source .venv_marl/bin/activate
pip install -U pip
pip install -r requirements.txt
python3 jackal_framework_ingest_check.py
```

### Current local status

- `pettingzoo` import: OK
- `numpy` import: OK
- `ray` import: OK
- `jackal_framework_ingest_check.py`: runnable inside `.venv_marl`
- next recommended check: `python3 jackal_stub_smoke_test.py`

## ROS2 handoff for Ubuntu real-backend work

The existing real backend script (`test_real_env.py`) is ROS1-era. For Ubuntu execution planning, use the ROS2 handoff artifacts added in this repo:

- `ROS2_MIGRATION_NOTES.md` (ROS1->ROS2/Nav2 mapping + scope)
- `jackal_ros2_preflight.py` (endpoint readiness gate)

Run preflight on Ubuntu after sourcing ROS2 and launching Jackal/Nav2:

```bash
python3 jackal_ros2_preflight.py
```

If you require EKF-filtered odometry specifically:

```bash
python3 jackal_ros2_preflight.py --strict-odometry-filtered
```
