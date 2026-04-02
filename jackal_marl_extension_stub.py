#!/usr/bin/env python3
"""First executable draft: Jackal -> MARL extension stub (Branch B).

Branch B interpretation:
- Actions in the existing env are high-level sensing/costmap toggles (local/cloud/oracle),
  not direct wheel steering commands.
- This stub provides a PettingZoo-style multi-agent facade with step(action_dict),
  keeping the existing Jackal action semantics intact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import random
import numpy as np

# A minimal spaces shim without hard-depending on gymnasium in the stub
class DiscreteStub:
    def __init__(self, n: int):
        self.n = n

class BoxStub:
    def __init__(self, low: float, high: float, shape: Tuple[int, ...], dtype: type = np.float32):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype

UGV_AGENT = "ugv_jackal"
UAV_AGENT = "uav_scout"


@dataclass(frozen=True)
class ActionSemantics:
    """Discrete actions inherited from test_real_env.py semantics."""

    local_costmap: int = 0
    cloud_costmap: int = 1
    oracle_costmap: int = 2


class JackalMarlExtensionStub:
    """Small MARL bridge wrapper around NavigationEnvROS semantics.

    This is intentionally a stub:
    - UGV action remains a map/sensing mode selector.
    - UAV action is a high-level supervisor hint (currently only policy shaping).
    - No low-level velocity control is exposed yet.
    """

    observation_mapping: List[str] = [
        "front_nearest_scan_norm",
        "velocity_norm",
        "temperature",
        "distance_to_goal_norm",
        "angle_to_goal_deg",
        "distance_from_network_norm",
        "information_gain_norm",
        "last_cloud_execute_time_s",
    ]

    # UGV chooses sensing mode; UAV gives lightweight hint (0=hold,1=encourage cloud,2=save cloud)
    ugv_action_space_n: int = 2
    uav_action_space_n: int = 3

    def __init__(self, backend: str = "mock", allow_oracle: bool = False, priority: int = 2):
        self.agents = [UGV_AGENT, UAV_AGENT]
        self.backend = backend
        self.allow_oracle = allow_oracle
        self.priority = priority
        self._t = 0
        self._max_steps = 200
        self._last_state = [1.0, 0.0, 35.0, 10.0, 0.0, 0.0, 0.0, 5.0]
        self._env = None

        if self.backend == "ros":
            # Lazy dependency path to keep this file parseable/runnable in non-ROS setups.
            from test_real_env import NavigationEnvROS  # type: ignore

            self._env = NavigationEnvROS(priority=self.priority)

        self.action_spaces = {
            UGV_AGENT: DiscreteStub(self.ugv_action_space_n),
            UAV_AGENT: DiscreteStub(self.uav_action_space_n),
        }

        # Approximating boundaries based on NavigationEnvROS properties
        self.observation_spaces = {
            UGV_AGENT: BoxStub(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32),
            # UAV gets a mock 2D occupancy grid overhead projection (e.g., 5x5)
            UAV_AGENT: BoxStub(low=0.0, high=1.0, shape=(5, 5), dtype=np.float32),
        }

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, List[float]], Dict[str, dict]]:
        if seed is not None:
            random.seed(seed)

        self._t = 0
        if self.backend == "ros" and self._env is not None:
            state = list(self._env.reset(episode=0))
        else:
            state = self._mock_state_reset()

        self._last_state = state
        obs = self._build_multi_obs(state)
        infos = {a: {"backend": self.backend} for a in self.agents}
        return obs, infos

    def step(self, action_dict: Dict[str, int]):
        """PettingZoo-style step using per-agent actions.

        action_dict keys:
        - 'ugv_jackal': {0,1} (local/cloud sensing mode)
        - 'uav_scout' : {0,1,2} (high-level hint only in this stub)
        """

        self._t += 1
        ugv_action = int(action_dict.get(UGV_AGENT, ActionSemantics.local_costmap))
        uav_hint = int(action_dict.get(UAV_AGENT, 0))

        chosen_action = self._resolve_supervisor_action(ugv_action=ugv_action, uav_hint=uav_hint)

        if self.backend == "ros" and self._env is not None:
            state, cumulative_reward, done, _ = self._env.step(chosen_action)
            state = list(state)
            shared_reward = float(cumulative_reward)
            terminated = bool(done)
        else:
            state, shared_reward, terminated = self._mock_step(chosen_action)

        self._last_state = state
        observations = self._build_multi_obs(state)
        terminations = {a: terminated for a in self.agents}
        truncations = {a: self._t >= self._max_steps for a in self.agents}

        # Keep rewards simple and explicit in this first draft.
        rewards = {
            UGV_AGENT: shared_reward,
            UAV_AGENT: shared_reward * 0.2 - (0.1 if uav_hint == 1 else 0.0),
        }

        infos = {
            UGV_AGENT: {
                "chosen_map_action": chosen_action,
                "raw_ugv_action": ugv_action,
                "raw_uav_hint": uav_hint,
                "action_semantics": "map-toggle/high-level-supervisor",
            },
            UAV_AGENT: {
                "chosen_map_action": chosen_action,
                "observation_projection": "[scan, goal_distance, network_distance, info_gain]",
            },
        }

        return observations, rewards, terminations, truncations, infos

    def _resolve_supervisor_action(self, ugv_action: int, uav_hint: int) -> int:
        if ugv_action not in (0, 1):
            raise ValueError(f"Invalid UGV action: {ugv_action}")
        if uav_hint not in (0, 1, 2):
            raise ValueError(f"Invalid UAV hint: {uav_hint}")

        chosen = ugv_action

        # Supervisor hint rules for first stub.
        if uav_hint == 1 and chosen == 0:
            chosen = 1  # encourage cloud sensing when UGV asks local only

        return chosen

    def _build_multi_obs(self, state: List[float]) -> Dict[str, List[float]]:
        # UGV sees full 8-dim mapping from NavigationEnvROS.state
        ugv_obs = state

        # UAV sees a simulated 2D overhead costmap/occupancy grid (5x5) instead of a state slice.
        # In a real ROS bridge, this would subscribe to a top-down /map or /octomap topic.
        np.random.seed(int(state[7] * 100))  # Deterministic mock based on time/state
        base_grid = np.zeros((5, 5), dtype=np.float32)
        
        # Center roughly represents the UGV; add a mock obstacle trace
        center_x, center_y = 2, 2
        base_grid[center_x, center_y] = 0.1 # UGV
        
        # If goal is close, put a blip in the grid
        dist = state[3]
        if dist < 5.0:
            base_grid[1, 1] = 0.9 # Mock goal or obstacle

        # Add noise to simulate sensor uncertainty
        uav_obs = base_grid + np.random.uniform(0.0, 0.1, (5, 5))
        uav_obs = np.clip(uav_obs, 0.0, 1.0).tolist() # return as nested list for JSON serialization ease

        return {UGV_AGENT: ugv_obs, UAV_AGENT: uav_obs}

    def _mock_state_reset(self) -> List[float]:
        return [1.0, 0.0, 35.0, 10.0, 0.0, 0.0, 0.0, 5.0]

    def _mock_step(self, chosen_action: int) -> Tuple[List[float], float, bool]:
        s = list(self._last_state)

        # Action affects sensing horizon / info gain tradeoff (Branch B semantics).
        if chosen_action == 0:  # local costmap
            s[0] = max(0.1, s[0] - 0.02)
            s[6] = max(0.0, s[6] - 0.01)
            action_penalty = -0.1
        elif chosen_action == 1:  # cloud costmap
            s[0] = min(1.0, s[0] + 0.05)
            s[6] = min(1.0, s[6] + 0.08)
            s[7] = 0.0
            action_penalty = -0.3
        else:  # oracle
            s[0] = 1.0
            s[6] = min(1.0, s[6] + 0.1)
            s[7] = 0.0
            action_penalty = -0.5

        # Goal distance trend (mock): gradually reduce with noise.
        s[3] = max(0.0, s[3] - 0.1 + random.uniform(-0.03, 0.02))
        s[1] = max(0.0, min(1.0, 0.3 + random.uniform(-0.1, 0.1)))
        s[4] = max(-180.0, min(180.0, s[4] + random.uniform(-5.0, 5.0)))
        s[5] = max(0.0, s[5] + random.uniform(0.0, 0.05))
        s[7] = s[7] + 1.0

        reward = (1.0 - s[3] / 10.0) + s[6] + action_penalty
        done = s[3] <= 0.3
        return s, float(reward), bool(done)


if __name__ == "__main__":
    env = JackalMarlExtensionStub(backend="mock")
    obs, infos = env.reset(seed=7)
    print("reset_obs_keys:", list(obs.keys()))
    print("ugv_obs_dim:", len(obs[UGV_AGENT]), "uav_obs_dim:", len(obs[UAV_AGENT]))
    out = env.step({UGV_AGENT: 0, UAV_AGENT: 1})
    print("step_ok:", isinstance(out, tuple), "len:", len(out))
