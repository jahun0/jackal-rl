#!/usr/bin/env python3
import sys
import numpy as np
from jackal_marl_extension_stub import JackalMarlExtensionStub, UGV_AGENT, UAV_AGENT

def run_smoke_test():
    print("=== Jackal MARL Extension Stub Smoke Test ===")
    try:
        env = JackalMarlExtensionStub(backend="mock")
        print("[+] Environment instantiated.")

        # Test Reset
        obs, infos = env.reset(seed=42)
        print("[+] Reset successful.")
        print(f"    - UGV Obs Shape: {np.shape(obs[UGV_AGENT])}")
        print(f"    - UAV Obs Shape: {np.shape(obs[UAV_AGENT])}")

        # Test Spaces
        print(f"    - UGV Action Space: {env.action_spaces[UGV_AGENT].n}")
        print(f"    - UAV Action Space: {env.action_spaces[UAV_AGENT].n}")

        # Test Step
        action_dict = {UGV_AGENT: 0, UAV_AGENT: 1}
        obs, rewards, terms, truncs, infos = env.step(action_dict)
        print(f"[+] Step successful with actions: {action_dict}")
        print(f"    - Rewards: {rewards}")
        print(f"    - Terminations: {terms}")
        print(f"    - Truncations: {truncs}")
        print(f"    - UGV Info: {infos[UGV_AGENT]}")
        
        print("=== Smoke Test PASS ===")
        return True
    except Exception as e:
        print(f"=== Smoke Test FAIL ===")
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
