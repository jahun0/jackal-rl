#!/usr/bin/env python3
import sys
import traceback

def run_smoke():
    print("=== RLlib Smoke Test ===")
    try:
        print("[*] Importing ray.rllib and pettingzoo...")
        import ray
        from ray.rllib.env.multi_agent_env import MultiAgentEnv
        
        print("[*] Importing Jackal stub...")
        from jackal_marl_extension_stub import JackalMarlExtensionStub
        
        print("[*] Instantiating mock environment...")
        env = JackalMarlExtensionStub(backend="mock")
        
        print("[*] Checking attributes for RLlib/PettingZoo translation...")
        assert hasattr(env, 'reset'), "Missing reset()"
        assert hasattr(env, 'step'), "Missing step()"
        assert hasattr(env, 'observation_spaces'), "Missing observation_spaces"
        assert hasattr(env, 'action_spaces'), "Missing action_spaces"
        assert hasattr(env, 'agents'), "Missing agents list"
        
        print("[+] SUCCESS: Environment exposes required base attributes for RLlib MultiAgentEnv wrapping.")
        return True
    except Exception as e:
        print("[-] FAIL: RLlib smoke test encountered an error.")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_smoke()
    sys.exit(0 if success else 1)
