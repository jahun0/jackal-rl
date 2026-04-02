#!/usr/bin/env python3
import sys
from jackal_marl_extension_stub import JackalMarlExtensionStub

def check_duck_typing(env):
    print("[*] Checking PettingZoo ParallelEnv duck-typing...")
    required_attrs = ['agents', 'observation_spaces', 'action_spaces', 'reset', 'step']
    missing = [attr for attr in required_attrs if not hasattr(env, attr)]
    if missing:
        print(f"[!] Duck-typing failed. Missing: {missing}")
        return False
    print("[+] Duck-typing passed. Basic API signature is compliant.")
    return True

def check_pettingzoo_import():
    print("[*] Attempting pettingzoo import...")
    try:
        import pettingzoo
        from pettingzoo.utils.env import ParallelEnv
        print("[+] pettingzoo is installed locally.")
        return "READY"
    except ImportError:
        print("[-] pettingzoo not found locally (expected for lightweight testing).")
        return "LOCAL_LIB_MISSING"

def check_rllib_import():
    print("[*] Attempting ray.rllib import...")
    try:
        import ray
        from ray.rllib.env.multi_agent_env import MultiAgentEnv
        print("[+] ray.rllib is installed locally.")
        return "READY"
    except ImportError:
        print("[-] ray.rllib not found locally (expected).")
        return "LOCAL_LIB_MISSING"

if __name__ == "__main__":
    print("=== Jackal MARL Framework Ingest Check ===")
    env = JackalMarlExtensionStub(backend="mock")
    
    duck_ok = check_duck_typing(env)
    pz_res = check_pettingzoo_import()
    rl_res = check_rllib_import()

    print("\n=== Final Status ===")
    if not duck_ok:
        print("Result: BLOCKED")
        sys.exit(1)
    elif pz_res == "LOCAL_LIB_MISSING" and rl_res == "LOCAL_LIB_MISSING":
        print("Result: LOCAL_LIB_MISSING")
        sys.exit(0)
    else:
        print("Result: INGEST_READY")
        sys.exit(0)
