#!/usr/bin/env python3
import sys
import traceback
from jackal_marl_extension_stub import JackalMarlExtensionStub

def run_api_test():
    print("=== PettingZoo API Compliance Test ===")
    try:
        from pettingzoo.test import parallel_api_test
    except ImportError:
        print("[-] FAIL: pettingzoo.test module not found. Run pip install pettingzoo.")
        return False

    env = JackalMarlExtensionStub(backend="mock")
    
    try:
        print("[*] Running parallel_api_test...")
        parallel_api_test(env, num_cycles=10)
        print("[+] SUCCESS: PettingZoo parallel_api_test passed.")
        return True
    except Exception as e:
        print("[-] FAIL: PettingZoo parallel_api_test failed.")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_api_test()
    sys.exit(0 if success else 1)
