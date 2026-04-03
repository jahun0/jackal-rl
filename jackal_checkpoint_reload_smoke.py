#!/usr/bin/env python3
import sys
import os
import traceback
import shutil

def run_checkpoint_reload_smoke():
    print("=== RLlib Checkpoint Reload Smoke Test ===")
    checkpoint_dir = os.path.abspath("smoke_checkpoint_reload_out")
    
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        
    try:
        import ray
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.rllib.algorithms.algorithm import Algorithm
        from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
        from ray.tune.registry import register_env
        from jackal_marl_extension_stub import JackalMarlExtensionStub

        print("[*] Initializing local Ray...")
        ray.init(local_mode=True, ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)

        def env_creator(args):
            return ParallelPettingZooEnv(JackalMarlExtensionStub(backend="mock"))

        register_env("jackal_marl_stub", env_creator)

        print("[*] Building minimal PPO config...")
        config = (
            PPOConfig()
            .environment("jackal_marl_stub")
            .env_runners(num_env_runners=0)
            .training(train_batch_size=200, minibatch_size=64)
            .multi_agent(
                policies={"ugv_policy", "uav_policy"},
                policy_mapping_fn=lambda agent_id, episode, worker, **kw: (
                    "ugv_policy" if "ugv" in agent_id else "uav_policy"
                ),
            )
            .debugging(logger_config={"type": "ray.tune.logger.NoopLogger"})
        )
        
        algo = config.build()
        print("[+] Algorithm built successfully. Running 1 iteration step...")
        algo.train()
        
        print(f"[*] Saving checkpoint to {checkpoint_dir}...")
        saved_path = algo.save(checkpoint_dir)
        print(f"[+] Checkpoint saved at: {saved_path}")
        algo.stop()

        print("[*] Restoring algorithm from checkpoint...")
        restored_algo = Algorithm.from_checkpoint(saved_path)
        print("[+] Checkpoint restored. Running 1 iteration step to prove functionality...")
        restored_algo.train()
        print("[+] SUCCESS: Restored algorithm completed an iteration without shape or state errors.")
        restored_algo.stop()
        
        ray.shutdown()
        return True

    except Exception as e:
        print("[-] FAIL: RLlib checkpoint reload test encountered an error.")
        traceback.print_exc()
        if 'ray' in sys.modules and ray.is_initialized():
            ray.shutdown()
        return False

if __name__ == "__main__":
    success = run_checkpoint_reload_smoke()
    sys.exit(0 if success else 1)
