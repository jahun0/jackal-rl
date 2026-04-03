#!/usr/bin/env python3
import sys
import traceback

def run_multiseed_smoke():
    print("=== RLlib Multi-Seed Smoke Test ===")
    try:
        import ray
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
        from ray.tune.registry import register_env
        from jackal_marl_extension_stub import JackalMarlExtensionStub

        print("[*] Initializing local Ray...")
        ray.init(local_mode=True, ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)

        def env_creator(args):
            return ParallelPettingZooEnv(JackalMarlExtensionStub(backend="mock"))

        register_env("jackal_marl_stub", env_creator)

        for seed in [42, 999]:
            print(f"[*] Building minimal PPO config for seed {seed}...")
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
                .debugging(seed=seed, logger_config={"type": "ray.tune.logger.NoopLogger"})
            )
            
            algo = config.build()
            print(f"[+] Algorithm built successfully. Running 1 iteration step for seed {seed}...")
            
            results = algo.train()
            reward_mean = results.get("env_runners", {}).get("episode_reward_mean", results.get("episode_reward_mean", "N/A"))
            print(f"    - Seed {seed} Iteration 1 | Reward mean: {reward_mean}")
            
            algo.stop()

        print(f"[+] SUCCESS: Multi-seed PPO iterations completed cleanly.")
        ray.shutdown()
        return True
    except Exception as e:
        print("[-] FAIL: RLlib multi-seed test encountered an error.")
        traceback.print_exc()
        if 'ray' in sys.modules and ray.is_initialized():
            ray.shutdown()
        return False

if __name__ == "__main__":
    success = run_multiseed_smoke()
    sys.exit(0 if success else 1)
