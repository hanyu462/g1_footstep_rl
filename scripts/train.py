"""Train a footstep policy with RSL-RL PPO on Isaac Lab (headless).

  python scripts/train.py --num_envs 4096                                    # full run (G1-GoalConditioned-Foot-Flat)
  python scripts/train.py --num_envs 256 --max_iterations 20 --run_name smoke  # smoke run
  python scripts/train.py --resume logs/rsl_rl/g1_goal_conditioned_foot/<run>/model_1000.pt
  python scripts/train.py --device cuda:1 ...

Output in logs/rsl_rl/<experiment>/<date_time>[_<run_name>]/: model_<it>.pt (deploy spec inside, infos["deploy"]),
deploy.json, params/env.yaml, params/agent.yaml, TensorBoard events (tensorboard --logdir logs/rsl_rl).
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="G1-GoalConditioned-Foot-Flat")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--save_interval", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--run_name", type=str, default="")
parser.add_argument("--resume", type=str, default=None, help="checkpoint (.pt) to continue from (same task / spec)")
parser.add_argument("--logger", choices=("tensorboard", "wandb"), default=None)
parser.add_argument("--log_root", default="logs/rsl_rl")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
sys.stdout.reconfigure(line_buffering=True)
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embodiments.g1.runner import G1OnPolicyRunner, load_cfgs  # noqa: E402


def main():
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True
  env_cfg, agent_cfg = load_cfgs(args.task)
  if args.num_envs is not None:
    env_cfg.scene.num_envs = args.num_envs
  for k in ("max_iterations", "save_interval", "seed", "logger"):
    if getattr(args, k) is not None:
      setattr(agent_cfg, k, getattr(args, k))
  agent_cfg.run_name = args.run_name
  agent_cfg.device = args.device
  env_cfg.seed = agent_cfg.seed
  env_cfg.sim.device = args.device
  env_cfg.commands.footstep.debug_vis = False

  stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + (f"_{args.run_name}" if args.run_name else "")
  log_dir = os.path.abspath(os.path.join(args.log_root, agent_cfg.experiment_name, stamp))
  print(f"[train] {args.task}: {env_cfg.scene.num_envs} envs, {agent_cfg.max_iterations} iterations, {args.device}, log {log_dir}")

  env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
  runner = G1OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
  if args.resume:
    print(f"[train] resuming from {args.resume}")
    runner.load(args.resume)
  dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
  dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
  runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
  print(f"[train] done: {log_dir}")


if __name__ == "__main__":
  # SimulationApp.close() can hang for minutes after a headless run (also after an exception): always exit directly
  try:
    main()
    code = 0
  except BaseException:  # noqa: BLE001
    import traceback
    traceback.print_exc()
    code = 1
  sys.stdout.flush()
  os._exit(code)
