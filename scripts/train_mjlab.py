"""Train a footstep policy with RSL-RL PPO on mjlab.

  python scripts/train_mjlab.py G1-GoalConditioned-Foot-Flat --env.scene.num-envs 4096
  python scripts/train_mjlab.py G1-GoalConditioned-Foot-Flat --env.scene.num-envs 256 --agent.max-iterations 20      # smoke run
  python scripts/train_mjlab.py G1-GoalConditioned-Foot-Flat --resume logs/rsl_rl/g1_goal_conditioned_foot/<run>/model_1000.pt
Any field of the environment / agent config can be overridden with --env.* / --agent.* (tyro).
"""
from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends


@dataclass
class TrainArgs:
  env: ManagerBasedRlEnvCfg
  agent: RslRlOnPolicyRunnerCfg
  device: str = "cuda:0"
  log_root: str = "logs/rsl_rl"
  resume: str | None = None
  """Checkpoint (.pt) to continue training from."""


def train(task: str, args: TrainArgs) -> Path:
  os.environ.setdefault("MUJOCO_GL", "egl")
  configure_torch_backends()
  args.env.seed = args.agent.seed
  stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + (f"_{args.agent.run_name}" if args.agent.run_name else "")
  log_dir = Path(args.log_root) / args.agent.experiment_name / stamp
  print(f"[train] task {task}, device {args.device}, {args.env.scene.num_envs} envs, log {log_dir}", flush=True)
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=args.env, device=args.device), clip_actions=args.agent.clip_actions)
  runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(args.agent), str(log_dir), args.device)
  if args.resume:
    print(f"[train] resuming from {args.resume}", flush=True)
    runner.load(args.resume)
  dump_yaml(log_dir / "params" / "env.yaml", asdict(args.env))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(args.agent))
  runner.learn(num_learning_iterations=args.agent.max_iterations, init_at_random_ep_len=True)
  env.close()
  return log_dir


def main():
  import mjlab.tasks  # noqa: F401
  import embodiments.g1.tasks  # noqa: F401  (registers G1-GoalConditioned-*)

  task, rest = tyro.cli(tyro.extras.literal_type_from_choices(list_tasks()), add_help=False, return_unknown_args=True,
                        config=mjlab.TYRO_FLAGS)
  args = tyro.cli(TrainArgs, args=rest, default=TrainArgs(env=load_env_cfg(task), agent=load_rl_cfg(task)),
                  prog=f"{sys.argv[0]} {task}", config=mjlab.TYRO_FLAGS)
  train(task, args)


if __name__ == "__main__":
  main()
