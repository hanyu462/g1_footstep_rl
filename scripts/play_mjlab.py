"""Watch a footstep policy in the MuJoCo viewer (native window if a display exists, else the viser web viewer).

  python scripts/play_mjlab.py G1-GoalConditioned-Foot-Flat --checkpoint-file logs/rsl_rl/g1_goal_conditioned_foot/<run>/model_4999.pt
  python scripts/play_mjlab.py G1-GoalConditioned-Foot-Flat --zero-policy True                    # environment only, zero action
"""
from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from typing import Literal

import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class PlayArgs:
  checkpoint_file: str | None = None
  zero_policy: bool = False
  num_envs: int = 16
  device: str = "cuda:0"
  viewer: Literal["auto", "native", "viser"] = "auto"


def play(task: str, args: PlayArgs) -> None:
  configure_torch_backends()
  env_cfg, agent_cfg = load_env_cfg(task, play=True), load_rl_cfg(task)
  env_cfg.scene.num_envs = args.num_envs
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=args.device), clip_actions=agent_cfg.clip_actions)
  if args.zero_policy:
    shape = env.unwrapped.action_space.shape
    policy = lambda obs: torch.zeros(shape, device=args.device)  # noqa: E731
  else:
    assert args.checkpoint_file, "--checkpoint-file or --zero-policy"
    runner = (load_runner_cls(task) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device=args.device)
    runner.load(args.checkpoint_file, load_cfg={"actor": True}, strict=True, map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)
  viewer = args.viewer
  if viewer == "auto":
    viewer = "native" if (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) else "viser"
  (NativeMujocoViewer if viewer == "native" else ViserPlayViewer)(env, policy).run()
  env.close()


def main():
  import mjlab.tasks  # noqa: F401
  import embodiments.g1.tasks  # noqa: F401

  task, rest = tyro.cli(tyro.extras.literal_type_from_choices(list_tasks()), add_help=False, return_unknown_args=True,
                        config=mjlab.TYRO_FLAGS)
  play(task, tyro.cli(PlayArgs, args=rest, prog=f"{sys.argv[0]} {task}", config=mjlab.TYRO_FLAGS))


if __name__ == "__main__":
  main()
