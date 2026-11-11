"""Precision evaluation of a trained footstep policy on a fixed goal grid.

Every env gets one goal from the grid {L, R} x dx x dy (repeated to fill num_envs; random heading and joint
offsets as in training).  Terminations are disabled and the episode runs a fixed time so the final step state of
every env can be read; falls are detected here (base height < 0.45 m or tilt > 60 deg at any step).

  python scripts/evaluate_mjlab.py G1-GoalConditioned-Foot-Flat --checkpoint-file logs/rsl_rl/g1_goal_conditioned_foot/<run>/model_4999.pt
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import asdict, dataclass

import numpy as np
import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class EvalConfig:
  checkpoint_file: str | None = None
  """Trained model .pt (omit with --zero-policy for a pipeline check)."""
  zero_policy: bool = False
  num_envs: int = 1000
  duration_s: float = 4.0
  dx: tuple[float, ...] = (0.10, 0.15, 0.20, 0.25, 0.30)
  dy: tuple[float, ...] = (-0.10, -0.05, 0.0, 0.05, 0.10)
  device: str | None = None
  seed: int = 0
  out: str | None = None
  """JSON output (default: next to the checkpoint)."""


def run_eval(task_id: str, cfg: EvalConfig) -> dict:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.terminations = {}
  env_cfg.episode_length_s = 1.0e6
  env_cfg.seed = cfg.seed
  for c in env_cfg.commands.values():
    c.debug_vis = False
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  grid = [(f, dx, dy) for f in (0, 1) for dx in cfg.dx for dy in cfg.dy]
  assign = [grid[i % len(grid)] for i in range(cfg.num_envs)]
  term = env.command_manager.get_term("footstep")
  term.fixed_foot = torch.tensor([a[0] for a in assign], device=device, dtype=torch.long)
  term.fixed_delta = torch.tensor([[a[1], a[2]] for a in assign], device=device, dtype=torch.float32)

  if cfg.zero_policy:
    policy = lambda obs: torch.zeros(env.action_space.shape, device=device)  # noqa: E731
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
    runner.load(cfg.checkpoint_file, load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

  obs, _ = wrapped.reset()
  robot = env.scene["robot"]
  fell = torch.zeros(cfg.num_envs, dtype=torch.bool, device=device)
  steps = int(round(cfg.duration_s / env.step_dt))
  with torch.inference_mode():
    for _ in range(steps):
      obs, _, _, _ = wrapped.step(policy(obs))
      g = robot.data.projected_gravity_b
      tilt = torch.acos(torch.clamp(-g[:, 2], -1.0, 1.0))
      fell |= (robot.data.root_link_pos_w[:, 2] < 0.45) | (tilt > math.radians(60.0))
  _, st = term.swing_stance_pos_w()
  rows = {
    "foot": [("L", "R")[a[0]] for a in assign],
    "dx": term.delta[:, 0].tolist(), "dy_cmd": [a[2] for a in assign], "dy": term.delta[:, 1].tolist(),
    "touched": term.touched.tolist(), "td_err_cm": (term.td_err * 100).tolist(), "clearance_cm": (term.peak_height * 100).tolist(),
    "second_swing": term.second_swing.tolist(), "stance_lift": term.stance_lift.tolist(),
    "stance_drift_cm": (torch.norm((st - term.st0_w)[:, :2], dim=-1) * 100).tolist(),
    "settled": (term.still_time >= term.cfg.settle_time).tolist(), "fell": fell.tolist(),
    "success": (term.success & ~fell).tolist(),
  }
  env.close()
  return rows


def summarize(rows: dict) -> dict:
  n = len(rows["foot"])
  arr = {k: np.array(v) for k, v in rows.items()}
  td = np.where(arr["touched"], arr["td_err_cm"], np.nan)

  def group(mask):
    e = td[mask]
    return {"n": int(mask.sum()), "success": float(arr["success"][mask].mean()), "touched": float(arr["touched"][mask].mean()),
            "td_err_mean_cm": float(np.nanmean(e)) if np.isfinite(e).any() else float("nan"),
            "td_err_median_cm": float(np.nanmedian(e)) if np.isfinite(e).any() else float("nan"),
            "td_lt3cm": float(np.mean(np.nan_to_num(e, nan=99) < 3)), "clearance_cm": float(arr["clearance_cm"][mask].mean()),
            "stance_drift_cm": float(arr["stance_drift_cm"][mask].mean()), "second_swing": float(arr["second_swing"][mask].mean()),
            "stance_lift": float(arr["stance_lift"][mask].mean()), "settled": float(arr["settled"][mask].mean()),
            "fell": float(arr["fell"][mask].mean())}
  out = {"all": group(np.ones(n, bool))}
  for f in ("L", "R"):
    out[f"foot {f}"] = group(arr["foot"] == f)
  for dx in sorted(set(np.round(arr["dx"], 3))):
    out[f"dx {dx * 100:.0f} cm"] = group(np.isclose(arr["dx"], dx))
  for dy in sorted(set(arr["dy_cmd"])):
    out[f"dy_cmd {dy * 100:+.0f} cm"] = group(np.isclose(arr["dy_cmd"], dy))
  return out


def main():
  import mjlab.tasks  # noqa: F401
  import embodiments.g1.tasks  # noqa: F401

  task, rest = tyro.cli(tyro.extras.literal_type_from_choices(list_tasks()), add_help=False, return_unknown_args=True,
                        config=mjlab.TYRO_FLAGS)
  cfg = tyro.cli(EvalConfig, args=rest, prog=sys.argv[0] + f" {task}", config=mjlab.TYRO_FLAGS)
  assert cfg.zero_policy or cfg.checkpoint_file, "--checkpoint-file or --zero-policy"
  rows = run_eval(task, cfg)
  summ = summarize(rows)
  for k, s in summ.items():
    print(f"{k:16s} n {s['n']:4d}  success {s['success']:.2f}  touched {s['touched']:.2f}  td<3cm {s['td_lt3cm']:.2f}  "
          f"td mean {s['td_err_mean_cm']:5.2f} med {s['td_err_median_cm']:5.2f} cm  clear {s['clearance_cm']:4.1f}  "
          f"drift {s['stance_drift_cm']:4.1f}  2nd {s['second_swing']:.2f}  st-lift {s['stance_lift']:.2f}  "
          f"settled {s['settled']:.2f}  fell {s['fell']:.2f}")
  out = cfg.out or (os.path.join(os.path.dirname(cfg.checkpoint_file), "eval_grid.json") if cfg.checkpoint_file else "eval_grid.json")
  with open(out, "w") as f:
    json.dump({"summary": summ, "episodes": rows, "config": asdict(cfg)}, f, indent=1)
  print("saved", out)


if __name__ == "__main__":
  main()
