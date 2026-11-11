"""Precision evaluation of a footstep policy on a goal grid (Isaac Lab port of evaluate_mjlab.py, same grid and output).

Envs are assigned {L, R} x dx {10, 15, 20, 25, 30} cm x dy {-10, -5, 0, +5, +10} cm round robin; terminations are
off and every env runs one episode of --duration_s; falls are recorded (pelvis < 0.45 m or tilt > 60 deg).

  python scripts/evaluate.py --checkpoint logs/rsl_rl/g1_goal_conditioned_foot/<run>/model_<it>.pt --num_envs 1000
  python scripts/evaluate.py --zero_policy --num_envs 100          # pipeline check
Output: per foot / dx / dy table and eval_grid.json next to the checkpoint (or --out).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="G1-GoalConditioned-Foot-Flat-Play")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--zero_policy", action="store_true")
parser.add_argument("--num_envs", type=int, default=1000)
parser.add_argument("--duration_s", type=float, default=4.0)
parser.add_argument("--dx", type=float, nargs="+", default=[0.10, 0.15, 0.20, 0.25, 0.30])
parser.add_argument("--dy", type=float, nargs="+", default=[-0.10, -0.05, 0.0, 0.05, 0.10])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
sys.stdout.reconfigure(line_buffering=True)
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embodiments.g1.runner import G1OnPolicyRunner, load_cfgs  # noqa: E402


def run_eval() -> dict:
  env_cfg, agent_cfg = load_cfgs(args.task)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.sim.device = args.device
  env_cfg.seed = args.seed
  env_cfg.episode_length_s = 1.0e6
  env_cfg.terminations.fell_over = None
  env_cfg.terminations.base_too_low = None
  env_cfg.commands.footstep.debug_vis = False
  agent_cfg.device = args.device
  wrapped = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
  env = wrapped.unwrapped
  dev = env.device

  grid = [(f, dx, dy) for f in (0, 1) for dx in args.dx for dy in args.dy]
  assign = [grid[i % len(grid)] for i in range(args.num_envs)]
  term = env.command_manager.get_term("footstep")
  term.fixed_foot = torch.tensor([a[0] for a in assign], device=dev, dtype=torch.long)
  term.fixed_delta = torch.tensor([[a[1], a[2]] for a in assign], device=dev, dtype=torch.float32)

  if args.zero_policy:
    policy = lambda obs: torch.zeros(env.num_envs, 29, device=dev)  # noqa: E731
  else:
    runner = G1OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint, load_optimizer=False)
    policy = runner.get_inference_policy(device=dev)

  obs, _ = wrapped.reset()       # samples the fixed goals, start positions captured on this reset
  robot = env.scene["robot"]
  fell = torch.zeros(args.num_envs, dtype=torch.bool, device=dev)
  with torch.inference_mode():
    for _ in range(int(round(args.duration_s / env.step_dt))):
      obs, _, _, _ = wrapped.step(policy(obs))
      g = robot.data.projected_gravity_b
      tilt = torch.acos(torch.clamp(-g[:, 2], -1.0, 1.0))
      fell |= (robot.data.root_link_pos_w[:, 2] < 0.45) | (tilt > math.radians(60.0))
  _, st = term.swing_stance_pos_w()
  return {
    "foot": [("L", "R")[a[0]] for a in assign],
    "dx": term.delta[:, 0].tolist(), "dy_cmd": [a[2] for a in assign], "dy": term.delta[:, 1].tolist(),
    "touched": term.touched.tolist(), "td_err_cm": (term.td_err * 100).tolist(), "clearance_cm": (term.peak_height * 100).tolist(),
    "second_swing": term.second_swing.tolist(), "stance_lift": term.stance_lift.tolist(),
    "stance_drift_cm": (torch.norm((st - term.st0_w)[:, :2], dim=-1) * 100).tolist(),
    "settled": (term.still_time >= term.cfg.settle_time).tolist(), "fell": fell.tolist(),
    "success": (term.success & ~fell).tolist(),
  }


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
  assert args.zero_policy or args.checkpoint, "--checkpoint or --zero_policy"
  rows = run_eval()
  summ = summarize(rows)
  for k, s in summ.items():
    print(f"{k:16s} n {s['n']:4d}  success {s['success']:.2f}  touched {s['touched']:.2f}  td<3cm {s['td_lt3cm']:.2f}  "
          f"td mean {s['td_err_mean_cm']:5.2f} med {s['td_err_median_cm']:5.2f} cm  clear {s['clearance_cm']:4.1f}  "
          f"drift {s['stance_drift_cm']:4.1f}  2nd {s['second_swing']:.2f}  st-lift {s['stance_lift']:.2f}  "
          f"settled {s['settled']:.2f}  fell {s['fell']:.2f}")
  out = args.out or (os.path.join(os.path.dirname(args.checkpoint), "eval_grid.json") if args.checkpoint else None)
  if out:
    with open(out, "w") as f:
      json.dump({"summary": summ, "episodes": rows, "config": vars(args)}, f, indent=1)
    print("saved", out)


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
