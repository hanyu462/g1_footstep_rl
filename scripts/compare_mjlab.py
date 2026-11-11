"""Compare the Isaac Lab footstep rewards / terminations with the mjlab version (run in the mjlab venv, .venv).

  1  formulas: every footstep reward function of foot/mdp/rewards.py (Isaac Lab) and foot/mdp_mjlab/rewards.py
     evaluated on the same synthetic command / robot states (random positions, contacts, flags, velocities);
     they must agree exactly.  Weights, parameters and terminations of the two env configs are compared by name.
  2  zero-action rollout of the mjlab env (training config, N envs, until each env's first termination), the same
     statistics as `scripts/inspect_task.py --part free --stats isaac.json`, printed side by side when --isaac is
     given.  The two simulators and robot models differ (MuJoCo vs PhysX, 33.3 vs 34.4 kg, hip motors), so the
     rollout numbers should be similar, not equal.

  .venv/bin/python scripts/compare_mjlab.py [--isaac isaac_stats.json] [--num-envs 256]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from types import SimpleNamespace

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
FOOT = os.path.join(REPO, "embodiments", "g1", "tasks", "goal_conditioned", "foot")

parser = argparse.ArgumentParser()
parser.add_argument("--isaac", type=str, default=None, help="stats json from scripts/inspect_task.py --part free --stats")
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--seconds", type=float, default=3.0)
args = parser.parse_args()

FAILS = [0]


def check(name, ok, msg=""):
  FAILS[0] += not ok
  print(f"   {'ok' if ok else 'FAIL':4s} {name:34s} {msg}")


def load(path, name):
  spec = importlib.util.spec_from_file_location(name, path)
  m = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(m)
  return m


# ---- 1. formulas ---------------------------------------------------------------------------------------------
def fake_state(n: int, g: torch.Generator):
  r = lambda *s: torch.rand(*s, generator=g)  # noqa: E731
  sw, st = (r(n, 3) - 0.5) * 0.6, (r(n, 3) - 0.5) * 0.6
  contact = r(n, 2) > 0.5
  t = r(n) * 4.0
  vel = (r(n, 2, 3) - 0.5) * 2.0
  common = dict(
    target_w=(r(n, 3) - 0.5) * 0.6, sw0_w=(r(n, 3) - 0.5) * 0.1, st0_w=(r(n, 3) - 0.5) * 0.1,
    touched=r(n) > 0.5, td_err=r(n) * 0.2, td_time=r(n) * 2.0,
    sw_air_steps=(r(n) * 4).long(), st_air_steps=(r(n) * 3).long(),
    swing_stance_pos_w=lambda: (sw, st), feet_contact=lambda: contact, time_s=lambda: t,
  )
  root = dict(root_link_lin_vel_w=(r(n, 3) - 0.5), root_link_ang_vel_w=(r(n, 3) - 0.5) * 2)
  # mjlab: foot site velocity; Isaac: link velocity + w x r with w = 0 (so both are the same point velocity)
  mj = SimpleNamespace(**common, site_ids=[0, 1], robot=SimpleNamespace(data=SimpleNamespace(site_lin_vel_w=vel, **root)))
  quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(n, 2, 4)
  il = SimpleNamespace(**common, feet_cfg=SimpleNamespace(body_ids=[0, 1]), robot=SimpleNamespace(data=SimpleNamespace(
    body_link_quat_w=quat, body_link_lin_vel_w=vel, body_link_ang_vel_w=torch.zeros(n, 2, 3), **root)))
  joints = SimpleNamespace(data=SimpleNamespace(joint_pos=r(n, 29), default_joint_pos=r(n, 29)))
  env = lambda c: SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: c), scene={"robot": joints})  # noqa: E731
  return env(mj), env(il)


def compare_formulas():
  print("\n== 1. reward formulas on identical states (8192 random states)")
  il = load(os.path.join(FOOT, "mdp", "rewards.py"), "isaac_rewards")
  from embodiments.g1.tasks.goal_conditioned.foot.mdp_mjlab import rewards as mj
  env_mj, env_il = fake_state(8192, torch.Generator().manual_seed(0))
  fns = ["foot_target_tracking", "touchdown_accuracy", "swing_clearance", "second_swing", "stance_foot_lift",
         "stance_foot_drift", "foot_slip", "settle", "double_support_after_touchdown"]
  for f in fns:
    a, b = getattr(mj, f)(env_mj, "footstep"), getattr(il, f)(env_il, "footstep")
    check(f, a.shape == b.shape and torch.allclose(a, b, atol=1e-6), f"max |diff| {float((a - b).abs().max()):.1e}, mean {float(a.mean()):.4f}")
  cfg = SimpleNamespace(name="robot", joint_ids=list(range(29)))
  a, b = mj.joint_posture(env_mj, cfg), il.joint_posture(env_il, cfg)
  check("joint_posture", torch.allclose(a, b), f"max |diff| {float((a - b).abs().max()):.1e}")


def compare_configs(il: dict | None):
  print("\n== config: reward weights / parameters and terminations by term (Isaac Lab values from the --isaac json)")
  if il is None:
    print("   (skipped: no --isaac json)")
    return
  from embodiments.g1.tasks.goal_conditioned.foot.env_cfgs_mjlab import g1_footstep_env_cfg
  mj = g1_footstep_env_cfg()
  num = lambda p: {k: float(v) for k, v in p.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}  # noqa: E731
  ir, it = il["config"]["rewards"], il["config"]["terminations"]
  bad = [n for n, t in mj.rewards.items() if n not in ir or ir[n]["weight"] != t.weight]
  check("reward terms and weights", not bad and len(ir) == len(mj.rewards), f"{len(mj.rewards)} mjlab / {len(ir)} Isaac Lab {bad}")
  bad = [n for n, t in mj.rewards.items() if n in ir and num(t.params) != ir[n]["params"]]
  check("reward numeric parameters", not bad, f"{bad}" if bad else "k, target_height, delay, force_threshold")
  bad = [n for n, t in mj.terminations.items() if n not in it or num(t.params) != it[n]]
  check("terminations and parameters", not bad and len(it) == len(mj.terminations),
        ", ".join(f"{n} {it[n]}" for n in it))
  check("step dt and episode length", abs(il["config"]["step_dt"] - mj.decimation * mj.sim.mujoco.timestep) < 1e-9
        and il["config"]["episode_length_s"] == mj.episode_length_s, f"{il['config']['step_dt']} s, {mj.episode_length_s} s")


# ---- 2. rollout --------------------------------------------------------------------------------------------
def rollout_mjlab() -> dict:
  import mjlab.tasks  # noqa: F401
  import embodiments.g1.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg
  cfg = load_env_cfg("G1-GoalConditioned-Foot-Flat")
  cfg.scene.num_envs = args.num_envs
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  rm, tm = env.reward_manager, env.termination_manager
  names = list(rm._term_names)
  w = {n: rm._term_cfgs[i].weight for i, n in enumerate(names)}
  env.reset()
  N = env.num_envs
  zero = torch.zeros(N, env.action_manager.total_action_dim, device=env.device)
  first_end = torch.full((N,), -1.0, device=env.device)
  stats = {"steps": 0, "sum": dict.fromkeys(names, 0.0), "reasons": dict.fromkeys(tm.active_terms, 0)}
  for k in range(int(args.seconds / env.step_dt)):
    _, _, term, trunc, _ = env.step(zero)
    alive = first_end < 0
    if k == 0:   # the first step after env.reset() is reported separately
      stats["first_step"] = {n: float(rm._step_reward[:, i].mean()) for i, n in enumerate(names)}
    else:
      for i, n in enumerate(names):
        stats["sum"][n] += float((rm._step_reward[:, i])[alive].sum())
      stats["steps"] += int(alive.sum())
    done = (term | trunc) & alive
    for n in tm.active_terms:
      stats["reasons"][n] += int(tm.get_term(n)[done].sum())
    first_end = torch.where(done, (k + 1) * env.step_dt, first_end)
    if (first_end >= 0).all():
      break
  stats["first_end_s"] = first_end[first_end > 0].tolist()
  stats["mean_per_step"] = {n: s / max(stats["steps"], 1) for n, s in stats["sum"].items()}
  stats["weights"] = w
  env.close()
  return stats


def print_side_by_side(mj: dict, il: dict | None):
  print(f"\n== 2. zero-action rollout until the first termination ({args.num_envs} mjlab envs)")
  def fe(s):
    t = torch.tensor(s["first_end_s"])
    return f"{float(t.mean()):.2f} s (min {float(t.min()):.2f}, max {float(t.max()):.2f}, n {len(t)})"
  print(f"   first termination   mjlab {fe(mj)}" + (f"   isaac {fe(il)}" if il else ""))
  print(f"   reasons             mjlab {mj['reasons']}" + (f"   isaac {il['reasons']}" if il else ""))
  print(f"   {'term (raw x weight)':20s} {'step 2.. (mean per alive step)':>32s} {'first step after env.reset()':>32s}")
  print(f"   {'':20s} {'mjlab':>16s}{'isaac':>16s} {'mjlab':>16s}{'isaac':>16s}")
  for n, v in mj["mean_per_step"].items():
    o, f0 = (il["mean_per_step"].get(n), il["first_step"].get(n)) if il else (float("nan"), float("nan"))
    print(f"   {n:20s} {v:16.4f}{o:16.4f} {mj['first_step'][n]:16.4f}{f0:16.4f}")
  print("   (rollouts until each env's first termination; no balance control, so both fall)")


if __name__ == "__main__":
  il = json.load(open(args.isaac)) if args.isaac else None
  compare_formulas()
  compare_configs(il)
  mj = rollout_mjlab()
  print_side_by_side(mj, il)
  print(f"\n== {'all formula / config checks passed' if FAILS[0] == 0 else f'{FAILS[0]} checks failed'}")
