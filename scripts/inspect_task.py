"""Check the Isaac Lab footstep task env (foot/env_cfgs.py + embodiments/g1/env.py): reset kinematics, rewards,
terminations.  Two parts, each in its own process (one Isaac Sim stage per process):

  free    the task env as trained (32 envs, zero action, no balance control -> the robots fall):
            reset   link poses on the reset step: stale before the refresh, correct after it, no physics step,
                    first observation goal correct, for the first reset, terminations and time-outs
            terms   every reward term (N,), finite, reward = sum(raw x weight x dt), raw values vs independent formulas
            ends    fell_over / base_too_low terminations (is_terminated reward), forced time-outs (no penalty)
  pinned  the same env with the pelvis fixed (soles on the ground), scripted steps in four groups (normal, second
          swing, stance lift, far target): landing, target deviation, stance stability and the step terms, success
          flag vs reward

  python scripts/inspect_task.py [--part free|pinned|all] [--stats out.json] [--json out.json]

--stats writes per-term mean rewards and termination statistics of the zero-action free rollout (compared with the
mjlab env by scripts/compare_mjlab.py).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--part", choices=("free", "pinned", "all"), default="all")
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--stats", type=str, default=None)
parser.add_argument("--json", type=str, default=None)
args, rest = parser.parse_known_args()

if args.part == "all":
  ok = True
  for part in ("free", "pinned"):
    cmd = [sys.executable, __file__, "--part", part, "--num-envs", str(args.num_envs)] + rest
    if args.stats and part == "free":
      cmd += ["--stats", args.stats]
    if args.json:
      cmd += ["--json", args.json.replace(".json", f"_{part}.json")]
    ok &= subprocess.run(cmd).returncode == 0
  sys.exit(0 if ok else 1)

from isaaclab.app import AppLauncher  # noqa: E402

app_parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(app_parser)
app_args = app_parser.parse_args(rest)
app_args.headless = True
sys.stdout.reconfigure(line_buffering=True)
app = AppLauncher(app_args).app

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embodiments.g1 import env as G  # noqa: E402
from embodiments.g1 import params as P  # noqa: E402
from embodiments.g1 import state as S  # noqa: E402
from embodiments.g1.tasks.goal_conditioned.foot import mdp  # noqa: E402
from embodiments.g1.tasks.goal_conditioned.foot.env_cfgs import CMD, UPPER_BODY, G1FootstepEnvCfg  # noqa: E402

LIFT = {"hip_pitch_joint": -0.8, "knee_joint": 1.2}
RES, FAILS = {}, [0]


def check(name, ok, msg=""):
  FAILS[0] += not ok
  RES[name] = {"ok": bool(ok), "msg": msg}
  print(f"   {'ok' if ok else 'FAIL':4s} {name:38s} {msg}")


class ProbeEnv(G.G1RLEnv):
  """G1RLEnv that records, for every reset, the PhysX link poses before / after the kinematics refresh."""

  def _reset_idx(self, env_ids):
    ManagerBasedRLEnv._reset_idx(self, env_ids)
    robot = self.scene[S.ROBOT]
    view = robot.root_physx_view
    ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
    before = view.get_link_transforms().clone()[ids]
    q_before, steps = view.get_dof_positions().clone()[ids], self._sim_step_counter
    G.refresh_kinematics(self)
    after = view.get_link_transforms().clone()[ids]
    root = robot.data.root_link_pose_w[ids]
    self.reset_log.append({
      "ids": ids,
      "stale_m": (before[..., :3] - after[..., :3]).norm(dim=-1).amax(-1),  # worst link, before vs after
      "pelvis_err_m": (after[:, 0, :3] - (root[:, :3])).norm(dim=-1),       # after vs written root pose
      "q_changed": float((view.get_dof_positions()[ids] - q_before).abs().max()),
      "physics_steps": self._sim_step_counter - steps,
    })
    for term in self.command_manager._terms.values():
      if hasattr(term, "post_reset"):
        term.post_reset(env_ids)


def make_env(pinned: bool) -> ProbeEnv:
  cfg = G1FootstepEnvCfg()
  cfg.scene.num_envs = args.num_envs
  cfg.sim.device = app_args.device
  cfg.commands.footstep.debug_vis = False
  if pinned:
    r = cfg.scene.robot
    cfg.scene.robot = r.replace(spawn=r.spawn.replace(articulation_props=r.spawn.articulation_props.replace(fix_root_link=True)))
  ProbeEnv.reset_log = []
  return ProbeEnv(cfg)


def goal_consistent(env, cmd, obs, ids) -> tuple[bool, float]:
  """On the step where ids were (re)set: target - swing (heading frame) == delta, target-relative entries non-zero."""
  g = obs["goal"][ids, 0:8]
  err = float((g[:, 2:4] - cmd.delta[ids]).abs().max())
  return bool(err < 1e-4 and (g[:, 2:6].abs().sum(-1) > 0).all() and not cmd.need_init[ids].any()), err


def raw_terms(env) -> dict[str, torch.Tensor]:
  rm = env.reward_manager
  return {n: rm._step_reward[:, i] / rm._term_cfgs[i].weight for i, n in enumerate(rm.active_terms)}


# ---- free ----------------------------------------------------------------------------------------------------
def part_free():
  env = make_env(pinned=False)
  robot, cmd, rm, tm = env.scene[S.ROBOT], env.command_manager.get_term(CMD), env.reward_manager, env.termination_manager
  N, dev = env.num_envs, env.device
  zero = torch.zeros(N, 29, device=dev)
  feet = S.FEET_BODIES.replace()
  feet.resolve(env.scene)

  print("\n== reset: link poses and the first goal")
  obs, _ = env.reset()
  log = env.reset_log[-1]
  ok, err = goal_consistent(env, cmd, obs, torch.arange(N, device=dev))
  check("first reset: goal valid at obs 0", ok, f"target - swing vs delta max {err:.1e}; target-relative entries non-zero")
  sole0 = S.feet_sole_pos_w(env, feet).clone()
  check("first reset: sw0 = sole at reset", torch.allclose(cmd.sw0_w, cmd.swing_stance_pos_w()[0]), "")
  check("refresh: no physics step, q unchanged", log["physics_steps"] == 0 and log["q_changed"] == 0.0,
        f"steps {log['physics_steps']}, max |dq| {log['q_changed']:.1e}")
  check("refresh: pelvis = written root pose", float(log["pelvis_err_m"].max()) < 1e-5, f"{float(log['pelvis_err_m'].max()):.1e} m")
  env.step(zero)
  d1 = float((S.feet_sole_pos_w(env, feet) - sole0).norm(dim=-1).max())
  check("first reset: soles vs one step later", d1 < 0.01, f"max {d1 * 1000:.1f} mm in 20 ms")

  # zero action until every env has terminated at least once (no balance control); then force time-outs
  first_end = torch.full((N,), -1.0, device=dev)
  term_rows, mid_ok, mid_err, stale, next_d, n_mid = [], True, 0.0, [], [], 0
  for k in range(int(3.0 / env.step_dt)):
    nlog = len(env.reset_log)
    obs, rew, term, trunc, extras = env.step(zero)
    raw = raw_terms(env)
    done = term | trunc
    if done.any():
      ids = done.nonzero().flatten()
      first_end = torch.where(done & (first_end < 0), (k + 2) * env.step_dt, first_end)
      term_rows.append((ids, raw["terminated"][ids].clone(), term[ids].clone()))
      ok, e = goal_consistent(env, cmd, obs, ids)
      mid_ok &= ok
      mid_err = max(mid_err, e)
      n_mid += len(ids)
      stale.append(env.reset_log[-1]["stale_m"])
      sole_r = S.feet_sole_pos_w(env, feet)[ids].clone()
      env.step(zero)
      next_d.append((S.feet_sole_pos_w(env, feet)[ids] - sole_r).norm(dim=-1).max())
      if (first_end >= 0).all():
        break
  stale = torch.cat(stale)
  check("termination resets: goal valid at reset obs", mid_ok and n_mid > 0, f"{n_mid} resets, target - swing vs delta max {mid_err:.1e}")
  print(f"   info link poses read before the refresh: worst link {float(stale.median()) * 100:.1f} cm (median), "
        f"{float(stale.max()) * 100:.1f} cm (max) from the refreshed poses")
  check("termination resets: soles vs one step later", float(torch.stack(next_d).max()) < 0.01,
        f"max {float(torch.stack(next_d).max()) * 1000:.1f} mm")
  t_ok = all(bool((r[t] == 1.0).all()) for _, r, t in term_rows)
  check("terminated reward = 1 on terminations", t_ok, "raw is_terminated (weight -200)")

  # forced time-outs on the 4 most recently reset envs (standing, a few steps into the episode)
  for _ in range(3):
    env.step(zero)
  ids = torch.argsort(env.episode_length_buf)[:4]
  env.episode_length_buf[ids] = env.max_episode_length - 1
  obs, rew, term, trunc, extras = env.step(zero)
  raw = raw_terms(env)
  check("time-out: truncated, not terminated", bool(trunc[ids].all() and not term[ids].any()), "")
  check("time-out: no termination penalty", bool((raw["terminated"][ids] == 0).all()), "")
  ok, e = goal_consistent(env, cmd, obs, ids)
  check("time-out reset: goal valid at reset obs", ok and bool((env.episode_length_buf[ids] == 0).all()), f"max {e:.1e}")
  check("time-out reset: metrics logged", any("footstep" in k for k in extras.get("log", {})), "")

  print("\n== reward terms (one step, zero action)")
  for _ in range(3):
    *_, term, trunc, _ = env.step(zero)
  raw = raw_terms(env)
  live = ~(term | trunc)          # envs reset in this step: rewards were computed on the pre-reset state
  shapes = all(v.shape == (N,) and bool(torch.isfinite(v).all()) for v in raw.values())
  check("terms: shape (N,), finite", shapes, f"{len(raw)} terms")
  tot = sum(rm._step_reward[:, i] for i in range(len(rm.active_terms))) * env.step_dt
  check("reward = sum raw x weight x dt", torch.allclose(rm._reward_buf, tot, atol=1e-5), f"dt {env.step_dt}")
  sw, st = cmd.swing_stance_pos_w()
  ref = {
    "foot_target": torch.exp(-100.0 * ((sw - cmd.target_w)[:, :2] ** 2).sum(-1)),
    "stance_drift": ((st - cmd.st0_w)[:, :2] ** 2).sum(-1),
    "upright": (robot.data.projected_gravity_b[:, :2] ** 2).sum(-1),
    "action_rate": torch.zeros(N, device=dev),
  }
  ub = UPPER_BODY.replace()
  ub.resolve(env.scene)
  ref["posture"] = ((robot.data.joint_pos[:, ub.joint_ids] - env.default_joint_pos_nominal[:, ub.joint_ids]) ** 2).sum(-1)
  for n, v in ref.items():
    e = float((raw[n] - v)[live].abs().max())
    check(f"term {n} = formula", e < 1e-5, f"max |diff| {e:.1e}, mean {float(v[live].mean()):.4f} ({int(live.sum())} envs)")
  jn = [robot.joint_names[i] for i in ub.joint_ids]
  check("posture joints: waist + arms (17)", len(jn) == 17 and not any("hand" in n for n in jn), f"{len(jn)}")
  print("   " + "  ".join(f"{n} {float(v.mean()):+.3g}" for n, v in raw.items()))

  if args.stats:
    rollout_stats(env)


def rollout_stats(env):
  """Zero-action rollout until each env's first termination, same statistics as scripts/compare_mjlab.py."""
  rm, tm = env.reward_manager, env.termination_manager
  N, dev = env.num_envs, env.device
  zero = torch.zeros(N, 29, device=dev)
  env.reset()
  first_end = torch.full((N,), -1.0, device=dev)
  stats = {"steps": 0, "sum": dict.fromkeys(rm.active_terms, 0.0), "reasons": dict.fromkeys(tm.active_terms, 0)}
  for k in range(int(3.0 / env.step_dt)):
    _, _, term, trunc, _ = env.step(zero)
    alive = first_end < 0
    if k == 0:   # the first step after env.reset() is reported separately
      stats["first_step"] = {n: float(rm._step_reward[:, i].mean()) for i, n in enumerate(rm.active_terms)}
    else:
      for i, n in enumerate(rm.active_terms):
        stats["sum"][n] += float(rm._step_reward[:, i][alive].sum())
      stats["steps"] += int(alive.sum())
    done = (term | trunc) & alive
    for n in tm.active_terms:
      stats["reasons"][n] += int(tm.get_term(n)[done].sum())
    first_end = torch.where(done, (k + 1) * env.step_dt, first_end)
    if (first_end >= 0).all():
      break
  stats["first_end_s"] = first_end[first_end > 0].tolist()
  stats["mean_per_step"] = {n: s / max(stats["steps"], 1) for n, s in stats["sum"].items()}
  num = lambda p: {k: float(v) for k, v in p.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}  # noqa: E731
  stats["config"] = {
    "rewards": {n: {"weight": float(c.weight), "params": num(c.params)} for n, c in zip(rm.active_terms, rm._term_cfgs)},
    "terminations": {n: num(c.params) for n, c in zip(tm.active_terms, tm._term_cfgs)},
    "step_dt": env.step_dt, "episode_length_s": env.cfg.episode_length_s,
  }
  ft = torch.tensor(stats["first_end_s"])
  print(f"\n   info zero-action rollout ({N} envs): first termination {float(ft.mean()):.2f} s "
        f"(min {float(ft.min()):.2f}, max {float(ft.max()):.2f}), reasons {stats['reasons']}")
  with open(args.stats, "w") as f:
    json.dump(stats, f, indent=1)
  print("saved", args.stats)


# ---- pinned --------------------------------------------------------------------------------------------------
def part_pinned():
  env = make_env(pinned=True)
  robot, cmd, rm = env.scene[S.ROBOT], env.command_manager.get_term(CMD), env.reward_manager
  act = env.action_manager.get_term("joint_pos")
  N, dev = env.num_envs, env.device
  idx = torch.arange(N, device=dev)
  zero = torch.zeros(N, 29, device=dev)
  env.reset()
  env.step(zero)
  feet = S.FEET_BODIES.replace()
  feet.resolve(env.scene)
  sole_z = float(S.feet_sole_pos_w(env, feet)[..., 2].mean())
  robot.data.default_root_state[:, 2] -= sole_z + 0.003
  print(f"\n   fixed pelvis height {float(robot.data.default_root_state[0, 2]):.3f} m (sole z was {sole_z * 1000:.1f} mm)")

  grp = idx % 4
  names = ("normal", "second swing", "stance lift", "far target")
  fixed_delta = torch.zeros(N, 2, device=dev)
  fixed_delta[grp == 3, 0] = 0.20
  cmd.fixed_foot, cmd.fixed_delta = (idx // 4) % 2, fixed_delta
  env.reset()
  lift_foot = torch.where(grp == 2, 1 - cmd.foot, cmd.foot)
  a_lift = zero.clone()
  jn = list(act._joint_names)
  for e in range(N):
    sd = "left" if int(lift_foot[e]) == 0 else "right"
    for j, dq in LIFT.items():
      i = jn.index(f"{sd}_{j}")
      a_lift[e, i] = dq / float(act._scale[e, i])
  hist = {n: [] for n in rm.active_terms}
  air_pre, td_seen, sole_prev = [], [], None
  slip_err = 0.0
  drift_err = 0.0
  relift = torch.full((N,), -1, device=dev)         # group 1: lift again for 15 steps from the first step >= 40 after touchdown
  for k in range(150):
    relift = torch.where((grp == 1) & (relift < 0) & (k >= 40) & cmd.touched, k, relift)
    lifting = torch.full((N,), k < 15, device=dev) | ((relift >= 0) & (k >= relift) & (k < relift + 15))
    # rewards are computed before the command update of the same step (as in mjlab): reference values use the
    # command state from the previous step and the poses after this step's physics
    prev = {a: getattr(cmd, a).clone() for a in ("touched", "td_err", "sw_air_steps", "sw0_w", "st0_w")}
    env.step(torch.where(lifting[:, None], a_lift, zero))
    raw = raw_terms(env)
    for n in rm.active_terms:
      hist[n].append(raw[n].clone())
    sw, st = cmd.swing_stance_pos_w()
    h = sw[:, 2] - prev["sw0_w"][:, 2]
    airborne = (prev["sw_air_steps"] > 0) & ~prev["touched"]
    exp_cl = airborne.float() * (0.5 + 0.5 * torch.clamp(h / 0.08, 0, 1))
    air_pre.append(float((raw["swing_clearance"] - exp_cl).abs().max()))
    exp_td = prev["touched"].float() * torch.exp(-100 * prev["td_err"] ** 2)
    td_seen.append(float((raw["touchdown"] - exp_td).abs().max()))
    drift_err = max(drift_err, float((raw["stance_drift"] - ((st - prev["st0_w"])[:, :2] ** 2).sum(-1)).abs().max()))
  H = {n: torch.stack(v) for n, v in hist.items()}                          # (T, N)
  dt = env.step_dt
  print("\n== pinned scripted steps (episode sums of raw x weight x dt per group)")
  print(f"   {'term':16s}" + "".join(f"{g:>14s}" for g in names))
  for n in rm.active_terms:
    w = rm._term_cfgs[rm.active_terms.index(n)].weight
    print(f"   {n:16s}" + "".join(f"{float((H[n][:, grp == gi] * w * dt).sum(0).mean()):14.3f}" for gi in range(4)))
  tot = sum(H[n] * rm._term_cfgs[rm.active_terms.index(n)].weight * dt for n in rm.active_terms).sum(0)
  print(f"   {'total':16s}" + "".join(f"{float(tot[grp == gi].mean()):14.3f}" for gi in range(4)))
  g0, g1, g2, g3 = (grp == i for i in range(4))
  print("   " + "  ".join(f"{g}: success {int(cmd.success[grp == i].sum())}/{int((grp == i).sum())}, td_err "
                         f"{float(cmd.td_err[grp == i].mean()) * 100:.1f} cm" for i, g in enumerate(names)))

  print("\n== checks")
  check("swing_clearance = formula", max(air_pre) < 1e-5, f"max |diff| {max(air_pre):.1e} over 150 steps")
  check("touchdown = touched x exp(-100 e^2)", max(td_seen) < 1e-5, f"max |diff| {max(td_seen):.1e}")
  c = cmd.cfg
  tdz = float(cmd.td_pos_w[~g2, 2].abs().max())
  # touchdown = first ground contact; the sole reference point is the sole centre, which is still above the ground
  # when the tilted foot lands on its toe or heel
  check("landing: touched, foot on the ground", bool(cmd.touched[~g2].all()) and tdz < 0.04,
        f"{int(cmd.touched[~g2].sum())}/{int((~g2).sum())}, sole centre at touchdown max {tdz * 1000:.1f} mm (toe / heel first)")
  # open-loop landings scatter (the foot can also sweep forward while it comes down), so the check is per landing:
  # a landing off the target (>= 3 cm) never counts as success and earns less touchdown reward
  off = cmd.touched & (cmd.td_err >= c.success_tol)
  check("target deviation: off-target landings", bool(off[g3].sum() >= 6 and not (cmd.success & off).any()
        and H["touchdown"][-1, g3].mean() < H["touchdown"][-1, g0].mean()),
        f"far target td_err {float(cmd.td_err[g3].mean()) * 100:.1f} cm ({int(off[g3].sum())}/8 off target), touchdown raw "
        f"{float(H['touchdown'][-1, g3].mean()):.2f} vs normal {float(H['touchdown'][-1, g0].mean()):.2f}; "
        f"success among {int(off.sum())} off-target landings: {int((cmd.success & off).sum())}")
  ss = H["second_swing"].amax(0) == 1
  # the penalty applies on every airborne step after touchdown; the second_swing flag (success) needs 3 steps
  check("second_swing: fires for every re-lift", bool(ss[g1].all() and (ss >= cmd.second_swing).all()),
        f"group 1 {int(ss[g1].sum())}/{int(g1.sum())}; others penalized {int(ss[~g1].sum())} (bounce after touchdown), "
        f"flagged {int(cmd.second_swing[~g1].sum())}")
  check("stance_lift fires only for group 2", bool((H["stance_lift"][:, g2].amax(0) == 1).all() and (H["stance_lift"][:, ~g2] == 0).all()), "")
  dmax = (H["stance_drift"][:, ~g2].amax(0)) ** 0.5
  check("stance_drift = formula", drift_err < 1e-6,
        f"max |diff| {drift_err:.1e}; stance drift max " + ", ".join(f"{names[i]} {float(dmax[(grp[~g2] == i)].max()) * 1000:.1f} mm" for i in (0, 1, 3)))
  ds = H["double_support"][-1]
  check("double_support after touchdown", bool((ds[cmd.touched] == 1).all() and (ds[~cmd.touched] == 0).all()), "")
  check("settle 0 with the pelvis fixed", float(H["settle"].abs().max()) < 1e-6, "")
  exp_s = (cmd.touched & (cmd.td_err < c.success_tol) & (cmd.peak_height >= c.min_clearance) & ~cmd.second_swing
           & ~cmd.stance_lift & (cmd.still_time >= c.settle_time))
  check("success flag (metric) = definition", bool((cmd.success == exp_s).all()),
        "touched, td_err < 3 cm, clearance >= 5 cm, no 2nd swing / stance lift, still 0.5 s; success "
        + ", ".join(f"{names[i]} {int(cmd.success[grp == i].sum())}" for i in range(4)))
  src = open(os.path.join(os.path.dirname(mdp.__file__), "rewards.py")).read()
  check("no reward term reads the success flag", ".success" not in src, "success is a metric, the reward is shaped")
  # sole velocity used by foot_slip vs the finite difference over one physics step (PhysX integrates positions with
  # the end-of-step velocity) while the swing foot is being lifted
  env.reset()
  for _ in range(5):
    env.step(a_lift)
  p0 = S.feet_sole_pos_w(env, feet).clone()
  v0 = mdp.feet_sole_vel_w(cmd).clone()
  env.action_manager.apply_action()
  env.scene.write_data_to_sim()
  env.sim.step(render=False)
  env.scene.update(env.physics_dt)
  v_fd = (S.feet_sole_pos_w(env, feet) - p0) / env.physics_dt
  v1 = mdp.feet_sole_vel_w(cmd)
  vmax = float(v_fd.norm(dim=-1).max())
  rel = lambda v: float(((v - v_fd).norm(dim=-1) / v_fd.norm(dim=-1).clamp(min=0.05)).median())  # noqa: E731
  e1, em = rel(v1), rel(0.5 * (v0 + v1))
  d = env.scene[S.ROBOT].data
  e_link = rel(d.body_link_lin_vel_w[:, feet.body_ids])
  # exact rigid-body check through PhysX's centre-of-mass velocity: v_sole = v_com + w x (p_sole - p_com)
  pc, vc = d.body_com_pos_w[:, feet.body_ids], d.body_com_lin_vel_w[:, feet.body_ids]
  v_rb = vc + torch.cross(d.body_link_ang_vel_w[:, feet.body_ids], S.feet_sole_pos_w(env, feet) - pc, dim=-1)
  e_rb = float((v1 - v_rb).norm(dim=-1).max())
  check("foot_slip sole velocity = rigid body", e_rb < 1e-4, f"vs COM velocity + w x r: max |diff| {e_rb:.1e} m/s")
  # finite difference (reference only): PhysX integrates joint coordinates, so the difference of link points over
  # 5 ms is off by O(dt) (~3 %), as large as the w x r part here; the rigid-body check above is the exact one
  print(f"   info sole velocity vs finite diff: |v| up to {vmax:.2f} m/s; median relative error end-of-step {e1:.3f}, "
        f"mean of start/end {em:.3f} (ankle-link velocity without w x r {e_link:.3f})")

  # self collision: drive both arms into the torso (shoulder roll inwards, elbow bent) for 0.5 s
  env.reset()
  a_arm = zero.clone()
  for n_, dq in (("left_shoulder_roll_joint", -1.5), ("right_shoulder_roll_joint", 1.5),
                 ("left_elbow_joint", 1.0), ("right_elbow_joint", 1.0)):
    i = jn.index(n_)
    a_arm[:, i] = dq / act._scale[:, i]
  sc = []
  for _ in range(25):
    env.step(a_arm)
    sc.append(raw_terms(env)["self_collision"].clone())
  sc = torch.stack(sc)
  check("self_collision: arms into the torso", bool((sc.amax(0) > 0).float().mean() > 0.9) and float(H["self_collision"].max()) == 0.0,
        f"{int((sc.amax(0) > 0).sum())}/{N} envs penalized (max {int(sc.max())} of 4 substeps); 0 in the stepping scenarios")

if __name__ == "__main__":
  (part_free if args.part == "free" else part_pinned)()
  print(f"\n== {args.part}: {len(RES) - FAILS[0]}/{len(RES)} checks passed")
  if args.json:
    with open(args.json, "w") as fp:
      json.dump(RES, fp, indent=1)
  os._exit(0 if FAILS[0] == 0 else 1)   # SimulationApp.close() can hang for minutes after a headless run
