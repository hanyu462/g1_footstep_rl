"""Check the Isaac Lab footstep goal (foot task mdp/commands.py, mdp/observations.py) in a minimal RL env.

The env has the G1 state, the goal group, the critic terms, the footstep command, standard reset events (random yaw,
joint offsets) and a time-out only (no rewards).  The pelvis is fixed (fix_root_link) with the soles just on the
ground, so the open-loop leg motions below cannot tip the robot over.  Three parts:

  A  sampling: target placement, start positions captured one step after the reset, goal observation values
  B  scripted steps, four env groups (target = swing start + delta):
       normal        lift the commanded foot 0.3 s, put it down, stand (delta 0)       -> success
       second swing  as normal, then lift the same foot again                         -> second_swing, no success
       stance lift   lift the stance foot instead                                     -> stance_lift, not lifted
       far target    as normal with delta (0.10, 0)                                   -> td_err ~ 10 cm, no success
  C  episode time-out: flags cleared, a new command, metrics logged

  python scripts/inspect_goal.py [--num-envs 32] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--json", type=str, default=None, help="dump the per-check results")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
sys.stdout.reconfigure(line_buffering=True)
app = AppLauncher(args).app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, TerminationTermCfg  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import ContactSensorCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embodiments.g1 import params as P  # noqa: E402
from embodiments.g1 import state as S  # noqa: E402
from embodiments.g1.robot import ACTION_SCALE, G1_CFG  # noqa: E402
from embodiments.g1.tasks.goal_conditioned.foot import mdp  # noqa: E402

CMD = "footstep"
LIFT = {"hip_pitch_joint": -0.8, "knee_joint": 1.2}   # joint offsets that bend one leg up (rad)
EPISODE_S = 1.6


def group(terms: dict, noisy: bool) -> ObservationGroupCfg:
  g = ObservationGroupCfg()
  for k, v in terms.items():
    setattr(g, k, v)
  g.enable_corruption, g.concatenate_terms = noisy, True
  return g


@configclass
class SceneCfg(InteractiveSceneCfg):
  ground = AssetBaseCfg(prim_path=S.GROUND_PRIM, spawn=sim_utils.GroundPlaneCfg())
  light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
  robot: ArticulationCfg = G1_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=G1_CFG.spawn.replace(articulation_props=G1_CFG.spawn.articulation_props.replace(fix_root_link=True)))
  feet_ground_contact: ContactSensorCfg = S.feet_contact_sensor_cfg()


@configclass
class ActionsCfg:
  joint_pos = mdp.JointPositionActionCfg(asset_name=S.ROBOT, joint_names=list(P.JOINT_NAMES), scale=ACTION_SCALE,
                                         use_default_offset=True)


@configclass
class ObservationsCfg:
  state: ObservationGroupCfg = group(S.actor_state_terms(noisy=True), noisy=True)
  goal: ObservationGroupCfg = group(mdp.goal_terms(CMD), noisy=False)
  critic: ObservationGroupCfg = group({**S.actor_state_terms(noisy=False), **mdp.goal_terms(CMD), **S.privileged_terms(),
                                       **mdp.task_privileged_terms(CMD)}, noisy=False)


@configclass
class CommandsCfg:
  footstep = mdp.FootstepCommandCfg()


@configclass
class EventsCfg:
  encoder_offset = S.encoder_offset_event()
  reset_base = EventTermCfg(func=mdp.reset_root_state_uniform, mode="reset",
                            params={"pose_range": {"yaw": (-math.pi, math.pi)}, "velocity_range": {}})
  reset_joints = EventTermCfg(func=mdp.reset_joints_by_offset, mode="reset",
                              params={"position_range": (-0.05, 0.05), "velocity_range": (0.0, 0.0), "asset_cfg": S.BODY_JOINTS})


@configclass
class TerminationsCfg:
  time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)


@configclass
class RewardsCfg:
  pass


@configclass
class EnvCfg(ManagerBasedRLEnvCfg):
  scene: SceneCfg = SceneCfg(num_envs=32, env_spacing=2.5)
  observations: ObservationsCfg = ObservationsCfg()
  actions: ActionsCfg = ActionsCfg()
  commands: CommandsCfg = CommandsCfg()
  events: EventsCfg = EventsCfg()
  terminations: TerminationsCfg = TerminationsCfg()
  rewards: RewardsCfg = RewardsCfg()

  def __post_init__(self):
    self.decimation = 4
    self.sim.dt = 0.005
    self.episode_length_s = EPISODE_S


def flag(ok: bool) -> str:
  return "ok" if ok else "FAIL"


def slices(om, g: str) -> dict[str, slice]:
  out, i = {}, 0
  for n, d in zip(om.active_terms[g], om.group_obs_term_dim[g]):
    out[n] = slice(i, i + d[0])
    i += d[0]
  return out


def main():
  cfg = EnvCfg()
  cfg.scene.num_envs = args.num_envs
  cfg.sim.device = args.device
  env = ManagerBasedRLEnv(cfg)
  robot, om = env.scene[S.ROBOT], env.observation_manager
  act = env.action_manager.get_term("joint_pos")
  cmd: mdp.FootstepCommand = env.command_manager.get_term(CMD)
  N, dev = env.num_envs, env.device
  idx = torch.arange(N, device=dev)
  zero = torch.zeros(N, P.NUM_JOINTS, device=dev)
  res, fails = {}, 0

  def check(name, ok, msg):
    nonlocal fails
    fails += not ok
    res[name] = {"ok": bool(ok), "msg": msg}
    print(f"   {flag(ok):4s} {name:32s} {msg}")

  gs, cs = slices(om, "goal"), slices(om, "critic")
  print("\n== observation groups")
  for g in ("state", "goal", "critic"):
    dims = dict(zip(om.active_terms[g], [d[0] for d in om.group_obs_term_dim[g]]))
    print(f"   {g:7s} {sum(dims.values()):4d} = " + " + ".join(f"{n} {d}" for n, d in dims.items()))
  gd, cd = om.group_obs_dim["goal"][0], om.group_obs_dim["critic"][0]
  check("dims", gd == 11 and cd == S.STATE_DIM + 11 + S.PRIVILEGED_DIM + 8, f"goal {gd} (11), critic {cd} (131)")

  # pelvis height that puts the soles 3 mm into the ground at the default pose (stance contact with a fixed root)
  env.reset()
  env.step(zero)
  sole_z = float(S.feet_sole_pos_w(env, cmd.feet_cfg)[..., 2].mean())
  robot.data.default_root_state[:, 2] -= sole_z + 0.003
  print(f"\n   fixed pelvis height {float(robot.data.default_root_state[0, 2]):.3f} m (sole z was {sole_z * 1000:.1f} mm)")

  # ---- A: sampling and start positions ----------------------------------------------------------------------
  print("\n== A: sampling (reset, then one zero-action step)")
  obs, _ = env.reset()
  t0 = obs["goal"][:, gs["target"]]
  check("reset step: target pending", bool(cmd.need_init.all()) and float(t0[:, 2:6].abs().max()) == 0.0,
        f"need_init all, target-relative entries 0 (one-hot {t0[0, :2].tolist()}, delta {[round(float(x), 3) for x in t0[0, 6:]]})")
  raw = cmd.delta.clone()
  obs, *_ = env.step(zero)
  g = obs["goal"][:, gs["target"]]
  sw, st = cmd.swing_stance_pos_w()
  check("start captured after 1 step", (not bool(cmd.need_init.any())) and torch.allclose(cmd.sw0_w, sw) and torch.allclose(cmd.st0_w, st),
        "sw0 / st0 == sole points at the init step")
  R = mdp.rot2(cmd.yaw0)
  tgt = cmd.sw0_w[:, :2] + torch.einsum("nij,nj->ni", R, cmd.delta)
  check("target = sw0 + R(yaw0) delta", torch.allclose(cmd.target_w[:, :2], tgt, atol=1e-6)
        and torch.allclose(cmd.target_w[:, 2], cmd.sw0_w[:, 2]), "xy in the start heading frame, z = swing start z")
  check("goal: target - swing == delta", torch.allclose(g[:, 2:4], cmd.delta, atol=1e-5),
        f"max |diff| {float((g[:, 2:4] - cmd.delta).abs().max()):.1e} (same step, heading = start heading)")
  rel_st = torch.einsum("nji,nj->ni", R, (st - sw)[:, :2])
  check("goal: target - stance", torch.allclose(g[:, 4:6], cmd.delta - rel_st, atol=1e-5), "== delta - stance offset")
  side = torch.where(cmd.foot == 0, 1.0, -1.0)
  gap = side * g[:, 5]
  clamped = int((~torch.isclose(raw[:, 1], cmd.delta[:, 1])).sum())
  check("lateral gap to stance >= 0.10", float(gap.min()) >= cmd.cfg.min_lateral_gap - 1e-5,
        f"min {float(gap.min()):.3f} m; dy raised for {clamped}/{N} envs")
  dx_ok = bool(((cmd.delta[:, 0] >= cmd.cfg.ranges.dx[0]) & (cmd.delta[:, 0] <= cmd.cfg.ranges.dx[1])).all())
  dy_ok = bool(((raw[:, 1] >= cmd.cfg.ranges.dy[0]) & (raw[:, 1] <= cmd.cfg.ranges.dy[1])).all())
  check("ranges", dx_ok and dy_ok, f"dx in {cmd.cfg.ranges.dx}, sampled dy in {cmd.cfg.ranges.dy}")
  nl = int((cmd.foot == 0).sum())
  check("foot one-hot", torch.equal(g[:, :2], torch.nn.functional.one_hot(cmd.foot, 2).float()) and 0 < nl < N,
        f"left {nl} / right {N - nl}")
  ph = obs["goal"][:, gs["phase"]]
  check("phase", torch.allclose(ph[:, 0], torch.full((N,), env.step_dt / 2.0, device=dev)), f"t {float(ph[0, 0]):.3f} after 1 step")
  yaw_spread = float(cmd.yaw0.max() - cmd.yaw0.min())
  check("random heading", yaw_spread > 3.0, f"yaw0 spread {yaw_spread:.2f} rad")

  # ---- B: scripted steps --------------------------------------------------------------------------------
  print("\n== B: scripted steps, four groups (lift 0.3 s, put down; group 1 lifts again at 0.8 s)")
  grp = idx % 4
  names = ("normal", "second swing", "stance lift", "far target")
  fixed_delta = torch.zeros(N, 2, device=dev)
  fixed_delta[grp == 3, 0] = 0.10
  cmd.fixed_foot, cmd.fixed_delta = cmd.foot.clone(), fixed_delta
  env.reset()
  env.step(zero)
  lift_foot = torch.where(grp == 2, 1 - cmd.foot, cmd.foot)
  a_lift = zero.clone()
  jn = list(act._joint_names)
  for e in range(N):
    sd = "left" if int(lift_foot[e]) == 0 else "right"
    for j, dq in LIFT.items():
      i = jn.index(f"{sd}_{j}")
      a_lift[e, i] = dq / float(act._scale[e, i])
  peak = torch.zeros(N, device=dev)
  for k in range(60):
    lifting = torch.full((N,), k < 15, device=dev) | ((grp == 1) & (k >= 40) & (k < 55))
    env.step(torch.where(lifting[:, None], a_lift, zero))
    sw, _ = cmd.swing_stance_pos_w()
    peak = torch.where(~cmd.touched, torch.maximum(peak, sw[:, 2] - cmd.sw0_w[:, 2]), peak)
  for gi, gname in enumerate(names):
    m = grp == gi
    print(f"   {gname:13s} lifted {int(cmd.lifted[m].sum())}  touched {int(cmd.touched[m].sum())}  "
          f"td_err {float(cmd.td_err[m].mean()) * 100:5.1f} cm  peak {float(cmd.peak_height[m].mean()) * 100:4.1f} cm  "
          f"second_swing {int(cmd.second_swing[m].sum())}  stance_lift {int(cmd.stance_lift[m].sum())}  "
          f"success {int(cmd.success[m].sum())} / {int(m.sum())}")
  g0, g1, g2, g3 = (grp == i for i in range(4))
  t = cmd.touched
  check("td_err = |td_pos - target|", torch.allclose(cmd.td_err[t], torch.norm((cmd.td_pos_w - cmd.target_w)[t, :2], dim=-1), atol=1e-6), "")
  check("peak height", torch.allclose(cmd.peak_height[~g2], peak[~g2], atol=1e-6), "== independent max before touchdown")
  check("touchdown on the ground", float(cmd.td_pos_w[t, 2].abs().max()) < 0.01, f"max |td z| {float(cmd.td_pos_w[t, 2].abs().max()) * 1000:.1f} mm")
  # the open-loop put-down lands a few cm from the start, so success must follow the touchdown error exactly
  exp = (cmd.td_err < cmd.cfg.success_tol)[g0]
  check("normal: success <=> td_err < tol", bool((cmd.success[g0] == exp).all() and not (cmd.second_swing | cmd.stance_lift)[g0].any()),
        f"success {int(cmd.success[g0].sum())}/{int(g0.sum())}, td_err {[round(float(x) * 100, 1) for x in cmd.td_err[g0]]} cm, "
        f"clearance min {float(cmd.peak_height[g0].min()) * 100:.1f} cm, still min {float(cmd.still_time[g0].min()):.2f} s")
  check("second swing", bool(cmd.second_swing[g1].all() and not cmd.success[g1].any()), "")
  check("stance lift", bool(cmd.stance_lift[g2].all() and not cmd.lifted[g2].any() and not cmd.success[g2].any()), "")
  d_w = cmd.target_w[g3, :2] - cmd.sw0_w[g3, :2]
  ref = torch.einsum("nij,nj->ni", mdp.rot2(cmd.yaw0[g3]), fixed_delta[g3])
  check("far target: target 10 cm ahead", torch.allclose(d_w, ref, atol=1e-6) and not bool(cmd.success[g3].any()),
        f"|target - start| {float(d_w.norm(dim=-1).mean()) * 100:.1f} cm, td_err {[round(float(x) * 100, 1) for x in cmd.td_err[g3]]} cm")

  # ---- C: time-out reset ---------------------------------------------------------------------------------
  print(f"\n== C: time-out after {EPISODE_S} s")
  extras_log = {}
  for _ in range(int(EPISODE_S / env.step_dt) + 2):
    _, _, _, trunc, extras = env.step(zero)
    if bool(trunc.any()):
      extras_log = extras.get("log", {})
      reset_ids = trunc.nonzero().flatten()
      break
  check("time-out reset", bool(trunc.all()), f"{int(trunc.sum())}/{N} envs truncated")
  check("flags cleared, target pending", bool(cmd.need_init[reset_ids].all() and not cmd.touched[reset_ids].any()
                                              and not cmd.lifted[reset_ids].any()), "")
  env.step(zero)
  check("new start captured", not bool(cmd.need_init.any()), "")
  keys = sorted(k for k in extras_log if "footstep" in k)
  check("metrics logged", len(keys) == 6, ", ".join(f"{k.split('/')[-1]} {float(extras_log[k]):.2f}" for k in keys))

  print(f"\n== {len(res) - fails}/{len(res)} checks passed")
  if args.json:
    with open(args.json, "w") as fp:
      json.dump(res, fp, indent=1)
    print("saved", args.json)


if __name__ == "__main__":
  main()
  # SimulationApp.close() can hang for minutes after a headless run
  os._exit(0)
