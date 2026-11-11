"""Check the Isaac Lab G1 state (embodiments/g1/state.py) in a minimal env: shapes, joint order, encoder offset,
noise, IMU frames (against finite differences), foot sole height / contact / force, and the [L, R] feet order.

The env holds the state terms, a joint position action over the 29 body joints and the encoder offset event only
(no goal, reward or termination).  Env 0 lifts the left leg, env 1 the right leg, the others stand.

  python scripts/inspect_state.py [--num-envs 8] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--json", type=str, default=None, help="dump the per-check results")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
sys.stdout.reconfigure(line_buffering=True)
app = AppLauncher(args).app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg, mdp  # noqa: E402
from isaaclab.managers import ObservationGroupCfg  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import ContactSensorCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import axis_angle_from_quat, quat_apply, quat_apply_inverse, quat_inv, quat_mul  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embodiments.g1 import params as P  # noqa: E402
from embodiments.g1 import state as S  # noqa: E402
from embodiments.g1.robot import ACTION_SCALE, G1_CFG  # noqa: E402

LIFT = {"hip_pitch_joint": -0.8, "knee_joint": 1.2}   # joint offsets that bend one leg up (rad)


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
  robot: ArticulationCfg = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
  feet_ground_contact: ContactSensorCfg = S.feet_contact_sensor_cfg()


@configclass
class ActionsCfg:
  joint_pos = mdp.JointPositionActionCfg(asset_name=S.ROBOT, joint_names=list(P.JOINT_NAMES), scale=ACTION_SCALE,
                                         use_default_offset=True)


@configclass
class ObservationsCfg:
  state: ObservationGroupCfg = group(S.actor_state_terms(noisy=True), noisy=True)
  critic: ObservationGroupCfg = group({**S.actor_state_terms(noisy=False), **S.privileged_terms()}, noisy=False)


@configclass
class EventsCfg:
  encoder_offset = S.encoder_offset_event()


@configclass
class EnvCfg(ManagerBasedEnvCfg):
  scene: SceneCfg = SceneCfg(num_envs=8, env_spacing=2.5)
  observations: ObservationsCfg = ObservationsCfg()
  actions: ActionsCfg = ActionsCfg()
  events: EventsCfg = EventsCfg()

  def __post_init__(self):
    self.decimation = 4
    self.sim.dt = 0.005


def flag(ok: bool) -> str:
  return "ok" if ok else "FAIL"


def slices(om, g: str) -> dict[str, slice]:
  out, i = {}, 0
  for n, d in zip(om.active_terms[g], om.group_obs_term_dim[g]):
    out[n] = slice(i, i + d[0])
    i += d[0]
  return out


def reset_robot(env, root_pos_offset=(0.0, 0.0, 0.0), root_vel=None):
  robot = env.scene[S.ROBOT]
  root = robot.data.default_root_state.clone()
  root[:, :3] += env.scene.env_origins + torch.tensor(root_pos_offset, device=env.device)
  if root_vel is not None:
    root[:, 7:] = root_vel
  robot.write_root_link_pose_to_sim(root[:, :7])
  robot.write_root_link_velocity_to_sim(root[:, 7:])
  robot.write_joint_state_to_sim(robot.data.default_joint_pos, torch.zeros_like(robot.data.default_joint_vel))
  robot.reset()
  env.scene.write_data_to_sim()
  env.sim.forward()
  env.scene.update(0.0)


def main():
  cfg = EnvCfg()
  cfg.scene.num_envs = max(args.num_envs, 3)
  cfg.sim.device = args.device
  env = ManagerBasedEnv(cfg)
  robot, om, act = env.scene[S.ROBOT], env.observation_manager, env.action_manager.get_term("joint_pos")
  N, dev = env.num_envs, env.device
  res, fails = {}, 0

  def check(name, ok, msg):
    nonlocal fails
    fails += not ok
    res[name] = {"ok": bool(ok), "msg": msg}
    print(f"   {flag(ok):4s} {name:28s} {msg}")

  # ---- shapes ----------------------------------------------------------------------------------------------
  print("\n== observation groups")
  for g in ("state", "critic"):
    dims = dict(zip(om.active_terms[g], [d[0] for d in om.group_obs_term_dim[g]]))
    print(f"   {g:7s} {sum(dims.values()):4d} = " + " + ".join(f"{n} {d}" for n, d in dims.items()))
  sd, cd = om.group_obs_dim["state"][0], om.group_obs_dim["critic"][0]
  check("dims", sd == S.STATE_DIM and cd == S.STATE_DIM + S.PRIVILEGED_DIM,
        f"state {sd} (expect {S.STATE_DIM}), critic {cd} (expect {S.STATE_DIM + S.PRIVILEGED_DIM})")

  # ---- joint order -----------------------------------------------------------------------------------------
  print("\n== joint order (policy order = simulator order of the 29 body joints)")
  obs_ids = S.BODY_JOINTS.replace()
  obs_ids.resolve(env.scene)
  obs_names = [robot.joint_names[i] for i in obs_ids.joint_ids]
  act_names = list(act._joint_names)
  check("body joints first", list(obs_ids.joint_ids) == list(range(P.NUM_JOINTS)),
        f"articulation indices {obs_ids.joint_ids[0]}..{obs_ids.joint_ids[-1]}, fingers after")
  check("obs order == action order", obs_names == act_names, "")
  motor = [P.JOINT_NAMES.index(n) for n in obs_names]
  print("   policy order -> motor index: " + " ".join(f"{n.removesuffix('_joint')}:{m}" for n, m in zip(obs_names, motor)))
  res["policy_joint_names"] = obs_names

  # ---- encoder offset --------------------------------------------------------------------------------------
  print("\n== encoder offset")
  b = env.encoder_offset
  jid = torch.tensor(obs_ids.joint_ids, device=dev)
  q0 = torch.tensor([P.DEFAULT_JOINT_POS[n] for n in obs_names], device=dev)
  check("range", float(b.abs().max()) <= P.ENCODER_OFFSET_RANGE and float(b.std()) > 0.4 * P.ENCODER_OFFSET_RANGE,
        f"|b| max {float(b.abs().max()):.4f}, std {float(b.std()):.4f} (U(+-{P.ENCODER_OFFSET_RANGE}) std "
        f"{P.ENCODER_OFFSET_RANGE / 3 ** 0.5:.4f}), envs differ: {bool((b[0] != b[1]).any())}")
  check("nominal default kept", torch.allclose(env.default_joint_pos_nominal[:, jid], q0.expand(N, -1)), "== params.DEFAULT_JOINT_POS")
  check("action offset", torch.allclose(act._offset, q0 - b, atol=1e-6),
        f"max |offset - (q0 - b)| {float((act._offset - (q0 - b)).abs().max()):.1e}")
  reset_robot(env)
  env.step(torch.zeros(N, P.NUM_JOINTS, device=dev))
  tgt = robot.data.joint_pos_target[:, jid]
  check("PD target at a = 0", torch.allclose(tgt, q0 - b, atol=1e-6), f"max |target - (q0 - b)| {float((tgt - (q0 - b)).abs().max()):.1e}")
  o = om.compute()
  cs, ss = slices(om, "critic"), slices(om, "state")
  enc = o["critic"][:, cs["joint_pos"]] - (robot.data.joint_pos[:, jid] - q0)
  check("joint_pos = q + b - q0", torch.allclose(enc, b, atol=1e-5), f"max |obs - (q - q0) - b| {float((enc - b).abs().max()):.1e}")

  # ---- noise -----------------------------------------------------------------------------------------------
  print("\n== noise: actor term - critic term, max |diff| (bound)")
  bounds = {"joint_pos": 0.01, "joint_vel": 1.5, "pelvis_ang_vel": 0.2, "pelvis_projected_gravity": 0.05, "actions": 0.0}
  for n, lo in bounds.items():
    d = float((o["state"][:, ss[n]] - o["critic"][:, cs[n]]).abs().max())
    check(f"noise {n}", d <= lo + 1e-6 and (lo == 0 or d > 0.5 * lo), f"{d:.4f} ({lo})")

  # ---- stand / lift one leg --------------------------------------------------------------------------------
  print("\n== 0.3 s: env 0 lifts the left leg, env 1 the right leg, the others stand (default pose)")
  reset_robot(env)
  a = torch.zeros(N, P.NUM_JOINTS, device=dev)
  for e, side in ((0, "left"), (1, "right")):
    for j, dq in LIFT.items():
      i = act_names.index(f"{side}_{j}")
      a[e, i] = dq / float(act._scale[e, i])
  for _ in range(15):
    env.step(a)
  o = om.compute()["critic"]
  fc, fh = o[:, cs["feet_contact"]], o[:, cs["feet_height"]]
  f = S.feet_ground_force_w(env)
  st = slice(2, N)
  fz = f[st, :, 2].sum(-1)
  mg = P.TOTAL_MASS * 9.81
  check("stand: both feet contact", bool((fc[st] == 1).all()), f"feet_contact {fc[2].tolist()}")
  check("stand: sole height ~ 0", float(fh[st].abs().max()) < 0.005, f"max |z| {float(fh[st].abs().max()) * 1000:.1f} mm")
  check("stand: ground force ~ m g", float((fz / mg - 1).abs().max()) < 0.15,
        f"sum Fz {float(fz.mean()):.0f} N vs m g {mg:.0f} N")
  net = env.scene["feet_ground_contact"].data.net_forces_w[st, :, 2].sum(-1)
  check("ground filter", float((net - fz).abs().max()) < 0.05 * mg,
        f"net (all contacts) {float(net.mean()):.0f} N vs ground-only {float(fz.mean()):.0f} N")
  for e, side, k in ((0, "left", 0), (1, "right", 1)):
    up, down = fh[e, k], fh[e, 1 - k]
    ok = bool(up > down + 0.03 and fc[e, k] == 0 and fc[e, 1 - k] == 1)
    check(f"[L, R] order: lift {side}", ok, f"feet_height {[round(float(x), 3) for x in fh[e]]}  feet_contact {fc[e].tolist()}")
  # sole points vs ankle links in the pelvis frame: left must have y > 0
  feet = S.FEET_BODIES.replace()
  feet.resolve(env.scene)
  sole_b = quat_apply_inverse(robot.data.root_link_quat_w[st, None].expand(-1, 2, -1),
                              S.feet_sole_pos_w(env, feet)[st] - robot.data.root_link_pos_w[st, None])
  check("[L, R] order: left at +y", bool((sole_b[..., 0, 1] > 0).all() and (sole_b[..., 1, 1] < 0).all()),
        f"sole y in pelvis frame {[round(float(x), 3) for x in sole_b[0, :, 1]]}")

  # ---- IMU frames vs finite differences (robot in the air, random root velocity) -----------------------------
  # PhysX integrates positions with the end-of-step velocity (semi-implicit Euler), so the finite difference over one
  # physics step is compared with the velocity after the step
  print("\n== IMU terms vs finite differences over one 5 ms physics step (robot in free flight, random root velocity)")
  vel = torch.cat([torch.empty(N, 3, device=dev).uniform_(-1, 1), torch.empty(N, 3, device=dev).uniform_(-3, 3)], -1)
  reset_robot(env, root_pos_offset=(0.0, 0.0, 1.0), root_vel=vel)
  for _ in range(3):
    env.step(torch.zeros(N, P.NUM_JOINTS, device=dev))
  ids = {"pelvis": robot.find_bodies("pelvis")[0][0], "torso": robot.find_bodies("torso_link")[0][0]}

  def snap():
    d = robot.data
    o = om.compute()["critic"]
    pq = {k: (d.body_link_pos_w[:, i].clone(), d.body_link_quat_w[:, i].clone()) for k, i in ids.items()}
    return o, pq
  o0, s0 = snap()
  env.sim.step(render=False)
  env.scene.update(env.physics_dt)
  o1, s1 = snap()
  dt = env.physics_dt
  avg = lambda n: o1[:, cs[n]]  # noqa: E731
  for body, term in (("pelvis", "pelvis_ang_vel"), ("torso", "torso_ang_vel")):
    (_, qa), (_, qb) = s0[body], s1[body]
    w_fd = quat_apply_inverse(qb, axis_angle_from_quat(quat_mul(qb, quat_inv(qa))) / dt)   # world rate, end frame
    e = float((avg(term) - w_fd).norm(dim=-1).max())
    check(f"{term} frame", e < 0.02 * float(w_fd.norm(dim=-1).max()) + 1e-3,
          f"max |obs - fd| {e:.4f} rad/s (|w| up to {float(w_fd.norm(dim=-1).max()):.2f})")
  (pa, qa), (pb, qb) = s0["pelvis"], s1["pelvis"]
  r = torch.tensor(P.PELVIS_IMU_POS, device=dev).expand(N, 3)
  v_fd_w = ((pb + quat_apply(qb, r)) - (pa + quat_apply(qa, r))) / dt
  v_fd = quat_apply_inverse(qb, v_fd_w)
  e = float((avg("pelvis_lin_vel") - v_fd).norm(dim=-1).max())
  check("pelvis_lin_vel (IMU point)", e < 0.01, f"max |obs - fd| {e:.4f} m/s (|v| up to {float(v_fd.norm(dim=-1).max()):.2f})")
  for body, term in (("pelvis", "pelvis_projected_gravity"), ("torso", "torso_projected_gravity")):
    g_ref = quat_apply_inverse(s1[body][1], torch.tensor([0.0, 0.0, -1.0], device=dev).expand(N, 3))
    e = float((o1[:, cs[term]] - g_ref).norm(dim=-1).max())
    check(f"{term}", e < 1e-4, f"max |obs - R^T (0,0,-1)| {e:.1e}")
  e = float((o1[:, cs["pelvis_ang_vel"]] - robot.data.root_ang_vel_b).norm(dim=-1).max())
  check("pelvis_ang_vel == root_ang_vel_b", e < 1e-4, f"{e:.1e}")

  print(f"\n== {len(res) - 1 - fails}/{len(res) - 1} checks passed")
  if args.json:
    with open(args.json, "w") as fp:
      json.dump(res, fp, indent=1)
    print("saved", args.json)


if __name__ == "__main__":
  main()
  # SimulationApp.close() can hang for minutes after a headless run
  os._exit(0)
