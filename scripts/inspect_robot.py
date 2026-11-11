"""Check the Isaac Lab G1 (embodiments/g1/robot.py) against embodiments/g1/params.py.

Reads the values PhysX actually uses (masses, joint gains, armature, limits) and lets the robot stand on flat
ground holding the default pose.

  python scripts/inspect_robot.py [--seconds 3] [--num-envs 4] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=3.0)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--json", type=str, default=None, help="dump bodies / joints / standing result")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
sys.stdout.reconfigure(line_buffering=True)
app = AppLauncher(args).app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embodiments.g1 import params as P  # noqa: E402
from embodiments.g1.robot import G1_CFG, G1_USD  # noqa: E402


@configclass
class SceneCfg(InteractiveSceneCfg):
  ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
  light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
  robot: ArticulationCfg = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
  # same robot with the pelvis fixed in the air, 1 m to the side: joint tracking under gravity
  pinned: ArticulationCfg = G1_CFG.replace(
    prim_path="{ENV_REGEX_NS}/RobotPinned",
    spawn=G1_CFG.spawn.replace(
      articulation_props=G1_CFG.spawn.articulation_props.replace(fix_root_link=True)),
    init_state=G1_CFG.init_state.replace(pos=(0.0, 1.0, 1.0)),
  )


def flag(ok: bool) -> str:
  return "ok" if ok else "MISMATCH"


def main():
  dt, decimation = 0.005, 4
  sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=args.device))
  scene = InteractiveScene(SceneCfg(num_envs=args.num_envs, env_spacing=2.0))
  sim.reset()
  robot = scene["robot"]
  view = robot.root_physx_view
  print(f"\nUSD: {G1_USD}")

  # ---- bodies --------------------------------------------------------------------------------------------
  masses = view.get_masses()[0].tolist()
  total = sum(masses)
  print(f"\n== bodies: {len(robot.body_names)}   total mass {total:.3f} kg   (reference {P.TOTAL_MASS:.2f}) "
        f"{flag(abs(total - P.TOTAL_MASS) < 0.05)}")
  for n, m in zip(robot.body_names, masses):
    print(f"   {n:34s} {m:7.4f}")

  # ---- joints --------------------------------------------------------------------------------------------
  names = robot.joint_names
  kp, kd = view.get_dof_stiffnesses()[0], view.get_dof_dampings()[0]
  arm, eff, vel = view.get_dof_armatures()[0], view.get_dof_max_forces()[0], view.get_dof_max_velocities()[0]
  q0 = robot.data.default_joint_pos[0].cpu()
  body = [n for n in names if n in P.JOINT_NAMES]
  fingers = [n for n in names if n not in P.JOINT_NAMES]
  print(f"\n== joints: {len(names)} = {len(body)} body + {len(fingers)} finger   "
        f"{flag(sorted(body) == sorted(P.JOINT_NAMES))}")
  print(f"   {'joint (sim order)':28s} {'kp':>7s} {'kd':>6s} {'armature':>9s} {'effort':>7s} {'vel':>5s} {'q0':>7s}")
  bad = 0
  for i, n in enumerate(names):
    row = (float(kp[i]), float(kd[i]), float(arm[i]), float(eff[i]), float(vel[i]), float(q0[i]))
    if n in P.JOINT_NAMES:
      ref = (P.KP[n], P.KD[n], P.ARMATURE_J[n], P.EFFORT_J[n], P.VELOCITY_J[n], P.DEFAULT_JOINT_POS[n])
      ok = all(abs(a - b) <= 1e-4 * max(1.0, abs(b)) for a, b in zip(row, ref))
      bad += not ok
    else:
      ok = True
    print(f"   {n:28s} {row[0]:7.2f} {row[1]:6.3f} {row[2]:9.6f} {row[3]:7.1f} {row[4]:5.1f} {row[5]:7.3f}"
          + ("" if ok else "   MISMATCH vs params"))
  print(f"   body joints matching params: {len(body) - bad}/{len(body)}")

  # ---- hold the default pose -----------------------------------------------------------------------------
  # pinned robot (pelvis fixed in the air): gravity sag of the body joints (reference) and finger hold (pass / fail)
  # free robot on the ground: an open-loop PD stand has no balance feedback and eventually tips over (the mjlab G1
  # does too); the time until it does is printed for reference only
  pinned = scene["pinned"]
  steps = int(args.seconds / (dt * decimation))
  bi = [names.index(n) for n in body]
  fi = [names.index(n) for n in fingers]
  b = {"pelvis_z": [], "tilt_deg": []}
  a = {"finger_err": 0.0}
  for k in range(steps):
    for _ in range(decimation):
      robot.set_joint_position_target(robot.data.default_joint_pos)
      pinned.set_joint_position_target(pinned.data.default_joint_pos)
      scene.write_data_to_sim()
      sim.step()
      scene.update(dt)
    g = robot.data.projected_gravity_b
    b["tilt_deg"].append(float(torch.rad2deg(torch.acos(torch.clamp(-g[:, 2], -1, 1))).max()))
    b["pelvis_z"].append(float(robot.data.root_pos_w[:, 2].mean()))
    if fi and k >= steps // 2:
      a["finger_err"] = max(a["finger_err"], float((pinned.data.joint_pos - pinned.data.default_joint_pos)[:, fi].abs().max()))
  sag = (pinned.data.joint_pos - pinned.data.default_joint_pos)[0].cpu()
  a["sag"] = {names[i]: float(sag[i]) for i in bi}
  top = sorted(a["sag"].items(), key=lambda kv: -abs(kv[1]))[:6]
  print(f"\n== pelvis pinned, default pose held {args.seconds:.1f} s, {args.num_envs} envs")
  print("   gravity sag q - q_default (rad), largest: " + "  ".join(f"{n} {v:+.3f}" for n, v in top))
  print("   (expected: the 10 Hz gains are soft; waist pitch carries the whole upper body)")
  print(f"   fingers max |q| {a['finger_err']:.4f} rad   {flag(a['finger_err'] < 0.05)}")
  fall = next((i for i, (z, t) in enumerate(zip(b["pelvis_z"], b["tilt_deg"])) if z < 0.5 or t > 45), None)
  settle = b["pelvis_z"][min(len(b["pelvis_z"]) - 1, int(0.5 / (dt * decimation)))]
  print(f"\n== free on flat ground, default pose held (no balance control), {args.num_envs} envs")
  print(f"   pelvis z at spawn {b['pelvis_z'][0]:.3f} m, after 0.5 s {settle:.3f} m;   "
        + (f"tips over at {fall * dt * decimation:.2f} s" if fall is not None else f"still up after {args.seconds:.1f} s"))

  if args.json:
    with open(args.json, "w") as f:
      json.dump({"usd": G1_USD, "bodies": dict(zip(robot.body_names, masses)), "total_mass": total,
                 "joints": names, "pinned": a, "free": b}, f, indent=1)
    print("saved", args.json)


if __name__ == "__main__":
  main()
  # SimulationApp.close() can hang for minutes after a headless run; nothing is left to save at this point
  os._exit(0)
