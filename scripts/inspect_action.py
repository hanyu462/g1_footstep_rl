"""Check the Isaac Lab G1 action (embodiments/g1/action.py) and the deployment spec (embodiments/g1/deploy.py).

Minimal env: G1 with the pelvis fixed (random actions cannot tip it over), the actor state group, the joint position
action and the encoder offset event.  Checks: action space (29 body joints, fingers excluded and held), PD gains,
action scale, joint mapping policy <-> Unitree motor index (one-hot action per env), zero / random action PD
targets, the encoder offset applied exactly once, and a deploy.json round trip.

  python scripts/inspect_action.py [--out deploy.json] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=str, default=None, help="where to write the deploy spec (default: a temp file)")
parser.add_argument("--json", type=str, default=None, help="dump the per-check results")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
sys.stdout.reconfigure(line_buffering=True)
app = AppLauncher(args).app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg  # noqa: E402
from isaaclab.managers import ObservationGroupCfg  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embodiments.g1 import action as A  # noqa: E402
from embodiments.g1 import deploy as D  # noqa: E402
from embodiments.g1 import params as P  # noqa: E402
from embodiments.g1 import state as S  # noqa: E402
from embodiments.g1.robot import FINGER_KD, FINGER_KP, G1_CFG  # noqa: E402

NUM_ENVS = 32   # env e < 29 gets a one-hot action at policy slot e


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
    spawn=G1_CFG.spawn.replace(articulation_props=G1_CFG.spawn.articulation_props.replace(fix_root_link=True)),
    init_state=G1_CFG.init_state.replace(pos=(0.0, 0.0, 1.2)))


@configclass
class ActionsCfg:
  joint_pos = A.joint_position_action_cfg()


@configclass
class ObservationsCfg:
  state: ObservationGroupCfg = group(S.actor_state_terms(noisy=False), noisy=False)


@configclass
class EventsCfg:
  encoder_offset = S.encoder_offset_event(A.ACTION_NAME)


@configclass
class EnvCfg(ManagerBasedEnvCfg):
  scene: SceneCfg = SceneCfg(num_envs=NUM_ENVS, env_spacing=2.5)
  observations: ObservationsCfg = ObservationsCfg()
  actions: ActionsCfg = ActionsCfg()
  events: EventsCfg = EventsCfg()

  def __post_init__(self):
    self.decimation = 4
    self.sim.dt = 0.005


def flag(ok: bool) -> str:
  return "ok" if ok else "FAIL"


def main():
  cfg = EnvCfg()
  cfg.sim.device = args.device
  env = ManagerBasedEnv(cfg)
  robot, term, om = env.scene[S.ROBOT], env.action_manager.get_term(A.ACTION_NAME), env.observation_manager
  N, dev = env.num_envs, env.device
  res, fails = {}, 0

  def check(name, ok, msg=""):
    nonlocal fails
    fails += not ok
    res[name] = {"ok": bool(ok), "msg": msg}
    print(f"   {flag(ok):4s} {name:34s} {msg}")

  names = list(term._joint_names)
  jid = torch.arange(robot.num_joints, device=dev)[term._joint_ids]
  fingers = [i for i, n in enumerate(robot.joint_names) if n not in P.JOINT_NAMES]
  q0 = torch.tensor([P.DEFAULT_JOINT_POS[n] for n in names], device=dev)
  b = env.encoder_offset
  scale = term._scale

  # ---- action space ------------------------------------------------------------------------------------------
  print("\n== action space")
  check("dim 29", env.action_manager.total_action_dim == A.ACTION_DIM == 29, f"{env.action_manager.total_action_dim}")
  check("order = simulator order", names == robot.joint_names[:29] and jid.tolist() == list(range(29)),
        "policy slots 0..28 = articulation joints 0..28")
  sj = S.BODY_JOINTS.replace()
  sj.resolve(env.scene)
  check("order = state joint order", [robot.joint_names[i] for i in sj.joint_ids] == names)
  check("fingers excluded", not any(n in names for n in [robot.joint_names[i] for i in fingers]) and len(fingers) == 14,
        f"{len(fingers)} finger joints, none in the action")
  view = robot.root_physx_view
  kp, kd = view.get_dof_stiffnesses()[0].to(dev), view.get_dof_dampings()[0].to(dev)
  check("PD gains body = params", torch.allclose(kp[jid], torch.tensor([P.KP[n] for n in names], device=dev), rtol=1e-5)
        and torch.allclose(kd[jid], torch.tensor([P.KD[n] for n in names], device=dev), rtol=1e-5))
  check("PD gains fingers", bool((kp[fingers] == FINGER_KP).all() and (kd[fingers] == FINGER_KD).all()), f"kp {FINGER_KP}, kd {FINGER_KD}")
  ref_scale = torch.tensor([P.ACTION_SCALE[n] for n in names], device=dev)
  eff = view.get_dof_max_forces()[0].to(dev)[jid]
  check("action scale = params", torch.allclose(scale, ref_scale.expand(N, -1), rtol=1e-6),
        f"range {float(ref_scale.min()):.3f} .. {float(ref_scale.max()):.3f}")
  check("action scale = 0.25 effort / kp", torch.allclose(ref_scale, 0.25 * eff / kp[jid], rtol=1e-4), "from PhysX values")

  def step(a, n=1):
    for _ in range(n):
      env.step(a)
    return robot.data.joint_pos_target

  # ---- joint mapping ------------------------------------------------------------------------------------------
  print("\n== joint mapping (env e < 29: action 1.0 at policy slot e)")
  a = torch.zeros(N, 29, device=dev)
  a[torch.arange(29), torch.arange(29)] = 1.0
  d = step(a)[:, jid] - (q0 - b)
  hit = d[:29].abs() > 1e-6
  check("one-hot moves only its joint", torch.equal(hit, torch.eye(29, dtype=torch.bool, device=dev)), "")
  check("by exactly its scale", torch.allclose(d[torch.arange(29), torch.arange(29)], ref_scale, atol=1e-6), "")
  mi = D.motor_index(names)
  motor_hits = [max(range(29), key=lambda m: abs(D.to_motor(d[e].tolist(), {"motor_index": mi})[m])) for e in range(29)]
  check("policy slot -> motor index", motor_hits == [P.JOINT_NAMES.index(n) for n in names] == mi,
        " ".join(f"{e}:{m}" for e, m in enumerate(mi)))
  x = torch.randn(29).tolist()
  check("to_policy(to_motor(x)) = x", D.to_policy(D.to_motor(x, {"motor_index": mi}), {"motor_index": mi}) == x)

  # ---- zero / random actions, encoder offset ------------------------------------------------------------------
  print("\n== PD targets")
  tgt = step(torch.zeros(N, 29, device=dev))
  check("zero action: q0 - b", torch.allclose(tgt[:, jid], q0 - b, atol=1e-6), f"max err {float((tgt[:, jid] - (q0 - b)).abs().max()):.1e}")
  errs, last = [], 0.0
  for _ in range(3):
    a = torch.randn(N, 29, device=dev)
    tgt = step(a)
    errs.append(float((tgt[:, jid] - (q0 - b + scale * a)).abs().max()))
    last = max(last, float((om.compute()["state"][:, 64:93] - a).abs().max()))
  check("random action: q0 - b + S a", max(errs) < 1e-5, f"max err {max(errs):.1e} over 3 draws a ~ N(0, 1)")
  check("last_action obs = raw a", last == 0.0, "state[64:93]")
  off = tgt[:, jid] - (q0 + scale * a)
  check("encoder offset applied once", torch.allclose(off, -b, atol=1e-5),
        f"target - (q0 + S a) = -b: max err {float((off + b).abs().max()):.1e}, |b| mean {float(b.abs().mean()):.4f}")
  check("fingers: target 0", bool((tgt[:, fingers] == 0).all()))
  enc = om.compute()["state"][:, :29] - (robot.data.joint_pos[:, jid] - q0)
  check("state uses the same b", torch.allclose(enc, b, atol=1e-5), f"joint_pos obs - (q - q0) - b: {float((enc - b).abs().max()):.1e}")
  # Fingers under random body actions.  Random arm motion swings the hands into the body and the contact pushes the
  # fingers (with self-collisions off they stay below 0.05 rad even at a ~ N(0, 2)): that is contact on a non-rigid
  # finger, not a weak drive.  The pass / fail check therefore moves legs and waist only (hands clear of the body);
  # whole-body random actions are printed for reference.
  arm = torch.tensor([("shoulder" in n or "elbow" in n or "wrist" in n) for n in names], device=dev)

  def finger_max(sigma, arms, steps=50):
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, torch.zeros_like(robot.data.default_joint_vel))
    env.step(torch.zeros(N, 29, device=dev))
    w = 0.0
    for _ in range(steps):
      a = (torch.randn(N, 29, device=dev) * sigma).clamp(-2 * sigma, 2 * sigma)
      env.step(a if arms else a * ~arm)
      w = max(w, float(robot.data.joint_pos[:, fingers].abs().max()))
    return w
  fq = finger_max(1.0, arms=False)
  check("fingers held (legs + waist random)", fq < 0.02, f"a ~ N(0, 1): max |q| {fq:.4f} rad over 1 s")
  print(f"   info fingers, whole-body random: max |q| a ~ N(0, 0.5) {finger_max(0.5, True):.3f}, "
        f"N(0, 2) {finger_max(2.0, True):.3f} rad (hands hit the body)")

  # ---- deploy spec ---------------------------------------------------------------------------------------------
  print("\n== deploy spec")
  with tempfile.TemporaryDirectory() as tmp:
    path = args.out or os.path.join(tmp, "deploy.json")
    spec = D.write_deploy_spec(env, path)
    spec2 = D.load_deploy_spec(path)
  check("write / load / check vs params", spec2 == spec, args.out or "(temp file)")
  check("fields", all(k in spec for k in ("joint_names", "motor_index", "default_joint_pos", "action_scale", "kp", "kd",
                                          "effort_limit", "observation", "control_dt")),
        f"observation {spec['observation']['state']}, control_dt {spec['control_dt']}")
  # deploy-side target from the spec alone, in motor order, vs the simulator target with the encoder offset removed
  a = torch.randn(N, 29, device=dev)
  tgt = step(a)[:, jid] + b
  e = 3
  dep = D.to_motor([q + s * float(x) for q, s, x in zip(spec["default_joint_pos"], spec["action_scale"], a[e])], spec)
  sim = D.to_motor(tgt[e].tolist(), spec)
  err = max(abs(u - v) for u, v in zip(dep, sim))
  check("deploy target = sim target + b", err < 1e-5, f"motor order, max err {err:.1e}")

  print(f"\n== {len(res) - fails}/{len(res)} checks passed")
  if args.json:
    with open(args.json, "w") as fp:
      json.dump(res, fp, indent=1)
    print("saved", args.json)


if __name__ == "__main__":
  main()
  os._exit(0)   # SimulationApp.close() can hang for minutes after a headless run
