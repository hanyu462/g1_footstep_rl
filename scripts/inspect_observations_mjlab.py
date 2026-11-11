"""Inspect and verify the observation groups of a task: per-term shapes, IMU frames, noise, actor / critic input sizes.

  python scripts/inspect_observations_mjlab.py G1-GoalConditioned-Foot-Flat --num-envs 64
"""
from __future__ import annotations

import sys
from dataclasses import asdict, dataclass

import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.lab_api.math import quat_apply_inverse


@dataclass(frozen=True)
class Args:
  num_envs: int = 64
  steps: int = 25
  device: str = "cuda:0"
  check_feet_order: bool = False
  """Lift the left, then the right leg and check that every two-foot term reports [left, right]."""


def check_feet_order(task: str, args: Args) -> None:
  """Static check of names + dynamic check: bend one leg (hip flexion + knee) and see which slot changes."""
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = 1
  cfg.terminations = {}
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  robot, om = env.scene["robot"], env.observation_manager
  cs = term_slices(om, "critic")
  term = env.command_manager.get_term("footstep")
  print("\n== feet order ==")
  print("   command foot sites      :", [robot.site_names[i] for i in term.site_ids.tolist()])
  act = env.action_manager.get_term("joint_pos")
  names = list(robot.joint_names)
  for side in ("left", "right"):
    env.reset()
    a = torch.zeros(env.action_space.shape, device=args.device)
    for j, delta in ((f"{side}_hip_pitch_joint", -0.8), (f"{side}_knee_joint", 1.2)):
      i = names.index(j)
      a[:, i] = delta / float(act._scale[0, i]) if torch.is_tensor(act._scale) else delta / act._scale
    for _ in range(12):
      env.step(a)
    o = om.compute()["critic"][0]
    fc, fh = o[cs["feet_contact"][0]].tolist(), o[cs["feet_height"][0]].tolist()
    p_b = quat_apply_inverse(robot.data.root_link_quat_w.expand(2, 4),
                             robot.data.site_pos_w[0, term.site_ids] - robot.data.root_link_pos_w)
    print(f"   lift {side:5s}: feet_contact [L, R] = {fc}   feet_height [L, R] = {[round(h, 3) for h in fh]}   "
          f"foot sites y in pelvis frame [L, R] = {[round(float(v), 3) for v in p_b[:, 1]]}")
  env.close()


def term_slices(om, group: str) -> dict:
  names, dims = om.active_terms[group], om.group_obs_term_dim[group]
  out, i = {}, 0
  for n, d in zip(names, dims):
    k = int(torch.tensor(d).prod())
    out[n] = (slice(i, i + k), tuple(d)); i += k
  return out


def main():
  import mjlab.tasks  # noqa: F401
  import embodiments.g1.tasks  # noqa: F401

  task, rest = tyro.cli(tyro.extras.literal_type_from_choices(list_tasks()), add_help=False, return_unknown_args=True,
                        config=mjlab.TYRO_FLAGS)
  args = tyro.cli(Args, args=rest, prog=f"{sys.argv[0]} {task}", config=mjlab.TYRO_FLAGS)
  if args.check_feet_order:
    check_feet_order(task, args)
    return
  cfg, agent = load_env_cfg(task), load_rl_cfg(task)
  cfg.scene.num_envs = args.num_envs
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  om = env.observation_manager

  print("\n== observation groups (per env) ==")
  for g in om.active_terms:
    sl = term_slices(om, g)
    total = sum(s.stop - s.start for s, _ in sl.values())
    noise = {n: (c.noise.n_max if c.noise is not None else None) for n, c in zip(om.active_terms[g], om._group_obs_term_cfgs[g])}
    print(f"[{g}] total {total}")
    for n, (s, shape) in sl.items():
      print(f"   {n:26s} {str(shape):8s} cols {s.start:3d}:{s.stop:3d}   noise ±{noise[n]}" if noise[n] is not None else
            f"   {n:26s} {str(shape):8s} cols {s.start:3d}:{s.stop:3d}   noise -")

  # random actions so that velocities are non-zero, then compare IMU terms with body kinematics (noise-free group)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  wrapped.reset()
  for _ in range(args.steps):
    wrapped.step(0.3 * torch.randn(env.action_space.shape, device=args.device))
  obs = om.compute()
  robot = env.scene["robot"]
  torso = robot.find_bodies(["torso_link"])[0][0]
  q_t = robot.data.body_link_quat_w[:, torso]
  ref = {
    "pelvis_ang_vel": robot.data.root_link_ang_vel_b,
    "pelvis_projected_gravity": quat_apply_inverse(robot.data.root_link_quat_w, robot.data.gravity_vec_w),
    "torso_ang_vel": quat_apply_inverse(q_t, robot.data.body_link_ang_vel_w[:, torso]),
    "torso_projected_gravity": quat_apply_inverse(q_t, robot.data.gravity_vec_w),
    "joint_pos": robot.data.joint_pos - robot.data.default_joint_pos,
    "joint_vel": robot.data.joint_vel,
  }
  cs, ss = term_slices(om, "critic"), term_slices(om, "state")
  print("\n== frame check: critic (noise-free) term vs body kinematics, max |diff| ==")
  for n, r in ref.items():
    v = obs["critic"][:, cs[n][0]]
    print(f"   {n:26s} max|diff| {float((v - r).abs().max()):.2e}   mean|value| {float(r.abs().mean()):.3f}")
  print("\n== noise check: actor state term - critic term (max |diff|) ==")
  for n in ss:
    d = (obs["state"][:, ss[n][0]] - obs["critic"][:, cs[n][0]]).abs().max()
    print(f"   {n:26s} {float(d):.3f}")
  g = obs["critic"][:, cs["torso_projected_gravity"][0]]
  print(f"\n   |torso gravity| = {float(g.norm(dim=-1).min()):.4f}..{float(g.norm(dim=-1).max()):.4f}")

  runner = MjlabOnPolicyRunner(wrapped, asdict(agent), device=args.device)
  print("\n== RSL-RL inputs (obs_groups) ==", agent.obs_groups)
  o = wrapped.get_observations()
  for role, groups in agent.obs_groups.items():
    print(f"   {role:6s} <- {groups}: {sum(o[gg].shape[-1] for gg in groups)} dims")
  print(runner.alg.actor if hasattr(runner.alg, "actor") else runner.alg.policy)
  env.close()


if __name__ == "__main__":
  main()
