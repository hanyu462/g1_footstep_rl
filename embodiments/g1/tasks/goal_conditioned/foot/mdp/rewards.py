"""Footstep rewards (Isaac Lab port of mdp_mjlab/rewards.py; same formulas, units and frames).

Every function returns the raw per-env value (shape (num_envs,)); the reward manager multiplies it by the term
weight and the policy step dt (0.02 s), as in mjlab.  Positive weight = reward, negative = penalty.  Positions are
the sole reference points (params.FOOT_SOLE_POS) in the world frame (m), velocities world frame (m/s, rad/s).

The success flag of the command is a metric only; no reward term reads it.

Only torch is imported at run time, so scripts/compare_mjlab.py can load this file next to the mjlab version and
evaluate both on the same states.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from isaaclab.envs import ManagerBasedRLEnv
  from isaaclab.managers import SceneEntityCfg

  from .commands import FootstepCommand


def _cmd(env: ManagerBasedRLEnv, command_name: str) -> FootstepCommand:
  return env.command_manager.get_term(command_name)


# ---- task: foothold ------------------------------------------------------------------------------------------
def foot_target_tracking(env: ManagerBasedRLEnv, command_name: str, k: float = 100.0) -> torch.Tensor:
  """exp(-k |p_swing,xy - p*_xy|^2) (k in 1/m^2): 0.91 at 3 cm, 0.78 at 5 cm, 0.37 at 10 cm (k = 100)."""
  c = _cmd(env, command_name)
  sw, _ = c.swing_stance_pos_w()
  return torch.exp(-k * torch.sum(torch.square((sw - c.target_w)[:, :2]), dim=-1))


def touchdown_accuracy(env: ManagerBasedRLEnv, command_name: str, k: float = 100.0) -> torch.Tensor:
  """After touchdown, every step: exp(-k e_TD^2) of the frozen touchdown error (rewards the actual landing spot)."""
  c = _cmd(env, command_name)
  return c.touched.float() * torch.exp(-k * torch.square(c.td_err))


def swing_clearance(env: ManagerBasedRLEnv, command_name: str, target_height: float = 0.08) -> torch.Tensor:
  """Before touchdown, while the swing foot is airborne: 0.5 + 0.5 * min(h / target_height, 1), h above its start (m)."""
  c = _cmd(env, command_name)
  sw, _ = c.swing_stance_pos_w()
  h = sw[:, 2] - c.sw0_w[:, 2]
  airborne = (c.sw_air_steps > 0) & ~c.touched
  return airborne.float() * (0.5 + 0.5 * torch.clamp(h / target_height, 0.0, 1.0))


# ---- one step with one foot ------------------------------------------------------------------------------------
def second_swing(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
  """Swing foot airborne again after touchdown (one step only): 0 / 1."""
  c = _cmd(env, command_name)
  return (c.touched & (c.sw_air_steps > 0)).float()


def stance_foot_lift(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
  """Stance foot off the ground (only the commanded foot may move): 0 / 1."""
  c = _cmd(env, command_name)
  return (c.st_air_steps > 0).float()


def stance_foot_drift(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
  """|p_stance,xy - p_stance0,xy|^2 (m^2)."""
  c = _cmd(env, command_name)
  _, st = c.swing_stance_pos_w()
  return torch.sum(torch.square((st - c.st0_w)[:, :2]), dim=-1)


def feet_sole_vel_w(c: FootstepCommand) -> torch.Tensor:
  """World velocity of the sole points, [L, R]: v_link + w x (R r)  (N, 2, 3)."""
  from embodiments.g1 import params as P
  d, ids = c.robot.data, c.feet_cfg.body_ids
  q = d.body_link_quat_w[:, ids]
  r = torch.tensor(P.FOOT_SOLE_POS, device=q.device).expand(q.shape[:-1] + (3,))
  # quaternion (w, x, y, z) rotation of r into the world frame
  w, u = q[..., :1], q[..., 1:]
  t = 2.0 * torch.cross(u, r, dim=-1)
  r_w = r + w * t + torch.cross(u, t, dim=-1)
  return d.body_link_lin_vel_w[:, ids] + torch.cross(d.body_link_ang_vel_w[:, ids], r_w, dim=-1)


def foot_slip(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
  """Sum over feet in contact of |v_sole,xy|^2 (m^2/s^2): stance sliding, swing dragging, post-touchdown slip."""
  c = _cmd(env, command_name)
  v = feet_sole_vel_w(c)[..., :2]
  return torch.sum(torch.sum(torch.square(v), dim=-1) * c.feet_contact().float(), dim=-1)


# ---- finish: stand still on the new stance ---------------------------------------------------------------------
def settle(env: ManagerBasedRLEnv, command_name: str, delay: float = 0.3) -> torch.Tensor:
  """After touchdown + delay (s): pelvis xy speed^2 (m^2/s^2) + 0.1 yaw rate^2 (rad^2/s^2), world frame."""
  c = _cmd(env, command_name)
  active = c.touched & (c.time_s() > c.td_time + delay)
  v = c.robot.data.root_link_lin_vel_w[:, :2]
  w = c.robot.data.root_link_ang_vel_w[:, 2]
  return active.float() * (torch.sum(torch.square(v), dim=-1) + 0.1 * torch.square(w))


def double_support_after_touchdown(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
  """Both feet on the ground after touchdown: 0 / 1."""
  c = _cmd(env, command_name)
  contact = c.feet_contact()
  return (c.touched & contact[:, 0] & contact[:, 1]).float()


# ---- balance / regularization ----------------------------------------------------------------------------------
def joint_posture(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """|q - q_default|^2 (rad^2) over asset_cfg's joints, against the nominal default pose (no encoder offset)."""
  a = env.scene[asset_cfg.name]
  ids = asset_cfg.joint_ids
  q0 = getattr(env, "default_joint_pos_nominal", a.data.default_joint_pos)
  return torch.sum(torch.square(a.data.joint_pos[:, ids] - q0[:, ids]), dim=-1)


def self_collision_cost(env: ManagerBasedRLEnv, sensor_name: str, force_threshold: float = 10.0) -> torch.Tensor:
  """Number of physics substeps of the last policy step (sensor history) in which some body's self-contact force
  exceeds force_threshold (N).  The sensor covers the robot bodies with a ground filter, so per body the self-contact
  force is the net contact force minus the ground contact force (the scene has no other objects).  Same count as the
  mjlab self_collision_cost (substeps with any self contact > 10 N)."""
  d = env.scene[sensor_name].data
  net = d.net_forces_w_history                                               # (N, H, B, 3)
  ground = d.force_matrix_w_history.sum(dim=3)                              # (N, H, B, 3)
  hit = torch.norm(net - ground, dim=-1) > force_threshold                   # (N, H, B)
  return hit.any(dim=-1).sum(dim=-1).float()
