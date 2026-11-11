"""Footstep rewards (all read the FootstepCommand state; positive = reward, sign of the weight decides)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from .commands import FootstepCommand


def _cmd(env: ManagerBasedRlEnv, command_name: str) -> FootstepCommand:
  return env.command_manager.get_term(command_name)


def foot_target_tracking(env: ManagerBasedRlEnv, command_name: str, k: float = 100.0) -> torch.Tensor:
  """exp(-k |p_swing,xy - p*_xy|^2): 0.91 at 3 cm, 0.78 at 5 cm, 0.37 at 10 cm (k = 100)."""
  c = _cmd(env, command_name)
  sw, _ = c.swing_stance_pos_w()
  return torch.exp(-k * torch.sum(torch.square((sw - c.target_w)[:, :2]), dim=-1))


def touchdown_accuracy(env: ManagerBasedRlEnv, command_name: str, k: float = 100.0) -> torch.Tensor:
  """After touchdown, every step: exp(-k e_TD^2) of the frozen touchdown error (rewards the actual landing spot)."""
  c = _cmd(env, command_name)
  return c.touched.float() * torch.exp(-k * torch.square(c.td_err))


def swing_clearance(env: ManagerBasedRlEnv, command_name: str, target_height: float = 0.08) -> torch.Tensor:
  """Before touchdown, while the swing foot is airborne: 0.5 + 0.5 * min(h / target_height, 1)."""
  c = _cmd(env, command_name)
  sw, _ = c.swing_stance_pos_w()
  h = sw[:, 2] - c.sw0_w[:, 2]
  airborne = (c.sw_air_steps > 0) & ~c.touched
  return airborne.float() * (0.5 + 0.5 * torch.clamp(h / target_height, 0.0, 1.0))


def second_swing(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Swing foot airborne again after touchdown (one step only)."""
  c = _cmd(env, command_name)
  return (c.touched & (c.sw_air_steps > 0)).float()


def stance_foot_lift(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Stance foot off the ground (only the commanded foot may move)."""
  c = _cmd(env, command_name)
  return (c.st_air_steps > 0).float()


def stance_foot_drift(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """|p_stance,xy - p_stance0,xy|^2 (m^2)."""
  c = _cmd(env, command_name)
  _, st = c.swing_stance_pos_w()
  return torch.sum(torch.square((st - c.st0_w)[:, :2]), dim=-1)


def foot_slip(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Sum over feet in contact of |v_foot,xy|^2 (stance sliding, swing dragging, post-touchdown slip)."""
  c = _cmd(env, command_name)
  v = c.robot.data.site_lin_vel_w[:, c.site_ids, :2]
  return torch.sum(torch.sum(torch.square(v), dim=-1) * c.feet_contact().float(), dim=-1)


def settle(env: ManagerBasedRlEnv, command_name: str, delay: float = 0.3) -> torch.Tensor:
  """After touchdown + delay: base xy speed^2 + 0.1 yaw rate^2 (stand still on the new stance)."""
  c = _cmd(env, command_name)
  active = c.touched & (c.time_s() > c.td_time + delay)
  v = c.robot.data.root_link_lin_vel_w[:, :2]
  w = c.robot.data.root_link_ang_vel_w[:, 2]
  return active.float() * (torch.sum(torch.square(v), dim=-1) + 0.1 * torch.square(w))


def double_support_after_touchdown(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Both feet on the ground after touchdown."""
  c = _cmd(env, command_name)
  contact = c.feet_contact()
  return (c.touched & contact[:, 0] & contact[:, 1]).float()


def joint_posture(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """|q - q_default|^2 over the joints of asset_cfg (used for waist / arms)."""
  a = env.scene[asset_cfg.name]
  ids = asset_cfg.joint_ids
  return torch.sum(torch.square(a.data.joint_pos[:, ids] - a.data.default_joint_pos[:, ids]), dim=-1)
