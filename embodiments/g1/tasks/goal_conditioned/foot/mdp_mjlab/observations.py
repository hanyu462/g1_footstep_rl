"""Footstep observations."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from .commands import FootstepCommand


def _cmd(env, command_name) -> FootstepCommand:
  return env.command_manager.get_term(command_name)


def step_phase(env: ManagerBasedRlEnv, command_name: str, horizon_s: float = 2.0) -> torch.Tensor:
  """Time since the command, clipped to horizon_s and scaled to [0, 1], plus its sin / cos (pi * t / horizon)."""
  t = torch.clamp(_cmd(env, command_name).time_s() / horizon_s, 0.0, 1.0)
  return torch.stack([t, torch.sin(torch.pi * t), torch.cos(torch.pi * t)], dim=-1)


def swing_error_3d(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """(privileged) target - swing foot, world frame (m)."""
  c = _cmd(env, command_name)
  sw, _ = c.swing_stance_pos_w()
  return c.target_w - sw


def step_state(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """(privileged) lifted, touched, touchdown error (m), peak height (m), stance drift (m)."""
  c = _cmd(env, command_name)
  _, st = c.swing_stance_pos_w()
  drift = torch.norm((st - c.st0_w)[:, :2], dim=-1)
  return torch.stack([c.lifted.float(), c.touched.float(), c.td_err, c.peak_height, drift], dim=-1)
