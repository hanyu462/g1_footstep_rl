"""Footstep goal and task-privileged observations (Isaac Lab)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from isaaclab.envs import ManagerBasedRLEnv

  from .commands import FootstepCommand


def _cmd(env, command_name) -> FootstepCommand:
  return env.command_manager.get_term(command_name)


def step_phase(env: ManagerBasedRLEnv, command_name: str, horizon_s: float = 2.0) -> torch.Tensor:
  """Time since the command, clipped to horizon_s and scaled to [0, 1], plus its sin / cos (pi * t / horizon)."""
  t = torch.clamp(_cmd(env, command_name).time_s() / horizon_s, 0.0, 1.0)
  return torch.stack([t, torch.sin(torch.pi * t), torch.cos(torch.pi * t)], dim=-1)


def swing_error_3d(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
  """(privileged) target - swing foot, world frame (m); 0 until the target is set (one step after reset)."""
  c = _cmd(env, command_name)
  sw, _ = c.swing_stance_pos_w()
  return (~c.need_init).float()[:, None] * (c.target_w - sw)


def step_state(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
  """(privileged) lifted, touched, touchdown error (m), peak height (m), stance drift (m)."""
  c = _cmd(env, command_name)
  _, st = c.swing_stance_pos_w()
  drift = (~c.need_init).float() * torch.norm((st - c.st0_w)[:, :2], dim=-1)
  return torch.stack([c.lifted.float(), c.touched.float(), c.td_err, c.peak_height, drift], dim=-1)


def goal_terms(command_name: str) -> dict:
  """Goal group (11): target 8 (command) + phase 3.  Actor and critic both get it."""
  from isaaclab.envs import mdp
  from isaaclab.managers import ObservationTermCfg
  return {
    "target": ObservationTermCfg(func=mdp.generated_commands, params={"command_name": command_name}),
    "phase": ObservationTermCfg(func=step_phase, params={"command_name": command_name}),
  }


def task_privileged_terms(command_name: str) -> dict:
  """Foot-task privileged terms for the critic (8): swing error 3 + step state 5."""
  from isaaclab.managers import ObservationTermCfg
  return {
    "swing_error": ObservationTermCfg(func=swing_error_3d, params={"command_name": command_name}),
    "step_state": ObservationTermCfg(func=step_state, params={"command_name": command_name}),
  }
