"""Footstep command: one foot, one target, one step per episode.

At each reset the command picks the swing foot (L / R) and a displacement (dx forward, dy left) relative to that
foot's start position, expressed in the robot's start heading frame.  The world target is frozen for the episode.
The term also keeps the per-episode stepping state that rewards, observations and metrics read:

    lifted       swing foot has been airborne (no contact and > lift_height above its start) for >= min_air_steps
    touched      first ground contact after liftoff happened; td_pos / td_err are frozen at that step
    peak_height  maximum swing-foot height above its start before touchdown (clearance)
    second_swing swing foot airborne again after touchdown;  stance_lift: stance foot airborne at any time
    success      touched, td_err < success_tol, peak_height >= min_clearance, no second swing / stance lift,
                 and the robot has stood still (base speed < settle_speed) for settle_time after touchdown
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

LEFT, RIGHT = 0, 1


def yaw_of(quat_wxyz: torch.Tensor) -> torch.Tensor:
  w, x, y, z = quat_wxyz.unbind(-1)
  return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def rot2(yaw: torch.Tensor) -> torch.Tensor:
  """(N,) yaw -> (N, 2, 2) rotation whose columns are the heading frame's forward / left axes in world."""
  c, s = torch.cos(yaw), torch.sin(yaw)
  return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)


class FootstepCommand(CommandTerm):
  cfg: FootstepCommandCfg

  def __init__(self, cfg: FootstepCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.contact: ContactSensor = env.scene[cfg.contact_sensor_name]
    site_ids, site_names = self.robot.find_sites(cfg.foot_site_names, preserve_order=True)
    assert list(site_names) == list(cfg.foot_site_names), site_names
    self.site_ids = torch.tensor(site_ids, device=self.device)
    n, dev = self.num_envs, self.device
    self.foot = torch.zeros(n, dtype=torch.long, device=dev)            # 0 = left, 1 = right
    self.delta = torch.zeros(n, 2, device=dev)                           # commanded (dx, dy) in the start heading frame
    self.yaw0 = torch.zeros(n, device=dev)
    self.target_w = torch.zeros(n, 3, device=dev)
    self.sw0_w = torch.zeros(n, 3, device=dev)
    self.st0_w = torch.zeros(n, 3, device=dev)
    self.td_pos_w = torch.zeros(n, 3, device=dev)
    self.td_err = torch.zeros(n, device=dev)
    self.td_time = torch.zeros(n, device=dev)
    self.peak_height = torch.zeros(n, device=dev)
    self.sw_air_steps = torch.zeros(n, dtype=torch.long, device=dev)
    self.st_air_steps = torch.zeros(n, dtype=torch.long, device=dev)
    self.still_time = torch.zeros(n, device=dev)
    self.need_init = torch.ones(n, dtype=torch.bool, device=dev)
    self.fixed_foot: torch.Tensor | None = None                          # evaluation: forced goals (N,) / (N, 2)
    self.fixed_delta: torch.Tensor | None = None
    for k in ("lifted", "touched", "second_swing", "stance_lift", "success"):
      setattr(self, k, torch.zeros(n, dtype=torch.bool, device=dev))
    for k in ("td_err_cm", "clearance_cm", "lifted", "touched", "success", "stance_drift_cm"):
      self.metrics[k] = torch.zeros(n, device=dev)

  # ---- geometry helpers ------------------------------------------------------------------------------------
  def feet_pos_w(self) -> torch.Tensor:
    return self.robot.data.site_pos_w[:, self.site_ids]                 # (N, 2, 3): left, right

  def swing_stance_pos_w(self) -> tuple[torch.Tensor, torch.Tensor]:
    p = self.feet_pos_w()
    idx = torch.arange(self.num_envs, device=self.device)
    return p[idx, self.foot], p[idx, 1 - self.foot]

  def feet_contact(self) -> torch.Tensor:
    assert self.contact.data.found is not None
    return self.contact.data.found > 0                                   # (N, 2): left, right

  def time_s(self) -> torch.Tensor:
    return self._env.episode_length_buf.float() * self._env.step_dt

  def to_heading(self, v_w_xy: torch.Tensor) -> torch.Tensor:
    """World xy vectors -> current base heading frame (forward, left)."""
    yaw = yaw_of(self.robot.data.root_link_quat_w)
    return torch.einsum("nji,nj->ni", rot2(yaw), v_w_xy)

  @property
  def command(self) -> torch.Tensor:
    """Policy goal (8): foot one-hot | target - swing foot (xy, heading frame) | target - stance foot | (dx, dy)."""
    sw, st = self.swing_stance_pos_w()
    one_hot = torch.nn.functional.one_hot(self.foot, 2).float()
    return torch.cat([one_hot, self.to_heading((self.target_w - sw)[:, :2]), self.to_heading((self.target_w - st)[:, :2]),
                      self.delta], dim=-1)

  # ---- command sampling ------------------------------------------------------------------------------------
  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    r = torch.empty(n, device=self.device)
    self.foot[env_ids] = (r.uniform_(0.0, 1.0) < 0.5).long()
    self.delta[env_ids, 0] = r.uniform_(*self.cfg.ranges.dx)
    self.delta[env_ids, 1] = r.uniform_(*self.cfg.ranges.dy)
    if self.fixed_foot is not None:
      self.foot[env_ids] = self.fixed_foot[env_ids]
      self.delta[env_ids] = self.fixed_delta[env_ids]
    self.need_init[env_ids] = True                                       # positions are captured after the reset settles
    for k in ("lifted", "touched", "second_swing", "stance_lift", "success"):
      getattr(self, k)[env_ids] = False
    for k in ("td_err", "td_time", "peak_height", "still_time"):
      getattr(self, k)[env_ids] = 0.0
    self.sw_air_steps[env_ids] = 0
    self.st_air_steps[env_ids] = 0

  def _init_targets(self, env_ids: torch.Tensor) -> None:
    sw, st = self.swing_stance_pos_w()
    yaw = yaw_of(self.robot.data.root_link_quat_w[env_ids])
    self.yaw0[env_ids] = yaw
    self.sw0_w[env_ids], self.st0_w[env_ids] = sw[env_ids], st[env_ids]
    # keep a minimum lateral gap to the stance foot (no foot crossing / collision)
    R = rot2(yaw)
    rel_st = torch.einsum("nji,nj->ni", R, (st[env_ids] - sw[env_ids])[:, :2])           # stance foot in the start frame
    d = self.delta[env_ids].clone()
    lat = d[:, 1] - rel_st[:, 1]                                                          # target lateral offset from stance
    side = torch.where(self.foot[env_ids] == LEFT, 1.0, -1.0)                           # left foot stays on the +y side
    lat = side * torch.clamp(side * lat, min=self.cfg.min_lateral_gap)
    d[:, 1] = lat + rel_st[:, 1]
    self.delta[env_ids] = d
    tgt = sw[env_ids].clone()
    tgt[:, :2] += torch.einsum("nij,nj->ni", R, d)
    self.target_w[env_ids] = tgt
    self.need_init[env_ids] = False

  # ---- per-step state update -------------------------------------------------------------------------------
  def _update_command(self) -> None:
    init_ids = self.need_init.nonzero().flatten()
    if len(init_ids) > 0:
      self._init_targets(init_ids)
    c = self.cfg
    idx = torch.arange(self.num_envs, device=self.device)
    contact = self.feet_contact()
    sw, st = self.swing_stance_pos_w()
    sw_contact, st_contact = contact[idx, self.foot], contact[idx, 1 - self.foot]
    h_sw = sw[:, 2] - self.sw0_w[:, 2]
    h_st = st[:, 2] - self.st0_w[:, 2]
    sw_air = (~sw_contact) & (h_sw > c.lift_height)
    st_air = (~st_contact) & (h_st > c.lift_height)
    self.sw_air_steps = torch.where(sw_air, self.sw_air_steps + 1, torch.zeros_like(self.sw_air_steps))
    self.st_air_steps = torch.where(st_air, self.st_air_steps + 1, torch.zeros_like(self.st_air_steps))
    before_td = ~self.touched
    self.peak_height = torch.where(before_td, torch.maximum(self.peak_height, h_sw), self.peak_height)
    self.lifted |= self.sw_air_steps >= c.min_air_steps
    td_now = self.lifted & before_td & sw_contact
    if td_now.any():
      self.td_pos_w[td_now] = sw[td_now]
      self.td_err[td_now] = torch.norm((sw - self.target_w)[td_now, :2], dim=-1)
      self.td_time[td_now] = self.time_s()[td_now]
      self.touched |= td_now
    self.second_swing |= self.touched & ~td_now & (self.sw_air_steps >= c.min_air_steps)
    self.stance_lift |= self.st_air_steps >= c.min_air_steps
    v = torch.norm(self.robot.data.root_link_lin_vel_w[:, :2], dim=-1)
    still = self.touched & sw_contact & st_contact & (v < c.settle_speed)
    self.still_time = torch.where(still, self.still_time + self._env.step_dt, torch.zeros_like(self.still_time))
    self.success = (self.touched & (self.td_err < c.success_tol) & (self.peak_height >= c.min_clearance)
                    & ~self.second_swing & ~self.stance_lift & (self.still_time >= c.settle_time))

  def _update_metrics(self) -> None:
    # latest value per env; CommandTerm.reset() averages them over the finished episodes
    st = self.swing_stance_pos_w()[1]
    self.metrics["td_err_cm"] = torch.where(self.touched, self.td_err * 100.0, torch.full_like(self.td_err, 30.0))
    self.metrics["clearance_cm"] = self.peak_height * 100.0
    self.metrics["lifted"] = self.lifted.float()
    self.metrics["touched"] = self.touched.float()
    self.metrics["success"] = self.success.float()
    self.metrics["stance_drift_cm"] = torch.norm((st - self.st0_w)[:, :2], dim=-1) * 100.0

  # ---- viewer ------------------------------------------------------------------------------------------------
  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    for i in visualizer.get_env_indices(self.num_envs):
      t = self.target_w[i].cpu().numpy()
      visualizer.add_sphere(t, radius=0.03, color=(0.1, 0.8, 0.2, 0.8))
      s = self.sw0_w[i].cpu().numpy()
      visualizer.add_sphere(s, radius=0.02, color=(0.6, 0.6, 0.6, 0.6))
      if bool(self.touched[i]):
        visualizer.add_sphere(self.td_pos_w[i].cpu().numpy(), radius=0.025, color=(0.95, 0.8, 0.1, 0.9))


@dataclass(kw_only=True)
class FootstepCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  contact_sensor_name: str = "feet_ground_contact"
  foot_site_names: tuple[str, str] = ("left_foot", "right_foot")
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)      # one command per episode (resampled at reset)
  lift_height: float = 0.015
  min_air_steps: int = 3
  min_lateral_gap: float = 0.10
  success_tol: float = 0.03
  min_clearance: float = 0.05
  settle_speed: float = 0.1
  settle_time: float = 0.5

  @dataclass
  class Ranges:
    dx: tuple[float, float] = (0.10, 0.30)
    dy: tuple[float, float] = (-0.10, 0.10)

  ranges: Ranges = field(default_factory=Ranges)

  def build(self, env: ManagerBasedRlEnv) -> FootstepCommand:
    return FootstepCommand(self, env)


__all__ = ["FootstepCommand", "FootstepCommandCfg", "LEFT", "RIGHT", "rot2", "yaw_of"]
