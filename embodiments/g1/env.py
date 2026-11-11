"""Isaac Lab RL env for G1 tasks: ManagerBasedRLEnv with link poses refreshed right after a reset.

Isaac Lab writes the reset state (root pose, joint positions) into PhysX but does not recompute the link poses
until the next physics step, so anything read on the reset step (commands that capture start positions, the first
observation of the new episode) would see the links where the previous episode ended (measured: ~1.15 m off for the
legs after a fall; the root link itself is written directly and is correct).  After the managers have reset, this
env recomputes the link poses of the written state (PhysX update_articulations_kinematic, the physics part of
sim.forward(): no physics step, the state is unchanged), drops the cached pose buffers, and calls post_reset(env_ids)
on every command term that defines it, so a command captures its start positions on the reset step itself.  This
holds for the first reset (env.reset()) and for resets inside env.step() (terminations, time-outs).

It also logs the outcome of the finished episodes (extras["log"], TensorBoard "Episode/..."): fall_rate (terminated,
not by time-out), time_out_rate and length_s, as fractions / means over the episodes reset in that step.
"""
from __future__ import annotations

from collections.abc import Sequence

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv

# world-frame link pose / velocity buffers of ArticulationData that are cached per sim timestamp
_POSE_BUFFERS = ("_root_link_pose_w", "_root_link_vel_w", "_root_com_pose_w", "_root_com_vel_w", "_root_state_w",
                 "_root_link_state_w", "_root_com_state_w", "_body_link_pose_w", "_body_link_vel_w", "_body_com_pose_w",
                 "_body_com_vel_w", "_body_state_w", "_body_link_state_w", "_body_com_state_w")


def refresh_kinematics(env) -> None:
  """Recompute link poses of the state written to PhysX (no physics step) and drop the cached copies, so the next
  read of body / root poses returns them even if something read the old poses after the reset write."""
  env.scene.write_data_to_sim()
  # what sim.forward() does for the physics, but also without fabric (sim.forward() skips it when use_fabric=False)
  env.sim.physics_sim_view.update_articulations_kinematic()
  for asset in env.scene.articulations.values():
    assert isinstance(asset, Articulation)
    for name in _POSE_BUFFERS:
      getattr(asset.data, name).timestamp = -1.0


class G1RLEnv(ManagerBasedRLEnv):

  def _reset_idx(self, env_ids: Sequence[int]):
    # outcome of the finished episodes (before the managers reset their buffers), logged as fractions
    tm = self.termination_manager
    term, tout = tm.terminated[env_ids], tm.time_outs[env_ids]
    length = self.episode_length_buf[env_ids].float()
    super()._reset_idx(env_ids)
    if len(env_ids) > 0 and "log" in self.extras:
      self.extras["log"]["Episode/fall_rate"] = term.float().mean().item()          # terminated, not by time-out
      self.extras["log"]["Episode/time_out_rate"] = tout.float().mean().item()
      self.extras["log"]["Episode/length_s"] = (length.mean() * self.step_dt).item()
    refresh_kinematics(self)
    for term in self.command_manager._terms.values():
      if hasattr(term, "post_reset"):
        term.post_reset(env_ids)
