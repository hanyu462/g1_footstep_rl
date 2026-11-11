"""Unitree G1 state definition shared by every G1 task (foot / hand / whole-body goals).

Actor state s_t (93): what a real G1 measures and what the GEAR-SONIC deployment (sim matched real) feeds its
policy: joints, pelvis IMU (rt/lowstate imu_state), previous action.  Joint terms in joint space; pelvis IMU terms
in the pelvis body frame (site imu_in_pelvis has no rotation).  Noise = uniform, same magnitudes as the unitree G1
velocity task.  The torso IMU (rt/secondary_imu) is NOT in the actor: GEAR-SONIC only logs it, so it is not
sim-to-real validated; it is given to the critic instead.

    joint_pos                 q + b_enc - q_default    29   joint space          +-0.01  (b_enc: encoder bias, fixed per env)
    joint_vel                 dq                       29   joint space          +-1.5
    pelvis_ang_vel            pelvis IMU gyro           3   pelvis body          +-0.2
    pelvis_projected_gravity  gravity direction         3   pelvis body          +-0.05
    actions                   a_{t-1} (raw, unscaled)  29   action space         -

Privileged G1 information for the critic (simulation only), task-independent:
    torso_ang_vel 3, torso_projected_gravity 3 (torso_link frame) |
    feet_contact 2 (exact L / R ground contact) | pelvis_lin_vel 3 (pelvis IMU velocimeter, pelvis frame) |
    feet_height 2 (world z of the foot sites) | feet_forces 6 (L / R contact force, world, sign * log1p)
Task-specific privileged terms (e.g. distance to a foot target) belong to the task.

Foot order: every two-foot quantity is [left, right] (contact sensor primary bodies and foot sites, checked by
scripts/inspect_observations.py --check-feet-order).
Foot height is world z: correct on flat ground only; on uneven terrain it must become height above the terrain
under the foot (e.g. a ray-cast / height-scan sensor).

To check on the real robot: the torso IMU corresponds to rt/secondary_imu -- whether it is published, its axes
relative to torso_link, its latency and bias (none of these are modelled yet); joint encoder offsets (modelled as a
fixed per-env encoder bias of +-0.015 rad).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensor, ContactSensorCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.noise import UniformNoiseCfg as Unoise

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

STATE_DIM = 93
FOOT_SITES = ("left_foot", "right_foot")
FEET_CONTACT_SENSOR = "feet_ground_contact"
PELVIS_GYRO = "robot/imu_ang_vel"           # gyro on site imu_in_pelvis (defined in the mjlab G1 model)
PELVIS_VELOCIMETER = "robot/imu_lin_vel"    # velocimeter on site imu_in_pelvis
TORSO = SceneEntityCfg("robot", body_names=("torso_link",))   # site imu_in_torso is fixed to torso_link


# ---- observation functions ---------------------------------------------------------------------------------
def body_ang_vel(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Angular velocity of asset_cfg's (single) body in its own frame = a gyro on a site fixed to it without rotation."""
  a = env.scene[asset_cfg.name]
  b = asset_cfg.body_ids[0]
  return quat_apply_inverse(a.data.body_link_quat_w[:, b], a.data.body_link_ang_vel_w[:, b])


def body_projected_gravity(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Unit gravity direction in the frame of asset_cfg's (single) body (what an IMU quaternion gives)."""
  a = env.scene[asset_cfg.name]
  return quat_apply_inverse(a.data.body_link_quat_w[:, asset_cfg.body_ids[0]], a.data.gravity_vec_w)


def contact_flags(env: ManagerBasedRlEnv, sensor_name: str = FEET_CONTACT_SENSOR) -> torch.Tensor:
  """(L, R) ground contact flags of a contact sensor (1 = in contact)."""
  s: ContactSensor = env.scene[sensor_name]
  assert s.data.found is not None
  return (s.data.found > 0).float()


def site_height(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """World z of asset_cfg's sites (e.g. the foot sites)."""
  return env.scene[asset_cfg.name].data.site_pos_w[:, asset_cfg.site_ids, 2]


# ---- sensors the G1 state needs (added to the scene by each task) ------------------------------------------
def feet_contact_sensor_cfg() -> ContactSensorCfg:
  """Net ground-contact force / flag of each foot (subtree of the ankle roll links vs. the terrain), L then R."""
  return ContactSensorCfg(
    name=FEET_CONTACT_SENSOR,
    primary=ContactMatch(mode="subtree", pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )


# ---- observation term sets ---------------------------------------------------------------------------------
def actor_state_terms(noisy: bool = True) -> dict[str, ObservationTermCfg]:
  """The 93-D G1 actor state, in this order.  noisy=False for the critic copy."""
  n = (lambda lo: Unoise(n_min=-lo, n_max=lo)) if noisy else (lambda lo: None)
  return {
    # actor: encoder reading (q + encoder bias, see the encoder_bias event); critic copy: true joint position
    "joint_pos": ObservationTermCfg(func=envs_mdp.joint_pos_rel, params={"biased": noisy}, noise=n(0.01)),
    "joint_vel": ObservationTermCfg(func=envs_mdp.joint_vel_rel, noise=n(1.5)),
    "pelvis_ang_vel": ObservationTermCfg(func=envs_mdp.builtin_sensor, params={"sensor_name": PELVIS_GYRO}, noise=n(0.2)),
    "pelvis_projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity, noise=n(0.05)),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
  }


def privileged_terms() -> dict[str, ObservationTermCfg]:
  """Task-independent privileged G1 information for the critic (requires feet_contact_sensor_cfg in the scene)."""
  return {
    # torso IMU: not part of the GEAR-SONIC policy input, so not sim-to-real validated -> critic only for now
    "torso_ang_vel": ObservationTermCfg(func=body_ang_vel, params={"asset_cfg": TORSO}),
    "torso_projected_gravity": ObservationTermCfg(func=body_projected_gravity, params={"asset_cfg": TORSO}),
    "feet_contact": ObservationTermCfg(func=contact_flags, params={"sensor_name": FEET_CONTACT_SENSOR}),
    "pelvis_lin_vel": ObservationTermCfg(func=envs_mdp.builtin_sensor, params={"sensor_name": PELVIS_VELOCIMETER}),
    "feet_height": ObservationTermCfg(func=site_height,
                                      params={"asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITES, preserve_order=True)}),
    "feet_forces": ObservationTermCfg(func=vel_mdp.foot_contact_forces, params={"sensor_name": FEET_CONTACT_SENSOR}),
  }
