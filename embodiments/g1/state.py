"""Unitree G1 state definition for Isaac Lab, shared by every G1 task (foot / hand / whole-body goals).

Actor state s_t (93): what a real G1 measures and what the GEAR-SONIC deployment (sim matched real) feeds its
policy: joints, pelvis IMU (rt/lowstate imu_state), previous action.  Noise = uniform, same magnitudes as the mjlab
version.  The torso IMU (rt/secondary_imu) is not in the actor (GEAR-SONIC only logs it, so it is not sim-to-real
validated); the critic gets it.

    joint_pos                 q + b - q_default        29   joint space          +-0.01  (b: encoder offset, per env)
    joint_vel                 dq                       29   joint space          +-1.5
    pelvis_ang_vel            pelvis IMU gyro           3   pelvis frame         +-0.2
    pelvis_projected_gravity  gravity direction         3   pelvis frame         +-0.05
    actions                   a_{t-1} (raw, unscaled)  29   action space         -

Privileged G1 information for the critic (simulation only, 19), task-independent:
    torso_ang_vel 3, torso_projected_gravity 3 (torso_link frame) | feet_contact 2 (ground contact, L / R) |
    pelvis_lin_vel 3 (velocity of the pelvis IMU point, pelvis frame) | feet_height 2 (world z of the sole points) |
    feet_forces 6 (L / R ground contact force, world, sign * log1p)
Task-specific privileged terms (e.g. distance to a foot target) belong to the task.

Joint order: the 29 body joints in the simulator's (Isaac Lab) order, which is also the GEAR-SONIC policy order;
the 14 finger joints come after them in the articulation and are not part of the state.  Deployment maps to the
Unitree motor order by joint name (params.JOINT_NAMES), using the joint names saved with the policy.

Encoder offset: randomize_encoder_offset shifts default_joint_pos by -b (as GEAR-SONIC does).  Every quantity
measured from default_joint_pos then sees the encoder reading: joint_pos_rel gives q + b - q0, the joint position
action (offset synced here) commands q0 - b + scale * a, which is what a PD loop on the biased encoder does.
Anything that needs the true default pose (rewards, deployment) uses params.DEFAULT_JOINT_POS instead.

Feet: every two-foot quantity is [left, right].  Foot height is world z (flat ground only; on uneven terrain it
must become height above the terrain under the foot).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, sample_uniform
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from . import params as P

if TYPE_CHECKING:
  from isaaclab.assets import Articulation
  from isaaclab.envs import ManagerBasedEnv

STATE_DIM = 93
PRIVILEGED_DIM = 19
ROBOT = "robot"
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
FEET_CONTACT_SENSOR = "feet_ground_contact"
GROUND_PRIM = "/World/ground"                                   # sim_utils.GroundPlaneCfg spawned here
GROUND_COLLIDER = f"{GROUND_PRIM}/GroundPlane/CollisionPlane"   # its collision prim (contact filter target)

# Managers resolve the SceneEntityCfg objects passed in a term's params (not function defaults), so every term below
# passes them explicitly; outside a manager call cfg.resolve(env.scene) first.
BODY_JOINTS = SceneEntityCfg(ROBOT, joint_names=list(P.JOINT_NAMES))    # simulator order (preserve_order=False)
PELVIS = SceneEntityCfg(ROBOT, body_names="pelvis")
TORSO = SceneEntityCfg(ROBOT, body_names="torso_link")
FEET_BODIES = SceneEntityCfg(ROBOT, body_names=list(FEET), preserve_order=True)


# ---- encoder offset (event, mode="startup") ----------------------------------------------------------------
def randomize_encoder_offset(env: ManagerBasedEnv, env_ids: torch.Tensor | None, asset_cfg: SceneEntityCfg,
                             offset_range: float = P.ENCODER_OFFSET_RANGE, action_name: str = "joint_pos") -> None:
  """Sample a per-env encoder offset b and shift default_joint_pos by -b (see the module docstring).

  The joint position action copied default_joint_pos into its offset when it was built (before startup events),
  so its offset is updated here too.  b is kept in env.encoder_offset (num_envs x 29, asset_cfg joint order).
  """
  asset: Articulation = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=asset.device)
  jid = torch.as_tensor(asset_cfg.joint_ids, device=asset.device)
  if not hasattr(env, "encoder_offset"):
    env.encoder_offset = torch.zeros(env.num_envs, len(jid), device=asset.device)
    env.default_joint_pos_nominal = asset.data.default_joint_pos.clone()
  b = sample_uniform(-offset_range, offset_range, (len(env_ids), len(jid)), asset.device)
  env.encoder_offset[env_ids] = b
  q0 = env.default_joint_pos_nominal[env_ids][:, jid]
  asset.data.default_joint_pos[env_ids[:, None], jid[None, :]] = q0 - b

  if action_name in env.action_manager.active_terms:
    term = env.action_manager.get_term(action_name)
    if getattr(term.cfg, "use_default_offset", False):
      tid = term._joint_ids
      tid = torch.arange(asset.num_joints, device=asset.device) if isinstance(tid, slice) else torch.as_tensor(tid, device=asset.device)
      term._offset[env_ids] = asset.data.default_joint_pos[env_ids][:, tid]


def encoder_offset_event(action_name: str = "joint_pos") -> EventTermCfg:
  """The encoder offset event every G1 task adds (SceneEntityCfg defaults are not resolved, so pass it here)."""
  return EventTermCfg(func=randomize_encoder_offset, mode="startup",
                      params={"asset_cfg": BODY_JOINTS, "action_name": action_name})


# ---- observation functions ---------------------------------------------------------------------------------
def _body(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg):
  a = env.scene[asset_cfg.name]
  return a, a.data, asset_cfg.body_ids[0]


def imu_ang_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Gyro of an IMU fixed to asset_cfg's (single) body without rotation: body angular velocity in the body frame."""
  _, d, b = _body(env, asset_cfg)
  return quat_apply_inverse(d.body_link_quat_w[:, b], d.body_link_ang_vel_w[:, b])


def imu_projected_gravity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Unit gravity direction in the frame of asset_cfg's (single) body (what an IMU orientation gives)."""
  _, d, b = _body(env, asset_cfg)
  return quat_apply_inverse(d.body_link_quat_w[:, b], d.GRAVITY_VEC_W)


def imu_lin_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, pos: tuple[float, float, float]) -> torch.Tensor:
  """Linear velocity of the point `pos` (body frame) of asset_cfg's body, in the body frame: v_b + w_b x r."""
  _, d, b = _body(env, asset_cfg)
  q = d.body_link_quat_w[:, b]
  r = torch.tensor(pos, device=q.device).expand(q.shape[0], 3)
  return quat_apply_inverse(q, d.body_link_lin_vel_w[:, b]) + torch.cross(quat_apply_inverse(q, d.body_link_ang_vel_w[:, b]), r, dim=-1)


def feet_sole_pos_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """World position of the sole reference point (params.FOOT_SOLE_POS) of each foot, [L, R]: (num_envs, 2, 3)."""
  d = env.scene[asset_cfg.name].data
  q, p = d.body_link_quat_w[:, asset_cfg.body_ids], d.body_link_pos_w[:, asset_cfg.body_ids]
  r = torch.tensor(P.FOOT_SOLE_POS, device=q.device).expand_as(p)
  return p + quat_apply(q, r)


def feet_height(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """World z of the sole points, [L, R]."""
  return feet_sole_pos_w(env, asset_cfg)[..., 2]


def feet_ground_force_w(env: ManagerBasedEnv, sensor_name: str = FEET_CONTACT_SENSOR) -> torch.Tensor:
  """Ground contact force on each foot (world frame, N), [L, R]: (num_envs, 2, 3)."""
  s: ContactSensor = env.scene[sensor_name]
  ids = s.find_bodies(list(FEET), preserve_order=True)[0]
  return s.data.force_matrix_w[:, ids].sum(dim=2)


def feet_contact(env: ManagerBasedEnv, sensor_name: str = FEET_CONTACT_SENSOR, threshold: float = 1.0) -> torch.Tensor:
  """(L, R) ground contact flags (1 = ground force above threshold N)."""
  return (feet_ground_force_w(env, sensor_name).norm(dim=-1) > threshold).float()


def feet_forces(env: ManagerBasedEnv, sensor_name: str = FEET_CONTACT_SENSOR) -> torch.Tensor:
  """(L, R) ground contact force, world frame, compressed as sign * log1p(|f|): 6 values."""
  f = feet_ground_force_w(env, sensor_name).flatten(1)
  return torch.sign(f) * torch.log1p(f.abs())


# ---- sensor the G1 state needs (added to the scene by each task as `feet_ground_contact`) -----------------
def feet_contact_sensor_cfg(ground_collider: str = GROUND_COLLIDER) -> ContactSensorCfg:
  """Contact forces of both ankle roll links, filtered to contacts with the ground (force_matrix_w).

  The filter must name the ground's collision prim itself; other terrains pass theirs.
  """
  return ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link",
    filter_prim_paths_expr=[ground_collider],
    history_length=3,
    track_air_time=True,
  )


# ---- observation term sets ---------------------------------------------------------------------------------
def actor_state_terms(noisy: bool = True) -> dict[str, ObservationTermCfg]:
  """The 93-D G1 actor state, in this order.  noisy=False for the critic copy (same encoder offset, no noise)."""
  n = (lambda lo: Unoise(n_min=-lo, n_max=lo)) if noisy else (lambda lo: None)
  return {
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, params={"asset_cfg": BODY_JOINTS}, noise=n(0.01)),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel, params={"asset_cfg": BODY_JOINTS}, noise=n(1.5)),
    "pelvis_ang_vel": ObservationTermCfg(func=imu_ang_vel, params={"asset_cfg": PELVIS}, noise=n(0.2)),
    "pelvis_projected_gravity": ObservationTermCfg(func=imu_projected_gravity, params={"asset_cfg": PELVIS}, noise=n(0.05)),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }


def privileged_terms() -> dict[str, ObservationTermCfg]:
  """Task-independent privileged G1 information for the critic (needs feet_contact_sensor_cfg in the scene)."""
  return {
    "torso_ang_vel": ObservationTermCfg(func=imu_ang_vel, params={"asset_cfg": TORSO}),
    "torso_projected_gravity": ObservationTermCfg(func=imu_projected_gravity, params={"asset_cfg": TORSO}),
    "feet_contact": ObservationTermCfg(func=feet_contact),
    "pelvis_lin_vel": ObservationTermCfg(func=imu_lin_vel, params={"asset_cfg": PELVIS, "pos": P.PELVIS_IMU_POS}),
    "feet_height": ObservationTermCfg(func=feet_height, params={"asset_cfg": FEET_BODIES}),
    "feet_forces": ObservationTermCfg(func=feet_forces),
  }
