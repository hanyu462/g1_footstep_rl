"""Unitree G1 (29 DoF) footstep environment: flat ground, one commanded foot, 29-D joint-position action.

Policy at 50 Hz (sim dt 5 ms x decimation 4), episode 4 s: the robot starts standing (default pose, random yaw,
small joint offsets), steps the commanded foot to the target and must stand still on the new stance.
"""
from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from embodiments.g1 import state_mjlab as g1_state
from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg     # G1 29-DoF model shipped with mjlab
from embodiments.g1.state_mjlab import feet_contact_sensor_cfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig


from embodiments.g1.tasks.goal_conditioned.foot import mdp_mjlab as mdp
from embodiments.g1.tasks.goal_conditioned.foot.mdp_mjlab import FootstepCommandCfg

CMD = "footstep"
FOOT_GEOMS = tuple(f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8))
UPPER_BODY = (r".*waist.*", r".*shoulder.*", r".*elbow.*", r".*wrist.*")


def g1_footstep_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  feet_ground = feet_contact_sensor_cfg()
  self_collision = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )

  # ---- observations: state (99, embodiments/g1/state_mjlab.py) | goal (11, this task) | critic = state + goal + privileged
  goal = {
    "target": ObservationTermCfg(func=mdp.generated_commands, params={"command_name": CMD}),          # 8
    "phase": ObservationTermCfg(func=mdp.step_phase, params={"command_name": CMD}),                   # 3
  }
  task_privileged = {
    "swing_error": ObservationTermCfg(func=mdp.swing_error_3d, params={"command_name": CMD}),         # 3
    "step_state": ObservationTermCfg(func=mdp.step_state, params={"command_name": CMD}),              # 5
  }
  observations = {
    "state": ObservationGroupCfg(terms=g1_state.actor_state_terms(noisy=True), concatenate_terms=True,
                                 enable_corruption=not play, history_length=1),
    "goal": ObservationGroupCfg(terms=goal, concatenate_terms=True, enable_corruption=False, history_length=1),
    "critic": ObservationGroupCfg(terms={**g1_state.actor_state_terms(noisy=False), **goal, **g1_state.privileged_terms(),
                                         **task_privileged},
                                  concatenate_terms=True, enable_corruption=False, history_length=1),
  }

  actions = {"joint_pos": JointPositionActionCfg(entity_name="robot", actuator_names=(".*",), scale=G1_ACTION_SCALE,
                                                 use_default_offset=True)}

  commands = {CMD: FootstepCommandCfg(entity_name="robot", contact_sensor_name=g1_state.FEET_CONTACT_SENSOR, foot_site_names=g1_state.FOOT_SITES,
                                      debug_vis=True)}

  # ---- events: a standing start with random heading and small joint offsets; light domain randomization ----
  events = {
    "reset_base": EventTermCfg(func=mdp.reset_root_state_uniform, mode="reset",
                               params={"pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "yaw": (-math.pi, math.pi)},
                                       "velocity_range": {}}),
    "reset_robot_joints": EventTermCfg(func=mdp.reset_joints_by_offset, mode="reset",
                                       params={"position_range": (-0.05, 0.05), "velocity_range": (0.0, 0.0),
                                               "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))}),
    "foot_friction": EventTermCfg(mode="startup", func=dr.geom_friction,
                                  params={"asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOMS), "operation": "abs",
                                          "ranges": (0.6, 1.2), "shared_random": True}),
    # joint encoder calibration error, fixed per env: actor sees q + b (joint_pos biased), the PD target is shifted by -b
    "encoder_bias": EventTermCfg(mode="startup", func=dr.encoder_bias,
                                 params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.015, 0.015)}),
    "base_com": EventTermCfg(mode="startup", func=dr.body_com_offset,
                             params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)), "operation": "add",
                                     "ranges": {0: (-0.02, 0.02), 1: (-0.02, 0.02), 2: (-0.02, 0.02)}}),
  }

  # ---- rewards ----
  r = RewardTermCfg
  rewards = {
    # task: foothold
    "foot_target": r(func=mdp.foot_target_tracking, weight=5.0, params={"command_name": CMD, "k": 100.0}),
    "touchdown": r(func=mdp.touchdown_accuracy, weight=5.0, params={"command_name": CMD, "k": 100.0}),
    "swing_clearance": r(func=mdp.swing_clearance, weight=1.0, params={"command_name": CMD, "target_height": 0.08}),
    # one step with one foot
    "second_swing": r(func=mdp.second_swing, weight=-2.0, params={"command_name": CMD}),
    "stance_lift": r(func=mdp.stance_foot_lift, weight=-2.0, params={"command_name": CMD}),
    "stance_drift": r(func=mdp.stance_foot_drift, weight=-20.0, params={"command_name": CMD}),
    "foot_slip": r(func=mdp.foot_slip, weight=-0.5, params={"command_name": CMD}),
    # finish: stand still on the new stance
    "settle": r(func=mdp.settle, weight=-2.0, params={"command_name": CMD, "delay": 0.3}),
    "double_support": r(func=mdp.double_support_after_touchdown, weight=0.5, params={"command_name": CMD}),
    # balance / regularization
    "upright": r(func=mdp.flat_orientation_l2, weight=-1.0),
    "posture": r(func=mdp.joint_posture, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", joint_names=UPPER_BODY)}),
    "self_collision": r(func=vel_mdp.self_collision_cost, weight=-1.0, params={"sensor_name": "self_collision", "force_threshold": 10.0}),
    "action_rate": r(func=mdp.action_rate_l2, weight=-0.05),
    "joint_acc": r(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_limits": r(func=mdp.joint_pos_limits, weight=-10.0),
    "terminated": r(func=mdp.is_terminated, weight=-200.0),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": math.radians(60.0)}),
    "base_too_low": TerminationTermCfg(func=mdp.root_height_below_minimum, params={"minimum_height": 0.45}),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), entities={"robot": get_g1_robot_cfg()},
                   sensors=(feet_ground, self_collision), num_envs=1, extent=2.0),
    observations=observations, actions=actions, commands=commands, events=events, rewards=rewards,
    terminations=terminations, curriculum={}, metrics={},
    viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot", body_name="torso_link",
                        distance=2.5, elevation=-15.0, azimuth=135.0),
    sim=SimulationCfg(njmax=300, contact_sensor_maxmatch=64,
                      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20, ccd_iterations=50)),
    decimation=4,
    episode_length_s=4.0,
  )
  if play:
    cfg.events.pop("foot_friction", None)
    cfg.events.pop("base_com", None)
    cfg.events.pop("encoder_bias", None)
  return cfg
