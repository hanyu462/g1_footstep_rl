"""Unitree G1 footstep environment for Isaac Lab (port of env_cfgs_mjlab.py): flat ground, one commanded foot,
29-D joint-position action.

Policy at 50 Hz (sim dt 5 ms x decimation 4), episode 4 s: the robot starts standing (default pose, random yaw,
small joint offsets), steps the commanded foot to the target and must stand still on the new stance.  Run it with
embodiments.g1.env.G1RLEnv (link poses refreshed on the reset step, see that module).

Reward terms (raw value x weight x dt; formulas, units and frames in mdp/rewards.py), same weights as mjlab:

    foot_target      +5.0    exp(-100 |swing - target|_xy^2)
    touchdown        +5.0    after touchdown: exp(-100 e_TD^2)
    swing_clearance  +1.0    swing airborne before touchdown: 0.5 + 0.5 min(h / 0.08, 1)
    second_swing     -2.0    swing foot airborne after touchdown
    stance_lift      -2.0    stance foot airborne
    stance_drift    -20.0    |stance - stance0|_xy^2
    foot_slip        -0.5    sum over feet in contact of |v_sole,xy|^2
    settle           -2.0    after touchdown + 0.3 s: |v_pelvis,xy|^2 + 0.1 yaw rate^2
    double_support   +0.5    both feet down after touchdown
    upright          -1.0    |projected gravity (pelvis)_xy|^2
    posture          -0.1    |q - q_default|^2 over waist and arms (nominal default pose)
    self_collision   -1.0    substeps with a self contact > 10 N
    action_rate      -0.05   |a_t - a_{t-1}|^2
    joint_acc        -2.5e-7 |qdd|^2 over the 29 body joints
    joint_limits    -10.0    soft-limit violation (rad) over the 29 body joints
    terminated     -200.0    terminated, not by time-out

Terminations: time_out (4 s), fell_over (pelvis tilt > 60 deg), base_too_low (pelvis z < 0.45 m).
"""
from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from embodiments.g1 import action as A
from embodiments.g1 import state as S
from embodiments.g1.robot import G1_CFG
from embodiments.g1.tasks.goal_conditioned.foot import mdp

CMD = "footstep"
CMD_PARAMS = {"command_name": CMD}
UPPER_BODY = SceneEntityCfg(S.ROBOT, joint_names=[r".*waist.*", r".*shoulder.*", r".*elbow.*", r".*wrist.*"])
BODY_CONTACT_SENSOR = "body_contact"


def _group(terms: dict, noisy: bool) -> ObservationGroupCfg:
  g = ObservationGroupCfg()
  for k, v in terms.items():
    setattr(g, k, v)
  g.enable_corruption, g.concatenate_terms = noisy, True
  return g


@configclass
class SceneCfg(InteractiveSceneCfg):
  # ground friction 1 with "multiply": the contact friction is the foot's (randomized) value
  ground = AssetBaseCfg(prim_path=S.GROUND_PRIM, spawn=sim_utils.GroundPlaneCfg(
    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0,
                                                    friction_combine_mode="multiply", restitution_combine_mode="multiply")))
  light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
  robot: ArticulationCfg = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
  feet_ground_contact: ContactSensorCfg = S.feet_contact_sensor_cfg()
  # all robot links (direct children of the robot prim), ground-filtered, 4 substeps of history (self collision)
  body_contact: ContactSensorCfg = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", filter_prim_paths_expr=[S.GROUND_COLLIDER],
                                                    history_length=4)


@configclass
class ActionsCfg:
  joint_pos = A.joint_position_action_cfg()


@configclass
class ObservationsCfg:
  # state (93) | goal (11) | critic = state + goal + privileged (19 G1 + 8 task) = 131
  state: ObservationGroupCfg = _group(S.actor_state_terms(noisy=True), noisy=True)
  goal: ObservationGroupCfg = _group(mdp.goal_terms(CMD), noisy=False)
  critic: ObservationGroupCfg = _group({**S.actor_state_terms(noisy=False), **mdp.goal_terms(CMD), **S.privileged_terms(),
                                        **mdp.task_privileged_terms(CMD)}, noisy=False)


@configclass
class CommandsCfg:
  footstep = mdp.FootstepCommandCfg(debug_vis=True)


@configclass
class EventsCfg:
  # startup: encoder offset (before the reset events, which start from the shifted default pose), foot friction,
  # torso centre-of-mass offset
  encoder_offset = S.encoder_offset_event(A.ACTION_NAME)
  foot_friction = EventTermCfg(func=mdp.randomize_feet_friction, mode="startup",
                               params={"asset_cfg": SceneEntityCfg(S.ROBOT, body_names=list(S.FEET)), "friction_range": (0.6, 1.2)})
  base_com = EventTermCfg(func=mdp.randomize_rigid_body_com, mode="startup",
                          params={"asset_cfg": SceneEntityCfg(S.ROBOT, body_names="torso_link"),
                                  "com_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.02, 0.02)}})
  # reset: standing start with a random heading and small joint offsets
  reset_base = EventTermCfg(func=mdp.reset_root_state_uniform, mode="reset",
                            params={"pose_range": {"yaw": (-math.pi, math.pi)}, "velocity_range": {}})
  reset_joints = EventTermCfg(func=mdp.reset_joints_by_offset, mode="reset",
                              params={"position_range": (-0.05, 0.05), "velocity_range": (0.0, 0.0), "asset_cfg": S.BODY_JOINTS})


@configclass
class RewardsCfg:
  # task: foothold
  foot_target = RewardTermCfg(func=mdp.foot_target_tracking, weight=5.0, params={**CMD_PARAMS, "k": 100.0})
  touchdown = RewardTermCfg(func=mdp.touchdown_accuracy, weight=5.0, params={**CMD_PARAMS, "k": 100.0})
  swing_clearance = RewardTermCfg(func=mdp.swing_clearance, weight=1.0, params={**CMD_PARAMS, "target_height": 0.08})
  # one step with one foot
  second_swing = RewardTermCfg(func=mdp.second_swing, weight=-2.0, params=CMD_PARAMS)
  stance_lift = RewardTermCfg(func=mdp.stance_foot_lift, weight=-2.0, params=CMD_PARAMS)
  stance_drift = RewardTermCfg(func=mdp.stance_foot_drift, weight=-20.0, params=CMD_PARAMS)
  foot_slip = RewardTermCfg(func=mdp.foot_slip, weight=-0.5, params=CMD_PARAMS)
  # finish: stand still on the new stance
  settle = RewardTermCfg(func=mdp.settle, weight=-2.0, params={**CMD_PARAMS, "delay": 0.3})
  double_support = RewardTermCfg(func=mdp.double_support_after_touchdown, weight=0.5, params=CMD_PARAMS)
  # balance / regularization
  upright = RewardTermCfg(func=mdp.flat_orientation_l2, weight=-1.0)
  posture = RewardTermCfg(func=mdp.joint_posture, weight=-0.1, params={"asset_cfg": UPPER_BODY})
  self_collision = RewardTermCfg(func=mdp.self_collision_cost, weight=-1.0,
                                 params={"sensor_name": BODY_CONTACT_SENSOR, "force_threshold": 10.0})
  action_rate = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05)
  joint_acc = RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": S.BODY_JOINTS})
  joint_limits = RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0, params={"asset_cfg": S.BODY_JOINTS})
  terminated = RewardTermCfg(func=mdp.is_terminated, weight=-200.0)


@configclass
class TerminationsCfg:
  time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
  fell_over = TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": math.radians(60.0)})
  base_too_low = TerminationTermCfg(func=mdp.root_height_below_minimum, params={"minimum_height": 0.45})


@configclass
class G1FootstepEnvCfg(ManagerBasedRLEnvCfg):
  scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)
  observations: ObservationsCfg = ObservationsCfg()
  actions: ActionsCfg = ActionsCfg()
  commands: CommandsCfg = CommandsCfg()
  events: EventsCfg = EventsCfg()
  rewards: RewardsCfg = RewardsCfg()
  terminations: TerminationsCfg = TerminationsCfg()
  viewer: ViewerCfg = ViewerCfg(origin_type="asset_root", asset_name=S.ROBOT, eye=(1.8, 1.8, 0.6), lookat=(0.0, 0.0, 0.3))

  def __post_init__(self):
    self.decimation = 4
    self.episode_length_s = 4.0
    self.sim.dt = 0.005
    self.sim.render_interval = self.decimation
    self.sim.physics_material = self.scene.ground.spawn.physics_material
    self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15


@configclass
class G1FootstepEnvCfg_PLAY(G1FootstepEnvCfg):
  def __post_init__(self):
    super().__post_init__()
    self.scene.num_envs = 32
    self.observations.state.enable_corruption = False
    self.events.foot_friction = None
    self.events.base_com = None
    self.events.encoder_offset = None
