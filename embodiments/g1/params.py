"""Unitree G1 (29 DoF) physical parameters: motors, PD gains, limits, action scale, default pose.

Plain numbers only (no simulator import), so the same values serve the Isaac Lab robot config (robot.py), a MuJoCo
sim2sim run and the real-robot deployment.  The values follow the GEAR-SONIC training configuration, whose policy
transferred to the real G1 without changes:

  * four motor types; armature = reflected rotor inertia at the joint (kg m^2)
  * PD gains from a critically over-damped 2nd-order joint model  J q'' + kd q' + kp q = 0:
        kp = J w^2,  kd = 2 zeta J w,  w = 2 pi 10 Hz,  zeta = 2
  * ankles and waist roll / pitch are driven by two 5020 motors through a linkage: 2x armature, gains, effort
  * hip pitch / roll and knee use the 7520-22 motor (139 Nm); hip yaw and waist yaw the 7520-14 (88 Nm)
  * action scale = 0.25 * effort / kp  (joint target = default + scale * action)

Joint order below is the Unitree motor order (rt/lowcmd index); simulators may order joints differently, so
everything is keyed by joint name.
"""
from __future__ import annotations

import math

# ---- motors ------------------------------------------------------------------------------------------------
ARMATURE = {"5020": 0.003609725, "7520_14": 0.010177520, "7520_22": 0.025101925, "4010": 0.00425}
EFFORT = {"5020": 25.0, "7520_14": 88.0, "7520_22": 139.0, "4010": 5.0}        # Nm

NATURAL_FREQ = 2.0 * math.pi * 10.0   # rad/s
DAMPING_RATIO = 2.0

# ---- joints: (name, motor type, number of motors acting on the joint, velocity limit rad/s) -------------
JOINTS = (
  ("left_hip_pitch_joint", "7520_22", 1, 20.0),
  ("left_hip_roll_joint", "7520_22", 1, 20.0),
  ("left_hip_yaw_joint", "7520_14", 1, 32.0),
  ("left_knee_joint", "7520_22", 1, 20.0),
  ("left_ankle_pitch_joint", "5020", 2, 37.0),
  ("left_ankle_roll_joint", "5020", 2, 37.0),
  ("right_hip_pitch_joint", "7520_22", 1, 20.0),
  ("right_hip_roll_joint", "7520_22", 1, 20.0),
  ("right_hip_yaw_joint", "7520_14", 1, 32.0),
  ("right_knee_joint", "7520_22", 1, 20.0),
  ("right_ankle_pitch_joint", "5020", 2, 37.0),
  ("right_ankle_roll_joint", "5020", 2, 37.0),
  ("waist_yaw_joint", "7520_14", 1, 32.0),
  ("waist_roll_joint", "5020", 2, 37.0),
  ("waist_pitch_joint", "5020", 2, 37.0),
  ("left_shoulder_pitch_joint", "5020", 1, 37.0),
  ("left_shoulder_roll_joint", "5020", 1, 37.0),
  ("left_shoulder_yaw_joint", "5020", 1, 37.0),
  ("left_elbow_joint", "5020", 1, 37.0),
  ("left_wrist_roll_joint", "5020", 1, 37.0),
  ("left_wrist_pitch_joint", "4010", 1, 22.0),
  ("left_wrist_yaw_joint", "4010", 1, 22.0),
  ("right_shoulder_pitch_joint", "5020", 1, 37.0),
  ("right_shoulder_roll_joint", "5020", 1, 37.0),
  ("right_shoulder_yaw_joint", "5020", 1, 37.0),
  ("right_elbow_joint", "5020", 1, 37.0),
  ("right_wrist_roll_joint", "5020", 1, 37.0),
  ("right_wrist_pitch_joint", "4010", 1, 22.0),
  ("right_wrist_yaw_joint", "4010", 1, 22.0),
)
JOINT_NAMES = tuple(j[0] for j in JOINTS)
NUM_JOINTS = len(JOINTS)   # 29

ARMATURE_J = {n: k * ARMATURE[m] for n, m, k, _ in JOINTS}
EFFORT_J = {n: k * EFFORT[m] for n, m, k, _ in JOINTS}
VELOCITY_J = {n: v for n, _, _, v in JOINTS}
KP = {n: ARMATURE_J[n] * NATURAL_FREQ ** 2 for n in JOINT_NAMES}
KD = {n: 2.0 * DAMPING_RATIO * ARMATURE_J[n] * NATURAL_FREQ for n in JOINT_NAMES}
ACTION_SCALE = {n: 0.25 * EFFORT_J[n] / KP[n] for n in JOINT_NAMES}

# ---- default pose (knees bent, arms slightly out) ----------------------------------------------------------
DEFAULT_JOINT_POS = dict.fromkeys(JOINT_NAMES, 0.0) | {
  "left_hip_pitch_joint": -0.312, "right_hip_pitch_joint": -0.312,
  "left_knee_joint": 0.669, "right_knee_joint": 0.669,
  "left_ankle_pitch_joint": -0.363, "right_ankle_pitch_joint": -0.363,
  "left_shoulder_pitch_joint": 0.2, "right_shoulder_pitch_joint": 0.2,
  "left_shoulder_roll_joint": 0.2, "right_shoulder_roll_joint": -0.2,
  "left_elbow_joint": 0.6, "right_elbow_joint": 0.6,
}
DEFAULT_ROOT_HEIGHT = 0.76   # pelvis height at spawn (m); the robot settles slightly lower

# ---- sensors and reference points (link frames; the Unitree rev_1_0 description) --------------------------
PELVIS_IMU_POS = (0.04525, 0.0, -0.08339)       # imu_in_pelvis in pelvis (rt/lowstate imu_state), no rotation
TORSO_IMU_POS = (-0.03959, -0.00224, 0.14792)   # imu_in_torso in torso_link (rt/secondary_imu), no rotation
# Foot reference point in <side>_ankle_roll_link: centre of the four sole contact spheres (x -0.02 .. 0.12,
# y +-0.026, centre z -0.031, radius 0.005), at the sole surface.  Touches the ground when the foot is flat.
FOOT_SOLE_POS = (0.05, 0.0, -0.036)

# ---- encoder calibration error (domain randomization) -----------------------------------------------------
# Per env, fixed for the whole run: reading = q + b, b ~ U(-r, r) per joint.  Same range as GEAR-SONIC training.
ENCODER_OFFSET_RANGE = 0.01   # rad

# ---- whole-robot reference --------------------------------------------------------------------------------
# rev_1_0 body with Dex3 three-finger hands (fingers rigid); used by scripts/inspect_robot.py as the expected value.
TOTAL_MASS = 34.39   # kg


def per_joint(table: dict[str, float]) -> list[float]:
  """A per-joint dict as a list in Unitree motor order."""
  return [table[n] for n in JOINT_NAMES]
