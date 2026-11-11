"""Unitree G1 action for Isaac Lab, shared by every G1 task.

a_t (29): joint position offsets of the 29 body joints, in the simulator joint order (= the state's joint order and
the GEAR-SONIC policy order).  The 14 Dex3 finger joints are not actuated by the policy; their PD drives hold them at
0 (robot.py).  The PD target applied in simulation is

    q_target = (q_default - b) + scale * a        scale = params.ACTION_SCALE (0.25 * effort / kp), no clipping

where b is the per-env encoder offset of embodiments/g1/state.py (randomize_encoder_offset shifts the default pose
the action term adds by -b, exactly once).  On the real robot the motor PD loop runs on the biased encoder itself, so
deployment commands q_default + scale * a with the nominal default pose (see deploy.py).
"""
from __future__ import annotations

from isaaclab.envs import mdp

from . import params as P
from .robot import ACTION_SCALE
from .state import ROBOT

ACTION_NAME = "joint_pos"          # the encoder offset event syncs the offset of the action term with this name
ACTION_DIM = P.NUM_JOINTS


def joint_position_action_cfg() -> mdp.JointPositionActionCfg:
  """Joint position action over the 29 body joints (simulator order, default pose as offset)."""
  return mdp.JointPositionActionCfg(
    asset_name=ROBOT,
    joint_names=list(P.JOINT_NAMES),
    preserve_order=False,          # simulator order, same as the state's BODY_JOINTS
    scale=ACTION_SCALE,
    use_default_offset=True,
  )
