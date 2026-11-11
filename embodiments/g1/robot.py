"""Unitree G1 articulation for Isaac Lab, built from params.py.

Model: the Unitree G1 rev_1_0 USD with Dex3 three-finger hands that ships with Isaac Sim
(Robots/Unitree/G1/g1.usd).  All 46 links have the masses of the GEAR-SONIC training model once the left palm is
corrected (see _spawn_g1); the 14 finger joints exist in the USD but are held at 0 by PD drives (GEAR-SONIC trains
with rigid fingers).

The USD is loaded from G1_USD (local copy, see scripts/fetch_assets.py) when it exists, else from the Isaac Sim
asset server (slow: ~50 MB per process start).
"""
from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import params as P

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
G1_USD_REMOTE = f"{ISAAC_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd"
G1_USD_LOCAL = os.path.join(_REPO, "assets", "isaac", "Robots", "Unitree", "G1", "g1.usd")
G1_USD = os.environ.get("G1_USD", G1_USD_LOCAL if os.path.exists(G1_USD_LOCAL) else G1_USD_REMOTE)

FINGER_JOINTS = r".*_hand_.*_joint"
FINGER_KP, FINGER_KD = 20.0, 1.0   # holds the fingers at 0 (they are rigid in the reference model)

# The USD authors no mass for left_hand_palm_link, so PhysX derives 0.215 kg from the collider; the right palm is
# authored with the reference value.  The left palm gets the right palm's properties mirrored in y
# (centre of mass y -> -y, principal axes quaternion (w, x, y, z) -> (w, -x, y, -z)).
LEFT_PALM = "left_hand/left_hand_palm_link"
LEFT_PALM_MASS = 0.3728385
LEFT_PALM_COM = (0.0621463, -0.0005087, -0.0005817)
LEFT_PALM_INERTIA = (0.00027434, 0.00054048, 0.00039628)                  # principal moments
LEFT_PALM_AXES = (0.9994980, -0.0006596, -0.0100376, 0.0300434)          # (w, x, y, z)


def _spawn_g1(prim_path: str, cfg: sim_utils.UsdFileCfg, *args, **kwargs):
  """Spawn the USD (cloned to every env), then author the left palm mass properties on every copy."""
  from pxr import Gf, UsdPhysics

  prim = sim_utils.spawn_from_usd(prim_path, cfg, *args, **kwargs)
  palms = sim_utils.find_matching_prims(f"{prim_path}/{LEFT_PALM}")
  assert palms, f"no {LEFT_PALM} under {prim_path}"
  for p in palms:
    m = UsdPhysics.MassAPI.Apply(p)
    m.CreateMassAttr().Set(LEFT_PALM_MASS)
    m.CreateCenterOfMassAttr().Set(Gf.Vec3f(*LEFT_PALM_COM))
    m.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*LEFT_PALM_INERTIA))
    w, x, y, z = LEFT_PALM_AXES
    m.CreatePrincipalAxesAttr().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
  return prim


def _exact(name: str) -> str:
  return f"^{name}$"


def _body_actuator(names: tuple[str, ...]) -> ImplicitActuatorCfg:
  """One implicit PD actuator group with per-joint values from params (keys are exact-match regexes)."""
  return ImplicitActuatorCfg(
    joint_names_expr=[_exact(n) for n in names],
    stiffness={_exact(n): P.KP[n] for n in names},
    damping={_exact(n): P.KD[n] for n in names},
    armature={_exact(n): P.ARMATURE_J[n] for n in names},
    effort_limit_sim={_exact(n): P.EFFORT_J[n] for n in names},
    velocity_limit_sim={_exact(n): P.VELOCITY_J[n] for n in names},
  )


_LEGS = tuple(n for n in P.JOINT_NAMES if "hip" in n or "knee" in n or "ankle" in n)
_WAIST = tuple(n for n in P.JOINT_NAMES if n.startswith("waist"))
_ARMS = tuple(n for n in P.JOINT_NAMES if n not in _LEGS and n not in _WAIST)

G1_CFG = ArticulationCfg(
  spawn=sim_utils.UsdFileCfg(
    func=_spawn_g1,
    usd_path=G1_USD,
    activate_contact_sensors=True,
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
      disable_gravity=False,
      retain_accelerations=False,
      linear_damping=0.0,
      angular_damping=0.0,
      max_linear_velocity=1000.0,
      max_angular_velocity=1000.0,
      max_depenetration_velocity=1.0,
    ),
    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
      enabled_self_collisions=True,
      solver_position_iteration_count=8,
      solver_velocity_iteration_count=4,
    ),
  ),
  init_state=ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, P.DEFAULT_ROOT_HEIGHT),
    joint_pos={_exact(n): q for n, q in P.DEFAULT_JOINT_POS.items()} | {FINGER_JOINTS: 0.0},
    joint_vel={".*": 0.0},
  ),
  soft_joint_pos_limit_factor=0.9,
  actuators={
    "legs": _body_actuator(_LEGS),
    "waist": _body_actuator(_WAIST),
    "arms": _body_actuator(_ARMS),
    "fingers": ImplicitActuatorCfg(joint_names_expr=[FINGER_JOINTS], stiffness=FINGER_KP, damping=FINGER_KD),
  },
)
"""Unitree G1 29 DoF (+ 14 held finger joints) with the GEAR-SONIC actuator model."""

ACTION_SCALE = {_exact(n): s for n, s in P.ACTION_SCALE.items()}
"""Per-joint action scale for a JointPositionAction over the 29 body joints."""
