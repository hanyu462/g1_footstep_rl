"""Deployment spec of a trained G1 policy: everything outside the network that sim2sim / the real robot needs.

Saved as deploy.json next to every checkpoint (write_deploy_spec, called with the training env).  The values are
read from the running simulation and checked against params.py before writing, so the file describes exactly what
the policy was trained with:

    joint_names        policy joint order (state joint_pos / joint_vel / actions and the action output)
    motor_index        Unitree motor index (rt/lowcmd, rt/lowstate) of each policy joint
    default_joint_pos  nominal default pose (no encoder offset), policy order
    action_scale       per joint, policy order;  target = default_joint_pos + action_scale * a
    kp, kd             PD gains per joint, policy order (the GEAR-SONIC actuator model)
    effort_limit       Nm, policy order
    observation        actor input layout: groups, term names and sizes, in order
    control_dt         policy period (s)

Only plain Python / json here, so the deploy side can import it without Isaac Lab (motor <-> policy reordering by
name: to_motor / to_policy).
"""
from __future__ import annotations

import json
from typing import Any, Sequence

from . import params as P

SPEC_VERSION = 1


def motor_index(joint_names: Sequence[str]) -> list[int]:
  """Unitree motor index of each joint name (params.JOINT_NAMES is the motor order)."""
  return [P.JOINT_NAMES.index(n) for n in joint_names]


def to_motor(x_policy: Sequence[float], spec: dict) -> list[float]:
  """Policy-order vector -> Unitree motor order."""
  out = [0.0] * len(x_policy)
  for i, m in enumerate(spec["motor_index"]):
    out[m] = x_policy[i]
  return out


def to_policy(x_motor: Sequence[float], spec: dict) -> list[float]:
  """Unitree motor-order vector -> policy order."""
  return [x_motor[m] for m in spec["motor_index"]]


def build_deploy_spec(env, action_name: str = "joint_pos", actor_groups: Sequence[str] = ("state",)) -> dict[str, Any]:
  """Collect the spec from an Isaac Lab env and check it against params.py (raises on any mismatch)."""
  term = env.action_manager.get_term(action_name)
  asset = term._asset
  names = list(term._joint_names)
  ids = list(range(asset.num_joints)) if isinstance(term._joint_ids, slice) else list(term._joint_ids)
  view = asset.root_physx_view
  kp, kd, eff = (t[0].tolist() for t in (view.get_dof_stiffnesses(), view.get_dof_dampings(), view.get_dof_max_forces()))
  scale = term._scale[0].tolist() if not isinstance(term._scale, float) else [term._scale] * len(names)
  nominal = getattr(env, "default_joint_pos_nominal", asset.data.default_joint_pos)[0].tolist()
  spec = {
    "version": SPEC_VERSION,
    "robot": "unitree_g1_29dof_dex3_rev_1_0",
    "joint_names": names,
    "motor_index": motor_index(names),
    "default_joint_pos": [nominal[i] for i in ids],
    "action_scale": scale,
    "kp": [kp[i] for i in ids],
    "kd": [kd[i] for i in ids],
    "effort_limit": [eff[i] for i in ids],
    "observation": {g: [[n, int(d[0])] for n, d in zip(env.observation_manager.active_terms[g],
                                                         env.observation_manager.group_obs_term_dim[g])]
                    for g in actor_groups},
    "control_dt": float(env.step_dt),
  }
  check_spec(spec)
  return spec


def check_spec(spec: dict, tol: float = 1e-4) -> None:
  """Every per-joint value must equal params.py for the named joint."""
  names = spec["joint_names"]
  assert sorted(names) == sorted(P.JOINT_NAMES), "policy joints are not the 29 body joints"
  assert [P.JOINT_NAMES[m] for m in spec["motor_index"]] == names, "motor_index does not match joint_names"
  ref = {"default_joint_pos": P.DEFAULT_JOINT_POS, "action_scale": P.ACTION_SCALE, "kp": P.KP, "kd": P.KD,
         "effort_limit": P.EFFORT_J}
  for key, table in ref.items():
    for n, v in zip(names, spec[key]):
      assert abs(v - table[n]) <= tol * max(1.0, abs(table[n])), f"{key}[{n}] = {v}, params {table[n]}"


def write_deploy_spec(env, path: str, **kw) -> dict[str, Any]:
  spec = build_deploy_spec(env, **kw)
  with open(path, "w") as f:
    json.dump(spec, f, indent=1)
  return spec


def load_deploy_spec(path: str) -> dict[str, Any]:
  with open(path) as f:
    spec = json.load(f)
  assert spec["version"] == SPEC_VERSION, spec["version"]
  check_spec(spec)
  return spec
