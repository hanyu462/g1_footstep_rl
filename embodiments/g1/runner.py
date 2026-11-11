"""RSL-RL helpers for G1 tasks in Isaac Lab: config loading from the gym registry, the runner that keeps the
deployment spec with every checkpoint, and policy export.

Import after the Isaac Sim app is running (AppLauncher).
"""
from __future__ import annotations

import importlib
import json
import os

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from . import deploy as D

DEPLOY_FILE = "deploy.json"


def _load(entry: str):
  module, name = entry.split(":")
  return getattr(importlib.import_module(module), name)


def load_cfgs(task: str):
  """(env_cfg, agent_cfg) instances of a registered task."""
  import embodiments.g1.tasks  # noqa: F401  (registers the G1 tasks)
  kw = gym.spec(task).kwargs
  return _load(kw["env_cfg_entry_point"])(), _load(kw["rsl_rl_cfg_entry_point"])()


class G1OnPolicyRunner(OnPolicyRunner):
  """OnPolicyRunner whose checkpoints carry the deployment spec (embodiments/g1/deploy.py).

  The spec is built from the running env once, written to <log_dir>/deploy.json, and stored in every checkpoint
  under infos["deploy"], so a model_<it>.pt is self-contained for sim2sim / real-robot deployment.
  """

  def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"):
    super().__init__(env, train_cfg, log_dir, device)
    self.deploy_spec = D.build_deploy_spec(env.unwrapped, actor_groups=train_cfg["obs_groups"]["policy"])
    if log_dir is not None:
      os.makedirs(log_dir, exist_ok=True)
      with open(os.path.join(log_dir, DEPLOY_FILE), "w") as f:
        json.dump(self.deploy_spec, f, indent=1)

  def save(self, path: str, infos: dict | None = None) -> None:
    super().save(path, {**(infos or {}), "deploy": self.deploy_spec})

  def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
    infos = super().load(path, load_optimizer=load_optimizer, map_location=map_location)
    saved = (infos or {}).get("deploy")
    if saved is not None:
      # the checkpoint must have been trained with the same joints / defaults / scales / gains as this env
      for k in ("joint_names", "motor_index", "observation", "default_joint_pos", "action_scale", "kp", "kd"):
        a, b = saved[k], self.deploy_spec[k]
        same = a == b if k in ("joint_names", "motor_index", "observation") else all(abs(x - y) < 1e-5 for x, y in zip(a, b))
        assert same, f"checkpoint {path} was trained with a different {k} than this env"
    return infos


def export_policy(runner: OnPolicyRunner, out_dir: str) -> dict[str, str]:
  """TorchScript + ONNX actor (observation normalizer included; input = concatenated actor groups) and deploy.json."""
  from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx
  policy = runner.alg.policy
  norm = getattr(policy, "actor_obs_normalizer", None)
  os.makedirs(out_dir, exist_ok=True)
  export_policy_as_jit(policy, normalizer=norm, path=out_dir, filename="policy.pt")
  export_policy_as_onnx(policy, normalizer=norm, path=out_dir, filename="policy.onnx")
  spec_path = os.path.join(out_dir, DEPLOY_FILE)
  with open(spec_path, "w") as f:
    json.dump(getattr(runner, "deploy_spec", None), f, indent=1)
  return {"jit": os.path.join(out_dir, "policy.pt"), "onnx": os.path.join(out_dir, "policy.onnx"), "deploy": spec_path}
