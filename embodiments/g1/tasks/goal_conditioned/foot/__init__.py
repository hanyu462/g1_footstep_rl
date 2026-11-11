"""Foot goal task: lift the commanded foot, land it on the target, keep balance, stand still.

Isaac Lab (gymnasium registry, when Isaac Lab is installed):
    G1-GoalConditioned-Foot-Flat        training env (4096 envs, domain randomization)
    G1-GoalConditioned-Foot-Flat-Play   play / evaluation env (no observation noise or randomization)
The mjlab version (task_mjlab, env_cfgs_mjlab, rl_cfg_mjlab, mdp_mjlab/) is registered with mjlab when mjlab is
installed; it is kept until the migration is signed off.
"""
import importlib.util

if importlib.util.find_spec("isaaclab") is not None:
  import gymnasium as gym

  for _id, _cfg in (("G1-GoalConditioned-Foot-Flat", "G1FootstepEnvCfg"), ("G1-GoalConditioned-Foot-Flat-Play", "G1FootstepEnvCfg_PLAY")):
    gym.register(
      id=_id,
      entry_point="embodiments.g1.env:G1RLEnv",
      disable_env_checker=True,
      kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfgs:{_cfg}",
        "rsl_rl_cfg_entry_point": f"{__name__}.rl_cfg:G1FootstepPPORunnerCfg",
      },
    )

if importlib.util.find_spec("mjlab") is not None:
  from . import task_mjlab  # noqa: F401
