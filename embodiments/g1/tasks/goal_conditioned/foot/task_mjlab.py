"""mjlab registration of G1-GoalConditioned-Foot-Flat (kept until the Isaac Lab migration is finished)."""
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs_mjlab import g1_footstep_env_cfg
from .rl_cfg_mjlab import g1_footstep_ppo_runner_cfg

register_mjlab_task(
  task_id="G1-GoalConditioned-Foot-Flat",
  env_cfg=g1_footstep_env_cfg(),
  play_env_cfg=g1_footstep_env_cfg(play=True),
  rl_cfg=g1_footstep_ppo_runner_cfg(),
  runner_cls=MjlabOnPolicyRunner,
)
