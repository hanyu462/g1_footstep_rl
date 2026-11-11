"""Play back a trained footstep policy (Isaac Lab Play env: no observation noise, no randomization).

  python scripts/play.py --checkpoint logs/rsl_rl/g1_goal_conditioned_foot/<run>/model_<it>.pt            # viewer window
  python scripts/play.py --checkpoint <ckpt> --headless --video --video_length 400                       # mp4, no display
  python scripts/play.py --zero_policy --headless --video                                                # env only

Every run with a checkpoint exports the actor next to it (<run>/exported/: policy.pt TorchScript, policy.onnx,
deploy.json); the exported input is the concatenated actor observation [state 93 | goal 11] = 104 values.
Videos go to <run>/videos/play/ (or logs/rsl_rl/zero_policy/videos/).
"""
from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="G1-GoalConditioned-Foot-Flat-Play")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--zero_policy", action="store_true")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=None, help="stop after this many policy steps (default: run until closed)")
parser.add_argument("--video", action="store_true", help="record an mp4 (implies --enable_cameras)")
parser.add_argument("--video_length", type=int, default=400, help="policy steps per video (50 Hz)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
  args.enable_cameras = True
sys.stdout.reconfigure(line_buffering=True)
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embodiments.g1.runner import G1OnPolicyRunner, export_policy, load_cfgs  # noqa: E402


def main():
  assert args.checkpoint or args.zero_policy, "--checkpoint or --zero_policy"
  env_cfg, agent_cfg = load_cfgs(args.task)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.sim.device = args.device
  agent_cfg.device = args.device
  run_dir = os.path.dirname(os.path.abspath(args.checkpoint)) if args.checkpoint else os.path.abspath("logs/rsl_rl/zero_policy")

  env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
  if args.video:
    folder = os.path.join(run_dir, "videos", "play")
    env = gym.wrappers.RecordVideo(env, video_folder=folder, step_trigger=lambda s: s == 0,
                                   video_length=args.video_length, disable_logger=True)
    print(f"[play] recording {args.video_length} steps to {folder}")
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  if args.zero_policy:
    policy = lambda obs: torch.zeros(env.num_envs, env.num_actions, device=env.device)  # noqa: E731
  else:
    runner = G1OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    out = export_policy(runner, os.path.join(run_dir, "exported"))
    print(f"[play] loaded {args.checkpoint}; exported {out['jit']}, {out['onnx']}, {out['deploy']}")

  obs = env.get_observations()
  steps = args.steps or (args.video_length + 1 if args.video else None)
  k = 0
  while app.is_running() and (steps is None or k < steps):
    with torch.inference_mode():
      obs, _, _, _ = env.step(policy(obs))
    k += 1
  if args.video:
    recorder = env.env                       # gym RecordVideo under the RSL-RL wrapper
    if recorder.recording:
      recorder.stop_recording()              # writes the mp4 (env.close() is skipped: it can hang headless)
    print(f"[play] video saved in {os.path.join(run_dir, 'videos', 'play')}")


if __name__ == "__main__":
  # SimulationApp.close() can hang for minutes after a headless run (also after an exception): always exit directly
  try:
    main()
    code = 0
  except BaseException:  # noqa: BLE001
    import traceback
    traceback.print_exc()
    code = 1
  sys.stdout.flush()
  os._exit(code)
