"""Check the PPO pipeline (Isaac Lab + RSL-RL) on the training env with a small number of envs.

  dims         actor input 104 (state 93 + goal 11), critic input 131, action 29, obs groups as configured
  bootstrap    one rollout with forced time-outs and natural falls: the stored reward is r + gamma V(s_t) exactly on
               time-outs and r on terminations / other steps; dones cut the GAE recursion; returns / advantages
               recomputed independently equal the storage
  checkpoint   save -> fresh runner -> load: same actions, normalizer statistics, iteration and optimizer state;
               the deploy spec inside the checkpoint and deploy.json equal the env's spec
  export       TorchScript / ONNX actor = runner inference on the concatenated actor observation
  logging      two learning iterations; TensorBoard tags for reward, success, touchdown error, fall rate,
               episode length, reward terms

  python scripts/inspect_ppo.py [--num_envs 128]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="G1-GoalConditioned-Foot-Flat")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--json", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
sys.stdout.reconfigure(line_buffering=True)
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embodiments.g1 import deploy as D  # noqa: E402
from embodiments.g1.runner import DEPLOY_FILE, G1OnPolicyRunner, export_policy, load_cfgs  # noqa: E402

RES, FAILS = {}, [0]
ROLLOUT = 80    # policy steps in the bootstrap check (1.6 s: long enough for falls with the untrained policy)


def check(name, ok, msg=""):
  FAILS[0] += not ok
  RES[name] = {"ok": bool(ok), "msg": msg}
  print(f"   {'ok' if ok else 'FAIL':4s} {name:36s} {msg}")


def main():
  env_cfg, agent_cfg = load_cfgs(args.task)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.sim.device = args.device
  env_cfg.commands.footstep.debug_vis = False
  agent_cfg.device = args.device
  env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
  base = env.unwrapped
  tmp = tempfile.mkdtemp(prefix="inspect_ppo_")
  def train_cfg():   # RSL-RL pops keys from the dict it is given, so every runner gets a fresh copy
    c = agent_cfg.to_dict()
    c["num_steps_per_env"] = ROLLOUT
    return c
  runner = G1OnPolicyRunner(env, train_cfg(), log_dir=os.path.join(tmp, "run"), device=agent_cfg.device)
  pol, alg = runner.alg.policy, runner.alg
  N = env.num_envs

  # ---- dims ------------------------------------------------------------------------------------------------
  print("\n== dimensions")
  obs = env.get_observations()
  groups = {k: tuple(v.shape) for k, v in obs.items()}
  a_in, c_in = pol.actor[0].in_features, pol.critic[0].in_features
  check("obs groups", groups == {"state": (N, 93), "goal": (N, 11), "critic": (N, 131)}, f"{groups}")
  check("actor input = state + goal", a_in == 104 and runner.cfg["obs_groups"]["policy"] == ["state", "goal"], f"{a_in}")
  check("critic input = critic group", c_in == 131 and runner.cfg["obs_groups"]["critic"] == ["critic"], f"{c_in}")
  check("action 29", pol.actor[-1].out_features == 29 == env.num_actions, f"{pol.actor[-1].out_features}")
  check("time-outs passed to PPO", not base.cfg.is_finite_horizon, "is_finite_horizon False -> extras['time_outs']")

  # ---- bootstrap / GAE ------------------------------------------------------------------------------------
  print(f"\n== bootstrap: one {ROLLOUT}-step rollout, time-outs forced on 16 envs at step 30")
  rec = []
  gamma, lam = alg.gamma, alg.lam
  with torch.inference_mode():
    for k in range(ROLLOUT):
      if k == 30:
        base.episode_length_buf[:16] = base.max_episode_length - 1
      actions = alg.act(obs)
      values = alg.transition.values.clone()
      obs, rewards, dones, extras = env.step(actions)
      alg.process_env_step(obs, rewards, dones, extras)
      rec.append((rewards.clone(), dones.clone(), extras["time_outs"].clone(), values[:, 0]))
    alg.compute_returns(obs)
  st = alg.storage
  r_env = torch.stack([r for r, *_ in rec])
  d = torch.stack([x for _, x, _, _ in rec]).bool()
  to = torch.stack([t for *_, t, _ in rec]).bool()
  v = torch.stack([x for *_, x in rec])
  term = d & ~to
  exp_r = r_env + gamma * v * to.float()
  check("stored reward = r + gamma V on time-outs", torch.allclose(st.rewards[..., 0], exp_r, atol=1e-5),
        f"{int(to.sum())} time-outs, max |diff| {float((st.rewards[..., 0] - exp_r).abs().max()):.1e}")
  check("no bootstrap on terminations", torch.allclose(st.rewards[..., 0][term], r_env[term]),
        f"{int(term.sum())} terminations (falls)")
  check("time-outs forced on envs 0..15 at step 30", bool(to[30, :16].all()), "")
  check("dones = terminations | time-outs", torch.equal(st.dones[..., 0].bool(), d), "")
  # independent GAE
  with torch.inference_mode():
    last_v = pol.evaluate(obs)[:, 0]
  adv = torch.zeros(N, device=v.device)
  ret = torch.zeros_like(v)
  R = st.rewards[..., 0]
  for t in reversed(range(ROLLOUT)):
    nv = last_v if t == ROLLOUT - 1 else v[t + 1]
    nt = 1.0 - d[t].float()
    delta = R[t] + nt * gamma * nv - v[t]
    adv = delta + nt * gamma * lam * adv
    ret[t] = adv + v[t]
  check("GAE returns (independent)", torch.allclose(st.returns[..., 0], ret, atol=1e-4),
        f"max |diff| {float((st.returns[..., 0] - ret).abs().max()):.1e}; gamma {gamma}, lambda {lam}")
  st.clear()

  # ---- checkpoint save / restore ---------------------------------------------------------------------------
  print("\n== checkpoint")
  # (the normalizers were updated during the rollout above)
  runner.current_learning_iteration = 7
  runner._prepare_logging_writer()              # save() needs the logger set up (learn() does this)
  ckpt = os.path.join(tmp, "run", "model_7.pt")
  runner.save(ckpt)
  r2 = G1OnPolicyRunner(env, train_cfg(), log_dir=None, device=agent_cfg.device)
  r2.load(ckpt)
  with torch.inference_mode():
    a1 = runner.get_inference_policy()(obs)
    a2 = r2.get_inference_policy()(obs)
  n1, n2 = pol.actor_obs_normalizer, r2.alg.policy.actor_obs_normalizer
  check("same actions after load", torch.allclose(a1, a2), f"max |diff| {float((a1 - a2).abs().max()):.1e}")
  check("normalizer + iteration restored", torch.allclose(n1._mean, n2._mean) and torch.allclose(n1._std, n2._std)
        and r2.current_learning_iteration == 7, "")
  o1, o2 = alg.optimizer.state_dict(), r2.alg.optimizer.state_dict()
  check("optimizer state restored", o1["param_groups"][0]["lr"] == o2["param_groups"][0]["lr"]
        and len(o1["state"]) == len(o2["state"]), f"lr {o2['param_groups'][0]['lr']}")
  saved = torch.load(ckpt, weights_only=False, map_location="cpu")["infos"]["deploy"]
  spec = D.build_deploy_spec(base, actor_groups=["state", "goal"])
  check("deploy spec inside the checkpoint", saved == spec, f"{len(saved['joint_names'])} joints, obs {saved['observation']}")
  with open(os.path.join(tmp, "run", DEPLOY_FILE)) as f:
    js = json.load(f)
  check("deploy.json next to the checkpoint", js == json.loads(json.dumps(spec)) and D.load_deploy_spec(os.path.join(tmp, "run", DEPLOY_FILE)) == js, "")

  # ---- export ------------------------------------------------------------------------------------------------
  print("\n== export")
  out = export_policy(runner, os.path.join(tmp, "run", "exported"))
  jit = torch.jit.load(out["jit"]).to("cpu")
  x = torch.cat([obs["state"], obs["goal"]], dim=-1).cpu()
  with torch.inference_mode():
    a_jit = jit(x)
  check("TorchScript = inference policy", torch.allclose(a_jit, a1.cpu(), atol=1e-5), f"input {tuple(x.shape)}, max |diff| {float((a_jit - a1.cpu()).abs().max()):.1e}")
  try:
    import onnxruntime as ort
    sess = ort.InferenceSession(out["onnx"])
    a_onnx = torch.tensor(sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0])
    check("ONNX = inference policy", torch.allclose(a_onnx, a1.cpu(), atol=1e-4), f"max |diff| {float((a_onnx - a1.cpu()).abs().max()):.1e}")
  except ImportError:
    print(f"   info ONNX written ({os.path.getsize(out['onnx'])} bytes); onnxruntime not installed, not executed")

  # ---- logging ----------------------------------------------------------------------------------------------
  print("\n== logging: 2 learning iterations")
  cfg2 = agent_cfg.to_dict()
  log_dir = os.path.join(tmp, "learn")
  r3 = G1OnPolicyRunner(env, cfg2, log_dir=log_dir, device=agent_cfg.device)
  r3.learn(num_learning_iterations=2)
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
  ea = EventAccumulator(log_dir)
  ea.Reload()
  tags = set(ea.Tags()["scalars"])
  want = ["Train/mean_reward", "Train/mean_episode_length", "Metrics/footstep/success", "Metrics/footstep/td_err_cm",
          "Episode/fall_rate", "Episode/time_out_rate", "Episode/length_s", "Episode_Reward/foot_target",
          "Episode_Reward/terminated", "Episode_Termination/fell_over", "Loss/value_function", "Policy/mean_noise_std"]
  miss = [t for t in want if t not in tags]
  check("TensorBoard tags", not miss, f"missing {miss}" if miss else f"{len(tags)} scalar tags")
  saved = [f for f in os.listdir(log_dir) if f.startswith("model_")]
  check("checkpoint + deploy.json written by learn", bool(saved) and os.path.exists(os.path.join(log_dir, DEPLOY_FILE)), f"{saved}")

  shutil.rmtree(tmp, ignore_errors=True)
  print(f"\n== {len(RES) - FAILS[0]}/{len(RES)} checks passed")
  if args.json:
    with open(args.json, "w") as f:
      json.dump(RES, f, indent=1)


if __name__ == "__main__":
  # SimulationApp.close() can hang for minutes after a headless run (also after an exception): always exit directly
  try:
    main()
    code = 0 if FAILS[0] == 0 else 1
  except BaseException:  # noqa: BLE001
    import traceback
    traceback.print_exc()
    code = 1
  sys.stdout.flush()
  os._exit(code)
