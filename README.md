# G1 Footstep RL

Unitree G1(29 DoF)이 **지정한 발 하나를 목표 상대 위치로 옮겨 딛고, 균형을 유지한 채 멈추는** goal-conditioned 정책을 PPO로 학습합니다.

- GEAR-SONIC이나 motion token은 쓰지 않고, 정책이 29개 관절 위치 목표를 직접 출력합니다.
- 학습 프레임워크는 Isaac Lab 2.3.2 (Isaac Sim 5.1.0)와 RSL-RL PPO입니다. G1 모델은 Isaac Sim의 `Robots/Unitree/G1/g1.usd`이고, actuator·PD·action scale은 GEAR-SONIC 학습 설정과 같습니다(`embodiments/g1/params.py`).
- 처음 버전은 [mjlab](https://github.com/mujocolab/mjlab)(MuJoCo-Warp)으로 만들었습니다. 이전이 끝날 때까지 `*_mjlab` 파일로 남겨 두고, 비교에 씁니다.
- 다른 레포의 코드나 파일은 쓰지 않습니다. `unitree_rl_mjlab`과 GEAR-SONIC은 설정값을 참고만 했습니다.
- 설계 근거와 선행 연구는 [docs/design.md](docs/design.md)에 정리했습니다.

```
a_t ~ π_θ(s_t, g_t)     s_t (93): 관절 위치(default 기준, encoder 편향 포함)·속도, pelvis IMU 각속도와 중력 방향, 직전 action
                        g_t (11): 발 one-hot, 목표−swing발, 목표−stance발, (Δx, Δy), phase
                        a_t (29): 관절 위치 offset (default pose 기준, 관절별 scale) → PD
V_φ(s_t, g_t, z_t)      z_t (27): 시뮬레이터 privileged 정보 (torso IMU, 발 접촉, 몸통 선속도, 발 높이, 접촉력, 목표 오차, step 상태)
```

## 설치

### mjlab 환경 (이전 버전, 비교용)

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e .      # mjlab 1.2.0, mujoco-warp 3.5.0 (+ rsl-rl-lib, torch, tyro)
```

### Isaac Lab 환경 (학습용)

GEAR-SONIC과 같은 Isaac Lab 2.3.2 (Isaac Sim 5.1.0, Python 3.11, torch 2.7.0 cu128) 환경입니다. 레포 안의 `.venv_isaaclab`에 설치하고, Isaac Lab 소스는 `~/IsaacLab`(tag v2.3.2)에 둡니다. 아래 명령은 레포 루트에서 실행합니다.

```bash
uv venv --seed --python 3.11 .venv_isaaclab
PY=.venv_isaaclab/bin/python
uv pip install --python $PY "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
uv pip install --python $PY torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
git clone --depth 1 --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
# 설치 중 걸린 두 가지: flatdict 빌드(최신 setuptools에 pkg_resources 없음), EULA 대화형 입력
uv pip install --python $PY "setuptools<81" && uv pip install --python $PY --no-build-isolation flatdict==4.0.1
export OMNI_KIT_ACCEPT_EULA=YES          # NVIDIA Omniverse EULA 동의
source .venv_isaaclab/bin/activate && (cd ~/IsaacLab && ./isaaclab.sh --install rsl_rl)
uv pip install --python $PY "tensordict==0.9.1"   # 최신 tensordict는 torch 2.7과 맞지 않음
# 마지막에 Isaac Sim / Isaac Lab이 고정한 버전으로 되돌립니다. 앞 단계들이 numpy 2, pillow 13 등을 끌어오면
# 학습은 돌아도 렌더링(--video)이 replicator TypeError로 실패합니다.
uv pip install --python $PY "numpy==1.26.0" "pillow==11.3.0" "packaging==23.0" "markupsafe==2.1.3" "filelock==3.13.1" \
  "fsspec==2024.6.1" "networkx==3.3" "sympy==1.13.3" "typing-extensions==4.12.2" "psutil==5.9.8" "starlette==0.49.1"
uv pip check --python $PY   # 남는 5건(pytest, wheel, ipython, fastapi, pyvers)은 상류 패키지끼리의 충돌로 실행과 무관
```

설치 확인(G1 기본 예제, headless 3 iteration):
```bash
export OMNI_KIT_ACCEPT_EULA=YES && source .venv_isaaclab/bin/activate
cd ~/IsaacLab && ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-G1-v0 --headless --num_envs 256 --max_iterations 3
```

G1 모델(Isaac Sim USD, 약 150 MB)을 `assets/isaac/`에 한 번 내려받습니다. asset 서버가 연결당 느려서 범위를 나눠 병렬로 받습니다. 받지 않으면 실행할 때마다 서버에서 읽어 오므로 매우 느립니다.
```bash
python scripts/fetch_assets.py
```

G1 정의 검증 (질량, 관절 게인·한계, 손가락 고정, 처짐, 자유 기립, 약 10초):
```bash
python scripts/inspect_robot.py
```
G1 state 검증 (차원, 관절 순서, encoder 편향, 노이즈, 발 높이·접촉·[L, R] 순서, IMU 유한차분, 26개 항목):
```bash
python scripts/inspect_state.py
```
발 목표(goal) 검증 (목표 배치, 시작 위치 기록 시점, goal 관측값, 골반 고정 로봇의 scripted step 4그룹으로 들기·착지·2차 swing·stance 들기·성공 판정, time-out 리셋과 지표, 22개 항목):
```bash
python scripts/inspect_goal.py
```
G1 action과 배포 spec 검증 (action 차원·순서·손가락 제외, 게인, scale, 정책 slot → 모터 index, zero/무작위 action의 PD 목표, encoder 편향 1회 반영, deploy.json 왕복, 22개 항목):
```bash
python scripts/inspect_action.py [--out deploy.json]
```
발 task 환경 검증 (리셋 직후 링크 위치 갱신과 첫 관측 goal, reward 16항, 종료 조건; free 20개 + 골반 고정 scripted step 13개 항목):
```bash
python scripts/inspect_task.py [--part free|pinned|all] [--stats isaac_stats.json]
```
mjlab 버전과 비교 (mjlab venv에서; reward 식은 같은 상태에서 차이 0, 가중치·종료 조건, zero action rollout 통계):
```bash
.venv/bin/python scripts/compare_mjlab.py --isaac isaac_stats.json
```
Isaac Sim은 headless 실행 후 `close()`에서 수 분간 멈추는 경우가 있습니다. 그래서 검증 스크립트는 결과를 출력한 뒤 바로 종료합니다.

## 서버에서 학습 시작 전 점검

clone한 서버에서 위 설치를 마친 뒤 아래 순서로 점검합니다. 각 단계는 GPU를 1–5분 씁니다.

1. `python scripts/fetch_assets.py`: G1 USD 내려받기 (`assets/isaac/`, git에 포함하지 않음)
2. `python scripts/inspect_robot.py`, `inspect_state.py`, `inspect_goal.py`, `inspect_action.py`, `inspect_task.py`: 1–5단계 검증
   - `inspect_action.py`의 손가락 유지 항목은 무작위 동작이라 가끔 0.02 rad를 넘습니다. 기록된 문제이고, 학습에는 영향이 없습니다.
3. `python scripts/inspect_ppo.py`: PPO 파이프라인 (18개 항목)
4. `python scripts/train.py --num_envs 256 --max_iterations 50 --run_name smoke`: smoke 학습
5. 메인 학습

## 실행 (Isaac Lab)

`export OMNI_KIT_ACCEPT_EULA=YES && source .venv_isaaclab/bin/activate` 후 레포 루트에서 실행합니다. 모든 명령은 GPU를 쓰고, 기본은 `cuda:0`입니다. 다른 GPU는 `--device cuda:1`로 지정합니다.

| 단계 | 명령 |
|---|---|
| PPO 파이프라인 검증 (차원, time-out bootstrap·GAE, 체크포인트, export, TensorBoard; 18개 항목) | `python scripts/inspect_ppo.py` |
| smoke 학습 | `python scripts/train.py --num_envs 256 --max_iterations 50 --run_name smoke` |
| 학습 | `python scripts/train.py --num_envs 4096` |
| 이어서 학습 | `python scripts/train.py --resume logs/rsl_rl/g1_goal_conditioned_foot/<run>/model_<it>.pt` |
| 재생 (화면) | `python scripts/play.py --checkpoint <run>/model_<it>.pt` |
| 재생 (화면 없이 mp4) | `python scripts/play.py --checkpoint <run>/model_<it>.pt --headless --video --video_length 400` |
| 정밀도 평가 (goal grid) | `python scripts/evaluate.py --checkpoint <run>/model_<it>.pt --num_envs 1000` |
| TensorBoard | `tensorboard --logdir logs/rsl_rl` |

학습 출력은 `logs/rsl_rl/g1_goal_conditioned_foot/<날짜_시간>[_<run_name>]/`에 저장됩니다.
- `model_<it>.pt`: 체크포인트. 배포 spec이 `infos["deploy"]`에 들어 있습니다.
- `deploy.json`: 관절 이름과 모터 index, 원래 default 자세, action scale, Kp/Kd, actor 관측 구성, 제어 주기
- `params/env.yaml`, `params/agent.yaml`, TensorBoard 이벤트 파일
- 체크포인트를 불러올 때 spec이 현재 환경과 다르면 멈춥니다.

`play.py`는 체크포인트 옆 `exported/`에 아래 세 파일을 씁니다.
- `policy.pt`: TorchScript, 관측 정규화 포함. 입력은 [state 93 | goal 11] = 104
- `policy.onnx`
- `deploy.json`

TensorBoard 주요 항목:

| 항목 | 태그 |
|---|---|
| reward | `Train/mean_reward`, `Episode_Reward/<항>` |
| 성공률 | `Metrics/footstep/success` |
| 착지 오차 | `Metrics/footstep/td_err_cm` (착지하지 못하면 30) |
| 넘어짐 비율 | `Episode/fall_rate` (time-out 제외 종료), `Episode/time_out_rate` |
| episode 길이 | `Train/mean_episode_length` (step), `Episode/length_s` |
| 종료 사유 | `Episode_Termination/<항>` |

## 실행 (mjlab 버전, 이전이 끝날 때까지 참고용)

| 단계 | 명령 |
|---|---|
| 등록된 task 확인 | `.venv/bin/python -c "import mjlab.tasks, embodiments.g1.tasks; from mjlab.tasks.registry import list_tasks; print(list_tasks())"` |
| 파이프라인 점검 (zero action) | `.venv/bin/python scripts/evaluate_mjlab.py G1-GoalConditioned-Foot-Flat --zero-policy True --num-envs 100 --duration-s 1` |
| 학습 | `.venv/bin/python scripts/train_mjlab.py G1-GoalConditioned-Foot-Flat --env.scene.num-envs 4096` |
| 짧은 smoke 학습 | `.venv/bin/python scripts/train_mjlab.py G1-GoalConditioned-Foot-Flat --env.scene.num-envs 256 --agent.max-iterations 20` |
| 이어서 학습 | `.venv/bin/python scripts/train_mjlab.py G1-GoalConditioned-Foot-Flat --resume logs/rsl_rl/g1_goal_conditioned_foot/<run>/model_<it>.pt` |
| GPU 지정 | `--device cuda:1` (train / play / evaluate 공통) |
| 재생 (viewer) | `.venv/bin/python scripts/play_mjlab.py G1-GoalConditioned-Foot-Flat --checkpoint-file logs/rsl_rl/g1_goal_conditioned_foot/<run>/model_<it>.pt` (화면 없으면 viser 웹 viewer) |
| 관측 구성 검증 | `.venv/bin/python scripts/inspect_observations_mjlab.py G1-GoalConditioned-Foot-Flat` |
| 양발 [L, R] 순서 검증 | `.venv/bin/python scripts/inspect_observations_mjlab.py G1-GoalConditioned-Foot-Flat --check-feet-order True` |
| 정밀도 평가 | `.venv/bin/python scripts/evaluate_mjlab.py G1-GoalConditioned-Foot-Flat --checkpoint-file logs/rsl_rl/g1_goal_conditioned_foot/<run>/model_<it>.pt --num-envs 1000` |

- 학습 log와 checkpoint는 `logs/rsl_rl/g1_goal_conditioned_foot/<날짜_시간>/`에 저장됩니다(`model_<iter>.pt`, `params/env.yaml`, `params/agent.yaml`).
- 학습 중 episode 지표는 `Metrics/footstep/...`로 기록됩니다(tensorboard 또는 wandb): `success`, `td_err_cm`, `clearance_cm`, `stance_drift_cm`, `lifted`, `touched`.
- 평가 스크립트는 {L, R} × Δx {10, 15, 20, 25, 30} cm × Δy {−10, −5, 0, +5, +10} cm grid에서 성공률, 착지 오차, 들어올림 높이, stance 발 drift, 2차 swing, 넘어짐을 발·Δx·Δy별로 출력합니다. 결과는 checkpoint 옆 `eval_grid.json`에 저장됩니다.
- `--env.*`, `--agent.*`로 설정을 덮어쓸 수 있습니다(tyro). 예: `--env.commands.footstep.ranges.dx 0.1 0.2`

## 구조

```
embodiments/
└── g1/                                Unitree G1 (29 DoF)
    ├── params.py                      시뮬레이터와 무관한 숫자: 모터, PD 게인, 한계, action scale, 기본 자세
    │                                  (GEAR-SONIC 학습 설정 기준; sim2sim·실물 배포에서도 사용)
    ├── robot.py                       Isaac Lab ArticulationCfg (Isaac Sim G1 USD + params, 왼손 palm 질량 보정)
    ├── state.py                       [Isaac Lab] G1 state: actor state 93 (관절, pelvis IMU, 직전 action),
    │                                  G1 공통 privileged 19, 발 접촉 센서, encoder 편향 event (모든 G1 task가 공유)
    ├── state_mjlab.py                 mjlab 버전 state (이전이 끝나면 삭제)
    ├── action.py                      [Isaac Lab] action 29 (몸통 관절 위치, 시뮬레이터 순서, 손가락 제외)
    ├── env.py                         [Isaac Lab] G1RLEnv: 리셋 직후 링크 pose 갱신 + command post_reset
    ├── runner.py                      [Isaac Lab] 등록된 task 설정 읽기, 체크포인트에 배포 spec을 넣는 runner, 정책 export
    ├── deploy.py                      배포 spec (deploy.json): 관절 이름·모터 index·default·scale·게인; Isaac Lab 없이 import 가능
    └── tasks/
        ├── __init__.py                G1 task 등록 진입점 (scripts가 import)
        └── goal_conditioned/          지정한 신체 부위를 목표 점으로 옮기는 task 묶음
            ├── foot/                  발: 들어서 목표에 착지, 균형 유지, 정지 → G1-GoalConditioned-Foot-Flat
                ├── __init__.py        gym 등록 (Isaac Lab): G1-GoalConditioned-Foot-Flat, -Play; mjlab 등록 (mjlab 설치 시)
                ├── rl_cfg.py          [Isaac Lab] PPO 설정 (mjlab 버전과 같은 값)
                ├── env_cfgs.py        [Isaac Lab] 환경: 장면·센서, 관측, action, command, event, reward 16항, 종료 조건
                ├── mdp/               [Isaac Lab]
                │   ├── commands.py      목표 샘플링, 발 들기·착지·2차 swing·성공 판정, 지표, post_reset
                │   ├── observations.py  goal (target 8 + phase 3), (critic 전용) swing 목표 오차·step 상태
                │   ├── rewards.py       목표 추종, 착지 정확도, clearance, stance 고정, settle, 자기 충돌 등 (torch만 import)
                │   └── events.py        발 마찰 (env별 한 값, 양발 공유)
                ├── task_mjlab.py      mjlab task 등록                         ┐
                ├── env_cfgs_mjlab.py  mjlab 환경 (센서, 관측, action, event,  │ mjlab 버전
                │                      reward 가중치, 종료 조건)              │ (이전이 끝나면 삭제)
                ├── rl_cfg_mjlab.py    PPO 설정                                │
                └── mdp_mjlab/         commands, rewards, observations         ┘
            ├── hand/                  (예정) 손을 목표 점으로, 전신 균형 유지
            └── whole_body/            (예정) 여러 부위 목표를 동시에
assets/isaac/                          scripts/fetch_assets.py가 받은 Isaac Sim USD (레포에 포함하지 않음)
scripts/
├── fetch_assets.py                    [Isaac Lab] G1 USD와 하위 레이어 다운로드
├── inspect_robot.py                   [Isaac Lab] robot.py를 params.py와 대조 검증
├── inspect_state.py                   [Isaac Lab] state.py 검증 (최소 환경)
├── inspect_goal.py                    [Isaac Lab] 발 목표 command·goal 관측 검증 (최소 RL 환경)
├── inspect_action.py                  [Isaac Lab] action·배포 spec 검증
├── inspect_task.py                    [Isaac Lab] 발 task 환경 검증 (리셋, reward, 종료)
├── compare_mjlab.py                   [mjlab venv] Isaac Lab reward/종료와 mjlab 버전 비교
├── inspect_ppo.py                     [Isaac Lab] PPO 파이프라인 검증
├── train.py                           [Isaac Lab] PPO 학습 (headless, RSL-RL)
├── play.py                            [Isaac Lab] 재생 (화면 또는 mp4), 정책 export
├── evaluate.py                        [Isaac Lab] goal grid 정밀도 평가
├── train_mjlab.py, play_mjlab.py, evaluate_mjlab.py, inspect_observations_mjlab.py   mjlab 버전
```

- `hand/`, `whole_body/`는 자리만 만들어 둔 상태입니다(task 미등록). 구현할 때 발·손에 공통인 로직(목표 샘플링, 목표 오차)은 `goal_conditioned/mdp/`로 뺍니다.
- 다른 로봇은 `embodiments/<robot>/tasks/...`를 같은 형태로 만들고, scripts의 `import embodiments.g1.tasks` 옆에 import를 추가합니다.
