# G1 Footstep RL 설계 (Stage 1: 평지, 한 발, 한 걸음)

G1 모델, 시뮬레이션, PPO는 mjlab 라이브러리(`mjlab.asset_zoo.robots.unitree_g1`, MuJoCo-Warp, RSL-RL)를 그대로 씁니다. 다른 레포 코드에는 의존하지 않습니다.

## 1. 과제

- **목표**: 서 있는 G1이 지정된 발 하나를, 그 발의 시작 위치 기준 (Δx 앞, Δy 왼쪽)으로 옮겨 딛고, 착지 후 두 발로 멈춥니다.
- **좌표계**: Δx·Δy는 episode 시작 시 몸통 heading 기준입니다.
- **episode**: 한 episode에 명령 하나이고 길이는 4 s입니다(정책 50 Hz, 200 step).
- **성공 판정** (`FootstepCommand.success`): 아래 조건을 모두 만족해야 합니다.
  1. swing 발이 공중에 떴다가(접촉 없음 + 시작 높이보다 1.5 cm 이상, 3 step 이상) 다시 착지했습니다.
  2. 착지 위치 오차(xy)가 3 cm 미만입니다.
  3. 착지 전 최대 높이(clearance)가 5 cm 이상입니다.
  4. 착지 후 swing 발이 다시 뜨지 않았고(2차 swing 없음), stance 발이 뜬 적이 없습니다.
  5. 착지 후 두 발 접촉 상태에서 몸통 속도가 0.1 m/s 미만인 상태가 0.5 s 이상 지속됐습니다.
  - 평가에서는 여기에 넘어짐 없음을 추가합니다(몸통 높이 0.45 m 이상, 기울기 60° 이하).
- 이 판정 기준은 이전 token-policy 프로젝트의 CEM teacher 합격 기준과 같은 수준입니다(착지 < 3 cm, clearance ≥ 5 cm, 2차 swing 없음, 멈춤).

## 2. State / Goal / Action

Asymmetric actor-critic: actor는 실물 G1에서 측정 가능한 state와 goal만, critic은 여기에 시뮬레이터의 privileged 정보를 더 봅니다. critic은 학습 중에만 쓰고 배포에는 actor만 씁니다. 관측 그룹은 `state`(93), `goal`(11), `critic`(131)이고, `rl_cfg.obs_groups`가 actor ← (`state`, `goal`) = 104, critic ← (`critic`) = 131로 연결합니다.

**Actor state s_t (93)**: GEAR-SONIC 배포(시뮬과 실물 일치)의 정책 입력과 같은 범위(관절, pelvis IMU, 직전 action)로 맞췄습니다. `embodiments/g1/state.py`(Isaac Lab)에 정의되어 있고, 모든 G1 task가 공유합니다. 아래 표는 mjlab 버전(`state_mjlab.py`) 기준이며, Isaac Lab 버전과의 차이는 표 아래 "Isaac Lab 버전 state"에 정리했습니다. 노이즈는 unitree G1 velocity task와 같은 크기의 균등 노이즈입니다.

| 항목 | 차원 | 함수 / 출처 | 좌표계 | 노이즈 |
|---|---:|---|---|---|
| 관절 위치 q + b_enc − q_default | 29 | `joint_pos_rel(biased=True)`: encoder 읽기값(편향 포함). critic 쪽은 편향 없는 실제 값 | 관절 공간 | ±0.01 + encoder bias ±0.015 (env별 고정) |
| 관절 속도 q̇ | 29 | `joint_vel_rel` | 관절 공간 | ±1.5 |
| pelvis 각속도 | 3 | `builtin_sensor("robot/imu_ang_vel")`: `imu_in_pelvis` site의 gyro | pelvis body (site 회전 없음) | ±0.2 |
| pelvis 중력 방향 | 3 | `projected_gravity` (root = pelvis) | pelvis body | ±0.05 |
| 직전 action a_{t−1} | 29 | `last_action` (scale 전) | action 공간 | 없음 |

**Goal (11)**: state와 분리한 그룹입니다. actor와 critic 모두 받습니다.
- target 8: 발 one-hot, 목표−swing발 xy, 목표−stance발 xy(현재 heading 기준), 명령 (Δx, Δy)
- phase 3: 명령 후 시간/2 s, sin, cos

**Critic (131)** = 노이즈 없는 state 93 + goal 11 + privileged 27. 이 중 G1 공통 privileged 19(torso 각속도·중력 방향, 접촉, 선속도, 발 높이, 접촉력)는 `g1/state.py`, 발 task 전용 8(목표 오차, step 상태)은 task에 있습니다.

**torso IMU를 actor에서 뺀 이유**: GEAR-SONIC 배포 코드는 `rt/secondary_imu`를 구독하지만 로그에만 쓰고 정책 입력으로 쓰지 않습니다. 그래서 sim-to-real 대응이 검증되지 않았습니다(GEAR-SONIC 시뮬의 torso 값도 `cvel[3:6]` = 선속도라 gyro가 아님). 실물에서 장착 body·축·지연을 확인한 뒤 actor에 다시 넣습니다.

| privileged 항목 | 차원 | 내용 |
|---|---:|---|
| torso_ang_vel, torso_projected_gravity | 6 | torso_link 좌표계 (actor에서 이동) |
| feet_contact | 2 | 정확한 양발 접촉 (실기 접촉 추정기가 생기기 전까지 critic 전용) |
| pelvis_lin_vel | 3 | `imu_in_pelvis` velocimeter, pelvis 좌표계 |
| feet_height | 2 | 양발 높이 (world z). 평지에서만 지면 기준 높이와 같음. 불규칙 지형에서는 발 아래 지면 기준 높이(ray cast 등)로 바꿔야 함 |
| swing_error | 3 | 목표 − swing발 (world) |
| step_state | 5 | lifted, touched, 착지 오차, 최대 높이, stance drift |
| feet_forces | 6 | 양발 접촉력 (world, sign·log1p 압축) |

**양발 순서**: 모든 양발 항목(접촉, 높이, 접촉력, 명령의 발 site)은 [left, right] 순서입니다. `scripts/inspect_observations.py --check-feet-order True`로 검증했습니다. 한쪽 다리를 굽혀 들면 그쪽 슬롯만 접촉 0, 높이 상승으로 바뀝니다.

**실기에서 확인할 것**
- torso IMU(`rt/secondary_imu`): 실제 발행 여부, torso_link 대비 축 방향, 지연, 편향. 지연과 편향은 아직 모델링하지 않았습니다.
- 관절 encoder offset: env별 고정 편향 ±0.015 rad로 모델링했습니다. mjlab은 PD 목표에서 같은 편향을 빼서, 실물처럼 "편향된 encoder 기준으로 제어"됩니다.

**Isaac Lab 버전 state** (`embodiments/g1/state.py`, 이전 2단계). 항목과 차원(actor 93, G1 공통 privileged 19)은 같고, 아래가 다릅니다.

| 항목 | mjlab 버전 | Isaac Lab 버전 |
|---|---|---|
| 관절 순서 | mjlab 모델 순서 | 시뮬레이터(Isaac Lab) 순서 = GEAR-SONIC 정책 순서. 몸통 29개가 articulation 0–28번, 손가락 14개는 그 뒤 |
| encoder 편향 | ±0.015 rad, 관측에 +b, PD 목표에서 −b (mjlab `encoder_bias`) | **±0.01 rad**, `default_joint_pos`를 −b만큼 옮김 (GEAR-SONIC 방식). 관측 `q + b − q0`, PD 목표 `q0 − b + S·a`가 함께 성립 |
| critic 관절 위치 | 편향 없는 실제 값 | actor와 같은 encoder 값 (노이즈만 없음) |
| pelvis 각속도·선속도 | mjlab site 센서 | pelvis 링크 운동학으로 계산. 선속도는 IMU 점(`params.PELVIS_IMU_POS`)의 속도 `v + ω × r` |
| 발 기준점 | mjlab `left_foot`/`right_foot` site (0.04, 0, −0.037) | `params.FOOT_SOLE_POS` = (0.05, 0, −0.036): ankle_roll_link의 발바닥 접촉구 4개 중심, 발바닥 면 |
| 접촉 | 지면과의 접촉 (subtree) | ankle_roll_link의 지면 collider 접촉만 필터 (`force_matrix_w`), 1 N 초과면 접촉 |

- **배포 시 관절 순서**: 정책 입출력은 시뮬레이터 순서입니다. 체크포인트와 함께 관절 이름 29개와 원래 default 자세를 저장하고, 배포 쪽에서 `params.JOINT_NAMES`(Unitree 모터 순서)와 이름으로 대응시킵니다. 숫자 인덱스 표를 따로 두지 않습니다.
- **encoder 편향**: `default_joint_pos`가 env별로 옮겨져 있으므로, 원래 default 자세가 필요한 곳(보상, 배포)은 `params.DEFAULT_JOINT_POS`를 씁니다. action term은 생성 시 default를 복사해 두므로, startup event가 action offset도 함께 갱신합니다. 리셋 자세도 옮겨진 default에서 시작하는데, 실물에서 encoder 기준 default로 세운 자세와 같습니다.
- 검증: `scripts/inspect_state.py`. 최소 환경(state 항목, 몸통 29개 관절 위치 action, encoder 편향 event)에서 26개 항목을 확인합니다.
  - 차원과 관절 순서
  - encoder 편향 범위, action offset, a = 0일 때 PD 목표
  - 노이즈 범위
  - 서 있을 때 발바닥 높이 0 mm, 지면 힘 합 ≈ mg
  - 한 발 들기로 [L, R] 순서
  - IMU 항목과 한 physics step 유한차분 비교 (자유 비행, 무작위 속도)

**Isaac Lab 버전 goal** (`foot/mdp/commands.py`, `observations.py`, 이전 3단계). 목표 샘플링, 판정 규칙, goal 11과 task privileged 8은 mjlab 버전과 같습니다. 다른 점은 아래와 같습니다.
- 발 위치는 발바닥 기준점(`params.FOOT_SOLE_POS`)이고, 접촉은 state의 지면 접촉(1 N 초과)과 같은 기준입니다.
- **시작 위치는 리셋 다음 step에 기록합니다.** Isaac Lab은 리셋 직후 forward kinematics를 하지 않아서, 리셋 step에 읽는 링크 위치는 이전 episode 값입니다. 그 한 step 동안 goal의 목표 상대 위치(target 2–5)와 critic의 swing 오차·stance drift는 0입니다. 발 one-hot, (Δx, Δy), phase는 그대로 나갑니다.
- 검증: `scripts/inspect_goal.py`, 22개 항목. 골반을 고정한 로봇으로 아래를 확인합니다.
  - 목표 = 시작 + R(yaw0)·Δ, stance 발과의 횡 간격 ≥ 10 cm, heading 기준 goal 값
  - scripted step 4그룹 각각의 판정
    - 정상: 성공 ⇔ 착지 오차 < 3 cm
    - 같은 발을 다시 듦: 2차 swing
    - stance 발을 듦: stance 들기
    - 10 cm 앞 목표: 성공 없음
  - time-out 리셋 후 flag 초기화와 지표 기록

**Action**: 29개 관절 위치 offset × 관절별 scale(0.25 × effort/stiffness) + default pose → PD

**Isaac Lab 버전 action** (`embodiments/g1/action.py`, 이전 4단계)
- 몸통 29개 관절만 다룹니다. 순서는 시뮬레이터 순서(state와 같음)이고, Dex3 손가락 14개는 action에서 빠지며 PD로 0에 고정됩니다. clip은 없습니다.
- PD 목표: `q_target = (q_default − b) + scale · a`. b는 2단계의 env별 encoder 편향이며, startup event가 action offset에 한 번만 반영합니다.
- 실물에서는 모터 PD가 편향된 encoder로 직접 돌므로, 배포는 원래 default로 `q_default + scale · a`를 명령합니다.
- **배포 spec** (`embodiments/g1/deploy.py`): 체크포인트 옆에 `deploy.json`으로 저장합니다(학습 스크립트 연결은 6단계).
  - 내용: 관절 이름(정책 순서), Unitree 모터 index, 원래 default 자세, action scale, Kp/Kd, 토크 한계, actor 관측 구성, 제어 주기
  - 값은 실행 중인 시뮬레이터에서 읽고 `params.py`와 대조한 뒤 저장합니다.
  - 배포 쪽은 Isaac Lab 없이 `load_deploy_spec`, `to_motor`, `to_policy`로 이름 기준 재정렬을 합니다.
- 검증: `scripts/inspect_action.py`, 골반 고정 로봇으로 22개 항목.
  - action 차원과 순서, 손가락 제외
  - PD 게인, action scale (params 값과 PhysX 값 `0.25·effort/kp`)
  - env별 one-hot action으로 정책 slot → 관절 → 모터 index 대응
  - zero action과 무작위 action의 PD 목표 `q0 − b + S·a` (오차 0)
  - encoder 편향이 정확히 한 번 반영됨 (`target − (q0 + S·a) = −b`), state와 같은 b
  - `last_action` 관측, 손가락 목표 0과 유지
  - `deploy.json` 저장·읽기, spec만으로 계산한 모터 순서 목표 = 시뮬레이터 목표 + b

**PPO 학습 전에 해결하거나 영향을 확인할 것**
- **리셋 직후 1 step의 goal 0 (3단계에서 발견) → 5단계에서 해결**: `embodiments/g1/env.py`의 `G1RLEnv` 참고.
  - 리셋 뒤 PhysX 기구학을 갱신합니다(`update_articulations_kinematic`, `sim.forward()`의 물리 부분).
  - 캐시된 링크 pose도 무효화한 뒤, command의 `post_reset`이 그 step에 바로 시작 위치를 기록합니다.
  - 확인한 사실
    - 갱신 전에 읽은 다리 링크는 새 자세에서 중앙값 115 cm 떨어져 있습니다. pelvis는 root라 바로 맞습니다. 따라서 갱신이 필요합니다.
    - 갱신 중 physics step 수 0, 관절값 변화 0이고, pelvis는 기록한 root pose와 일치합니다.
    - 최초 reset, 넘어짐 종료 reset, time-out reset 모두 첫 관측부터 goal이 정확합니다(목표 − swing = Δ, 오차 < 1e-6). 리셋 step의 발 위치와 다음 step의 차이는 4 mm 이내입니다.
  - `scene.update(0)`는 시간 0을 더할 뿐이라 캐시를 갱신하지 못해서 쓰지 않습니다. `sim.forward()`는 fabric을 끄면 기구학 갱신을 건너뛰므로 PhysX 함수를 직접 부릅니다.
- **손가락이 완전히 고정되지 않음**: 기준 모델(GEAR-SONIC)은 손가락이 고정 관절이고, 여기서는 PD(Kp 20)로 0을 유지합니다.
  - 다리·허리만 무작위로 움직이면 0.012 rad 이내입니다.
  - 팔을 크게 무작위로 휘두르면 손이 몸에 부딪혀 0.15–0.5 rad까지 밀립니다. 자기 충돌을 끄면 N(0, 2) action에서도 0.05 rad 이내이므로, PD가 약해서가 아니라 접촉 때문입니다.
  - 발 task에서는 손이 몸에 닿는 일이 드물어 그대로 둡니다. 손 task(`hand/`)에서는 손가락을 고정 관절로 바꾸는 방안을 검토합니다.


**로봇 모델**: 기준은 GEAR-SONIC의 **학습** 설정입니다. 이 설정으로 학습한 정책이 실물에서 그대로 동작했습니다. GEAR-SONIC sim2sim용 MuJoCo 모델(`g1_29dof_with_hand.xml`, 36.17 kg, torso +19 mm)은 학습 모델과 다른 별도 모델이라 기준으로 쓰지 않습니다.

| | mjlab G1 (mjlab 버전에서 사용) | GEAR-SONIC 학습 (기준) | Isaac Lab 버전 (`embodiments/g1/robot.py`) |
|---|---|---|---|
| 모델 | Unitree rev_1_0, 손 없음 | rev_1_0 + Dex3 손 (손가락 고정) | Isaac Sim `Robots/Unitree/G1/g1.usd` (rev_1_0 + Dex3) |
| 총 질량 | 33.34 kg | 34.39 kg | 34.39 kg (왼손 palm 질량 보정 후) |
| 몸통 링크 질량·위치 | 기준과 같음 (손 제외) | — | 46개 링크 질량 일치, 몸통 30개 링크 위치 < 1 mm, 손 링크 약 3 mm |
| hip pitch 모터 | 7520_14 (Kp 40.2, 88 Nm) | 7520_22 (Kp 99.1, 139 Nm) | 7520_22 |
| 그 외 actuator | 기준과 같음 | 모터별 armature, Kp = J·ω², Kd = 2ζJω (10 Hz, ζ = 2) | 29개 관절 모두 기준과 일치 |
| 손가락 | 없음 | 고정 | 14개 관절을 PD(Kp 20)로 0에 고정 (오차 < 0.01 rad) |
| 기본 자세 | −0.312 / 0.669 / −0.363 | 같음 | 같음 |

- 숫자(모터, 게인, 한계, action scale, 기본 자세)는 시뮬레이터와 무관한 `embodiments/g1/params.py`에 두고, `robot.py`가 이를 Isaac Lab `ArticulationCfg`로 만듭니다. 나중에 MuJoCo sim2sim이나 실물 배포에서도 같은 `params.py`를 씁니다.
- Isaac Sim USD에는 왼손 palm 질량이 지정되어 있지 않습니다. 그래서 PhysX가 collider에서 0.215 kg를 계산합니다. `robot.py`가 생성 시 오른손 palm 값(0.373 kg, COM, 관성)을 y 대칭으로 넣습니다.
- 골반을 고정하고 기본 자세를 PD로 유지하면 허리 pitch가 약 0.39 rad 처집니다. 10 Hz 게인이 상체 무게에 비해 약하기 때문이며, 같은 게인의 MuJoCo 모델에서도 0.21–0.35 rad가 나옵니다. 균형 제어 없이 자유롭게 세우면 약 1.2 s 뒤 넘어지는데, mjlab G1도 정책 출력 0이면 모두 넘어집니다.
- 검증: `scripts/inspect_robot.py` (질량, 관절별 게인·한계, 손가락 고정, 처짐, 자유 기립)

`scripts/inspect_observations.py`로 그룹별 항목·차원·노이즈, IMU 좌표계(body 운동학과의 차이), actor/critic 입력 차원을 검증할 수 있습니다.

## 3. Reward

| 항목 | 식 | 가중치 | 의도 |
|---|---|---|---|
| foot_target | exp(−100·‖p_sw,xy − p\*‖²) | +5 | 목표 추종. 3 cm에서 0.91, 5 cm에서 0.78, 10 cm에서 0.37 |
| touchdown | 착지 후 매 step exp(−100·e_TD²) | +5 | 순간 통과가 아니라 실제 착지점을 최적화. 착지 오차를 고정해 유지 |
| swing_clearance | 착지 전 공중일 때 0.5 + 0.5·min(h/8 cm, 1) | +1 | 발을 들어 옮기기(끌기 방지), 높이 확보 |
| second_swing | 착지 후 swing 발이 다시 뜸 | −2 | 한 걸음만 |
| stance_lift | stance 발이 뜸 | −2 | 지정한 발만 움직임 |
| stance_drift | ‖p_st,xy − p_st0‖² | −20 | stance 발 고정 |
| foot_slip | 접촉 중인 발의 ‖v_xy‖² 합 | −0.5 | 미끄러짐, 끌기 방지 |
| settle | 착지 + 0.3 s 이후 ‖v_base,xy‖² + 0.1·ω_z² | −2 | 착지 후 정지 |
| double_support | 착지 후 두 발 접촉 | +0.5 | 새 stance로 마무리 |
| upright | 중력 방향 xy² (pelvis) | −1 | 균형 |
| posture | 허리·팔 관절 ‖q − q_default‖² | −0.1 | 상체를 쓰는 비정상 균형 억제 |
| self_collision, action_rate, joint_acc, joint_limits, terminated | unitree velocity task와 같음 | −1, −0.05, −2.5e−7, −10, −200 | 정규화, 넘어짐 |

**Isaac Lab 버전** (`foot/mdp/rewards.py`, `foot/env_cfgs.py`, 이전 5단계): 항목, 식, 가중치는 위 표와 같습니다.
- 실제 reward = 원래 값 × 가중치 × dt(0.02 s)입니다(mjlab과 같음).
- 위치는 발바닥 기준점의 world 좌표(m), 속도는 world 좌표(m/s, rad/s)입니다.
- 계산 순서는 mjlab과 같습니다. 같은 step 안에서 reward를 command 갱신보다 먼저 계산하므로, reward는 직전 step의 command 상태를 읽습니다.

mjlab과 다른 점:
- foot_slip의 발 속도: site 속도 대신 발바닥 기준점 속도 `v_link + ω × r`입니다.
- posture: 원래 default 자세(encoder 편향 없음) 기준입니다.
- self_collision: 모든 링크의 접촉력에서 지면 접촉력을 뺀 값으로 자기 접촉력을 구합니다. 마지막 4개 substep 중 10 N을 넘는 substep 수를 셉니다.
- 발 마찰: env마다 값 하나를 양발 모든 collision shape에 같이 줍니다. 지면은 마찰 1에 multiply로 결합해서, 실제 마찰은 발 값이 됩니다. mjlab의 foot priority와 같은 효과입니다.

성공 판정(`success`)은 지표일 뿐이고, 어떤 reward 항도 읽지 않습니다.

검증은 아래 두 스크립트로 합니다.

`scripts/inspect_task.py` (Isaac Lab)
- free (학습 환경 그대로, zero action, 20개 항목)
  - reset 갱신
  - reward 16항의 shape (N,)와 유한성, `reward = Σ 원래 값 × 가중치 × dt`, 5개 항의 독립 계산 비교
  - 넘어짐 종료와 −200, time-out에는 벌점 없음
- pinned (골반 고정, scripted step 4그룹, 13개 항목)
  - 착지, 목표 이탈 착지는 성공 없음
  - 2차 swing, stance 들기, stance drift
  - double support, 성공 판정 정의
  - 발바닥 속도 = 강체 관계식, 팔을 몸통에 밀어 자기 충돌 검출

`scripts/compare_mjlab.py` (mjlab venv)
- 같은 무작위 상태 8192개에서 발 task reward 함수 10개가 mjlab과 완전히 같습니다(차이 0).
- 가중치, 수치 파라미터, 종료 조건, dt, episode 길이가 같습니다.
- zero action rollout 비교 (256 env, 각 env의 첫 종료까지)

  | 항목 | mjlab | Isaac Lab |
  |---|---|---|
  | 첫 종료 | 1.45 s | 1.32 s |
  | upright (step당) | −0.106 | −0.118 |
  | joint_limits (step당) | −0.30 | −0.24 |
  | foot_target (step당) | 0.24 | 0.30 |

  두 시뮬레이터와 로봇 모델이 달라서(33.3 vs 34.4 kg, hip 모터) 같은 크기 수준이면 맞는 것으로 봅니다.

**mjlab 버전의 첫 step 문제 (비교 중 발견)**: mjlab은 최초 `env.reset()` 직후 첫 step의 reward를 시작 위치가 기록되기 전(원점 0)에 계산합니다.
- 그 step의 stance_drift가 env 평균 −3405, foot_target이 0입니다.
- 이후 리셋은 step 안에서 command가 갱신된 뒤라 정상입니다. 그래서 영향은 학습 시작 때 env당 한 step뿐입니다.
- Isaac Lab 버전은 `post_reset` 덕분에 이 문제가 없습니다(첫 step stance_drift −0.0005).

## 4. 종료 조건과 초기 상태

- **종료**: time out(4 s), 기울기 60° 초과, 몸통 높이 0.45 m 미만.
- **초기 상태**: default 서 있는 자세에서 시작합니다.
  - yaw: −π ~ π 무작위
  - 관절 offset: ±0.05 rad 무작위
- **domain randomization**: Stage 1에서는 가볍게만 넣습니다. 미는 힘은 아직 없습니다.
  - 발 마찰 0.6~1.2
  - torso 질량중심 ±2 cm
  - 관절 encoder 편향 ±0.015 rad (env별 고정, startup). Isaac Lab 버전은 GEAR-SONIC과 같은 ±0.01 rad
- **명령 분포**: 발 L/R은 균등입니다.
  - Δx는 10~30 cm, Δy는 −10~+10 cm에서 균등하게 뽑습니다.
  - 목표가 stance 발과 옆 방향으로 10 cm 이상 떨어지도록 보정합니다(발 교차와 충돌 방지).

## 5. 학습 설정

- **PPO**: RSL-RL입니다. 설정값은 unitree_rl_mjlab G1 velocity task를 참고해 같은 값으로 시작합니다.
  - actor/critic 512-256-128 ELU, 관측 정규화
  - γ 0.99, λ 0.95, clip 0.2, entropy 0.01, lr 1e−3(adaptive, KL 0.01)
  - env당 24 step × 4096 env
- **기대 학습량**: unitree velocity task가 4090에서 수천 iteration(1~2시간) 규모이니, 이 task도 우선 5000 iteration으로 시작합니다.

**Isaac Lab 학습 파이프라인** (이전 6단계): `foot/rl_cfg.py`, `embodiments/g1/runner.py`, `scripts/train.py`, `play.py`, `evaluate.py`
- PPO 설정값은 위와 같습니다(RSL-RL 3.1.2).
  - actor 입력: state + goal = 104
  - critic 입력: critic 그룹 131
  - action: 29
- **time-out과 종료 처리** (RSL-RL 동작)
  - time-out: `r + γ·V(s_t)`로 bootstrap
  - 넘어짐 종료: bootstrap 없음
  - 둘 다 done이라 GAE가 끊깁니다.
  - `scripts/inspect_ppo.py`로 확인: 강제 time-out 16건, 넘어짐 116건에서 저장된 reward와 따로 계산한 GAE return이 차이 0
- **체크포인트**: `G1OnPolicyRunner`가 모든 `model_<it>.pt`의 `infos["deploy"]`에 배포 spec을 넣고, 같은 폴더에 `deploy.json`을 씁니다. 불러올 때 spec(관절 이름·순서, default, scale, 게인, 관측 구성)이 현재 환경과 다르면 멈춥니다.
- **TensorBoard 기록**
  - RSL-RL 기본: `Train/mean_reward`, `Train/mean_episode_length`, 손실, 정책 std
  - Isaac Lab manager: `Episode_Reward/*`, `Metrics/footstep/*`, `Episode_Termination/*`
  - `G1RLEnv` 추가: `Episode/fall_rate`, `Episode/time_out_rate`, `Episode/length_s`
- **재생**: `play.py`가 actor를 TorchScript와 ONNX로 export합니다(관측 정규화 포함, 입력 104). 화면에 띄우거나 mp4로 녹화할 수 있습니다.
- **smoke 결과** (로컬 4090, 256 env, 50 iteration): 3.7k steps/s로 오류 없이 돌았습니다. 아직 학습 효과는 없습니다(넘어짐 100%).
  - `--resume`으로 이어 학습할 수 있습니다.
  - 재생 영상, export, goal grid 평가가 동작합니다.
- **설치 문제 (해결)**: torch 설치의 `-U`가 numpy 2, pillow 13 등을 끌어와 Isaac Sim 고정 버전과 22건이 어긋났습니다. 학습은 됐지만 영상 녹화가 replicator TypeError로 실패했습니다. 고정 버전으로 되돌린 뒤 1–6단계 검증을 모두 다시 통과했습니다. README 설치 절차에 반영했습니다.

## 6. 커리큘럼 (계획)

1. **Stage 1 (지금)**: 평지, 한 발 한 걸음, Δx 10~30 cm, Δy ±10 cm, 미는 힘 없음.
2. **Stage 2**: Δ 범위 확대(뒤로, 더 넓은 옆), 목표 방향(yaw), 연속 걸음(명령을 여러 번 resample).
3. **Stage 3**: 외란과 domain randomization 확대(미는 힘, 마찰, 질량, PD gain, 지연), sim-to-real 대비.
4. Teacher 성능이 확인된 뒤에만 language-conditioned Student로 distillation합니다: π_T(s, g) → π_S(s, text).

## 7. 선행 연구에서 가져온 것

- **unitree_rl_mjlab velocity task** (참고만 했고, 코드나 파일은 가져오지 않음)
  - 관측 구성(IMU, 관절, 직전 action), 관절 위치 action과 scale
  - 정규화 reward(종료 −200, action rate, 관절 한계, self collision, 미끄러짐)와 PPO 설정
- **foothold-conditioned 보행 연구** (예: Mind Your Steps)
  - 목표를 stance 발 기준으로 표현하기
  - 착지 순간의 오차를 착지 후에도 reward로 유지하기(touchdown reward hold)
  - 평지에서 먼저 학습한 뒤 확장하는 커리큘럼
- **ALLSTEPS** (Xie et al., 2020): 발 디딤 목표를 쉬운 분포에서 점점 넓히는 커리큘럼. Stage 2의 범위 확대 방식으로 쓸 예정입니다.
- **이전 token-policy 프로젝트의 결과**
  - 성공 기준은 CEM teacher 합격 기준을 따릅니다.
  - 그 프로젝트에서 상태별 보정이 핵심 난점이었습니다. 그래서 여기서는 시작 상태(yaw, 관절 offset)를 처음부터 무작위로 줍니다.

선행 연구 조사는 아직 간단한 수준입니다. 필요하면 별도로 넓히겠습니다.
