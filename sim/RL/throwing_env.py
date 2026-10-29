import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt


class FrictionRobustThrowingEnv(gym.Env):
    """
    제안서 기반 마찰 강인 투척 강화학습 시뮬레이션 환경 (2D 평면 / 질점 모델 / 모터 등속 회전)
    - State (관측 가능): s_t = [x_b, dx_b, theta, dtheta, prev_action] (mu, r, dr 제외)
    - Action (제어 입력): a_t = [dx_b_cmd] (베이스 목표 선속도 명령)
    - 내부 물리 (Hidden): 질점(Point-mass)의 반경 방향 미끄러짐(r, dr) 및 마찰(mu) 동역학
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        mu_range=(0.1, 0.5),
        target_pos=(0.8, 1.2),
        eval_multi_mu=True,
        num_mu_samples=15,
    ):
        super().__init__()

        # 1. 하드웨어 기반 제어 및 물리 파라미터 (근거: data_logger.py & config_manager.py)
        self.dt = 0.005                 # 200Hz 제어 주기 (s)
        self.g = 9.81                   # 중력가속도 (m/s^2)
        self.release_height = 0.8       # 투척 평면(엔드이펙터)의 지면으로부터 높이 h (m)
        
        # 투사체 비행 시간 tau = sqrt(2*h/g) (수평면 사이드 투척 기준)
        self.flight_time = np.sqrt(2.0 * self.release_height / self.g)

        # 2. 로봇팔 베이스(TCP) 및 AK60-6 모터 파라미터
        self.max_base_vel = 1.0         # 베이스 최대 선속도 (m/s)
        self.max_base_acc = 6.0         # 베이스 최대 선가속도 (m/s^2)
        self.omega_motor = 4.0         # 모터 등속 회전 각속도 (rad/s, ddtheta = 0)
        self.theta_init = -np.pi / 2    # 모터 시작 각도 (rad)
        self.max_rot_amount = np.pi     # 최대 회전 허용량 (3.14 rad)

        # 3. 그리퍼(가이드) 상의 질점 초기 위치 및 끝단(분리) 반경
        self.r_init = 0.2              # 초기 파지 반경 (m)
        self.r_tip = 0.3               # 분리(Release) 반경 (m)

        # 4. 마찰 도메인 랜덤화 및 목표 탄착 지점 (X, Y 평면)
        self.mu_min, self.mu_max = mu_range
        self.target_pos = np.array(target_pos, dtype=np.float32)
        
        # eval_multi_mu=True일 경우: 단일 궤적에서 N개의 마찰 샘플을 동시에 굴려 탄착 분산(Variance)을 직접 계산
        self.eval_multi_mu = eval_multi_mu
        self.num_mu_samples = num_mu_samples

        # Action Space: 목표 베이스 속도 dx_b_cmd [-1.0, 1.0] m/s
        self.action_space = spaces.Box(
            low=np.array([-self.max_base_vel], dtype=np.float32),
            high=np.array([self.max_base_vel], dtype=np.float32),
            dtype=np.float32,
        )

        # State Space: [x_b, dx_b, theta, dtheta, prev_action]
        obs_high = np.array([2.0, self.max_base_vel, np.pi, 60.0, self.max_base_vel], dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        # 내부 상태 변수 초기화
        self.x_b = 0.0
        self.dx_b = 0.0
        self.theta = 0.0
        self.dtheta = 0.0
        self.prev_action = 0.0

        # 질점 은닉 상태 (N개 마찰 샘플 동시 추적 지원)
        self.mu_samples = None
        self.r = None
        self.dr = None
        self.released = None
        self.landings = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 초기 베이스 속도 설정 (기본값 0.2 m/s = 200 mm/s)
        init_vel = options.get("init_base_vel", 0.2) if options else 0.2
        self.x_b = 0.0
        self.dx_b = float(init_vel)
        self.theta = self.theta_init
        self.dtheta = self.omega_motor  # 등속 회전 가정
        self.prev_action = float(init_vel)

        # 마찰계수 샘플링: mu ~ U[mu_min, mu_max]
        if self.eval_multi_mu:
            self.mu_samples = np.linspace(self.mu_min, self.mu_max, self.num_mu_samples)
        else:
            mu_val = self.np_random.uniform(self.mu_min, self.mu_max)
            self.mu_samples = np.array([mu_val])

        n = len(self.mu_samples)
        self.r = np.full(n, self.r_init, dtype=np.float64)
        self.dr = np.zeros(n, dtype=np.float64)
        self.released = np.zeros(n, dtype=bool)
        self.landings = np.zeros((n, 2), dtype=np.float64)

        return self._get_obs(), {"mu_samples": self.mu_samples.copy()}

    def _get_obs(self):
        """에이전트에게 제공되는 상태 벡터 (마찰계수 mu 및 질점 위치 r, dr 제외)"""
        return np.array(
            [self.x_b, self.dx_b, self.theta, self.dtheta, self.prev_action],
            dtype=np.float32,
        )

    def step(self, action):
        # 1. Action 클리핑 및 로봇팔 가속도 한계 반영
        dx_b_cmd = float(np.clip(action[0], -self.max_base_vel, self.max_base_vel))
        ddx_b = np.clip((dx_b_cmd - self.dx_b) / self.dt, -self.max_base_acc, self.max_base_acc)

        # 2. 아직 분리되지 않은 질점들에 대해 접촉-마찰 미끄러짐 동역학 업데이트
        active = ~self.released
        if np.any(active):
            # 수직항력 가속도: a_N = 2 * dr * omega - ddx_b * sin(theta)
            normal_acc = 2.0 * self.dr[active] * self.dtheta - ddx_b * np.sin(self.theta)

            # 반경 방향 구동 가속도: a_drive = r * omega^2 - ddx_b * cos(theta)
            radial_drive = self.r[active] * (self.dtheta ** 2) - ddx_b * np.cos(self.theta)

            # 마찰 감속도: F_f / m = mu * |a_N|
            friction_acc = self.mu_samples[active] * np.abs(normal_acc)

            # 미끄러짐 가속도 ddr 계산 (바깥쪽 방향 미끄러짐 dr >= 0)
            ddr = radial_drive - friction_acc
            ddr = np.where((self.dr[active] <= 0.0) & (ddr < 0.0), 0.0, ddr)

            # 질점 상태 적분 (Semi-implicit Euler)
            self.dr[active] = np.maximum(0.0, self.dr[active] + ddr * self.dt)
            self.r[active] += self.dr[active] * self.dt

            # 3. 분리(Release) 판정: 질점이 가이드 끝단(r_tip)에 도달한 순간
            newly_released = active & (self.r >= self.r_tip)
            if np.any(newly_released):
                self._record_landing(newly_released)
                self.released[newly_released] = True

        # 4. 로봇 베이스 및 모터 상태 업데이트 (다음 스텝 이동)
        self.dx_b += ddx_b * self.dt
        self.x_b += self.dx_b * self.dt
        self.theta += self.dtheta * self.dt
        self.prev_action = dx_b_cmd

        # 5. 에피소드 종료 조건 확인 (모든 마찰 샘플이 분리되었거나 최대 회전각 도달 시)
        rot_progress = self.theta - self.theta_init
        terminated = bool(np.all(self.released))
        truncated = bool(rot_progress >= self.max_rot_amount)

        reward = 0.0
        info = {}

        if terminated or truncated:
            # 아직 분리되지 못한 샘플이 있다면 현재 상태에서 강제 분리 처리
            still_attached = ~self.released
            if np.any(still_attached):
                self._record_landing(still_attached)
                self.released[still_attached] = True

            # 탄착군 평균 위치 및 분산(Dispersion) 산출
            mean_landing = np.mean(self.landings, axis=0)
            dispersion_var = np.mean(np.sum((self.landings - mean_landing) ** 2, axis=1))
            target_error_sq = np.sum((mean_landing - self.target_pos) ** 2)

            # 보상 함수: 탄착 분산(Dispersion) + 목표 지점 평균 오차 페널티
            w_disp = 5.0   # 탄착 군집도(분산 억제) 가중치
            w_targ = 1.0   # 목표 지점 정확도 가중치
            reward = -float(w_disp * dispersion_var + w_targ * target_error_sq)

            info = {
                "landings": self.landings.copy(),
                "mean_landing": mean_landing,
                "dispersion_std": float(np.sqrt(dispersion_var)),
                "target_error": float(np.sqrt(target_error_sq)),
            }

        return self._get_obs(), reward, terminated, truncated, info

    def _record_landing(self, mask):
        """분리 순간의 베이스 속도와 질점 상대 속도를 합산하여 탄착 좌표 계산"""
        r_rel = np.minimum(self.r[mask], self.r_tip)
        dr_rel = self.dr[mask]

        # 분리 순간 질점의 평면 위치 (x_p, y_p)
        x_p = self.x_b + r_rel * np.cos(self.theta)
        y_p = r_rel * np.sin(self.theta)

        # 분리 순간 질점의 평면 절대 속도 (vx_p, vy_p)
        # v_p = v_base + dr * e_r + (r * dtheta) * e_theta
        vx_p = self.dx_b + dr_rel * np.cos(self.theta) - r_rel * self.dtheta * np.sin(self.theta)
        vy_p = dr_rel * np.sin(self.theta) + r_rel * self.dtheta * np.cos(self.theta)

        # 비행 시간(flight_time) 동안 날아간 최종 지면 탄착 위치 p_land
        self.landings[mask, 0] = x_p + vx_p * self.flight_time
        self.landings[mask, 1] = y_p + vy_p * self.flight_time


# =====================================================================
# 시뮬레이션 동작 검증: 등속 베이스라인 vs. 비선형 가감속 베이스 기동 비교
# =====================================================================
if __name__ == "__main__":
    env = FrictionRobustThrowingEnv(mu_range=(0.1, 0.5), eval_multi_mu=True, num_mu_samples=20)

    # 1. 해석적 베이스라인: 등속 직선 운동 (dx_b = 0.2 m/s 고정, ddx_b = 0)
    obs, _ = env.reset(options={"init_base_vel": 0.2})
    done = False
    while not done:
        obs, reward, term, trunc, info_const = env.step(np.array([0.2], dtype=np.float32))
        done = term or trunc

    # 2. 비선형 베이스 기동 예시: 회전 각도에 맞춰 베이스를 감속/가속하는 프로파일
    obs, _ = env.reset(options={"init_base_vel": 0.5})
    done = False
    while not done:
        theta_curr = obs[2]
        # 회전 중반부에 베이스 속도를 줄여 음(-)의 가속도(ddx_b < 0)를 인가하는 예시
        cmd_vel = 0.5 - 0.8 * np.sin(theta_curr - env.theta_init)
        obs, reward, term, trunc, info_nonlin = env.step(np.array([cmd_vel], dtype=np.float32))
        done = term or trunc

    print("=== [1주 차 시뮬레이션 검증 결과] ===")
    print(f"1. 등속 베이스 기동 (Baseline)  -> 탄착 표준편차(Dispersion): {info_const['dispersion_std']*1000:.2f} mm")
    print(f"2. 비선형 가감속 기동 (Test)    -> 탄착 표준편차(Dispersion): {info_nonlin['dispersion_std']*1000:.2f} mm")

    # 탄착 분포 시각화
    fig, ax = plt.subplots(figsize=(7, 6))
    land_c = info_const["landings"]
    land_n = info_nonlin["landings"]

    ax.scatter(land_c[:, 0], land_c[:, 1], c=env.mu_samples, cmap="Blues", s=60, edgecolors="b", label="Constant Base Vel (Baseline)")
    ax.scatter(land_n[:, 0], land_n[:, 1], c=env.mu_samples, cmap="Oranges", s=60, edgecolors="r", marker="^", label="Non-Linear Base Maneuver (Test)")
    
    ax.set_title("Landing Distribution across Friction $\mu \in [0.1, 0.5]$")
    ax.set_xlabel("Landing X (m)")
    ax.set_ylabel("Landing Y (m)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.axis("equal")
    plt.tight_layout()
    plt.show()