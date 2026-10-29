import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    from scipy.optimize import minimize
    SB3_AVAILABLE = False
    print("[알림] stable-baselines3가 없어 Scipy 정책 최적화 모드로 대체 실행됩니다.")

# =====================================================================
# 1. 강화학습 투척 시뮬레이션 환경
# =====================================================================
class FrictionThrowingEnv(gym.Env):
    def __init__(self, mu_range=(0.1, 0.5), num_mu=20, target_x=1.65):
        super().__init__()
        self.dt = 0.005                 
        self.g = 9.81                   
        self.base_height = 0.45         

        self.max_vel = 1.2              
        self.max_acc = 8.0              
        self.omega_motor = 12.0         
        self.theta_init = -np.pi / 3    
        self.max_rot = np.pi * 0.85     

        self.r_init = 0.05              
        self.r_tip = 0.14               

        self.mu_min, self.mu_max = mu_range
        self.mu_samples = np.linspace(self.mu_min, self.mu_max, num_mu)
        self.num_mu = num_mu
        self.target_x = target_x

        self.action_space = spaces.Box(
            low=np.array([-self.max_vel], dtype=np.float32),
            high=np.array([self.max_vel], dtype=np.float32),
            dtype=np.float32,
        )

        high = np.array([2.0, self.max_vel, np.pi, 60.0, self.max_vel], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        init_vel = options.get("init_vel", 0.0) if options else 0.0

        self.x_b = 0.0
        self.dx_b = float(init_vel)
        self.theta = self.theta_init
        self.dtheta = self.omega_motor
        self.prev_action = float(init_vel)
        self.elapsed_time = 0.0

        self.r = np.full(self.num_mu, self.r_init, dtype=np.float64)
        self.dr = np.zeros(self.num_mu, dtype=np.float64)
        self.released = np.zeros(self.num_mu, dtype=bool)

        self.t_rel = np.zeros(self.num_mu, dtype=np.float64)
        self.tau_air = np.zeros(self.num_mu, dtype=np.float64)
        self.x_land = np.zeros(self.num_mu, dtype=np.float64)
        self.rel_state = np.zeros((self.num_mu, 4), dtype=np.float64) 

        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([self.x_b, self.dx_b, self.theta, self.dtheta, self.prev_action], dtype=np.float32)

    def step(self, action):
        dx_cmd = float(np.clip(action[0], -self.max_vel, self.max_vel))
        ddx_b = np.clip((dx_cmd - self.dx_b) / self.dt, -self.max_acc, self.max_acc)

        active = ~self.released
        if np.any(active):
            # 가속도(ddx_b)가 수직항력(a_N)과 미끄러짐에 실시간으로 반영되는 부분
            normal_acc = (2.0 * self.dr[active] * self.dtheta 
                          - ddx_b * np.sin(self.theta) 
                          + self.g * np.cos(self.theta))
            radial_drive = (self.r[active] * (self.dtheta ** 2) 
                            - ddx_b * np.cos(self.theta) 
                            - self.g * np.sin(self.theta))
            
            fric_acc = self.mu_samples[active] * np.abs(normal_acc)

            ddr = radial_drive - fric_acc
            ddr = np.where((self.dr[active] <= 0.0) & (ddr < 0.0), 0.0, ddr)

            self.dr[active] = np.maximum(0.0, self.dr[active] + ddr * self.dt)
            self.r[active] += self.dr[active] * self.dt

            newly_rel = active & (self.r >= self.r_tip)
            if np.any(newly_rel):
                self._compute_release_and_landing(newly_rel)
                self.released[newly_rel] = True

        self.dx_b += ddx_b * self.dt
        self.x_b += self.dx_b * self.dt
        self.theta += self.dtheta * self.dt
        self.elapsed_time += self.dt
        self.prev_action = dx_cmd

        terminated = bool(np.all(self.released))
        truncated = bool((self.theta - self.theta_init) >= self.max_rot)

        reward = -0.001 * ((dx_cmd - self.dx_b) ** 2) 
        info = {}

        if terminated or truncated:
            still_active = ~self.released
            if np.any(still_active):
                self._compute_release_and_landing(still_active)
                self.released[still_active] = True

            mean_x = float(np.mean(self.x_land))
            var_x = float(np.var(self.x_land))
            std_x = float(np.std(self.x_land))
            target_err_sq = (mean_x - self.target_x) ** 2

            reward += -float(40.0 * var_x + 2.0 * target_err_sq)
            info = {
                "x_land": self.x_land.copy(),
                "mean_x": mean_x,
                "std_x": std_x,
                "t_total": (self.t_rel + self.tau_air).copy(),
                "rel_state": self.rel_state.copy(),
            }

        return self._get_obs(), reward, terminated, truncated, info

    def _compute_release_and_landing(self, mask):
        r_rel = np.minimum(self.r[mask], self.r_tip)
        dr_rel = self.dr[mask]

        x_p = self.x_b + r_rel * np.cos(self.theta)
        z_p = np.maximum(0.02, self.base_height + r_rel * np.sin(self.theta))

        vx_p = self.dx_b + dr_rel * np.cos(self.theta) - r_rel * self.dtheta * np.sin(self.theta)
        vz_p = dr_rel * np.sin(self.theta) + r_rel * self.dtheta * np.cos(self.theta)

        tau = (vz_p + np.sqrt(vz_p ** 2 + 2.0 * self.g * z_p)) / self.g

        self.t_rel[mask] = self.elapsed_time
        self.tau_air[mask] = tau
        self.x_land[mask] = x_p + vx_p * tau
        self.rel_state[mask] = np.column_stack([x_p, z_p, vx_p, vz_p])


# =====================================================================
# 2. 헬퍼 함수
# =====================================================================
def evaluate_policy(env, policy_fn, init_vel=0.0):
    obs, _ = env.reset(options={"init_vel": init_vel})
    done = False
    time_log, vel_log, pos_log = [], [], []

    while not done:
        time_log.append(env.elapsed_time)
        vel_log.append(env.dx_b)
        pos_log.append(env.x_b)
        
        action = policy_fn(obs)
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc

    info["time_log"] = np.array(time_log)
    info["vel_log"] = np.array(vel_log)
    info["pos_log"] = np.array(pos_log)
    return info


# =====================================================================
# 3. 메인 실행 및 플롯
# =====================================================================
if __name__ == "__main__":
    env = FrictionThrowingEnv(mu_range=(0.1, 0.5), num_mu=25)

    # [Step 1] 정지 베이스라인 (v_b = 0)
    res_stat = evaluate_policy(env, lambda obs: np.array([0.0]), init_vel=0.0)
    
    # [Step 2] 해석적 최적 등속 베이스라인 (Analytical Constant-Vel)
    cov_mat = np.cov(res_stat["x_land"], res_stat["t_total"])
    v_opt_const = float(np.clip(-cov_mat[0, 1] / cov_mat[1, 1], -env.max_vel, env.max_vel))
    res_const = evaluate_policy(env, lambda obs: np.array([v_opt_const]), init_vel=v_opt_const)
    env.target_x = res_const["mean_x"]

    # [Step 3] PPO 비선형 베이스 기동 학습
    print("=================================================================")
    print("강화학습(PPO) 기반 비선형 베이스 기동 학습 시작...")
    if SB3_AVAILABLE:
        model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=512, batch_size=64, 
                    n_epochs=10, gamma=0.99, clip_range=0.2, seed=42, verbose=0)
        model.learn(total_timesteps=35_000)
        rl_policy_fn = lambda obs: model.predict(obs, deterministic=True)[0]
    else:
        def poly_policy(weights, obs):
            th_n = (obs[2] - env.theta_init) / env.max_rot
            cmd = weights[0] + weights[1]*th_n + weights[2]*(th_n**2) + weights[3]*np.sin(np.pi*th_n)
            return np.array([cmd], dtype=np.float32)
        def loss_fn(w):
            inf = evaluate_policy(env, lambda obs: poly_policy(w, obs), init_vel=w[0])
            return 40.0 * (inf["std_x"] ** 2) + 2.0 * ((inf["mean_x"] - env.target_x) ** 2)
        opt_res = minimize(loss_fn, x0=[v_opt_const, -0.5, 0.2, 0.2], method="Nelder-Mead", options={"maxiter": 300})
        rl_policy_fn = lambda obs: poly_policy(opt_res.x, obs)

    res_rl = evaluate_policy(env, rl_policy_fn, init_vel=0.0)
    print("학습 완료!")

    # [Step 4] 대시보드 시각화 (3D 포함)
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle("Friction-Robust Throwing: 3D Trajectory & Non-Linear Maneuver", fontsize=14, fontweight="bold")

    # (A) 좌상단: 시간에 따른 베이스 속도 프로파일
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(res_stat["time_log"]*1000, res_stat["vel_log"], "k:", linewidth=2, label="1. Stationary ($v_b=0$)")
    ax1.plot(res_const["time_log"]*1000, res_const["vel_log"], "b--", linewidth=2.2, label=f"2. Const-Vel ($v_b={v_opt_const:.2f}$ m/s)")
    ax1.plot(res_rl["time_log"]*1000, res_rl["vel_log"], "r-", linewidth=2.5, label="3. RL Non-Linear")
    ax1.set_title("(A) TCP Velocity Profile", fontweight="bold")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("TCP Forward Velocity (m/s)")
    ax1.legend(loc="best")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # (B) 우상단: 3D TCP 궤적 및 비행 탄착군 분포
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    # Target
    ax2.scatter(env.target_x, 0, 0, color='gold', marker='*', s=250, edgecolor='black', label="Target Spot", zorder=5)
    
    # TCP Path (RL Policy)
    tcp_x = res_rl["pos_log"]
    tcp_y = np.zeros_like(tcp_x)
    tcp_z = np.full_like(tcp_x, env.base_height)
    ax2.plot(tcp_x, tcp_y, tcp_z, 'r-', linewidth=3, alpha=0.8, label="TCP Path (RL)")
    ax2.scatter(tcp_x[0], 0, env.base_height, color='green', s=60, label="TCP Start")
    ax2.scatter(tcp_x[-1], 0, env.base_height, color='red', s=60, label="TCP End")

    # RL Policy - Parabolic Flights
    cmap = plt.cm.viridis
    for idx in [0, 6, 12, 18, 24]:
        mu_val = env.mu_samples[idx]
        x_p, z_p, vx_p, vz_p = res_rl["rel_state"][idx]
        t_air = np.linspace(0, res_rl["t_total"][idx] - env.t_rel[idx], 40)
        x_traj = x_p + vx_p * t_air
        z_traj = z_p + vz_p * t_air - 0.5 * env.g * (t_air ** 2)
        ax2.plot(x_traj, np.zeros_like(x_traj), z_traj, color=cmap((mu_val-env.mu_min)/(env.mu_max-env.mu_min)), linewidth=1.5, alpha=0.7)

    # Ground Distributions of all 3 methods (Y offset added strictly for visual separation)
    ax2.scatter(res_stat["x_land"], np.full_like(res_stat["x_land"], 0.2), np.zeros_like(res_stat["x_land"]), c='gray', s=15, label="Landing: Stationary")
    ax2.scatter(res_const["x_land"], np.full_like(res_const["x_land"], 0.0), np.zeros_like(res_const["x_land"]), c='blue', s=15, marker='s', label="Landing: Const-Vel")
    ax2.scatter(res_rl["x_land"], np.full_like(res_rl["x_land"], -0.2), np.zeros_like(res_rl["x_land"]), c='red', s=20, marker='^', label="Landing: RL")

    ax2.set_title("(B) 3D Trajectories & Dispersions", fontweight="bold")
    ax2.set_xlabel("X Position (m) - Forward")
    ax2.set_ylabel("Y Position (m) - Separation")
    ax2.set_zlabel("Z Height (m)")
    ax2.set_ylim([-0.5, 0.5])
    ax2.view_init(elev=15, azim=-55)
    ax2.legend(loc="upper left", fontsize=8)

    # (C) 좌하단: 마찰계수 변화에 따른 탄착 위치 (Target 대비)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(env.mu_samples, res_stat["x_land"] - res_stat["mean_x"] + env.target_x, "k:o", markersize=4, label=f"1. Stationary (Std: {res_stat['std_x']*1000:.1f}mm)")
    ax3.plot(env.mu_samples, res_const["x_land"] - res_const["mean_x"] + env.target_x, "b--s", markersize=4, label=f"2. Const-Vel (Std: {res_const['std_x']*1000:.1f}mm)")
    ax3.plot(env.mu_samples, res_rl["x_land"], "r-^", linewidth=2.2, markersize=5, label=f"3. RL (Std: {res_rl['std_x']*1000:.1f}mm)")
    ax3.axhline(env.target_x, color="gray", linestyle="-", alpha=0.5)
    ax3.set_title("(C) Landing Position vs. Friction", fontweight="bold")
    ax3.set_xlabel(r"Friction Coefficient ($\mu$)")
    ax3.set_ylabel("Landing Position X (m)")
    ax3.legend(loc="best")
    ax3.grid(True, linestyle="--", alpha=0.6)

    # (D) 우하단: 최종 탄착군 밀집도 Boxplot (traceback 오류 해결됨)
    ax4 = fig.add_subplot(2, 2, 4)
    methods = ["1. Stationary", "2. Const-Vel", "3. RL (Proposed)"]
    data_list = [
        (res_stat["x_land"] - res_stat["mean_x"]) * 1000,
        (res_const["x_land"] - res_const["mean_x"]) * 1000,
        (res_rl["x_land"] - res_rl["mean_x"]) * 1000,
    ]
    
    bp = ax4.boxplot(data_list, orientation="horizontal", patch_artist=True, widths=0.45)
    ax4.set_yticks(np.arange(1, len(methods) + 1))
    ax4.set_yticklabels(methods)
    for patch, color in zip(bp["boxes"], ["lightgray", "skyblue", "salmon"]):
        patch.set_facecolor(color)

    for i, d in enumerate(data_list, start=1):
        ax4.scatter(d, np.full_like(d, i) + np.random.uniform(-0.08, 0.08, size=len(d)), c=env.mu_samples, cmap="viridis", s=35, edgecolors="k", zorder=3)

    ax4.axvline(0.0, color="red", linestyle="--", linewidth=1.5)
    ax4.set_title("(D) Final Landing Dispersion (mm)", fontweight="bold")
    ax4.set_xlabel("Deviation from Target Center (mm)")
    ax4.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()