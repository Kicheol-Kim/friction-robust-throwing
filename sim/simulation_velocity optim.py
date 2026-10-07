import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ==========================================
# 1. 시스템 파라미터 세팅
# ==========================================
L = 0.3
omega = 8
r0 = 0.03
theta0 = np.pi
z_goal = -0.915
g = 9.81
mu_min = 0.1
mu_max = 0.3
num_samples = 21

t_flight = np.sqrt(2 * abs(z_goal) / g)
# mu_list = np.linspace(0.0, 1.0, 101)
mu_list = np.linspace(mu_min, mu_max, num_samples)

# ==========================================
# 2. 고유 탄착 데이터 수집
# ==========================================
p_land_0 = [] 
p_exit_0 = [] 
taus = []     
t_exits = []  
mu_used = []  

for mu in mu_list:
    def dynamics(t, y):
        r, r_dot = y
        r_ddot = r * omega**2 - 2 * mu * omega * r_dot - mu * g
        return [r_dot, r_ddot]

    def event_exit(t, y):
        return y[0] - L
    event_exit.terminal = True
    event_exit.direction = 1

    sol = solve_ivp(dynamics, [0, 20.0], [r0, 0.0], events=event_exit, dense_output=True)
    
    if len(sol.t_events[0]) == 0:
        continue
        
    t_exit = sol.t_events[0][0]
    r_dot_exit = sol.y_events[0][0][1]
    theta_exit = theta0 + omega * t_exit
    
    v_exit_local_x = r_dot_exit * np.cos(theta_exit) - L * omega * np.sin(theta_exit)
    v_exit_local_y = r_dot_exit * np.sin(theta_exit) + L * omega * np.cos(theta_exit)
    
    p_exit_0_x = L * np.cos(theta_exit)
    p_exit_0_y = L * np.sin(theta_exit)
    
    px_0 = p_exit_0_x + v_exit_local_x * t_flight
    py_0 = p_exit_0_y + v_exit_local_y * t_flight
    
    p_exit_0.append([p_exit_0_x, p_exit_0_y])
    p_land_0.append([px_0, py_0])
    taus.append(t_exit + t_flight)
    t_exits.append(t_exit)
    mu_used.append(mu)

p_land_0 = np.array(p_land_0)
p_exit_0 = np.array(p_exit_0)
taus = np.array(taus)
t_exits = np.array(t_exits)
mu_used = np.array(mu_used)

# ==========================================
# 3. 분산 최소화(Variance Minimization) 닫힌 해 도출
# ==========================================
# 분산(Variance) 및 공분산(Covariance) 계산[cite: 1]
var_tau = np.cov(taus)
cov_x_tau = np.cov(p_land_0[:, 0], taus)[0, 1]
cov_y_tau = np.cov(p_land_0[:, 1], taus)[0, 1]
cov_p_tau = np.array([cov_x_tau, cov_y_tau])

# 최적 베이스 속도(Eq. 30): v_b* = -Cov / Var[cite: 1]
v_b_opt = -cov_p_tau / var_tau

# ==========================================
# 4. 데이터 매핑 (보정 전 vs 보정 후)
# ==========================================
# 보정 후 발사(Release) 및 착탄(Landing) 위치
p_release_optimized = p_exit_0 + np.outer(t_exits, v_b_opt)
p_land_optimized = p_land_0 + np.outer(taus, v_b_opt)

# 군집 중심점(Centroid) 계산
centroid_0 = np.mean(p_land_0, axis=0)
centroid_opt = np.mean(p_land_optimized, axis=0)

error_0 = np.average(np.linalg.norm(p_land_0 - centroid_0, axis=1)**2)
error_opt = np.average(np.linalg.norm(p_land_optimized - centroid_opt, axis=1)**2)
# error_0 = np.average(np.sqrt(np.linalg.norm(p_land_0 - centroid_0, axis=1)**2))
# error_opt = np.average(np.sqrt(np.linalg.norm(p_land_optimized - centroid_opt, axis=1)**2))

# ==========================================
# 5. 시각화 (Subplots)
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
cmap = plt.cm.viridis
t_max_exit = np.max(t_exits)

def plot_variance(ax, p_release, p_land, centroid, title, v_b):
    # 군집 중심점
    ax.scatter(centroid[0], centroid[1], color='red', marker='+', s=300, label='Centroid', zorder=10)
    
    # 잠재적 발사 궤적 및 베이스 경로
    t_span = np.linspace(0, t_max_exit, 100)
    scoop_tip_x = v_b[0] * t_span + L * np.cos(theta0 + omega * t_span)
    scoop_tip_y = v_b[1] * t_span + L * np.sin(theta0 + omega * t_span)
    ax.plot(scoop_tip_x, scoop_tip_y, 'k--', alpha=0.4, label='Potential Release Trajectory')
    
    if np.linalg.norm(v_b) > 0:
        end_base = v_b * t_max_exit
        ax.annotate('', xy=(end_base[0], end_base[1]), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color='gray', ls='--', lw=2))
    
    # 궤적 및 위치 렌더링
    for i in range(len(mu_used)):
        color = cmap((mu_used[i] - min(mu_used)) / (max(mu_used) - min(mu_used)))
        ax.plot([p_release[i, 0], p_land[i, 0]], [p_release[i, 1], p_land[i, 1]], color=color, alpha=0.3, ls='-', lw=1)
    
    ax.scatter(p_release[:, 0], p_release[:, 1], c=mu_used, cmap=cmap, marker='^', s=60, edgecolor='k', label='Release', zorder=5)
    sc = ax.scatter(p_land[:, 0], p_land[:, 1], c=mu_used, cmap=cmap, marker='o', s=80, edgecolor='k', label='Landing', zorder=5)
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Global X Position (m)', fontsize=12)
    ax.set_ylabel('Global Y Position (m)', fontsize=12)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, linestyle=':')
    ax.set_aspect('equal', adjustable='box')
    return sc

print("="*60)
print(f"전체 샘플 통과 수: {len(mu_used)} / {len(mu_list)}")
print(f"최적 베이스 이동 속도: X = {v_b_opt[0]:.3f} m/s, Y = {v_b_opt[1]:.3f} m/s")
print(f"최적 베이스 이동 속력: {np.linalg.norm(v_b_opt):.3f} m/s")
print(f"베이스 이동 거리 t_exit*v_b: {np.linalg.norm((np.max(t_exits)) * v_b_opt):.2f} m")
print(f"보상 전 분산: {error_0:.4f} m^2")
print(f"보상 후 분산: {error_opt:.4f} m^2")
print("="*60)

# Subplot 1: Uncompensated
plot_variance(ax1, p_exit_0, p_land_0, centroid_0, 
              f'Uncompensated ($v_b = 0$)\nVariance: {error_0:.4f}', np.array([0.0, 0.0]))
ax1.legend(loc='lower left')

# Subplot 2: Optimized
sc2 = plot_variance(ax2, p_release_optimized, p_land_optimized, centroid_opt, 
                    f'Optimized Variance ($v_b = v_b^*$)\nVariance: {error_opt:.4f}', v_b_opt)

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(sc2, cax=cbar_ax, label=r'Friction Coefficient ($\mu$)')
plt.suptitle('Comparison: Minimizing Spatial Variance of Landing Positions', fontsize=18, y=0.95)
plt.subplots_adjust(right=0.9)
plt.show()