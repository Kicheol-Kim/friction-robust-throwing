import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

# =============================================================================
# --- Physical Constants & Configuration ---
# =============================================================================
L = 0.2              # Total length of the arm link (m)
r0 = 0.1             # Initial radial position of the marble (m) - 회전축 부근으로 수정
g = 9.81             # Acceleration due to gravity (m/s^2)

# Rotational Kinematics Parameters (XY Plane)
# omega_0 = np.deg2rad(100)  # Initial angular velocity of the arm (rad/s)
omega_0 = 5.0
alpha = 0.0                # Constant angular acceleration of the arm (rad/s^2)

# PARAMETRIZE START ANGLE HERE (in degrees)
theta_0_deg = 180
theta_0 = np.radians(theta_0_deg)

# --- 3D HEIGHT CONFIGURATION (NEW) ---
z_arm = 1.5          # 로봇 팔이 회전하는 XY 평면의 절대 높이 (m)
z_goal = 0.0         # 목표 착탄면 높이 (m) - 예: 지면 (기존 y_goal 대체)

# --- CONSTANT HORIZONTAL BASE VELOCITY ---
vel_base_x = -0.063 #3             # Constant speed of the base along the +X axis (m/s)
vel_base_y = -3.378 #6

# Friction coefficients to simulate
# mu_values = [0.125, 0.25]
# mu_values = [0, 0.15, 0.30, 0.45, 0.60]
mu_values = np.linspace(0.1, 0.2, 11)

# =============================================================================
# --- Simulation Pre-Computation Loop ---
# =============================================================================
def arm_dynamics(t, state, mu, omega_0, alpha, theta_0, g):
    # previously used simplified dynamics without gravity and friction:
    # r, v = state
    # omega_t = omega_0 + alpha * t
    # r_dot = v
    # v_dot = r * (omega_t**2) - 2 * mu * omega_t * v - mu * alpha * r\
    
    r, v = state
    omega_t = omega_0 + alpha * t
    theta_t = theta_0 + omega_0 * t + 0.5 * alpha * (t**2) # Current angle of the arm at time t

    r_dot = v
    f_centrifugal = r * (omega_t**2) # Centrifugal force component along the radial direction
    
    # In XY-plane rotation, gravity does not act in the radial direction
    # Normal acceleration is the vector sum of lateral (Coriolis/Euler) and vertical (Gravity)
    a_lateral = 2 * omega_t * v + alpha * r 
    a_vertical = g 
    a_normal = np.sqrt(a_lateral**2 + a_vertical**2) # Normal acceleration component

    v_scale = 0.005  
    f_friction = -mu * np.abs(a_normal) * np.tanh(v / v_scale) # Friction opposes motion with smooth transition near zero velocity
    
    v_dot = f_centrifugal + f_friction # Total radial acceleration
    return [r_dot, v_dot]

def hit_arm_tip(t, state, *args):
    return state[0] - L
hit_arm_tip.terminal = True

def hit_arm_base(t, state, *args):
    return state[0] - 0.0 
hit_arm_base.terminal = True   # 

# Data structures to store computed paths for animation framing
sim_data = []
max_global_time = 0.0

for mu in mu_values:
    # 1. Phase 1: Internal Slide
    t_span = (0.0, 10.0)
    
    # previously used simplified dynamics without gravity and friction:
    # sol = solve_ivp(arm_dynamics, t_span, [r0, 0.0],
    #                 events=hit_arm_tip, args=(mu, omega_0, alpha), rtol=1e-8, atol=1e-10)
    
    sol = solve_ivp(arm_dynamics, t_span, [r0, 0.0],
                events=[hit_arm_tip, hit_arm_base], args=(mu, omega_0, alpha, theta_0, g), rtol=1e-8, atol=1e-10)
   
    t_slide = sol.t
    r_slide = sol.y[0]
    v_slide = sol.y[1]
    t_exit = t_slide[-1]

    r_e = r_slide[-1]

    # Calculate state variables at exact exit boundary
    theta_e = theta_0 + omega_0 * t_exit + 0.5 * alpha * (t_exit**2)
    omega_e = omega_0 + alpha * t_exit
    r_dot_exit = v_slide[-1]
   
    # Base is at (vel_base_x * t_exit, vel_base_y * t_exit, z_arm) at the exit instant
    x_e = r_e * np.cos(theta_e) + vel_base_x * t_exit
    y_e = r_e * np.sin(theta_e) + vel_base_y * t_exit
    z_e = z_arm
   
    # Compound Launch Velocities (Rotational Components + Linear Base Velocity Shift)
    vx0 = r_dot_exit * np.cos(theta_e) - r_e * omega_e * np.sin(theta_e) + vel_base_x
    vy0 = r_dot_exit * np.sin(theta_e) + r_e * omega_e * np.cos(theta_e) + vel_base_y
    vz0 = 0.0  # Horizontal release in XY plane
   
    # 2. Phase 2: Ballistic Free Flight (Now evaluating Z-axis dropping)
    a_q = 0.5 * g
    b_q = -vz0
    c_q = z_goal - z_e
    discriminant = b_q**2 - 4 * a_q * c_q
   
    if discriminant >= 0:
        tau_1 = (-b_q + np.sqrt(discriminant)) / (2 * a_q)
        tau_2 = (-b_q - np.sqrt(discriminant)) / (2 * a_q)
        positive_roots = [t for t in [tau_1, tau_2] if t > 0]
        tau_impact = min(positive_roots) if positive_roots else 0.0
    else:
        tau_impact = 0.0
       
    t_total_lifecycle = t_exit + tau_impact
    if t_total_lifecycle > max_global_time:
        max_global_time = t_total_lifecycle
       
    sim_data.append({
        'mu': mu,
        't_exit': t_exit,
        't_total': t_total_lifecycle,
        't_slide': t_slide,
        'r_slide': r_slide,
        'theta_e': theta_e,
        'vx0': vx0, 'vy0': vy0, 'vz0': vz0,
        'x_e': x_e, 'y_e': y_e, 'z_e': z_e
    })

# =============================================================================
# --- Animation Environment Setup ---
# =============================================================================
# Create a 1x3 grid for 3D View, XY-Plane View (Top-Down), and XZ-Plane View (Side)

# Calculate the overall limits for axis of the plots based on the final positions of all marbles
end_x = [d['x_e'] + d['vx0'] * (d['t_total'] - d['t_exit']) for d in sim_data] + [vel_base_x * max_global_time]
end_y = [d['y_e'] + d['vy0'] * (d['t_total'] - d['t_exit']) for d in sim_data] + [vel_base_y * max_global_time]
end_z = [d['z_e'] + d['vz0'] * (d['t_total'] - d['t_exit']) - 0.5 * g * (d['t_total'] - d['t_exit'])**2 for d in sim_data] + [z_arm]

pad = 1.2*L # 화면 여백(m)
x_lims = (min(0, min(end_x)) - pad, max(0, max(end_x)) + pad)
y_lims = (min(0, min(end_y)) - pad, max(0, max(end_y)) + pad)
z_lims = (min(0, min(end_z)) - pad, max(0, max(end_z)) + pad)

fig = plt.figure(figsize=(18, 6))

ax_3d = fig.add_subplot(131, projection='3d')
ax_xy = fig.add_subplot(132)
ax_xz = fig.add_subplot(133)

colors = plt.cm.viridis(np.linspace(0, 0.85, len(mu_values)))

# Master time array running the animation frames
frames_t = np.linspace(0, max_global_time, 6000)

# -------------------------------------------------
# Structural Plot Elements for each axis
# -------------------------------------------------
# 3D Axis Setup
ax_3d.set_title('3D Diagonal View', fontsize=12)
ax_3d.set_xlim(x_lims)
ax_3d.set_ylim(y_lims)
ax_3d.set_zlim(z_lims)

# XY Axis Setup (Top-Down)
ax_xy.set_title('XY Plane View (Top-Down)', fontsize=12)
ax_xy.set_xlim(x_lims)
ax_xy.set_ylim(y_lims)
ax_xy.set_aspect('equal')
ax_xy.grid(True, linestyle=':', alpha=0.5)

# XZ Axis Setup (Side)
ax_xz.set_title('XZ Plane View (Side)', fontsize=12)
ax_xz.set_xlim(x_lims)
ax_xz.set_ylim(z_lims)
ax_xz.axhline(z_goal, color='red', linewidth=1.5, linestyle='-.', label=f'Goal Floor ({z_goal}m)')
ax_xz.grid(True, linestyle=':', alpha=0.5)
ax_xz.grid(True, linestyle=':', alpha=0.5)

# -------------------------------------------------
# Visual objects to update dynamically per frame
# -------------------------------------------------
# Base pivots
base_dot_3d, = ax_3d.plot([], [], [], 'ko', markersize=8)
base_dot_xy, = ax_xy.plot([], [], 'ko', markersize=8)
base_dot_xz, = ax_xz.plot([], [], 'ko', markersize=8)

# Arm links
arm_line_3d, = ax_3d.plot([], [], [], color='black', linestyle='-', linewidth=3, alpha=0.7)
arm_line_xy, = ax_xy.plot([], [], color='black', linestyle='-', linewidth=3, alpha=0.7)
arm_line_xz, = ax_xz.plot([], [], color='black', linestyle='-', linewidth=3, alpha=0.7)

# Lists to hold marble and trail objects for each view
marbles_3d, trails_3d = [], []
marbles_xy, trails_xy = [], []
marbles_xz, trails_xz = [], []

for dataset, color in zip(sim_data, colors):
    # 3D objects
    m_3d, = ax_3d.plot([], [], [], 'o', color=color, markersize=8)
    t_3d, = ax_3d.plot([], [], [], '-', color=color, linewidth=2)
    marbles_3d.append(m_3d)
    trails_3d.append(t_3d)
    
    # XY objects
    m_xy, = ax_xy.plot([], [], 'o', color=color, markersize=8)
    t_xy, = ax_xy.plot([], [], '-', color=color, linewidth=2, label=rf'$\mu$ = {dataset["mu"]:.2f}')
    marbles_xy.append(m_xy)
    trails_xy.append(t_xy)
    
    # XZ objects
    m_xz, = ax_xz.plot([], [], 'o', color=color, markersize=8)
    t_xz, = ax_xz.plot([], [], '-', color=color, linewidth=2)
    marbles_xz.append(m_xz)
    trails_xz.append(t_xz)

# ax_xy.legend(loc='upper right')

# =============================================================================
# --- Animation Core Loop Engine ---
# =============================================================================
def update(frame_time):
    # Update the linear position of the moving base pivot
    x_base_current = vel_base_x * frame_time
    y_base_current = vel_base_y * frame_time
    z_base_current = z_arm
    
    # Update Base Dots
    base_dot_3d.set_data([x_base_current], [y_base_current])
    base_dot_3d.set_3d_properties([z_base_current])
    
    base_dot_xy.set_data([x_base_current], [y_base_current])
    base_dot_xz.set_data([x_base_current], [z_base_current])
   
    # Update Arm Lines
    theta_current = theta_0 + omega_0 * frame_time + 0.5 * alpha * (frame_time**2)
    ax_tip = x_base_current + L * np.cos(theta_current)
    ay_tip = y_base_current + L * np.sin(theta_current)
    az_tip = z_arm
    
    arm_line_3d.set_data([x_base_current, ax_tip], [y_base_current, ay_tip])
    arm_line_3d.set_3d_properties([z_base_current, az_tip])
    
    arm_line_xy.set_data([x_base_current, ax_tip], [y_base_current, ay_tip])
    arm_line_xz.set_data([x_base_current, ax_tip], [z_base_current, az_tip])
   
    # Process each marble individually
    for i, data in enumerate(sim_data):
        # Case A: Marble is still inside the guided channel sliding out
        if frame_time <= data['t_exit']:
            idx = np.searchsorted(data['t_slide'], frame_time)
            idx = min(idx, len(data['t_slide']) - 1)
           
            r_curr = data['r_slide'][idx]
            t_curr = data['t_slide'][idx]
            theta_curr = theta_0 + omega_0 * t_curr + 0.5 * alpha * (t_curr**2)
           
            x_m = r_curr * np.cos(theta_curr) + vel_base_x * t_curr
            y_m = r_curr * np.sin(theta_curr) + vel_base_y * t_curr
            z_m = z_arm
           
            # Draw the path it has traced so far
            t_span_past = data['t_slide'][:idx+1]
            x_trail = data['r_slide'][:idx+1] * np.cos(theta_0 + omega_0 * t_span_past + 0.5 * alpha * (t_span_past**2)) + vel_base_x * t_span_past
            y_trail = data['r_slide'][:idx+1] * np.sin(theta_0 + omega_0 * t_span_past + 0.5 * alpha * (t_span_past**2)) + vel_base_y * t_span_past
            z_trail = np.full_like(x_trail, z_arm)
           
        # Case B: Marble has exited and is falling ballistically
        elif frame_time <= data['t_total']:
            tau_curr = frame_time - data['t_exit']
            x_m = data['x_e'] + data['vx0'] * tau_curr
            y_m = data['y_e'] + data['vy0'] * tau_curr
            z_m = data['z_e'] + data['vz0'] * tau_curr - 0.5 * g * (tau_curr**2)
           
            # Reconstruct the historical track: internal slide + ballistic flight
            t_span_past = data['t_slide']
            x_trail_internal = data['r_slide'] * np.cos(theta_0 + omega_0 * t_span_past + 0.5 * alpha * (t_span_past**2)) + vel_base_x * t_span_past
            y_trail_internal = data['r_slide'] * np.sin(theta_0 + omega_0 * t_span_past + 0.5 * alpha * (t_span_past**2)) + vel_base_y * t_span_past
            z_trail_internal = np.full_like(x_trail_internal, z_arm)
           
            tau_array = np.linspace(0, tau_curr, 100)
            x_trail_flight = data['x_e'] + data['vx0'] * tau_array
            y_trail_flight = data['y_e'] + data['vy0'] * tau_array
            z_trail_flight = data['z_e'] + data['vz0'] * tau_array - 0.5 * g * (tau_array**2)
           
            x_trail = np.concatenate([x_trail_internal, x_trail_flight])
            y_trail = np.concatenate([y_trail_internal, y_trail_flight])
            z_trail = np.concatenate([z_trail_internal, z_trail_flight])
           
        # Case C: Marble has hit the goal floor boundary and freezes
        else:
            tau_end = data['t_total'] - data['t_exit']
            x_m = data['x_e'] + data['vx0'] * tau_end
            y_m = data['y_e'] + data['vy0'] * tau_end
            z_m = z_goal
           
            t_span_past = data['t_slide']
            x_trail_internal = data['r_slide'] * np.cos(theta_0 + omega_0 * t_span_past + 0.5 * alpha * (t_span_past**2)) + vel_base_x * t_span_past
            y_trail_internal = data['r_slide'] * np.sin(theta_0 + omega_0 * t_span_past + 0.5 * alpha * (t_span_past**2)) + vel_base_y * t_span_past
            z_trail_internal = np.full_like(x_trail_internal, z_arm)
           
            tau_array = np.linspace(0, tau_end, 100)
            x_trail_flight = data['x_e'] + data['vx0'] * tau_array
            y_trail_flight = data['y_e'] + data['vy0'] * tau_array
            z_trail_flight = data['z_e'] + data['vz0'] * tau_array - 0.5 * g * (tau_array**2)
           
            x_trail = np.concatenate([x_trail_internal, x_trail_flight])
            y_trail = np.concatenate([y_trail_internal, y_trail_flight])
            z_trail = np.concatenate([z_trail_internal, z_trail_flight])
           
        # Apply data to 3D View
        marbles_3d[i].set_data([x_m], [y_m])
        marbles_3d[i].set_3d_properties([z_m])
        trails_3d[i].set_data(x_trail, y_trail)
        trails_3d[i].set_3d_properties(z_trail)
        
        # Apply data to XY View
        marbles_xy[i].set_data([x_m], [y_m])
        trails_xy[i].set_data(x_trail, y_trail)
        
        # Apply data to XZ View
        marbles_xz[i].set_data([x_m], [z_m])
        trails_xz[i].set_data(x_trail, z_trail)
       
    # Return all updated objects for blitting
    return [base_dot_3d, base_dot_xy, base_dot_xz, 
            arm_line_3d, arm_line_xy, arm_line_xz] + \
           marbles_3d + trails_3d + \
           marbles_xy + trails_xy + \
           marbles_xz + trails_xz

# Note: Setting blit=False is sometimes safer when animating 3D axes alongside 2D in Matplotlib.
ani = animation.FuncAnimation(fig, update, frames=frames_t, interval=25, blit=False, repeat=False)
plt.tight_layout()
plt.show()