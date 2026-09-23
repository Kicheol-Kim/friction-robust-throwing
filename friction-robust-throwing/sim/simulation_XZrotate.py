import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

# =============================================================================
# --- Physical Constants & Configuration ---
# =============================================================================
r0 = 0.9             # Initial radial position of the marble (m)
L = 1.0              # Total length of the arm link (m)
g = 9.81             # Acceleration due to gravity (m/s^2)

# Rotational Kinematics Parameters
omega_0 = np.deg2rad(100)        # Initial angular velocity of the arm (rad/s)
alpha = 0.0          # Constant angular acceleration of the arm (rad/s^2)

# PARAMETRIZE START ANGLE HERE (in degrees)
theta_0_deg = -20  
theta_0 = np.radians(theta_0_deg)

# --- GOAL HEIGHT PLANE ---
y_goal = -1        # Target horizontal plane altitude (m in lab coordinates)

# --- NEW PARAMETER: CONSTANT HORIZONTAL BASE VELOCITY ---
v_b = 0           # Constant speed of the base along the +X axis (m/s)

# Friction coefficients to simulate
mu_values = [0, 0.15, 0.30, 0.45, 0.60]
# mu_values = np.linspace(0.0, 0.5, 51)  # 11 values from 0 to 0.5

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
    f_gravity_radial = -g * np.sin(theta_t) # Radial component of gravity
    a_normal = 2 * omega_t * v + alpha * r + g * np.cos(theta_t) # Normal acceleration component along the radial direction

    v_scale = 0.005  
    f_friction = -mu * np.abs(a_normal) * np.tanh(v / v_scale) # Friction opposes motion with smooth transition near zero velocity
    
    v_dot = f_centrifugal + f_gravity_radial + f_friction # Total radial acceleration
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
    t_span = (0.0, 15.0)
    
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
   
    # Base is at (v_b * t_exit, 0) at the exit instant
    x_e = r_e * np.cos(theta_e) + v_b * t_exit
    y_e = r_e * np.sin(theta_e)
   
    # Compound Launch Velocities (Rotational Components + Linear Base Velocity Shift)
    vx0 = r_dot_exit * np.cos(theta_e) - r_e * omega_e * np.sin(theta_e) + v_b
    vy0 = r_dot_exit * np.sin(theta_e) + r_e * omega_e * np.cos(theta_e)
   
    # 2. Phase 2: Ballistic Free Flight
    a_q = 0.5 * g
    b_q = -vy0
    c_q = y_goal - y_e
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
        'vx0': vx0, 'vy0': vy0,
        'x_e': x_e, 'y_e': y_e
    })

# =============================================================================
# --- Animation Environment Setup ---
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 10))
colors = plt.cm.viridis(np.linspace(0, 0.85, len(mu_values)))

# Master time array running the animation frames
frames_t = np.linspace(0, max_global_time, 250)

# Structural Plot Elements
ax.axhline(y_goal, color='red', linewidth=1.5, linestyle='-.', label=f'Goal Height ({y_goal}m)')
ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)

# Visual objects to update dynamically per frame
base_dot, = ax.plot([], [], 'ko', markersize=10, label='Moving Base Pivot')
arm_line, = ax.plot([], [], color='black', linestyle='-', linewidth=3, alpha=0.7, label='Rotating Arm Link')

marble_plots = []
trail_plots = []

for dataset, color in zip(sim_data, colors):
    m_dot, = ax.plot([], [], 'o', color=color, markersize=8, zorder=5)
    t_line, = ax.plot([], [], '-', color=color, linewidth=2, label=rf'$\mu$ = {dataset["mu"]:.2f}')
    marble_plots.append(m_dot)
    trail_plots.append(t_line)

# Formatting bounds safely to encase the entire horizontal moving workspace
ax.set_xlim(min(-1.5, v_b * max_global_time - 5), max(1.5, v_b * max_global_time + 5))
ax.set_ylim(y_goal - 0.5, 2)
ax.set_aspect('equal')
ax.grid(True, linestyle=':', alpha=0.5)
ax.set_title(f'Moving-Base Dynamic Release Animation (v_b = {v_b} m/s Horizontal Scroll)', fontsize=12)
ax.set_xlabel('Lab Frame X Position (m)')
ax.set_ylabel('Lab Frame Y Position (m)')
ax.legend(loc='upper right')

# =============================================================================
# --- Animation Core Loop Engine ---
# =============================================================================
def update(frame_time):
    # Update the linear position of the moving base pivot
    x_base_current = v_b * frame_time
    base_dot.set_data([x_base_current], [0])
   
    # Update the arm orientation based on frame time
    theta_current = theta_0 + omega_0 * frame_time + 0.5 * alpha * (frame_time**2)
    ax_tip = x_base_current + L * np.cos(theta_current)
    ay_tip = L * np.sin(theta_current)
    arm_line.set_data([x_base_current, ax_tip], [0, ay_tip])
   
    # Process each marble individually
    for i, data in enumerate(sim_data):
        # Case A: Marble is still inside the guided channel sliding out
        if frame_time <= data['t_exit']:
            # Linearly sample or find closest pre-calculated structural step
            idx = np.searchsorted(data['t_slide'], frame_time)
            idx = min(idx, len(data['t_slide']) - 1)
           
            r_curr = data['r_slide'][idx]
            t_curr = data['t_slide'][idx]
            theta_curr = theta_0 + omega_0 * t_curr + 0.5 * alpha * (t_curr**2)
           
            x_m = r_curr * np.cos(theta_curr) + v_b * t_curr
            y_m = r_curr * np.sin(theta_curr)
           
            # Draw the path it has traced so far in lab frame
            t_span_past = data['t_slide'][:idx+1]
            x_trail = data['r_slide'][:idx+1] * np.cos(theta_0 + omega_0 * t_span_past + 0.5 * alpha * (t_span_past**2)) + v_b * t_span_past
            y_trail = data['r_slide'][:idx+1] * np.sin(theta_0 + omega_0 * t_span_past + 0.5 * alpha * (t_span_past**2))
           
        # Case B: Marble has exited and is falling ballistically
        elif frame_time <= data['t_total']:
            tau_curr = frame_time - data['t_exit']
            x_m = data['x_e'] + data['vx0'] * tau_curr
            y_m = data['y_e'] + data['vy0'] * tau_curr - 0.5 * g * (tau_curr**2)
           
            # Reconstruct the historical track: internal slide sequence + ballistic flight sequence
            t_span_past = data['t_slide']
            x_trail_internal = data['r_slide'] * np.cos(theta_0 + omega_0 * t_span_past + 0.5 * alpha * (t_span_past**2)) + v_b * t_span_past
            y_trail_internal = data['r_slide'] * np.sin(theta_0 + omega_0 * t_span_past + 0.5 * alpha * (t_span_past**2))
           
            tau_array = np.linspace(0, tau_curr, 100)
            x_trail_flight = data['x_e'] + data['vx0'] * tau_array
            y_trail_flight = data['y_e'] + data['vy0'] * tau_array - 0.5 * g * (tau_array**2)
           
            x_trail = np.concatenate([x_trail_internal, x_trail_flight])
            y_trail = np.concatenate([y_trail_internal, y_trail_flight])
           
        # Case C: Marble has hit the goal floor boundary and freezes
        else:
            tau_end = data['t_total'] - data['t_exit']
            x_m = data['x_e'] + data['vx0'] * tau_end
            y_m = y_goal
           
            t_span_past = data['t_slide']
            x_trail_internal = data['r_slide'] * np.cos(theta_0 + omega_0 * t_span_past + 0.5 * alpha * (t_span_past**2)) + v_b * t_span_past
            y_trail_internal = data['r_slide'] * np.sin(theta_0 + omega_0 * t_span_past + 0.5 * alpha * (t_span_past**2))
           
            tau_array = np.linspace(0, tau_end, 100)
            x_trail_flight = data['x_e'] + data['vx0'] * tau_array
            y_trail_flight = data['y_e'] + data['vy0'] * tau_array - 0.5 * g * (tau_array**2)
           
            x_trail = np.concatenate([x_trail_internal, x_trail_flight])
            y_trail = np.concatenate([y_trail_internal, y_trail_flight])
           
        marble_plots[i].set_data([x_m], [y_m])
        trail_plots[i].set_data(x_trail, y_trail)
       
    return [base_dot, arm_line] + marble_plots + trail_plots

ani = animation.FuncAnimation(fig, update, frames=frames_t, interval=25, blit=True)
plt.tight_layout()
plt.show()
