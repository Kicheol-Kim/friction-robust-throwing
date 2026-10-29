import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

class MujocoThrowingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, mu_range=(0.1, 0.5), target_pos=(1.65, 0.0, 0.0), omega_motor=15.0):
        super().__init__()
        
        xml_path = os.path.join(os.path.dirname(__file__), "assets", "throw_scene.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep
        
        self.max_vel = 1.2          
        self.max_acc = 8.0          
        
        self.omega_motor = omega_motor     
        self.theta_init = -np.pi/3  
        
        # [수정됨] 목표 지점을 X, Y, Z 3차원 좌표로 저장
        self.target_pos = np.array(target_pos, dtype=np.float32)
        self.mu_min, self.mu_max = mu_range
        
        self.action_space = spaces.Box(low=-self.max_vel, high=self.max_vel, shape=(3,), dtype=np.float32)
        
        obs_high = np.full((11,), 10.0, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)
        
        mocap_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_tcp_mocap")
        self.mocap_id = self.model.body_mocapid[mocap_body_id]
        self.motor_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ak60_motor")
        self.motor_act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "ak60_torque")
        self.ball_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "projectile_ball")
        
        self.gripper_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ["gripper_bottom", "gripper_left", "gripper_right"]
        ]
        
        self.base_mocap_pos = np.array([0.5, 0.0, 0.5])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        sampled_mu = self.np_random.uniform(self.mu_min, self.mu_max)
        for geom_id in self.gripper_geom_ids:
            self.model.geom_friction[geom_id][0] = sampled_mu
            
        self.tcp_pos = np.zeros(3, dtype=np.float32)
        self.tcp_vel = np.zeros(3, dtype=np.float32)
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.steps = 0
        
        self.data.mocap_pos[self.mocap_id] = self.base_mocap_pos
        self.data.qpos[self.motor_joint_id] = self.theta_init
        
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {"mu": sampled_mu}
        
    def _get_obs(self):
        theta = self.data.qpos[self.motor_joint_id]
        dtheta = self.data.qvel[self.motor_joint_id]
        return np.concatenate([
            self.tcp_pos, 
            self.tcp_vel, 
            [theta, dtheta], 
            self.prev_action
        ]).astype(np.float32)

    def step(self, action):
        cmd_vel = np.clip(action, self.action_space.low, self.action_space.high)
        ddx = np.clip((cmd_vel - self.tcp_vel) / self.dt, -self.max_acc, self.max_acc)
        
        self.tcp_vel += ddx * self.dt
        self.tcp_pos += self.tcp_vel * self.dt
        
        self.data.mocap_pos[self.mocap_id] = self.base_mocap_pos + self.tcp_pos
        
        motor_vel = self.data.qvel[self.motor_joint_id]
        torque = 2.0 * (self.omega_motor - motor_vel)
        self.data.ctrl[self.motor_act_id] = np.clip(torque, -12.0, 12.0)
        
        mujoco.mj_step(self.model, self.data)
        
        self.steps += 1
        self.prev_action = cmd_vel.copy()
        
        ball_pos = self.data.xpos[self.ball_body_id]
        
        # [핵심 수정] 공의 현재 Z 높이가 목표 Z 높이(target_pos[2]) 이하로 떨어지면 시뮬레이션 종료
        terminated = bool(ball_pos[2] <= self.target_pos[2] and self.steps > 20)
        truncated = bool(self.steps >= 500)
        
        reward = -0.001 * np.sum((cmd_vel - self.tcp_vel) ** 2)
        info = {}
        
        if terminated or truncated:
            # 3D 유클리디안 오차(Mean Squared Error)로 탄착 페널티 계산
            target_err_sq = np.sum((ball_pos - self.target_pos) ** 2)
            reward += -float(2.0 * target_err_sq)
            
            info["x_land"] = ball_pos[0]
            info["y_land"] = ball_pos[1]
            info["z_land"] = ball_pos[2]  # Z축 착지 정보 추가
            
        return self._get_obs(), reward, terminated, truncated, info