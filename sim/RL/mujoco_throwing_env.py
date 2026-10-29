import os

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


class MujocoThrowingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    ROBOT_JOINT_NAMES = ("base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3")
    ROBOT_HOME = np.array(
        [-0.950692785, -0.131682190, 1.193488540, -1.061806360, 0.950692785, 0.0],
        dtype=np.float64,
    )
    MAX_TCP_OFFSET = 0.08

    def __init__(
        self,
        mu_range=(0.1, 0.5),
        target_pos=(1.65, 0.0, 0.0),
        omega_motor=15.0,
    ):
        super().__init__()

        xml_path = os.path.join(os.path.dirname(__file__), "assets", "throw_scene.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep

        self.max_vel = 1.2
        self.max_acc = 8.0
        self.omega_motor = omega_motor
        self.theta_init = -np.pi / 3
        self.target_pos = np.asarray(target_pos, dtype=np.float64)
        self.mu_min, self.mu_max = mu_range

        self.action_space = spaces.Box(
            low=-self.max_vel, high=self.max_vel, shape=(3,), dtype=np.float32
        )
        obs_high = np.full((11,), 10.0, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
        )

        self.robot_joint_ids = np.asarray(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.ROBOT_JOINT_NAMES
            ],
            dtype=np.intp,
        )
        self.robot_qpos_ids = self.model.jnt_qposadr[self.robot_joint_ids].copy()
        self.robot_dof_ids = self.model.jnt_dofadr[self.robot_joint_ids].copy()
        self.robot_joint_limits = self.model.jnt_range[self.robot_joint_ids].copy()

        mocap_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target_tcp_mocap"
        )
        self.mocap_id = self.model.body_mocapid[mocap_body_id]
        self.tcp_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "link6"
        )
        self.thrower_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "thrower_assembly"
        )
        self.motor_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "ak60_motor"
        )
        self.motor_qpos_id = self.model.jnt_qposadr[self.motor_joint_id]
        self.motor_dof_id = self.model.jnt_dofadr[self.motor_joint_id]
        self.motor_act_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "ak60_torque"
        )
        self.ball_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "projectile_ball"
        )
        self.ball_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "projectile_freejoint"
        )
        self.ball_qpos_id = self.model.jnt_qposadr[self.ball_joint_id]
        self.ball_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "proj_geom"
        )
        self.ground_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground"
        )

        self.gripper_geom_ids = np.asarray(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                for name in ("gripper_bottom", "gripper_left", "gripper_right")
            ],
            dtype=np.intp,
        )
        self.fixed_tcp_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.joint_targets = self.ROBOT_HOME.copy()
        self.tcp_home_pos = np.zeros(3, dtype=np.float64)
        self.tcp_pos = np.zeros(3, dtype=np.float32)
        self.tcp_target_pos = np.zeros(3, dtype=np.float32)
        self.tcp_vel = np.zeros(3, dtype=np.float32)
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        sampled_mu = float(self.np_random.uniform(self.mu_min, self.mu_max))
        self.model.geom_friction[self.gripper_geom_ids, 0] = sampled_mu

        self.data.qpos[self.robot_qpos_ids] = self.ROBOT_HOME
        self.data.qpos[self.motor_qpos_id] = self.theta_init
        self.joint_targets = self.ROBOT_HOME.copy()
        self.tcp_pos = np.zeros(3, dtype=np.float32)
        self.tcp_target_pos = np.zeros(3, dtype=np.float32)
        self.tcp_vel = np.zeros(3, dtype=np.float32)
        self.commanded_tcp_vel = np.zeros(3, dtype=np.float32)
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.steps = 0

        mujoco.mj_forward(self.model, self.data)
        self.tcp_home_pos = self.data.xpos[self.tcp_body_id].copy()
        self.data.mocap_pos[self.mocap_id] = self.tcp_home_pos
        self.data.mocap_quat[self.mocap_id] = self.fixed_tcp_quat

        thrower_rotation = self.data.xmat[self.thrower_body_id].reshape(3, 3)
        local_ball_position = np.array([0.05, 0.0, 0.015])
        self.data.qpos[self.ball_qpos_id : self.ball_qpos_id + 3] = (
            self.data.xpos[self.thrower_body_id]
            + thrower_rotation @ local_ball_position
        )
        self.data.qpos[self.ball_qpos_id + 3 : self.ball_qpos_id + 7] = (
            self.data.xquat[self.thrower_body_id]
        )
        mujoco.mj_forward(self.model, self.data)

        self.sampled_mu = sampled_mu
        return self._get_obs(), {"mu": sampled_mu}

    def get_tcp_position(self):
        return (self.data.xpos[self.tcp_body_id] - self.tcp_home_pos).copy()

    def get_tcp_velocity(self):
        jacobian = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacBody(
            self.model,
            self.data,
            jacobian,
            np.zeros((3, self.model.nv), dtype=np.float64),
            self.tcp_body_id,
        )
        return jacobian @ self.data.qvel

    def _orientation_error(self):
        current_quat = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(
            current_quat, self.data.xmat[self.tcp_body_id].reshape(-1)
        )
        inverse_quat = np.empty(4, dtype=np.float64)
        mujoco.mju_negQuat(inverse_quat, current_quat)
        error_quat = np.empty(4, dtype=np.float64)
        mujoco.mju_mulQuat(error_quat, self.fixed_tcp_quat, inverse_quat)
        if error_quat[0] < 0:
            error_quat *= -1

        vector_norm = np.linalg.norm(error_quat[1:])
        if vector_norm < 1e-12:
            return np.zeros(3, dtype=np.float64)
        angle = 2.0 * np.arctan2(vector_norm, error_quat[0])
        return error_quat[1:] * (angle / vector_norm)

    def _solve_tcp_ik(self, target_position):
        actual_qpos = self.data.qpos[self.robot_qpos_ids].copy()
        candidate = self.joint_targets.copy()
        lower = self.robot_joint_limits[:, 0] + 1e-3
        upper = self.robot_joint_limits[:, 1] - 1e-3

        for _ in range(20):
            self.data.qpos[self.robot_qpos_ids] = candidate
            mujoco.mj_forward(self.model, self.data)

            position_error = target_position - self.data.xpos[self.tcp_body_id]
            rotation_error = self._orientation_error()
            error = np.concatenate((position_error, rotation_error))
            if (
                np.linalg.norm(position_error) < 2e-4
                and np.linalg.norm(rotation_error) < 1e-3
            ):
                break

            jacobian_position = np.zeros((3, self.model.nv), dtype=np.float64)
            jacobian_rotation = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jacBody(
                self.model,
                self.data,
                jacobian_position,
                jacobian_rotation,
                self.tcp_body_id,
            )
            jacobian = np.vstack(
                (
                    jacobian_position[:, self.robot_dof_ids],
                    jacobian_rotation[:, self.robot_dof_ids],
                )
            )
            damping = 0.01
            update = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping**2 * np.eye(6), error
            )
            update_norm = np.linalg.norm(update)
            if update_norm > 0.15:
                update *= 0.15 / update_norm
            candidate = np.clip(candidate + update, lower, upper)
        else:
            position_error_size = float(np.linalg.norm(position_error))
            rotation_error_size = float(np.linalg.norm(rotation_error))
            self.data.qpos[self.robot_qpos_ids] = actual_qpos
            mujoco.mj_forward(self.model, self.data)
            raise RuntimeError(
                "TCP inverse kinematics did not converge "
                f"(position error={position_error_size:.4f} m, "
                f"orientation error={rotation_error_size:.4f} rad)."
            )

        self.data.qpos[self.robot_qpos_ids] = actual_qpos
        mujoco.mj_forward(self.model, self.data)
        self.joint_targets = candidate
        return candidate.copy()

    def _get_obs(self):
        self.tcp_pos = self.get_tcp_position().astype(np.float32)
        self.tcp_vel = self.get_tcp_velocity().astype(np.float32)
        theta = self.data.qpos[self.motor_qpos_id]
        dtheta = self.data.qvel[self.motor_dof_id]
        return np.concatenate(
            [
                self.tcp_pos,
                self.tcp_vel,
                [theta, dtheta],
                self.prev_action,
            ]
        ).astype(np.float32)

    def _has_ground_contact(self):
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            if {contact.geom1, contact.geom2} == {
                self.ground_geom_id,
                self.ball_geom_id,
            }:
                return True
        return False

    def step(self, action):
        command = np.clip(action, self.action_space.low, self.action_space.high)
        acceleration = np.clip(
            (command - self.commanded_tcp_vel) / self.dt,
            -self.max_acc,
            self.max_acc,
        )
        self.commanded_tcp_vel += acceleration * self.dt
        next_tcp_pos = self.tcp_target_pos + self.commanded_tcp_vel * self.dt
        self.tcp_target_pos = np.clip(
            next_tcp_pos, -self.MAX_TCP_OFFSET, self.MAX_TCP_OFFSET
        ).astype(np.float32)
        clipped = np.abs(self.tcp_target_pos - next_tcp_pos) > 1e-8
        self.commanded_tcp_vel[clipped] = 0.0

        target_position = self.tcp_home_pos + self.tcp_target_pos
        self.data.mocap_pos[self.mocap_id] = target_position
        self.data.mocap_quat[self.mocap_id] = self.fixed_tcp_quat
        # IK drives the arm kinematically; the launcher and projectile stay dynamic.
        previous_configuration = self.data.qpos[self.robot_qpos_ids].copy()
        robot_configuration = self._solve_tcp_ik(target_position)
        robot_velocity = (robot_configuration - previous_configuration) / self.dt
        self.data.qpos[self.robot_qpos_ids] = robot_configuration
        self.data.qvel[self.robot_dof_ids] = robot_velocity
        mujoco.mj_forward(self.model, self.data)

        motor_velocity = self.data.qvel[self.motor_dof_id]
        torque = 2.0 * (self.omega_motor - motor_velocity)
        self.data.ctrl[self.motor_act_id] = np.clip(torque, -12.0, 12.0)
        mujoco.mj_step(self.model, self.data)
        self.data.qpos[self.robot_qpos_ids] = robot_configuration
        self.data.qvel[self.robot_dof_ids] = robot_velocity
        mujoco.mj_forward(self.model, self.data)

        self.steps += 1
        self.prev_action = command.copy()
        terminated = self._has_ground_contact()
        truncated = not terminated and self.steps >= 500
        ball_pos = self.data.xpos[self.ball_body_id].copy()

        reward = -0.001 * float(np.sum(command**2))
        info = {
            "mu": self.sampled_mu,
            "tcp_position": self.get_tcp_position().copy(),
            "tcp_orientation_error_rad": float(
                np.linalg.norm(self._orientation_error())
            ),
            "landed": terminated,
        }
        if terminated:
            xy_error = float(np.linalg.norm(ball_pos[:2] - self.target_pos[:2]))
            reward -= 2.0 * xy_error**2
            info.update(
                {
                    "x_land": float(ball_pos[0]),
                    "y_land": float(ball_pos[1]),
                    "z_land": float(ball_pos[2]),
                    "target_xy_error": xy_error,
                }
            )
        elif truncated:
            reward -= 1.0

        return self._get_obs(), reward, terminated, truncated, info

    def close(self):
        return None
