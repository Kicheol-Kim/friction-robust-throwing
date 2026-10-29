import os
import csv
import datetime
import numpy as np
from stable_baselines3 import PPO

# 무조코 환경 임포트
from mujoco_throwing_env import MujocoThrowingEnv

def extract_and_save_trajectory():
    print("=========================================================")
    print(" 🛠 학습된 PPO 정책 기반 3D 최적 궤적 추출 및 저장")
    print("=========================================================")
    
    motor_target_speed = 15.0
    # [수정] 메인 코드와 동일하게 Z축(0.5)이 포함된 3D 타겟 위치 지정
    target_3d_pos = (1.65, 0.0, 0.5) 
    
    env = MujocoThrowingEnv(
        mu_range=(0.3, 0.3), 
        target_pos=target_3d_pos, 
        omega_motor=motor_target_speed
    )
    
    model_path = os.path.join(os.path.dirname(__file__), "models", "ppo_throwing_model")
    if not os.path.exists(model_path + ".zip"):
        print("❌ 저장된 모델을 찾을 수 없습니다. main_mujoco_ppo.py를 먼저 실행하세요.")
        return
        
    model = PPO.load(model_path)
    
    log_data = []
    obs, info = env.reset()
    done = False
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        
        curr_time = env.steps * env.dt
        curr_tcp_pos = env.tcp_pos.copy()
        curr_tcp_vel = env.tcp_vel.copy()
        cmd_vel = action.copy()
        curr_motor_pos = obs[6]
        curr_motor_vel = obs[7]
        
        row = [
            curr_time, 
            curr_tcp_pos[0], curr_tcp_pos[1], curr_tcp_pos[2],
            curr_tcp_vel[0], curr_tcp_vel[1], curr_tcp_vel[2],
            cmd_vel[0], cmd_vel[1], cmd_vel[2],
            curr_motor_pos, curr_motor_vel
        ]
        log_data.append(row)
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    metadata = {
        "Script": "extract_trajectory.py",
        "Environment": "3D_MujocoThrowingEnv",
        "Algorithm": "PPO",
        "Target_Pos_XYZ": str(target_3d_pos),
        "Motor_Omega_Rads": motor_target_speed,
        "Friction_Coefficient": info.get("mu", 0.3),
        "Control_Frequency_Hz": int(1.0 / env.dt),
        "Final_Landing_X_m": round(info.get("x_land", 0.0), 3),
        "Final_Landing_Y_m": round(info.get("y_land", 0.0), 3),
        "Final_Landing_Z_m": round(info.get("z_land", 0.0), 3)
    }
    
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'rl_trajectories'))
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(log_dir, f"ppo_3d_trajectory_{timestamp}.csv")
    
    try:
        with open(filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            for key, value in metadata.items():
                writer.writerow([f"# {key}: {value}"])
            
            writer.writerow([
                "Time(s)", 
                "TCP_PosX(m)", "TCP_PosY(m)", "TCP_PosZ(m)", 
                "TCP_VelX(m/s)", "TCP_VelY(m/s)", "TCP_VelZ(m/s)", 
                "Cmd_VelX(m/s)", "Cmd_VelY(m/s)", "Cmd_VelZ(m/s)", 
                "Motor_Pos(rad)", "Motor_Vel(rad/s)"
            ])
            
            for row in log_data:
                writer.writerow([round(val, 4) for val in row])
                
        print(f"✅ 3D 궤적 추출 성공! 파일 위치: {filename}")
    except Exception as e:
        print(f"❌ 로그 저장 실패: {e}")

if __name__ == "__main__":
    extract_and_save_trajectory()