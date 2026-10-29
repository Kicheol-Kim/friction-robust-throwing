import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from mujoco_throwing_env import MujocoThrowingEnv

def main():
    print("=========================================================")
    print(" 🚀 3D MuJoCo 기반 마찰 강인 투척 PPO 학습 시작")
    print("=========================================================")
    
    # [수정됨] 목표 지점을 X, Y, Z 3차원 좌표로 설정 (예: 거리 1.65m, 중심축 0.0m, 높이 0.5m 타겟)
    env = MujocoThrowingEnv(
        mu_range=(0.1, 0.5), 
        target_pos=(1.65, 0.0, 0.5), 
        omega_motor=15.0
    )
    
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=3e-4, 
        n_steps=512, 
        batch_size=64, 
        n_epochs=10, 
        gamma=0.99, 
        clip_range=0.2, 
        seed=42, 
        verbose=1
    )
    
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=checkpoint_dir, name_prefix="ppo_throw")
    
    start_time = time.time()
    model.learn(total_timesteps=30_000, callback=checkpoint_callback)
    
    final_model_path = os.path.join(checkpoint_dir, "ppo_throwing_model")
    model.save(final_model_path)
    
    elapsed_time = time.time() - start_time
    print(f"\n✅ 학습 완료! (소요 시간: {elapsed_time:.1f}초)")

if __name__ == "__main__":
    main()