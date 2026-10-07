# hardware/config/config_manager.py

import json
import os

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "user_settings.json")

DEFAULT_METADATA = {
    "script_name": "main_side_throw.py",
    "robot_ip": "192.168.0.25",
    "can_port": "/dev/ttyACM0",
    "op_mode": "simulation",
    "playback_speed": 1.0,
    "start_pose": "-750.0, 400.0, 200.0, -180.0, 0.0, 0.0",
    "end_pose": "750.0, 400.0, 200.0, -180.0, 0.0, 0.0",
    "arm_speed": 200.0,
    "motor_start_pos": 0.0,
    "motor_rotation_amount": 3.14,
    "motor_target_vel": 10.0,
    "motor_kp": 30.0,  # 추가됨: 위치 게인
    "motor_kd": 2.0,   # 추가됨: 속도 게인
    "enable_logging": False
}

def load_metadata():
    if not os.path.exists(CONFIG_FILE):
        return DEFAULT_METADATA.copy()
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"설정 파일 로드 실패 (기본값 사용): {e}")
        return DEFAULT_METADATA.copy()

def save_metadata(metadata):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"설정 파일 저장 실패: {e}")