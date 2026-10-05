# hardware/config/config_manager.py

import json
import os

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "user_settings.json")

# 최초 실행 시 적용될 기본 메타데이터
DEFAULT_METADATA = {
    "script_name": "main_side_throw.py",
    "robot_ip": "192.168.0.25",
    "can_port": "/dev/ttyACM0",
    "op_mode": "simulation",
    "playback_speed": 1.0,
    "start_pose": "-750.0, 400.0, 200.0, -180.0, 0.0, 0.0",
    "end_pose": "750.0, 400.0, 200.0, -180.0, 0.0, 0.0",
    "arm_speed": 200.0,
    "motor_target_vel": 10.0,
    "enable_logging": False
}

def load_metadata():
    """저장된 JSON 설정 파일에서 메타데이터를 불러옵니다. 없으면 기본값을 반환합니다."""
    if not os.path.exists(CONFIG_FILE):
        return DEFAULT_METADATA.copy()
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"설정 파일 로드 실패 (기본값 사용): {e}")
        return DEFAULT_METADATA.copy()

def save_metadata(metadata):
    """현재 UI 상태의 메타데이터를 JSON 파일로 덮어씌워 저장합니다."""
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"설정 파일 저장 실패: {e}")