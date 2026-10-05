# config/settings.py

# --- AK60-6 V3 모터 제원 ---
P_MIN = -12.56
P_MAX = 12.56
V_MIN = -60.0
V_MAX = 60.0
T_MIN = -12.0
T_MAX = 12.0
Kp_MIN = 0.0
Kp_MAX = 500.0
Kd_MIN = 0.0
Kd_MAX = 5.0

MOTOR_ID = 104

# --- 로봇팔 기본 설정 ---
DEFAULT_ROBOT_IP = "192.168.0.25"
DEFAULT_CAN_PORT = "/dev/ttyACM0"