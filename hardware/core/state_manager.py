# hardware/core/state_manager.py

import queue
from dataclasses import dataclass, field  # field가 추가되었습니다.

@dataclass
class RobotState:
    arm_connected: bool = False
    motor_connected: bool = False
    motor_running: bool = False
    target_motor_vel: float = 0.0
    op_mode: str = "simulation"
    playback_speed: float = 1.0
    
    is_logging: bool = False
    log_time: list = field(default_factory=list)
    log_linear_vel: list = field(default_factory=list)
    log_j6_vel: list = field(default_factory=list)

# UI 모듈과 하드웨어 스레드 간 통신을 위한 큐
event_queue = queue.Queue()
shared_state = RobotState()