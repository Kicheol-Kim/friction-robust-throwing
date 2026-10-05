# core/state_manager.py

import queue
from dataclasses import dataclass

@dataclass
class RobotState:
    """전체 시스템의 상태를 중앙에서 관리하는 데이터 클래스"""
    arm_connected: bool = False
    motor_connected: bool = False
    motor_running: bool = False
    target_motor_vel: float = 0.0
    op_mode: str = "simulation"
    playback_speed: float = 1.0

# UI 스레드와 하드웨어 스레드 간의 안전한 통신을 위한 이벤트 큐
event_queue = queue.Queue()

# 전역 상태 객체
shared_state = RobotState()