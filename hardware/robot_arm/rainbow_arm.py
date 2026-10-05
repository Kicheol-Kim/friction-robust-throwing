# hardware/rainbow_arm.py

import time
import numpy as np
from core.state_manager import event_queue, shared_state

try:
    import rbpodo
except ImportError:
    rbpodo = None

class RobotController:
    def __init__(self):
        self.robot = None

    def connect(self, ip):
        if self.robot:
            event_queue.put({"type": "STATUS", "msg": "로봇팔이 이미 연결되어 있습니다.", "color": "green"})
            return

        if rbpodo is not None:
            try:
                self.robot = rbpodo.Cobot(ip)
                shared_state.arm_connected = True
                
                # 연결 성공 직후 현재 설정된 운용 모드 즉시 반영
                self.set_op_mode(shared_state.op_mode)
                event_queue.put({"type": "STATUS", "msg": f"로봇팔 연결 완료 ({shared_state.op_mode.upper()})", "color": "green"})
            except Exception as e:
                shared_state.arm_connected = False
                event_queue.put({"type": "ERROR", "title": "로봇팔 연결 오류", "msg": f"연결 실패:\n{e}"})
        else:
            shared_state.arm_connected = True
            event_queue.put({"type": "STATUS", "msg": "가상 로봇팔 모드 (rbpodo 없음)", "color": "green"})

    def set_op_mode(self, mode_str):
        shared_state.op_mode = mode_str
        if self.robot and hasattr(self.robot, 'set_operation_mode'):
            try:
                rc = rbpodo.ResponseCollector()
                mode_val = rbpodo.OperationMode.Simulation if mode_str == "simulation" else rbpodo.OperationMode.Real
                self.robot.set_operation_mode(rc, mode_val)
                event_queue.put({"type": "STATUS", "msg": f"모드 전환 완료: {mode_str.upper()}", "color": "green"})
            except Exception as e:
                event_queue.put({"type": "ERROR", "title": "모드 전환 오류", "msg": f"제어기 모드 전환 중 오류 발생:\n{e}"})
        else:
            event_queue.put({"type": "STATUS", "msg": f"모드 선택됨 (가상): {mode_str.upper()}", "color": "blue"})

    def move_j(self, joints, speed, acc):
        if self.robot:
            rc = rbpodo.ResponseCollector()
            target = np.array(joints, dtype=np.float64)
            self.robot.move_j(rc, target, speed, acc)

    def move_l(self, pose, speed, acc):
        if self.robot:
            rc = rbpodo.ResponseCollector()
            target = np.array(pose, dtype=np.float64)
            self.robot.move_l(rc, target, speed, acc)

    def get_tcp_position(self):
        """현재 TCP의 X, Y, Z 좌표를 Numpy 배열로 반환합니다."""
        if self.robot:
            try:
                # rbpodo API 버전에 따라 함수명이 get_tcp_pose, get_kinematics_info 등일 수 있습니다.
                if hasattr(self.robot, 'get_tcp_pose'):
                    pose = self.robot.get_tcp_pose()
                    return np.array(pose[0:3])
            except Exception as e:
                print(f"TCP 위치 읽기 실패: {e}")
        # 가상 모드이거나 에러 발생 시 더미 데이터 반환
        return np.array([0.0, 0.0, 0.0])

    def stop_immediately(self):
        if self.robot:
            try:
                rc = rbpodo.ResponseCollector()
                if hasattr(self.robot, 'move_stop'):
                    self.robot.move_stop(rc)
                elif hasattr(self.robot, 'task_stop'):
                    self.robot.task_stop(rc)
            except Exception as e:
                print(f"로봇 정지 실패: {e}")