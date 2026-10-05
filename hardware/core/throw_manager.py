import time
import threading
import os
from core.state_manager import event_queue, shared_state
from config.config_manager import save_metadata
from end_effector.ak60_motor import MotorController
from robot_arm.rainbow_arm import RobotController
from log_arm_velocity import DataLogger

class ThrowManager:
    """하드웨어를 총괄하고 투척 시퀀스를 제어하는 Controller 클래스"""
    def __init__(self):
        self.motor_ctrl = MotorController()
        self.robot_ctrl = RobotController()
        self.data_logger = DataLogger(self.robot_ctrl)

    def connect_all(self, ip, port, mode):
        self.robot_ctrl.connect(ip)
        self.robot_ctrl.set_op_mode(mode)
        self.motor_ctrl.connect(port)

    def stop_all(self):
        self.motor_ctrl.stop_immediately()
        self.robot_ctrl.stop_immediately()
        self.data_logger.is_logging = False
        event_queue.put({"type": "STATUS", "msg": "모든 동작 중지됨", "color": "red"})

    def execute_throw_sequence(self, current_meta):
        """별도 스레드에서 모션과 로깅을 실행합니다."""
        threading.Thread(target=self._motion_task, args=(current_meta,), daemon=True).start()

    def _motion_task(self, meta):
        try:
            # 1. 메타데이터 저장 및 로깅 시작
            save_metadata(meta)
            if meta.get("enable_logging"):
                self.data_logger.start(metadata=meta)
            
            # 2. 로봇 이동 및 모터 회전
            end_pose = [float(x.strip()) for x in meta["end_pose"].split(",")]
            shared_state.target_motor_vel = meta["motor_target_vel"]
            shared_state.motor_running = True
            
            self.robot_ctrl.move_l(end_pose, meta["arm_speed"], meta["arm_speed"] * 2.0)
            time.sleep(3.0) # 임시 대기
            
            # 3. 종료 및 데이터 저장
            shared_state.motor_running = False
            if meta.get("enable_logging"):
                saved_file = self.data_logger.stop_and_save()
                if saved_file:
                    event_queue.put({"type": "STATUS", "msg": f"완료: {os.path.basename(saved_file)} 저장됨", "color": "blue"})
            else:
                event_queue.put({"type": "STATUS", "msg": "모션 완료", "color": "blue"})
                
        except Exception as e:
            shared_state.motor_running = False
            self.data_logger.is_logging = False
            event_queue.put({"type": "ERROR", "title": "오류", "msg": str(e)})