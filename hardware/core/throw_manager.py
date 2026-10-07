# hardware/core/throw_manager.py

import time
import threading
import os
from core.state_manager import event_queue, shared_state
from config.config_manager import save_metadata
from end_effector.ak60_motor import MotorController
from robot_arm.rainbow_arm import RobotController
from core.data_logger import DataLogger

class ThrowManager:
    """하드웨어를 총괄하고 투척 시퀀스를 제어하는 Controller 클래스"""
    def __init__(self):
        self.motor_ctrl = MotorController()
        self.robot_ctrl = RobotController()
        self.data_logger = DataLogger(self.robot_ctrl, self.motor_ctrl)

    def connect_all(self, ip, port, mode):
        self.robot_ctrl.connect(ip)
        self.robot_ctrl.set_op_mode(mode)
        self.motor_ctrl.connect(port)

    def stop_all(self):
        shared_state.motor_running = False
        self.motor_ctrl.stop_immediately()
        self.robot_ctrl.stop_immediately()
        self.data_logger.is_logging = False
        event_queue.put({"type": "STATUS", "msg": "사용자 개입: 모든 동작 중지됨", "color": "red"})

    def disconnect_motor(self):
        shared_state.motor_running = False
        self.robot_ctrl.stop_immediately()
        self.motor_ctrl.disconnect()
        event_queue.put({"type": "STATUS", "msg": "모터 연결 해제됨", "color": "blue"})

    def disconnect_robot(self):
        shared_state.motor_running = False
        self.motor_ctrl.stop_immediately()
        self.robot_ctrl.disconnect()

    def disconnect_all(self):
        shared_state.motor_running = False
        self.data_logger.is_logging = False
        try:
            self.motor_ctrl.disconnect()
        finally:
            self.robot_ctrl.disconnect()
        event_queue.put({"type": "STATUS", "msg": "전체 장비 연결 해제됨", "color": "blue"})

    def move_to_start(self, current_meta):
        """별도 스레드에서 로봇팔을 시작점으로 보내고 모터를 초기 위치로 '천천히' 보냅니다."""
        threading.Thread(target=self._move_to_start_task, args=(current_meta,), daemon=True).start()

    def _move_to_start_task(self, meta):
        try:
            save_metadata(meta)
            if not shared_state.motor_connected:
                raise RuntimeError("모터 CAN 포트가 연결되지 않았습니다. 먼저 모터를 연결해 주세요.")
            if not shared_state.arm_connected:
                raise RuntimeError("로봇팔이 연결되지 않았습니다. 먼저 로봇팔을 연결해 주세요.")

            shared_state.motor_running = True
            
            target_pos = float(meta.get("motor_start_pos", 0.0))
            kp = float(meta.get("motor_kp", 30.0))
            kd = float(meta.get("motor_kd", 2.0))
            
            # 모터를 켜고 통신이 안정화될 때까지 잠시 대기 후 실제 현재 위치 확보
            self.motor_ctrl.enable()
            time.sleep(0.1) 
            curr_pos, _, _ = self.motor_ctrl.get_feedback()
            
            # 1. 2초에 걸쳐 천천히 시작 각도로 이동 (부드러운 선형 보간)
            duration = 2.0
            steps = int(duration / 0.01)
            pos_step = (target_pos - curr_pos) / steps if steps > 0 else 0
            
            for _ in range(steps):
                if not shared_state.motor_running:  # 정지 버튼 클릭 시 중단
                    break
                curr_pos += pos_step
                self.motor_ctrl.set_gains_and_targets(p=curr_pos, v=0.0, kp=kp, kd=kd, t=0.0)
                time.sleep(0.01)
                
            if shared_state.motor_running:
                # 최종 위치 고정
                self.motor_ctrl.set_gains_and_targets(p=target_pos, v=0.0, kp=kp, kd=kd, t=0.0)
                
                # 2. 로봇팔 시작점 이동
                start_pose = [float(x.strip()) for x in meta["start_pose"].split(",")]
                self.robot_ctrl.move_l(start_pose, meta["arm_speed"], meta["arm_speed"] * 2.0)
                
                event_queue.put({"type": "STATUS", "msg": f"시작점 세팅 완료 (모터 각도: {target_pos:.2f} Rad)", "color": "blue"})
            else:
                self.motor_ctrl.disable()
            
        except Exception as e:
            shared_state.motor_running = False
            self.motor_ctrl.disable()
            event_queue.put({"type": "ERROR", "title": "이동 오류", "msg": f"시작점 세팅 실패:\n{e}"})

    def execute_throw_sequence(self, current_meta):
        threading.Thread(target=self._motion_task, args=(current_meta,), daemon=True).start()

    def _motion_task(self, meta):
        try:
            save_metadata(meta)
            rot_amount = abs(float(meta.get("motor_rotation_amount", 3.14)))
            target_vel = float(meta.get("motor_target_vel", 10.0))
            if target_vel == 0.0:
                raise ValueError("모터 목표 각속도는 0이 아닌 값이어야 합니다.")
            if rot_amount == 0.0:
                raise ValueError("투척 회전량은 0보다 커야 합니다.")
            if not shared_state.motor_connected:
                raise RuntimeError("모터 CAN 포트가 연결되지 않았습니다. 먼저 모터를 연결해 주세요.")
            if not shared_state.arm_connected:
                raise RuntimeError("로봇팔이 연결되지 않았습니다. 먼저 로봇팔을 연결해 주세요.")

            if meta.get("enable_logging"):
                self.data_logger.start(metadata=meta)
            
            # 속도의 부호로 회전 방향 결정
            direction = 1.0 if target_vel >= 0 else -1.0
            actual_vel = abs(target_vel) * direction
            
            # 사용자 지정 Kp, Kd 로드 (투척 종료 후 홀딩용)
            kd_hold = float(meta.get("motor_kd", 2.0))
            deceleration_time = 0.5
            
            # =========================================================================
            # [핵심] 순수 속도 제어 모드 돌입 (위치 오차가 속도에 개입하지 못하도록 Kp=0 강제)
            # =========================================================================
            self.motor_ctrl.set_gains_and_targets(p=0.0, v=actual_vel, kp=0.0, kd=kd_hold, t=0.0)
            self.motor_ctrl.enable()
            shared_state.motor_running = True
            
            end_pose = [float(x.strip()) for x in meta["end_pose"].split(",")]
            arm_error = []

            def move_arm():
                try:
                    self.robot_ctrl.move_l(end_pose, meta["arm_speed"], meta["arm_speed"] * 2.0)
                except Exception as exc:
                    arm_error.append(exc)

            arm_thread = threading.Thread(target=move_arm, daemon=True)
            arm_thread.start()
            
            # --- 로봇 이동과 동시에 누적 모터 회전량 감시 ---
            accumulated_rotation = 0.0
            prev_time = time.time()
            
            while shared_state.motor_running:
                if arm_error:
                    raise RuntimeError(f"로봇팔 투척 동작 실패: {arm_error[0]}")

                curr_time = time.time()
                dt = curr_time - prev_time
                
                # 모터의 실제 피드백 속도를 가져옴
                _, vel_fb, _ = self.motor_ctrl.get_feedback()
                
                # 피드백 이상치가 회전 완료로 오인되지 않도록 명령 속도 상한을 적용합니다.
                measured_speed = min(abs(vel_fb), abs(target_vel))
                accumulated_rotation += measured_speed * dt
                prev_time = curr_time
                
                remaining_rotation = rot_amount - accumulated_rotation
                braking_distance = abs(target_vel) * deceleration_time * 0.5

                if remaining_rotation <= braking_distance:
                    deceleration_start = curr_time
                    while shared_state.motor_running:
                        elapsed = time.time() - deceleration_start
                        speed_scale = max(0.0, 1.0 - elapsed / deceleration_time)
                        self.motor_ctrl.set_gains_and_targets(
                            p=0.0,
                            v=actual_vel * speed_scale,
                            kp=0.0,
                            kd=kd_hold,
                            t=0.0,
                        )
                        if speed_scale == 0.0:
                            break
                        time.sleep(0.01)
                    break
                    
                time.sleep(0.005) # 200Hz 모니터링
            
            if shared_state.motor_running:
                arm_thread.join()
                if arm_error:
                    raise RuntimeError(f"로봇팔 투척 동작 실패: {arm_error[0]}")

                # 영점으로 끌어당기는 위치 목표를 주지 않고 속도 0으로 감쇠 상태를 유지합니다.
                self.motor_ctrl.set_gains_and_targets(p=0.0, v=0.0, kp=0.0, kd=kd_hold, t=0.0)
                shared_state.motor_running = False
                
                if meta.get("enable_logging"):
                    saved_file = self.data_logger.stop_and_save()
                    if saved_file:
                        event_queue.put({"type": "STATUS", "msg": f"모션 완료: {os.path.basename(saved_file)} 저장됨", "color": "blue"})
                else:
                    event_queue.put({"type": "STATUS", "msg": "모션 완료", "color": "blue"})
                
        except Exception as e:
            self.motor_ctrl.disable()
            shared_state.motor_running = False
            if hasattr(self, 'data_logger'):
                self.data_logger.is_logging = False
            event_queue.put({"type": "ERROR", "title": "오류", "msg": str(e)})