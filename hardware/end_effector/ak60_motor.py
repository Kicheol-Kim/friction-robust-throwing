# hardware/ak60_motor.py

import can
import time
import threading
from config.settings import *
from core.state_manager import event_queue, shared_state

def float_to_uint(x, x_min, x_max, bits):
    span = x_max - x_min
    if x < x_min: x = x_min
    elif x > x_max: x = x_max
    return int((x - x_min) * float((1 << bits) - 1) / span)

def pack_mit_cmd(p_des, v_des, kp, kd, t_ff):
    p_int = float_to_uint(p_des, P_MIN, P_MAX, 16)
    v_int = float_to_uint(v_des, V_MIN, V_MAX, 12)
    kp_int = float_to_uint(kp, Kp_MIN, Kp_MAX, 12)
    kd_int = float_to_uint(kd, Kd_MIN, Kd_MAX, 12)
    t_int = float_to_uint(t_ff, T_MIN, T_MAX, 12)

    buffer = bytearray(8)
    buffer[0] = kp_int >> 4
    buffer[1] = ((kp_int & 0xF) << 4) | (kd_int >> 8)
    buffer[2] = kd_int & 0xFF
    buffer[3] = p_int >> 8
    buffer[4] = p_int & 0xFF
    buffer[5] = v_int >> 4
    buffer[6] = ((v_int & 0xF) << 4) | (t_int >> 8)
    buffer[7] = t_int & 0xFF
    return buffer

class MotorController:
    def __init__(self):
        self.bus = None
        self.control_thread = None

    def connect(self, port):
        if self.bus:
            event_queue.put({"type": "STATUS", "msg": "모터가 이미 연결되어 있습니다.", "color": "green"})
            return

        try:
            self.bus = can.interface.Bus(bustype='slcan', channel=port, bitrate=1000000)
            shared_state.motor_connected = True
            event_queue.put({"type": "STATUS", "msg": "모터(CAN) 연결 완료", "color": "green"})
            
            # 연결 성공 시 제어 루프 스레드 시작
            self.control_thread = threading.Thread(target=self._loop, daemon=True)
            self.control_thread.start()
        except Exception as e:
            shared_state.motor_connected = False
            event_queue.put({"type": "ERROR", "title": "모터 연결 오류", "msg": f"CAN 연결 실패:\n{e}"})

    def stop_immediately(self):
        """즉시 모터를 비활성화 상태로 만듭니다."""
        shared_state.motor_running = False
        if self.bus:
            ext_id = (15 << 8) | MOTOR_ID
            msg = can.Message(arbitration_id=ext_id, data=bytearray(8), is_extended_id=True)
            try:
                self.bus.send(msg)
            except Exception:
                pass

    def _loop(self):
        """100Hz로 MIT 모드 명령을 지속적으로 송신하는 루프"""
        while True:
            if shared_state.motor_running and self.bus:
                # 상태 관리자에서 실시간 목표 속도를 읽어옵니다.
                target_vel = shared_state.target_motor_vel
                data = pack_mit_cmd(p_des=0.0, v_des=target_vel, kp=0.0, kd=2.0, t_ff=0.0)
                ext_id = (8 << 8) | MOTOR_ID
                msg = can.Message(arbitration_id=ext_id, data=data, is_extended_id=True)
                
                try:
                    self.bus.send(msg)
                except Exception as e:
                    # 너무 잦은 에러 출력을 막기 위해 연결이 끊어지면 처리
                    pass
            else:
                if self.bus and not shared_state.motor_running:
                    # 안전을 위한 Disable 명령 전송
                    ext_id = (15 << 8) | MOTOR_ID
                    msg = can.Message(arbitration_id=ext_id, data=bytearray(8), is_extended_id=True)
                    try:
                        self.bus.send(msg)
                    except Exception:
                        pass
            time.sleep(0.01)