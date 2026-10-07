# hardware/end_effector/ak60_motor.py

import can
import time
import threading
import struct
import math
from core.state_manager import event_queue, shared_state

# 아두이노 코드와 동일하게 맞춘 매뉴얼 제원 한계치
P_MIN, P_MAX = -12.5, 12.5
V_MIN, V_MAX = -45.0, 45.0
T_MIN, T_MAX = -18.0, 18.0
Kp_MIN, Kp_MAX = 0.0, 500.0
Kd_MIN, Kd_MAX = 0.0, 5.0

def float_to_uint(x, x_min, x_max, bits):
    """실수값을 CAN 통신용 정수형으로 변환 (아두이노 코드와 100% 동일한 연산)"""
    span = x_max - x_min
    if x < x_min: x = x_min
    elif x > x_max: x = x_max
    return int((x - x_min) * float((1 << bits) - 1) / span)

def pack_mit_cmd(p_des, v_des, kp, kd, t_ff):
    """
    아두이노 코드 기반 정확한 MIT 모드 패킹
    순서: Kp(12) -> Kd(12) -> 위치 P(16) -> 속도 V(12) -> 토크 T(12)
    """
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

def unpack_servo_response(data):
    """
    아두이노 코드 기반 피드백 데이터(6바이트) 언패킹
    V3 프로토콜은 피드백을 int16 형식으로 보냅니다.
    """
    if len(data) < 6:
        return 0.0, 0.0, 0.0
    
    # 빅엔디안(Big-endian) 16비트 부호있는 정수로 변환 (아두이노의 data[0]<<8 | data[1] 와 동일)
    pos_int, spd_int, cur_int = struct.unpack('>hhh', data[0:6])
    
    # 1. 위치: 아두이노는 0.1 degree 단위로 파싱 -> 파이썬에서 Radian으로 변환
    pos_deg = pos_int * 0.1
    pos_rad = pos_deg * (math.pi / 180.0)
    
    # 2. 속도: 아두이노는 10.0 RPM 단위로 파싱 -> 파이썬에서 Rad/s로 변환
    spd_rpm = spd_int * 10.0
    vel_rads = spd_rpm * (math.pi / 30.0)
    
    # 3. 토크(전류): 0.01 A 단위
    torque = cur_int * 0.01
    
    return pos_rad, vel_rads, torque

class MotorController:
    def __init__(self, motor_id=104):
        self.bus = None
        self.motor_id = motor_id
        
        # 상태 변수 (Radian, Rad/s)
        self.pos_fb = 0.0
        self.vel_fb = 0.0
        self.trq_fb = 0.0
        
        # 제어 목표값
        self.p_des = 0.0
        self.v_des = 0.0
        self.kp = 0.0
        self.kd = 0.0
        self.t_ff = 0.0
        
        self.is_enabled = False
        self.running = False
        self.thread = None
        self._lifecycle_lock = threading.Lock()

    def connect(self, port="/dev/ttyACM0", bustype='slcan'):
        with self._lifecycle_lock:
            if self.bus:
                event_queue.put({"type": "STATUS", "msg": "모터가 이미 연결되어 있습니다.", "color": "green"})
                return

            try:
                bus = can.interface.Bus(bustype=bustype, channel=port, bitrate=1000000)
                self.bus = bus
                self.running = True
                self.thread = threading.Thread(target=self._io_loop, daemon=True)
                self.thread.start()
                shared_state.motor_connected = True
                event_queue.put({"type": "STATUS", "msg": f"모터({port}) 연결 완료", "color": "green"})
            except Exception as e:
                self.running = False
                self.thread = None
                self.bus = None
                shared_state.motor_connected = False
                if "bus" in locals():
                    bus.shutdown()
                event_queue.put({"type": "ERROR", "title": "모터 연결 오류", "msg": f"CAN 포트 연결 실패:\n{e}"})

    def enable(self):
        self.is_enabled = True

    def disable(self):
        self.is_enabled = False
        if self.bus:
            # 안전을 위한 전류 차단 (모터 Disable 모드)
            ext_id = (15 << 8) | self.motor_id
            msg = can.Message(arbitration_id=ext_id, data=bytearray(8), is_extended_id=True)
            try:
                self.bus.send(msg)
            except can.CanError:
                pass

    def stop_immediately(self):
        self.disable()

    def disconnect(self):
        with self._lifecycle_lock:
            shared_state.motor_running = False
            self.running = False
            thread = self.thread
            if thread and thread is not threading.current_thread():
                thread.join()

            try:
                try:
                    self.disable()
                finally:
                    if self.bus:
                        self.bus.shutdown()
            finally:
                self.thread = None
                self.bus = None
                self.is_enabled = False
                shared_state.motor_connected = False

    def set_gains_and_targets(self, p, v, kp, kd, t=0.0):
        self.p_des = p
        self.v_des = v
        self.kp = kp
        self.kd = kd
        self.t_ff = t

    def get_feedback(self):
        return self.pos_fb, self.vel_fb, self.trq_fb

    def _io_loop(self):
        while self.running:
            if self.bus:
                # 1. 활성화 시 패킷 송신
                if self.is_enabled:
                    data = pack_mit_cmd(self.p_des, self.v_des, self.kp, self.kd, self.t_ff)
                    ext_id = (8 << 8) | self.motor_id
                    msg = can.Message(arbitration_id=ext_id, data=data, is_extended_id=True)
                    try:
                        self.bus.send(msg)
                    except can.CanError:
                        pass
                
                # 2. 피드백 패킷 수신
                msg_rx = self.bus.recv(timeout=0.005)
                # 아두이노 코드의 0x2968 확인 부분에 대응하여 하위 바이트로 ID 필터링
                if msg_rx and len(msg_rx.data) >= 6:
                    if (msg_rx.arbitration_id & 0xFF) == self.motor_id:
                        try:
                            # 올바른 정수형 언패킹 적용
                            self.pos_fb, self.vel_fb, self.trq_fb = unpack_servo_response(msg_rx.data)
                        except Exception:
                            pass
                        
            time.sleep(0.005) # 200Hz 통신 유지