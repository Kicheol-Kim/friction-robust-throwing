import tkinter as tk
from tkinter import messagebox
import can
import struct
import time
import threading
import os

# AK60-6 V3 모터 파라미터 제원 (매뉴얼 기준)
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

def float_to_uint(x, x_min, x_max, bits):
    """실수값을 CAN 통신용 정수형으로 변환"""
    span = x_max - x_min
    if x < x_min: x = x_min
    elif x > x_max: x = x_max
    return int((x - x_min) * float((1 << bits) - 1) / span)

def pack_mit_cmd(p_des, v_des, kp, kd, t_ff):
    """MIT 모드 제어 데이터 패킹 (8바이트)"""
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

class MotorTesterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AK60-6 V3 모터 UI 테스터")
        self.root.geometry("400x500")
        
        self.bus = None
        self.is_enabled = False
        self.running = True
        
        self.create_widgets()
        
        # 100Hz로 CAN 메시지를 쏘는 백그라운드 스레드 시작
        self.control_thread = threading.Thread(target=self.control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()

    def create_widgets(self):
        # --- 통신 설정 프레임 ---
        conn_frame = tk.LabelFrame(self.root, text="연결 설정", padx=10, pady=10)
        conn_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Label(conn_frame, text="포트 (예: /dev/ttyACM0):").grid(row=0, column=0, sticky="w")
        self.port_entry = tk.Entry(conn_frame, width=15)
        self.port_entry.insert(0, self.find_default_port())
        self.port_entry.grid(row=0, column=1)
        
        tk.Label(conn_frame, text="모터 ID:").grid(row=1, column=0, sticky="w")
        self.id_entry = tk.Entry(conn_frame, width=15)
        self.id_entry.insert(0, "104")
        self.id_entry.grid(row=1, column=1)
        
        self.btn_connect = tk.Button(conn_frame, text="연결", command=self.connect_can, bg="lightgray")
        self.btn_connect.grid(row=0, column=2, rowspan=2, padx=10, sticky="ns")

        # --- 제어 설정 프레임 ---
        ctrl_frame = tk.LabelFrame(self.root, text="모터 제어 (MIT Mode)", padx=10, pady=10)
        ctrl_frame.pack(fill="x", padx=10, pady=5)
        
        # 안전을 위해 Kp의 UI 최대값을 500이 아닌 50으로 제한합니다.
        tk.Label(ctrl_frame, text="Kp (강성) [0~50]:").pack(anchor="w")
        self.kp_scale = tk.Scale(ctrl_frame, from_=0, to=50, resolution=0.5, orient="horizontal")
        self.kp_scale.set(0.0)
        self.kp_scale.pack(fill="x")
        
        tk.Label(ctrl_frame, text="Kd (댐핑) [0~5]:").pack(anchor="w")
        self.kd_scale = tk.Scale(ctrl_frame, from_=0, to=5, resolution=0.1, orient="horizontal")
        self.kd_scale.set(0.5)
        self.kd_scale.pack(fill="x")
        
        tk.Label(ctrl_frame, text="목표 위치 (Position) [Rad]:").pack(anchor="w")
        self.pos_scale = tk.Scale(ctrl_frame, from_=-6.28, to=6.28, resolution=0.01, orient="horizontal")
        self.pos_scale.set(0.0)
        self.pos_scale.pack(fill="x")
        
        btn_frame = tk.Frame(ctrl_frame)
        btn_frame.pack(fill="x", pady=10)
        
        self.btn_enable = tk.Button(btn_frame, text="모터 활성화 (Enable)", command=self.enable_motor, state="disabled", bg="lightgreen")
        self.btn_enable.pack(side="left", expand=True, fill="x", padx=5)
        
        self.btn_disable = tk.Button(btn_frame, text="비활성화 (Disable)", command=self.disable_motor, state="disabled", bg="salmon")
        self.btn_disable.pack(side="right", expand=True, fill="x", padx=5)
        
        # --- 상태 표시창 ---
        self.status_label = tk.Label(self.root, text="상태: 연결 대기 중...", fg="blue")
        self.status_label.pack(pady=10)

    def find_default_port(self):
        """리눅스 환경에서 기본으로 잡히는 CANable 포트를 자동 탐색합니다."""
        for i in range(3):
            port = f"/dev/ttyACM{i}"
            if os.path.exists(port): return port
        return "/dev/ttyACM0"

    def connect_can(self):
        if self.bus is None:
            port = self.port_entry.get()
            try:
                # 리눅스 네트워크 세팅 없이 slcan 타입으로 시리얼 포트에 직접 연결
                self.bus = can.interface.Bus(bustype='slcan', channel=port, bitrate=1000000)
                self.motor_id = int(self.id_entry.get())
                self.status_label.config(text=f"상태: {port} 포트에 연결됨.", fg="green")
                self.btn_connect.config(text="연결 해제")
                self.btn_enable.config(state="normal")
                self.btn_disable.config(state="normal")
            except Exception as e:
                self.show_error_dialog("오류 상세 내용", f"입력값 오류 또는 로봇 제어 실패:\n\n{e}")
        else:
            self.is_enabled = False
            self.bus.shutdown()
            self.bus = None
            self.status_label.config(text="상태: 연결 해제됨.", fg="black")
            self.btn_connect.config(text="연결")
            self.btn_enable.config(state="disabled")
            self.btn_disable.config(state="disabled")

    def enable_motor(self):
        if not self.bus: return
        # MIT 제어 루프 활성화
        self.is_enabled = True
        self.status_label.config(text="상태: 모터 활성화됨 (MIT 모드 출력 중)", fg="green")

    def disable_motor(self):
        self.is_enabled = False
        if self.bus:
            # 매뉴얼 상 Disable 명령 (Control Mode ID: 15)
            ext_id = (15 << 8) | self.motor_id
            msg = can.Message(arbitration_id=ext_id, data=bytearray(8), is_extended_id=True)
            self.bus.send(msg)
            self.status_label.config(text="상태: 모터 비활성화됨 (Disable)", fg="red")

    def control_loop(self):
        """100Hz로 MIT 모드 명령을 지속적으로 쏘는 스레드"""
        while self.running:
            if self.is_enabled and self.bus is not None:
                p_des = self.pos_scale.get()
                kp = self.kp_scale.get()
                kd = self.kd_scale.get()
                
                # 목표속도 0, 전향토크 0으로 설정하여 오직 위치와 강성(Kp, Kd)으로 제어
                data = pack_mit_cmd(p_des=p_des, v_des=0.0, kp=kp, kd=kd, t_ff=0.0)
                
                # MIT 제어 모드 ID: 8
                ext_id = (8 << 8) | self.motor_id
                msg = can.Message(arbitration_id=ext_id, data=data, is_extended_id=True)
                
                try:
                    self.bus.send(msg)
                except Exception as e:
                    print("Send Error:", e)
            time.sleep(0.01)

    def on_closing(self):
        self.running = False
        if self.bus:
            self.disable_motor()
            time.sleep(0.1)
            self.bus.shutdown()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = MotorTesterGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()