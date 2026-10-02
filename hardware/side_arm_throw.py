import tkinter as tk
from tkinter import messagebox
import can
import time
import threading
import os
import numpy as np

try:
    import rbpodo
except ImportError:
    print("경고: rbpodo 라이브러리를 찾을 수 없습니다.")
    rbpodo = None

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

class ThrowSimulatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("로봇팔 직선 이동 + 모터 회전 통합 제어")
        self.root.geometry("600x700")
        
        self.can_bus = None
        self.robot = None
        self.motor_running = False
        self.target_motor_vel = 0.0
        
        self.create_widgets()
        
        self.motor_thread = threading.Thread(target=self.motor_control_loop)
        self.motor_thread.daemon = True
        self.motor_thread.start()

    def show_error_dialog(self, title, message):
        err_win = tk.Toplevel(self.root)
        err_win.title(title)
        err_win.geometry("600x250")
        err_win.transient(self.root)
        err_win.grab_set()
        
        txt = tk.Text(err_win, wrap="word", font=("Consolas", 10))
        txt.insert("1.0", message)
        txt.config(state="disabled")
        txt.pack(expand=True, fill="both", padx=10, pady=10)
        
        btn = tk.Button(err_win, text="확인", command=err_win.destroy, bg="lightgray")
        btn.pack(pady=10)

    def create_widgets(self):
        # 1. 환경 설정
        conn_frame = tk.LabelFrame(self.root, text="1. 환경 설정", padx=10, pady=10)
        conn_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Label(conn_frame, text="로봇팔 IP:").grid(row=0, column=0, sticky="w")
        self.ip_entry = tk.Entry(conn_frame, width=15)
        self.ip_entry.insert(0, "192.168.0.25")
        self.ip_entry.grid(row=0, column=1, padx=5)
        self.btn_connect_robot = tk.Button(conn_frame, text="로봇팔 연결", command=self.connect_robot_arm)
        self.btn_connect_robot.grid(row=0, column=2, padx=5, sticky="we")
        
        tk.Label(conn_frame, text="CAN 포트:").grid(row=1, column=0, sticky="w")
        self.port_entry = tk.Entry(conn_frame, width=15)
        self.port_entry.insert(0, "/dev/ttyACM0")
        self.port_entry.grid(row=1, column=1, padx=5)
        self.btn_connect_motor = tk.Button(conn_frame, text="모터 연결", command=self.connect_motor)
        self.btn_connect_motor.grid(row=1, column=2, padx=5, sticky="we")

        # 운용 모드 선택 (Simulation / Real)
        tk.Label(conn_frame, text="운용 모드:").grid(row=2, column=0, sticky="w", pady=5)
        mode_frame = tk.Frame(conn_frame)
        mode_frame.grid(row=2, column=1, columnspan=2, sticky="w")
        self.op_mode_var = tk.StringVar(value="simulation")
        tk.Radiobutton(mode_frame, text="Simulation", variable=self.op_mode_var, value="simulation", 
                       command=self.toggle_operation_mode).pack(side="left")
        tk.Radiobutton(mode_frame, text="Real", variable=self.op_mode_var, value="real", 
                       command=self.toggle_operation_mode).pack(side="left")

        tk.Label(conn_frame, text="재생 속도 (0.0~1.0):").grid(row=3, column=0, sticky="w")
        self.playback_speed_entry = tk.Entry(conn_frame, width=15)
        self.playback_speed_entry.insert(0, "1.0")
        self.playback_speed_entry.grid(row=3, column=1, padx=5, sticky="w")
        
        self.btn_connect_all = tk.Button(conn_frame, text="전체 연결", command=self.connect_all, bg="lightgray")
        self.btn_connect_all.grid(row=0, column=3, rowspan=4, padx=10, sticky="ns")

        # 2. 로봇팔 궤적 설정
        arm_frame = tk.LabelFrame(self.root, text="2. 로봇팔 선형 궤적 (X, Y, Z, Rx, Ry, Rz)", padx=10, pady=10)
        arm_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Label(arm_frame, text="시작점 (Start):").grid(row=0, column=0, sticky="w")
        self.start_entry = tk.Entry(arm_frame, width=35)
        self.start_entry.insert(0, "400.0, 0.0, 300.0, 180.0, 0.0, 0.0")
        self.start_entry.grid(row=0, column=1)
        
        tk.Label(arm_frame, text="도착점 (End):").grid(row=1, column=0, sticky="w")
        self.end_entry = tk.Entry(arm_frame, width=35)
        self.end_entry.insert(0, "600.0, 0.0, 500.0, 180.0, 0.0, 0.0")
        self.end_entry.grid(row=1, column=1)
        
        tk.Label(arm_frame, text="이동 속도 (mm/s):").grid(row=2, column=0, sticky="w")
        self.arm_speed_entry = tk.Entry(arm_frame, width=10)
        self.arm_speed_entry.insert(0, "200.0")
        self.arm_speed_entry.grid(row=2, column=1, sticky="w")

        # 3. 말단 모터 설정
        motor_frame = tk.LabelFrame(self.root, text="3. 말단 모터 회전 설정", padx=10, pady=10)
        motor_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Label(motor_frame, text="목표 각속도 [Rad/s]:").pack(anchor="w")
        self.vel_scale = tk.Scale(motor_frame, from_=-60.0, to=60.0, resolution=1.0, orient="horizontal")
        self.vel_scale.set(10.0)
        self.vel_scale.pack(fill="x")

        # 4. 실행 및 중지 버튼
        action_frame = tk.Frame(self.root)
        action_frame.pack(fill="x", pady=15)
        
        self.btn_move_home = tk.Button(action_frame, text="기본자세 이동", command=self.move_to_packaging, bg="thistle")
        self.btn_move_home.pack(side="left", expand=True, fill="x", padx=5)
        
        self.btn_move_start = tk.Button(action_frame, text="시작점 이동", command=self.move_to_start, bg="lightblue")
        self.btn_move_start.pack(side="left", expand=True, fill="x", padx=5)
        
        self.btn_execute = tk.Button(action_frame, text="투척 궤적 실행", command=self.execute_motion, bg="lightgreen")
        self.btn_execute.pack(side="left", expand=True, fill="x", padx=5)
        
        # 긴급 중지 버튼 추가
        self.btn_stop = tk.Button(action_frame, text="■ 정지 (Stop)", command=self.stop_motion, bg="red", fg="white", font=("Helvetica", 10, "bold"))
        self.btn_stop.pack(side="right", expand=True, fill="x", padx=5)
        
        self.status_label = tk.Label(self.root, text="상태: 대기 중", fg="blue")
        self.status_label.pack(pady=10)

    def set_status(self, msg, color="blue"):
        self.status_label.config(text=f"상태: {msg}", fg=color)

    def connect_motor(self):
        if not self.can_bus:
            try:
                self.can_bus = can.interface.Bus(bustype='slcan', channel=self.port_entry.get(), bitrate=1000000)
                self.set_status("모터(CAN) 연결 완료", "green")
            except Exception as e:
                self.show_error_dialog("연결 오류", f"CAN 연결 실패:\n{e}")
        else:
            self.set_status("모터가 이미 연결되어 있습니다.", "green")

    def connect_robot_arm(self):
        if not self.robot and rbpodo is not None:
            try:
                self.robot = rbpodo.Cobot(self.ip_entry.get())
                
                # 연결 성공 직후, 현재 UI에 선택된 모드를 로봇에 동기화
                self.toggle_operation_mode()
                
            except Exception as e:
                self.show_error_dialog("연결 오류", f"로봇팔 연결 실패:\n{e}")
        elif rbpodo is None:
            self.set_status("가상 로봇팔 모드 (rbpodo 없음)", "green")
        else:
            self.set_status("로봇팔이 이미 연결되어 있습니다.", "green")

    def toggle_operation_mode(self):
        """UI에서 모드(Sim/Real)를 토글할 때마다 로봇 제어기에 즉각 명령을 보냅니다."""
        mode_str = self.op_mode_var.get()
        
        # 로봇이 연결된 상태일 때만 제어기로 명령 전송
        if self.robot:
            try:
                rc = rbpodo.ResponseCollector()
                
                # 정수(0, 1) 대신 rbpodo.OperationMode 열거형(Enum) 객체를 전달합니다.
                if mode_str == "simulation":
                    mode_val = rbpodo.OperationMode.Simulation
                else:
                    mode_val = rbpodo.OperationMode.Real
                
                if hasattr(self.robot, 'set_operation_mode'):
                    self.robot.set_operation_mode(rc, mode_val)
                    self.set_status(f"운용 모드 전환 완료: {mode_str.upper()}", "green")
                else:
                    self.set_status("현재 API 버전에서 set_operation_mode를 지원하지 않습니다.", "orange")
            except Exception as e:
                self.show_error_dialog("모드 전환 오류", f"제어기 모드 전환 중 오류 발생:\n{e}")
        else:
            # 로봇이 연결되지 않은 상태라면 UI 변수만 바뀌고 대기
            self.set_status(f"모드 선택됨 (로봇 연결 시 적용): {mode_str.upper()}", "blue")

    def connect_all(self):
        self.connect_robot_arm()
        self.connect_motor()
        if self.can_bus and (self.robot or rbpodo is None):
            self.set_status("로봇팔 및 모터 전체 연결 완료", "green")

    def stop_motion(self):
        """즉시 모터를 비활성화하고 로봇팔의 동작을 중지합니다."""
        # 1. 모터 중지
        self.motor_running = False
        if self.can_bus:
            try:
                ext_id = (15 << 8) | MOTOR_ID
                msg = can.Message(arbitration_id=ext_id, data=bytearray(8), is_extended_id=True)
                self.can_bus.send(msg)
            except:
                pass
                
        # 2. 로봇팔 중지 (Real 모드일 때만 호출)
        if self.robot and self.op_mode_var.get() == "real":
            try:
                rc = rbpodo.ResponseCollector()
                # rbpodo 라이브러리의 정지 함수 호출 (예: move_stop, task_stop 등)[cite: 1]
                if hasattr(self.robot, 'move_stop'):
                    self.robot.move_stop(rc)
                elif hasattr(self.robot, 'task_stop'):
                    self.robot.task_stop(rc)
            except Exception as e:
                print(f"로봇 정지 실패: {e}")
                
        self.set_status("사용자 개입: 모든 동작 중지됨", "red")

    def parse_pose(self, entry_str):
        return [float(x.strip()) for x in entry_str.split(",")]

    def _get_actual_speed_acc(self, base_speed):
        try:
            ratio = float(self.playback_speed_entry.get())
        except:
            ratio = 1.0
        actual_speed = base_speed * ratio
        return actual_speed, actual_speed * 2.0

    def move_to_packaging(self):
        try:
            packaging_joints = [0.0, 0.0, 90.0, 0.0, 90.0, 0.0] 
            speed, acc = self._get_actual_speed_acc(float(self.arm_speed_entry.get()) * 0.5)
            
            # 모드(Real/Sim)에 상관없이 로봇 객체가 있으면 무조건 명령 전송!
            if self.robot:
                rc = rbpodo.ResponseCollector()
                target_joints = np.array(packaging_joints, dtype=np.float64)
                self.robot.move_j(rc, target_joints, speed, acc)
                self.set_status("기본자세(패키징)로 관절 이동 중...")
            else:
                self.set_status(f"[가상] 패키징 조인트 이동 (속도: {speed})")
        except Exception as e:
            self.show_error_dialog("오류 상세 내용", f"기본자세 이동 실패:\n\n{e}")

    def move_to_start(self):
        try:
            start_pose = self.parse_pose(self.start_entry.get())
            speed, acc = self._get_actual_speed_acc(float(self.arm_speed_entry.get()))
            
            # 모드에 상관없이 로봇 객체가 있으면 무조건 명령 전송!
            if self.robot:
                rc = rbpodo.ResponseCollector()
                target_point = np.array(start_pose, dtype=np.float64)
                self.robot.move_l(rc, target_point, speed, acc)
                self.set_status("시작점(직교좌표)으로 이동 중...")
            else:
                self.set_status(f"[가상] 시작점 이동 (속도: {speed}): {start_pose}")
        except Exception as e:
            self.show_error_dialog("오류 상세 내용", f"시작점 이동 실패:\n\n{e}")

    def execute_motion(self):
        threading.Thread(target=self._motion_sequence, daemon=True).start()

    def _motion_sequence(self):
        try:
            end_pose = self.parse_pose(self.end_entry.get())
            arm_speed, acc = self._get_actual_speed_acc(float(self.arm_speed_entry.get()))
            op_mode = self.op_mode_var.get()
            
            self.set_status(f"[{op_mode.upper()}] 투척 모션 실행 중", "red")
            
            self.target_motor_vel = self.vel_scale.get()
            self.motor_running = True
            
            # 모드에 상관없이 로봇 객체가 있으면 무조건 명령 전송!
            if self.robot:
                rc = rbpodo.ResponseCollector()
                target_point = np.array(end_pose, dtype=np.float64)
                self.robot.move_l(rc, target_point, arm_speed, acc)
                time.sleep(3.0) 
            else:
                time.sleep(3.0)
                
            self.motor_running = False
            self.set_status("모션 완료 (도착점 도달)", "blue")
            
        except Exception as e:
            self.motor_running = False
            self.show_error_dialog("오류 상세 내용", f"동작 중 오류 발생:\n\n{e}")

    def motor_control_loop(self):
        while True:
            if self.motor_running and self.can_bus:
                data = pack_mit_cmd(p_des=0.0, v_des=self.target_motor_vel, kp=0.0, kd=2.0, t_ff=0.0)
                ext_id = (8 << 8) | MOTOR_ID
                msg = can.Message(arbitration_id=ext_id, data=data, is_extended_id=True)
                
                try:
                    self.can_bus.send(msg)
                except Exception as e:
                    print("CAN 송신 에러:", e)
            else:
                if self.can_bus and not self.motor_running:
                    ext_id = (15 << 8) | MOTOR_ID
                    msg = can.Message(arbitration_id=ext_id, data=bytearray(8), is_extended_id=True)
                    try:
                        self.can_bus.send(msg)
                    except:
                        pass
            time.sleep(0.01)

if __name__ == "__main__":
    root = tk.Tk()
    app = ThrowSimulatorGUI(root)
    root.mainloop()