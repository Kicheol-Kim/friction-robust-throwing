# main.py

import tkinter as tk
from tkinter import messagebox
import threading
import time

from config.settings import DEFAULT_ROBOT_IP, DEFAULT_CAN_PORT
from core.state_manager import event_queue, shared_state
from end_effector.ak60_motor import MotorController
from robot_arm.rainbow_arm import RobotController

class ThrowSimulatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("로봇팔 직선 이동 + 모터 회전 통합 제어")
        self.root.geometry("600x700")
        
        # 하드웨어 컨트롤러 초기화
        self.motor_ctrl = MotorController()
        self.robot_ctrl = RobotController()
        
        self.create_widgets()
        
        # UI 업데이트용 큐 폴링 루프 시작 (100ms 간격)
        self.root.after(100, self.process_queue)

    def process_queue(self):
        """이벤트 큐를 확인하여 안전하게 UI를 업데이트합니다."""
        while not event_queue.empty():
            event = event_queue.get()
            if event["type"] == "STATUS":
                self.status_label.config(text=f"상태: {event['msg']}", fg=event.get("color", "blue"))
            elif event["type"] == "ERROR":
                self.show_error_dialog(event.get("title", "오류"), event["msg"])
        
        self.root.after(100, self.process_queue)

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
        conn_frame = tk.LabelFrame(self.root, text="1. 환경 설정", padx=10, pady=10)
        conn_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Label(conn_frame, text="로봇팔 IP:").grid(row=0, column=0, sticky="w")
        self.ip_entry = tk.Entry(conn_frame, width=15)
        self.ip_entry.insert(0, DEFAULT_ROBOT_IP)
        self.ip_entry.grid(row=0, column=1, padx=5)
        tk.Button(conn_frame, text="로봇팔 연결", command=lambda: self.robot_ctrl.connect(self.ip_entry.get())).grid(row=0, column=2, padx=5, sticky="we")
        
        tk.Label(conn_frame, text="CAN 포트:").grid(row=1, column=0, sticky="w")
        self.port_entry = tk.Entry(conn_frame, width=15)
        self.port_entry.insert(0, DEFAULT_CAN_PORT)
        self.port_entry.grid(row=1, column=1, padx=5)
        tk.Button(conn_frame, text="모터 연결", command=lambda: self.motor_ctrl.connect(self.port_entry.get())).grid(row=1, column=2, padx=5, sticky="we")

        tk.Label(conn_frame, text="운용 모드:").grid(row=2, column=0, sticky="w", pady=5)
        mode_frame = tk.Frame(conn_frame)
        mode_frame.grid(row=2, column=1, columnspan=2, sticky="w")
        self.op_mode_var = tk.StringVar(value="simulation")
        tk.Radiobutton(mode_frame, text="Simulation", variable=self.op_mode_var, value="simulation", command=self.on_mode_change).pack(side="left")
        tk.Radiobutton(mode_frame, text="Real", variable=self.op_mode_var, value="real", command=self.on_mode_change).pack(side="left")

        tk.Label(conn_frame, text="재생 속도 (0.0~1.0):").grid(row=3, column=0, sticky="w")
        self.playback_speed_entry = tk.Entry(conn_frame, width=15)
        self.playback_speed_entry.insert(0, "1.0")
        self.playback_speed_entry.grid(row=3, column=1, padx=5, sticky="w")
        
        tk.Button(conn_frame, text="전체 연결", command=self.connect_all, bg="lightgray").grid(row=0, column=3, rowspan=4, padx=10, sticky="ns")

        arm_frame = tk.LabelFrame(self.root, text="2. 로봇팔 선형 궤적 (X, Y, Z, Rx, Ry, Rz)", padx=10, pady=10)
        arm_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Label(arm_frame, text="시작점 (Start):").grid(row=0, column=0, sticky="w")
        self.start_entry = tk.Entry(arm_frame, width=35)
        self.start_entry.insert(0, "-750.0, 400.0, 200.0, -180.0, 0.0, 0.0")
        self.start_entry.grid(row=0, column=1)
        
        tk.Label(arm_frame, text="도착점 (End):").grid(row=1, column=0, sticky="w")
        self.end_entry = tk.Entry(arm_frame, width=35)
        self.end_entry.insert(0, "750.0, 400.0, 200.0, -180.0, 0.0, 0.0")
        self.end_entry.grid(row=1, column=1)
        
        tk.Label(arm_frame, text="이동 속도 (mm/s):").grid(row=2, column=0, sticky="w")
        self.arm_speed_entry = tk.Entry(arm_frame, width=10)
        self.arm_speed_entry.insert(0, "200.0")
        self.arm_speed_entry.grid(row=2, column=1, sticky="w")

        motor_frame = tk.LabelFrame(self.root, text="3. 말단 모터 회전 설정", padx=10, pady=10)
        motor_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Label(motor_frame, text="목표 각속도 [Rad/s]:").pack(anchor="w")
        self.vel_scale = tk.Scale(motor_frame, from_=-60.0, to=60.0, resolution=1.0, orient="horizontal")
        self.vel_scale.set(10.0)
        self.vel_scale.pack(fill="x")

        action_frame = tk.Frame(self.root)
        action_frame.pack(fill="x", pady=15)
        
        tk.Button(action_frame, text="기본자세 이동", command=self.cmd_move_home, bg="thistle").pack(side="left", expand=True, fill="x", padx=5)
        tk.Button(action_frame, text="시작점 이동", command=self.cmd_move_start, bg="lightblue").pack(side="left", expand=True, fill="x", padx=5)
        tk.Button(action_frame, text="투척 궤적 실행", command=self.cmd_execute_motion, bg="lightgreen").pack(side="left", expand=True, fill="x", padx=5)
        tk.Button(action_frame, text="■ 정지 (Stop)", command=self.cmd_stop_all, bg="red", fg="white", font=("Helvetica", 10, "bold")).pack(side="right", expand=True, fill="x", padx=5)
        
        self.status_label = tk.Label(self.root, text="상태: 대기 중", fg="blue")
        self.status_label.pack(pady=10)

    def on_mode_change(self):
        self.robot_ctrl.set_op_mode(self.op_mode_var.get())

    def connect_all(self):
        self.robot_ctrl.connect(self.ip_entry.get())
        self.motor_ctrl.connect(self.port_entry.get())

    def cmd_stop_all(self):
        self.motor_ctrl.stop_immediately()
        self.robot_ctrl.stop_immediately()
        event_queue.put({"type": "STATUS", "msg": "사용자 개입: 모든 동작 중지됨", "color": "red"})

    def parse_pose(self, entry_str):
        return [float(x.strip()) for x in entry_str.split(",")]

    def _get_actual_speed(self):
        base_speed = float(self.arm_speed_entry.get())
        try:
            ratio = float(self.playback_speed_entry.get())
        except ValueError:
            ratio = 1.0
        return base_speed * ratio

    def cmd_move_home(self):
        try:
            packaging_joints = [0.0, 0.0, 90.0, 0.0, 90.0, 0.0]
            speed = self._get_actual_speed() * 0.5
            self.robot_ctrl.move_j(packaging_joints, speed, speed * 2.0)
            event_queue.put({"type": "STATUS", "msg": "기본자세(패키징)로 관절 이동 중...", "color": "blue"})
        except Exception as e:
            event_queue.put({"type": "ERROR", "title": "오류", "msg": f"입력값 오류:\n{e}"})

    def cmd_move_start(self):
        try:
            start_pose = self.parse_pose(self.start_entry.get())
            speed = self._get_actual_speed()
            self.robot_ctrl.move_l(start_pose, speed, speed * 2.0)
            event_queue.put({"type": "STATUS", "msg": "시작점(직교좌표)으로 이동 중...", "color": "blue"})
        except Exception as e:
            event_queue.put({"type": "ERROR", "title": "오류", "msg": f"입력값 오류:\n{e}"})

    def cmd_execute_motion(self):
        threading.Thread(target=self._motion_sequence, daemon=True).start()

    def _motion_sequence(self):
        try:
            end_pose = self.parse_pose(self.end_entry.get())
            speed = self._get_actual_speed()
            
            event_queue.put({"type": "STATUS", "msg": "투척 모션 실행 중 (로봇 이동 + 모터 회전)", "color": "red"})
            
            # 모터 상태 동기화
            shared_state.target_motor_vel = self.vel_scale.get()
            shared_state.motor_running = True
            
            # 로봇팔 직선 궤적 실행
            self.robot_ctrl.move_l(end_pose, speed, speed * 2.0)
            time.sleep(3.0) # 가상의 대기 (추후 상태 폴링 로직으로 교체)
                
            shared_state.motor_running = False
            event_queue.put({"type": "STATUS", "msg": "모션 완료 (도착점 도달)", "color": "blue"})
            
        except Exception as e:
            shared_state.motor_running = False
            event_queue.put({"type": "ERROR", "title": "모션 오류", "msg": f"동작 중 오류 발생:\n{e}"})

if __name__ == "__main__":
    root = tk.Tk()
    app = ThrowSimulatorGUI(root)
    root.mainloop()