# hardware/ui/main_window.py

import tkinter as tk
from tkinter import ttk, messagebox
import threading
from queue import Empty

# shared_state 추가 임포트
from core.state_manager import event_queue, shared_state

class ThrowSimulatorGUI:
    def __init__(self, root, manager, initial_meta):
        self.root = root
        self.manager = manager
        self.current_meta = initial_meta.copy()
        
        self.meta_vars = {
            "robot_ip": tk.StringVar(value=str(initial_meta.get("robot_ip", ""))),
            "can_port": tk.StringVar(value=str(initial_meta.get("can_port", ""))),
            "op_mode": tk.StringVar(value=str(initial_meta.get("op_mode", "simulation"))),
            "playback_speed": tk.StringVar(value=str(initial_meta.get("playback_speed", 1.0))),
            "start_pose": tk.StringVar(value=str(initial_meta.get("start_pose", ""))),
            "end_pose": tk.StringVar(value=str(initial_meta.get("end_pose", ""))),
            "arm_speed": tk.StringVar(value=str(initial_meta.get("arm_speed", 200.0))),
            "motor_start_pos": tk.StringVar(value=str(initial_meta.get("motor_start_pos", 0.0))),
            "motor_rotation_amount": tk.StringVar(value=str(initial_meta.get("motor_rotation_amount", 3.14))),
            "motor_target_vel": tk.StringVar(value=str(initial_meta.get("motor_target_vel", 10.0))),
            "motor_kp": tk.StringVar(value=str(initial_meta.get("motor_kp", 30.0))), 
            "motor_kd": tk.StringVar(value=str(initial_meta.get("motor_kd", 2.0))),  
        }
        self.enable_logging_var = tk.BooleanVar(value=bool(initial_meta.get("enable_logging", False)))

        self.root.title("로봇팔 직선 이동 + 모터 회전 통합 제어")
        self.root.geometry("620x720") 
        self.root.minsize(600, 650)
        
        self._build_widgets()
        self.root.after(100, self.process_queue)

    def _build_widgets(self):
        main_frame = ttk.Frame(self.root, padding=15)
        main_frame.pack(fill="both", expand=True)

        # 1. 연결 설정 프레임
        conn_frame = ttk.LabelFrame(main_frame, text="1. 연결 설정", padding=10)
        conn_frame.pack(fill="x", pady=(0, 15))
        
        ttk.Label(conn_frame, text="로봇 IP:").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(conn_frame, textvariable=self.meta_vars["robot_ip"], width=15).grid(row=0, column=1, padx=10, pady=4)
        # 버튼을 객체 멤버로 저장하고 명령을 토글 함수로 매핑
        self.btn_connect_arm = ttk.Button(conn_frame, text="로봇 연결", command=self.cmd_toggle_arm)
        self.btn_connect_arm.grid(row=0, column=2, padx=5, pady=4)
        
        ttk.Label(conn_frame, text="모터 포트:").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(conn_frame, textvariable=self.meta_vars["can_port"], width=15).grid(row=1, column=1, padx=10, pady=4)
        # 버튼을 객체 멤버로 저장하고 명령을 토글 함수로 매핑
        self.btn_connect_motor = ttk.Button(conn_frame, text="모터 연결", command=self.cmd_toggle_motor)
        self.btn_connect_motor.grid(row=1, column=2, padx=5, pady=4)
        
        self.btn_connect_all = ttk.Button(conn_frame, text="전체 연결", command=self.cmd_toggle_all)
        self.btn_connect_all.grid(row=0, column=3, rowspan=2, padx=15, sticky="ns")

        # 2. 파라미터 및 궤적 설정 프레임
        param_frame = ttk.LabelFrame(main_frame, text="2. 로봇팔 궤적 설정", padding=10)
        param_frame.pack(fill="x", pady=(0, 15))
        
        ttk.Label(param_frame, text="운용 모드:").grid(row=0, column=0, sticky="w", pady=4)
        mode_frame = ttk.Frame(param_frame)
        mode_frame.grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(mode_frame, text="Simulation", variable=self.meta_vars["op_mode"], value="simulation", command=self.on_mode_change).pack(side="left", padx=(0, 10))
        ttk.Radiobutton(mode_frame, text="Real", variable=self.meta_vars["op_mode"], value="real", command=self.on_mode_change).pack(side="left")

        arm_fields = [
            ("재생 속도 (0.0~1.0):", "playback_speed", 10),
            ("시작 위치 (Start):", "start_pose", 40),
            ("종료 위치 (End):", "end_pose", 40),
            ("이동 속도 (mm/s):", "arm_speed", 10),
        ]
        
        for i, (label_text, var_key, width) in enumerate(arm_fields, start=1):
            ttk.Label(param_frame, text=label_text).grid(row=i, column=0, sticky="w", pady=4)
            ttk.Entry(param_frame, textvariable=self.meta_vars[var_key], width=width).grid(row=i, column=1, sticky="w", pady=4)

        # 3. 말단 모터 설정
        motor_frame = ttk.LabelFrame(main_frame, text="3. 말단 모터 설정", padding=10)
        motor_frame.pack(fill="x", pady=(0, 15))
        
        motor_fields = [
            ("초기 위치 (Start Angle) [Rad]:", "motor_start_pos", 15),
            ("목표 회전량 (Rotation Amount) [Rad]:", "motor_rotation_amount", 15),
            ("목표 각속도 [Rad/s]:", "motor_target_vel", 15),
            ("위치 게인 (Kp):", "motor_kp", 15),
            ("속도 게인 (Kd):", "motor_kd", 15),
        ]

        for i, (label_text, var_key, width) in enumerate(motor_fields):
            ttk.Label(motor_frame, text=label_text).grid(row=i, column=0, sticky="w", pady=4)
            ttk.Entry(motor_frame, textvariable=self.meta_vars[var_key], width=width).grid(row=i, column=1, sticky="w", pady=4, padx=10)

        # 4. 로깅 옵션 및 실행 액션 프레임
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill="x", pady=(5, 0))

        ttk.Checkbutton(action_frame, text="모션 실행 시 TCP 데이터 로깅", variable=self.enable_logging_var).pack(anchor="w", pady=(0, 10))

        btn_frame = ttk.Frame(action_frame)
        btn_frame.pack(fill="x")
        
        ttk.Button(btn_frame, text="기본자세 이동", command=self.cmd_move_home).pack(side="left", expand=True, fill="x", padx=3)
        ttk.Button(btn_frame, text="시작점 이동", command=self.cmd_move_start).pack(side="left", expand=True, fill="x", padx=3)
        ttk.Button(btn_frame, text="투척 모션 실행", command=self.cmd_execute_motion).pack(side="left", expand=True, fill="x", padx=3)
        ttk.Button(btn_frame, text="■ 정지 (Stop)", command=self.cmd_stop_all).pack(side="left", expand=True, fill="x", padx=3)

        ttk.Separator(main_frame).pack(fill="x", pady=15)
        
        self.status_text_widget = tk.Text(main_frame, height=3, wrap="word", font=("", 10), fg="blue", bg=self.root.cget("bg"), relief="flat")
        self.status_text_widget.pack(fill="x")
        self.update_status("대기 중")

    def update_status(self, message, color="blue"):
        self.status_text_widget.config(state="normal", fg=color)
        self.status_text_widget.delete("1.0", tk.END)
        self.status_text_widget.insert("1.0", message)
        self.status_text_widget.config(state="disabled")

    def _update_metadata_from_ui(self):
        self.current_meta.update({key: var.get().strip() for key, var in self.meta_vars.items()})
        
        self.current_meta["playback_speed"] = self._read_float("playback_speed", "재생 속도")
        self.current_meta["arm_speed"] = self._read_float("arm_speed", "로봇 속도")
        
        self.current_meta["motor_start_pos"] = self._read_float("motor_start_pos", "모터 초기 위치")
        self.current_meta["motor_rotation_amount"] = self._read_float("motor_rotation_amount", "투척 회전 량")
        self.current_meta["motor_target_vel"] = self._read_float("motor_target_vel", "모터 목표 각속도")
        self.current_meta["motor_kp"] = self._read_float("motor_kp", "위치 게인 (Kp)")
        self.current_meta["motor_kd"] = self._read_float("motor_kd", "속도 게인 (Kd)")
        
        self.current_meta["enable_logging"] = self.enable_logging_var.get()
        return self.current_meta

    def _read_float(self, key, label):
        try:
            return float(self.meta_vars[key].get().strip())
        except ValueError as exc:
            raise ValueError(f"'{label}' 필드에 올바른 숫자를 입력해 주세요.") from exc

    def _run_manager_action(self, action, *args):
        try:
            action(*args)
        except Exception as exc:
            event_queue.put({"type": "ERROR", "title": "실행 오류", "msg": str(exc)})

    # --- UI 이벤트 콜백 함수들 ---
    def on_mode_change(self):
        self.update_status("모드 전환 중...")
        threading.Thread(
            target=self._run_manager_action,
            args=(self.manager.robot_ctrl.set_op_mode, self.meta_vars["op_mode"].get()),
            daemon=True
        ).start()

    # ---- [수정됨] 토글 연결 기능 ----
    def cmd_toggle_arm(self):
        if shared_state.arm_connected:
            self.update_status("로봇팔 연결 해제 중...")
            threading.Thread(target=self._run_manager_action, args=(self.manager.disconnect_robot,), daemon=True).start()
        else:
            self.update_status("로봇팔 연결 시도 중...")
            threading.Thread(
                target=self._run_manager_action,
                args=(self.manager.robot_ctrl.connect, self.meta_vars["robot_ip"].get()),
                daemon=True
            ).start()

    def cmd_toggle_motor(self):
        if shared_state.motor_connected:
            self.update_status("모터 연결 해제 중...")
            threading.Thread(target=self._run_manager_action, args=(self.manager.disconnect_motor,), daemon=True).start()
        else:
            self.update_status("모터 연결 시도 중...")
            threading.Thread(
                target=self._run_manager_action,
                args=(self.manager.motor_ctrl.connect, self.meta_vars["can_port"].get()),
                daemon=True
            ).start()

    def cmd_toggle_all(self):
        if shared_state.arm_connected and shared_state.motor_connected:
            self.update_status("전체 장비 연결 해제 중...")
            threading.Thread(target=self._run_manager_action, args=(self.manager.disconnect_all,), daemon=True).start()
        else:
            try:
                self._update_metadata_from_ui()
            except ValueError as exc:
                messagebox.showerror("입력 오류", str(exc), parent=self.root)
                return
            self.update_status("전체 장비 연결 중...")
            threading.Thread(
                target=self._run_manager_action,
                args=(
                    self.manager.connect_all,
                    self.current_meta["robot_ip"],
                    self.current_meta["can_port"],
                    self.current_meta["op_mode"],
                ),
                daemon=True,
            ).start()
    # --------------------------------

    def cmd_move_home(self):
        try:
            self._update_metadata_from_ui()
            speed = self.current_meta["arm_speed"] * self.current_meta["playback_speed"] * 0.5
            self.update_status("기본자세로 이동 중...")
            threading.Thread(
                target=self._run_manager_action,
                args=(self.manager.robot_ctrl.move_j, [0.0, 0.0, 90.0, 0.0, 90.0, 0.0], speed, speed * 2.0),
                daemon=True
            ).start()
        except Exception as e:
            messagebox.showerror("입력 오류", str(e), parent=self.root)

    def cmd_move_start(self):
        try:
            self._update_metadata_from_ui()
            self.update_status("시작점 설정 중 (이동 및 모터 초기화)...")
            self.manager.move_to_start(self.current_meta.copy())
        except Exception as e:
            messagebox.showerror("입력 오류", str(e), parent=self.root)

    def cmd_execute_motion(self):
        try:
            self._update_metadata_from_ui()
        except ValueError as exc:
            messagebox.showerror("입력 오류", str(exc), parent=self.root)
            return
        self.update_status("투척 모션 시퀀스 실행 중...")
        self.manager.execute_throw_sequence(self.current_meta.copy())

    def cmd_stop_all(self):
        self.update_status("강제 정지 명령 전송 중...")
        threading.Thread(target=self._run_manager_action, args=(self.manager.stop_all,), daemon=True).start()

    def process_queue(self):
        while True:
            try:
                event = event_queue.get_nowait()
            except Empty:
                break
            if event.get("type") == "ERROR":
                self.update_status(event.get("msg", "오류가 발생했습니다."), color="red")
                messagebox.showerror(event.get("title", "오류"), event.get("msg", "오류가 발생했습니다."), parent=self.root)
            elif event.get("type") == "STATUS":
                self.update_status(event.get("msg", ""), color=event.get("color", "blue"))

        # 전역 상태(shared_state)를 감지하여 연결/해제 버튼 텍스트 자동 전환
        if shared_state.arm_connected:
            self.btn_connect_arm.config(text="로봇 해제")
        else:
            self.btn_connect_arm.config(text="로봇 연결")
            
        if shared_state.motor_connected:
            self.btn_connect_motor.config(text="모터 해제")
        else:
            self.btn_connect_motor.config(text="모터 연결")
            
        if shared_state.arm_connected and shared_state.motor_connected:
            self.btn_connect_all.config(text="전체 해제")
        else:
            self.btn_connect_all.config(text="전체 연결")

        self.root.after(100, self.process_queue)