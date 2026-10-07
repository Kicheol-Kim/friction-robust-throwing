import tkinter as tk
from tkinter import ttk, messagebox
import os
from end_effector.ak60_motor import CustomMotorController

class MotorTesterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("자체 CAN 모터 제어 툴 (AK60-6 V3)")
        self.root.geometry("450x620")
        
        self.motor = CustomMotorController(motor_id=104)
        self.is_connected = False
        
        self.port_var = tk.StringVar(value=self._find_default_port())
        self.id_var = tk.StringVar(value="104")
        self.target_pos_var = tk.DoubleVar(value=0.0)
        self.target_vel_var = tk.DoubleVar(value=5.0)
        self.kp_var = tk.DoubleVar(value=5.0)
        self.kd_var = tk.DoubleVar(value=2.0)
        
        self.status_var = tk.StringVar(value="상태: 연결 대기 중...")
        self.pos_fb_var = tk.StringVar(value="0.000")
        self.vel_fb_var = tk.StringVar(value="0.000")
        
        self._create_widgets()
        self._update_loop()

    def _create_widgets(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill="both", expand=True)

        conn_frame = ttk.LabelFrame(main_frame, text="1. 통신 설정 (slcan)", padding=10)
        conn_frame.pack(fill="x", pady=5)
        ttk.Label(conn_frame, text="포트:").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(conn_frame, textvariable=self.port_var, width=15).grid(row=0, column=1, padx=5)
        ttk.Label(conn_frame, text="모터 ID:").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(conn_frame, textvariable=self.id_var, width=15).grid(row=1, column=1, padx=5)
        self.btn_connect = ttk.Button(conn_frame, text="연결", command=self.toggle_connect)
        self.btn_connect.grid(row=0, column=2, rowspan=2, padx=10, sticky="ns")

        ctrl_frame = ttk.LabelFrame(main_frame, text="2. MIT 모드 제어", padding=10)
        ctrl_frame.pack(fill="x", pady=10)
        self._add_scale(ctrl_frame, "목표 위치 [Rad]", self.target_pos_var, -12.5, 12.5)
        self._add_scale(ctrl_frame, "목표 속도 [Rad/s]", self.target_vel_var, -50.0, 50.0)
        self._add_scale(ctrl_frame, "Kp (강성) [0~50]", self.kp_var, 0.0, 50.0)
        self._add_scale(ctrl_frame, "Kd (댐핑) [0~5]", self.kd_var, 0.0, 5.0)

        btn_frame = ttk.Frame(ctrl_frame)
        btn_frame.pack(fill="x", pady=10)
        self.btn_enable = tk.Button(btn_frame, text="모터 활성화 (Enable)", command=self.enable_motor, state="disabled", bg="lightgreen")
        self.btn_enable.pack(side="left", expand=True, fill="x", padx=5)
        self.btn_disable = tk.Button(btn_frame, text="비활성화 (Disable)", command=self.disable_motor, state="disabled", bg="salmon")
        self.btn_disable.pack(side="right", expand=True, fill="x", padx=5)

        fb_frame = ttk.LabelFrame(main_frame, text="3. 모터 상태 모니터링", padding=10)
        fb_frame.pack(fill="x", pady=5)
        ttk.Label(fb_frame, text="현재 위치 (Rad):").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Label(fb_frame, textvariable=self.pos_fb_var, font=("", 10, "bold")).grid(row=0, column=1, sticky="e", pady=2)
        ttk.Label(fb_frame, text="현재 속도 (Rad/s):").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Label(fb_frame, textvariable=self.vel_fb_var, font=("", 10, "bold")).grid(row=1, column=1, sticky="e", pady=2)

        ttk.Separator(main_frame).pack(fill="x", pady=10)
        ttk.Label(main_frame, textvariable=self.status_var, foreground="blue", wraplength=410).pack(anchor="w")

    def _find_default_port(self):
        for i in range(3):
            port = f"/dev/ttyACM{i}"
            if os.path.exists(port): return port
        return "/dev/ttyACM0"

    def _add_scale(self, parent, label, variable, minimum, maximum):
        ttk.Label(parent, text=label).pack(anchor="w")
        ttk.Scale(parent, from_=minimum, to=maximum, variable=variable, orient="horizontal").pack(fill="x", pady=2)
        ttk.Label(parent, textvariable=variable).pack(anchor="e")

    def toggle_connect(self):
        if self.is_connected:
            self.motor.disconnect()
            self.is_connected = False
            self.status_var.set("상태: 연결 해제됨")
            self.btn_connect.config(text="연결")
            self.btn_enable.config(state="disabled")
            self.btn_disable.config(state="disabled")
        else:
            try:
                self.motor.motor_id = int(self.id_var.get())
                # slcan으로 직접 연결 (네트워크 설정 불필요)
                self.motor.connect(port=self.port_var.get(), bustype='slcan')
                self.is_connected = True
                self.status_var.set(f"상태: {self.port_var.get()} 연결 성공")
                self.btn_connect.config(text="연결 해제")
                self.btn_enable.config(state="normal")
                self.btn_disable.config(state="normal")
            except Exception as e:
                messagebox.showerror("연결 오류", f"실패:\n{e}")

    def enable_motor(self):
        self.motor.enable()
        self.status_var.set("상태: 활성화 됨 (명령 전송 중)")

    def disable_motor(self):
        self.motor.disable()
        self.status_var.set("상태: 비활성화 됨 (대기)")

    def _update_loop(self):
        if self.is_connected:
            self.motor.set_gains_and_targets(
                self.target_pos_var.get(),
                self.target_vel_var.get(),
                self.kp_var.get(),
                self.kd_var.get()
            )
            pos, vel, _ = self.motor.get_feedback()
            self.pos_fb_var.set(f"{pos:.3f}")
            self.vel_fb_var.set(f"{vel:.3f}")
            
        self.root.after(20, self._update_loop)

    def on_closing(self):
        if self.is_connected:
            self.motor.disconnect()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = MotorTesterGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()