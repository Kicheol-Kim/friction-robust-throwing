# hardware/core/data_logger.py

import time
import threading
import csv
import os
import datetime
import numpy as np

class DataLogger:
    def __init__(self, robot_ctrl, motor_ctrl):
        self.robot_ctrl = robot_ctrl
        self.motor_ctrl = motor_ctrl
        self.is_logging = False
        
        # 수집 버퍼 초기화
        self.log_time = []
        self.log_linear_vel = []
        self.log_motor_pos = []
        self.log_motor_vel = []
        
        self.metadata = {}
        self._thread = None

    def start(self, metadata=None):
        self.metadata = metadata if metadata else {}
        self.log_time.clear()
        self.log_linear_vel.clear()
        self.log_motor_pos.clear()
        self.log_motor_vel.clear()
        
        self.is_logging = True
        self._thread = threading.Thread(target=self._log_loop, daemon=True)
        self._thread.start()

    def _log_loop(self):
        """200Hz로 로봇 TCP와 모터 데이터를 백그라운드에서 수집합니다."""
        start_time = time.time()
        prev_time = start_time
        prev_tcp = self.robot_ctrl.get_tcp_position()

        while self.is_logging:
            curr_time = time.time()
            curr_tcp = self.robot_ctrl.get_tcp_position()
            
            # 모터의 현재 위치와 속도 읽기 (토크 값은 사용하지 않으므로 _ 로 무시)
            curr_motor_pos, curr_motor_vel, _ = self.motor_ctrl.get_feedback()

            if curr_tcp is not None and prev_tcp is not None:
                dt = curr_time - prev_time
                if dt > 0:
                    # TCP 선속도 계산
                    lin_vel = np.linalg.norm(curr_tcp - prev_tcp) / dt
                    
                    self.log_time.append(curr_time - start_time)
                    self.log_linear_vel.append(lin_vel)
                    self.log_motor_pos.append(curr_motor_pos)
                    self.log_motor_vel.append(curr_motor_vel)
            
            prev_tcp = curr_tcp
            prev_time = curr_time
            time.sleep(0.005) # 200Hz 폴링

    def stop_and_save(self):
        self.is_logging = False
        if self._thread:
            self._thread.join(timeout=1.0)
        
        if not self.log_time:
            return None
        
        log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw_logs'))
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(log_dir, f"throw_log_with_motor_{timestamp}.csv")
        
        try:
            with open(filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                
                # 메타데이터 기록 (plot 시 파싱 용도)
                for key, value in self.metadata.items():
                    writer.writerow([f"# {key}: {value}"])
                
                # 데이터 헤더
                writer.writerow(["Time(s)", "TCP_Vel(mm/s)", "Motor_Pos(rad)", "Motor_Vel(rad/s)"])
                
                # 데이터 기록
                for t, tv, mp, mv in zip(self.log_time, self.log_linear_vel, self.log_motor_pos, self.log_motor_vel):
                    writer.writerow([t, tv, mp, mv])
                    
            return filename
        except Exception as e:
            print(f"로그 저장 실패: {e}")
            return None