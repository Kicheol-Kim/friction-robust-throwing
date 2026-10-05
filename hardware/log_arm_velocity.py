# hardware/log_arm_velocity.py

import time
import threading
import csv
import os
import datetime
import numpy as np

class DataLogger:
    def __init__(self, robot_ctrl):
        self.robot_ctrl = robot_ctrl
        self.is_logging = False
        self.log_time = []
        self.log_linear_vel = []
        self.log_j6_vel = []
        self.metadata = {}
        self._thread = None

    def start(self, metadata=None):
        """데이터 수집 스레드를 시작합니다. 메타데이터(딕셔너리)를 받아 추후 CSV에 기록합니다."""
        self.metadata = metadata if metadata else {}
        self.log_time.clear()
        self.log_linear_vel.clear()
        self.log_j6_vel.clear()
        
        self.is_logging = True
        self._thread = threading.Thread(target=self._log_loop, daemon=True)
        self._thread.start()

    def _log_loop(self):
        """200Hz로 로봇 TCP와 J6 관절 데이터를 백그라운드에서 수집합니다."""
        start_time = time.time()
        prev_time = start_time
        prev_tcp, prev_j6 = self.robot_ctrl.get_realtime_data()

        while self.is_logging:
            curr_time = time.time()
            curr_tcp, curr_j6 = self.robot_ctrl.get_realtime_data()

            if curr_tcp is not None and prev_tcp is not None and curr_j6 is not None and prev_j6 is not None:
                dt = curr_time - prev_time
                if dt > 0:
                    lin_vel = np.linalg.norm(curr_tcp[0:3] - prev_tcp[0:3]) / dt
                    j6_vel = (curr_j6 - prev_j6) / dt

                    self.log_time.append(curr_time - start_time)
                    self.log_linear_vel.append(lin_vel)
                    self.log_j6_vel.append(j6_vel)
            
            prev_tcp = curr_tcp
            prev_j6 = curr_j6
            prev_time = curr_time
            time.sleep(0.005) # 200Hz

    def stop_and_save(self):
        """수집을 중단하고 메타데이터와 함께 CSV 파일로 저장한 뒤 파일명을 반환합니다."""
        self.is_logging = False
        if self._thread:
            self._thread.join(timeout=1.0)
        
        if not self.log_time:
            return None
        
        log_dir = os.path.join(os.path.dirname(__file__), '../data/raw_logs')
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(log_dir, f"throw_log_{timestamp}.csv")
        
        try:
            with open(filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                
                # 메타데이터를 파일 최상단에 주석(#) 형태로 기록
                for key, value in self.metadata.items():
                    writer.writerow([f"# {key}: {value}"])
                
                # 실제 데이터 기록
                writer.writerow(["Time(s)", "TCP_Vel(mm/s)", "J6_Vel(deg/s)"])
                for t, v, j in zip(self.log_time, self.log_linear_vel, self.log_j6_vel):
                    writer.writerow([t, v, j])
                    
            return filename
        except Exception as e:
            print(f"로그 저장 실패: {e}")
            return None