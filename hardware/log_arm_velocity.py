import time
import threading
import math
import numpy as np
import matplotlib.pyplot as plt
import os

try:
    import rbpodo
except ImportError:
    print("경고: rbpodo 라이브러리가 없습니다.")
    rbpodo = None

# --- 설정값 ---
ROBOT_IP = "192.168.0.25"
START_POSE = [400.0, 0.0, 300.0, 180.0, 0.0, 0.0]
END_POSE = [600.0, 0.0, 500.0, 180.0, 0.0, 0.0]
MOVE_SPEED = 200.0  # 목표 선속도 (mm/s)
ACCELERATION = 400.0

class VelocityLogger:
    def __init__(self, ip):
        self.robot = rbpodo.Cobot(ip) if rbpodo else None
        self.is_moving = False
        
        self.time_data = []
        self.vel_data = []
        
        if self.robot:
            # 안전을 위해 Simulation 모드로 기본 연결
            rc = rbpodo.ResponseCollector()
            self.robot.set_operation_mode(rc, rbpodo.OperationMode.Simulation)
            print("로봇팔(Simulation) 연결 완료.")

    def get_current_tcp(self):
        """
        rbpodo에서 현재 TCP 위치(X, Y, Z)를 가져옵니다.
        (API 버전에 따라 함수명이 get_tcp_pose, get_kinematics_info 등으로 다를 수 있습니다.
         에러 발생 시 해당 라이브러리의 상태 반환 함수명으로 교체해 주세요.)
        """
        if not self.robot: return np.array([0.0, 0.0, 0.0])
        
        # 예시: get_tcp_pose()가 [x, y, z, rx, ry, rz]를 반환한다고 가정
        try:
            pose = self.robot.get_tcp_pose() 
            return np.array(pose[0:3])
        except AttributeError:
            # API 함수명이 다를 경우를 대비한 가상 데이터
            return np.array([0.0, 0.0, 0.0])

    def log_velocity_loop(self):
        """100Hz로 위치를 측정해 순간 선속도를 계산하고 기록합니다."""
        self.time_data.clear()
        self.vel_data.clear()
        
        prev_time = time.time()
        prev_pos = self.get_current_tcp()
        start_time = prev_time
        
        while self.is_moving:
            curr_time = time.time()
            curr_pos = self.get_current_tcp()
            
            dt = curr_time - prev_time
            if dt > 0:
                # 3차원 유클리디안 거리 계산 (mm)
                distance = np.linalg.norm(curr_pos - prev_pos)
                # 순간 선속도 (mm/s)
                velocity = distance / dt 
                
                self.time_data.append(curr_time - start_time)
                self.vel_data.append(velocity)
            
            prev_time = curr_time
            prev_pos = curr_pos
            time.sleep(0.01) # 100Hz 폴링

    def run_test(self):
        if not self.robot:
            print("로봇이 연결되지 않아 가상 데이터를 생성합니다.")
            self.generate_dummy_plot()
            return

        rc = rbpodo.ResponseCollector()
        
        # 1. 시작점으로 이동
        print("시작점으로 이동 중...")
        self.robot.move_l(rc, np.array(START_POSE, dtype=np.float64), MOVE_SPEED, ACCELERATION)
        time.sleep(2.0) # 도착 대기 (실제 적용시 상태 확인 로직 필요)
        
        # 2. 로깅 스레드 시작
        print("목표점 이동 및 데이터 로깅 시작...")
        self.is_moving = True
        log_thread = threading.Thread(target=self.log_velocity_loop, daemon=True)
        log_thread.start()
        
        # 3. 목표점으로 직선 이동 실행
        self.robot.move_l(rc, np.array(END_POSE, dtype=np.float64), MOVE_SPEED, ACCELERATION)
        
        # 4. 이동 시간 대기 후 로깅 종료
        # (이동 거리가 약 282mm, 속도가 200이므로 약 1.5초 소요 예상)
        time.sleep(2.0) 
        self.is_moving = False
        log_thread.join()
        
        # 5. 결과 그래프 그리기
        self.plot_results()

    def plot_results(self):
        plt.figure(figsize=(10, 5))
        
        # 원본 데이터는 노이즈가 있을 수 있으므로 플로팅
        plt.plot(self.time_data, self.vel_data, label='TCP Linear Velocity', color='b', linewidth=2)
        
        # 목표 속도 기준선 (200 mm/s)
        plt.axhline(y=MOVE_SPEED, color='r', linestyle='--', label=f'Target Speed ({MOVE_SPEED} mm/s)')
        
        plt.title('Robot Arm TCP Velocity Profile')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Velocity (mm/s)')
        plt.grid(True)
        plt.legend()
        
        # 프로젝트 폴더 내 저장
        save_path = os.path.join(os.path.dirname(__file__), '../data/plots/velocity_profile.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"그래프가 저장되었습니다: {save_path}")
        
        plt.show()

    def generate_dummy_plot(self):
        """로봇 연결 실패 시 테스트용 가상 사다리꼴 속도 프로파일 생성"""
        t = np.linspace(0, 2, 200)
        v = np.zeros_like(t)
        for i, time_val in enumerate(t):
            if time_val < 0.5: v[i] = (MOVE_SPEED / 0.5) * time_val
            elif time_val < 1.5: v[i] = MOVE_SPEED
            else: v[i] = max(0, MOVE_SPEED - (MOVE_SPEED / 0.5) * (time_val - 1.5))
        
        self.time_data = list(t)
        self.vel_data = list(v)
        self.plot_results()

if __name__ == "__main__":
    logger = VelocityLogger(ROBOT_IP)
    logger.run_test()