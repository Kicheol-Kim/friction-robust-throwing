import rbpodo as rb
import numpy as np
import time
import threading                  # 데이터 수집 스레드용
import matplotlib.pyplot as plt   # 그래프 출력용

ROBOT_IP = "192.168.0.25" # 로봇 제어기의 IP

def main():
    try:
        # 1. 로봇 커맨드 채널 연결
        robot = rb.Cobot(ROBOT_IP)
        rc = rb.ResponseCollector()

        # 2. 로봇 데이터(상태) 채널 연결
        data_channel = rb.CobotData(ROBOT_IP)
        
        # 현재 로봇 상태 데이터 읽어보기 (연결 확인용)
        sys_data = data_channel.request_data()
        if sys_data:
            print("현재 관절 각도(Reference):", np.array(sys_data.sdata.jnt_ref))

        # 3. 제어 모드 및 속도 설정
        # 초기 테스트이므로 실제 로봇이 움직이지 않는 Simulation 모드 유지
        # (실제로 로봇을 움직이려면 rb.OperationMode.Real 등으로 변경해야 합니다)
        robot.set_operation_mode(rc, rb.OperationMode.Real)
        
        # 안전을 위해 글로벌 속도를 10% (0.1)로 매우 낮게 설정
        # robot.set_speed_bar(rc, 0.5) 
        robot.set_speed_bar(rc, 1.0) 

        # 버퍼에 남아있는 이전 메시지 비우기 (Wait 함수 오류 방지)
        robot.flush(rc)

        # # 3.1. 초기 준비 자세로 관절 이동 (move_j) - 특이점 회피
        # # 로봇이 일자로 펴진 상태(0도)를 벗어나도록 패키징자세로 구부려줍니다.
        # ready_joint = np.array([90, -65, 155, -45, -90, 0], dtype=np.float64)
        # print("준비 자세(관절 이동)로 먼저 이동합니다...")
        # robot.move_j(rc, ready_joint, 100.0, 200.0) #입력 = 현재모터자세, 목표모터자세, 속도(deg/s), 가속도(deg/s^2)
        
        # if robot.wait_for_move_started(rc, 0.1).type() == rb.ReturnType.Success:
        #     robot.wait_for_move_finished(rc)

        # 3.2. 투척 초기위치 이동
        target_pose = np.array([350.0, -120.0, 120.0, -90.0, -90.0+45, -90.0], dtype=np.float64)
        tcp_speed = 150.0  # 150 mm/s
        tcp_accel = 300.0  # 300 mm/s^2
        robot.move_l(rc, target_pose, tcp_speed, tcp_accel)

        if robot.wait_for_move_started(rc, 0.1).type() == rb.ReturnType.Success:
            robot.wait_for_move_finished(rc)
        robot.flush(rc) # move_j가 끝난 후 버퍼를 비워줌

        # 4. 이전 블렌딩 버퍼 초기화
        robot.move_pb_clear(rc)

        # 5. 궤적 경유점 추가 (move_pb_add)
        # [핵심 수정] Rx를 175도로 살짝 비틀어 Wrist Singularity 완벽 회피
        # 블렌딩 비율(0.2)을 줄여서 관절 속도 초과 방지 및 안정적인 궤적 생성

        # pose1 = np.array([300.0, 0.0, 300.0, 180.0, 60.0, 0.0], dtype=np.float64) # Point 1: 시작점 (손목을 뒤로 -30도 젖힌 상태)
        # robot.move_pb_add(rc, pose1, 200.0, rb.BlendingOption.Ratio, 0.5)

        # pose2 = np.array([500.0, 0.0, 300.0, 180.0, 0.0, 0.0], dtype=np.float64) # Point 2: 중간 가속 구간 (자연스럽게 스윙하며 지나가는 점)
        # robot.move_pb_add(rc, pose2, 500.0, rb.BlendingOption.Ratio, 0.5)

        # pose3 = np.array([350.0, -120.0-300, 120.0+300, 90.0, -90.0+30, 90.0], dtype=np.float64) # Point 3: 릴리스 타점 (팔을 뻗으며 손목을 30도로 튕겨줌)
        pose3 = np.array([350.0, -120.0, 120.0, 90.0, -90.0+30, 90.0], dtype=np.float64)
        robot.move_pb_add(rc, pose3, 2000.0, rb.BlendingOption.Ratio, 0.5)
        
        # 6. 궤적 구동 실행 (move_pb_run)
        overall_accel = 2000.0 
        robot.move_pb_run(rc, overall_accel, rb.MovePBOption.Intended)

        # --- [수정 부분: 데이터 수집을 위한 변수 및 함수 정의] ---
        collecting = True
        time_data = []
        linear_vel_data = []
        j6_vel_data = []  # 각속도 데이터를 J6 전용으로 변경

        def data_logger():
            start_time = time.time()
            prev_time = start_time
            prev_tcp_ref = None
            prev_j6_ang = None

            while collecting:
                sys_data = data_channel.request_data()
                if sys_data:
                    curr_time = time.time()
                    t = curr_time - start_time
                    
                    # 1. TCP 선속도 계산용 위치 (X, Y, Z)
                    curr_tcp_ref = np.array(sys_data.sdata.tcp_ref) 
                    
                    # 2. [핵심] J6 각속도 계산용 관절 각도 (실제 인코더 값 사용)
                    # jnt_ang 배열: [J1, J2, J3, J4, J5, J6] (단위: deg)
                    curr_j6_ang = sys_data.sdata.jnt_ang[5] 
                    
                    if prev_tcp_ref is not None and prev_j6_ang is not None:
                        dt = curr_time - prev_time
                        
                        if dt > 0:
                            # --- 선속도 계산 ---
                            vel_array = (curr_tcp_ref - prev_tcp_ref) / dt
                            lin_vel = np.linalg.norm(vel_array[0:3])
                            
                            # --- J6 각속도 계산 ---
                            # 단순히 현재 인코더 각도와 이전 인코더 각도의 차이를 시간으로 나눔
                            j6_ang_vel = (curr_j6_ang - prev_j6_ang) / dt
                            
                            time_data.append(t)
                            linear_vel_data.append(lin_vel)
                            # 크기(스피드)만 보고 싶다면 abs() 적용, 방향을 보려면 그대로 append
                            j6_vel_data.append(abs(j6_ang_vel)) 
                    
                    # 다음 루프를 위해 상태 저장
                    prev_tcp_ref = curr_tcp_ref
                    prev_j6_ang = curr_j6_ang
                    prev_time = curr_time
                
                # 통신 부하 방지
                time.sleep(0.005) 

        # 7. 이동 시작 대기 및 스레드를 이용한 데이터 수집
        if robot.wait_for_move_started(rc, 0.1).type() == rb.ReturnType.Success:
            print("이동 시작 확인됨. 데이터 실시간 수집 중...")
            
            logger_thread = threading.Thread(target=data_logger)
            logger_thread.start()
            
            robot.wait_for_move_finished(rc)
            
            collecting = False
            logger_thread.join()
            print("이동 및 데이터 수집 완료!")

        # 8. 통신 과정 중 에러가 있었다면 예외 발생
        rc.error().throw_if_not_empty()

        # --- [수정 부분: 수집된 데이터 Plotting] ---
        if len(time_data) > 0:
            print(f"총 {len(time_data)}개의 데이터 포인트가 수집되었습니다. 그래프를 그립니다.")
            plt.figure(figsize=(10, 8))
            
            # 선속도(Linear Velocity) 서브플롯
            plt.subplot(2, 1, 1)
            plt.plot(time_data, linear_vel_data, label='TCP Linear Velocity', color='blue', linewidth=2)
            plt.title("Actual TCP Linear Velocity Profile")
            plt.xlabel("Time (s)")
            plt.ylabel("Velocity (mm/s)")
            plt.grid(True)
            plt.legend()

            # [핵심] J6 각속도(Angular Velocity) 서브플롯
            plt.subplot(2, 1, 2)
            plt.plot(time_data, j6_vel_data, label='J6 Joint Velocity', color='red', linewidth=2)
            plt.title("Actual J6 (Pitch) Angular Velocity Profile")
            plt.xlabel("Time (s)")
            plt.ylabel("Velocity (deg/s)")
            plt.grid(True)
            plt.legend()

            plt.tight_layout()
            plt.show()
        else:
            print("수집된 데이터가 없습니다.")

    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    main()