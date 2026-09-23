import rbpodo as rb
import numpy as np
import time
import threading
import matplotlib.pyplot as plt

ROBOT_IP = "192.168.0.25" # 로봇 제어기의 IP

def main():
    try:
        # 1. 로봇 채널 연결 및 초기화
        robot = rb.Cobot(ROBOT_IP)
        rc = rb.ResponseCollector()
        data_channel = rb.CobotData(ROBOT_IP)
        
        robot.set_operation_mode(rc, rb.OperationMode.Real)
        robot.set_speed_bar(rc, 1.0) 
        robot.flush(rc)

        # 2. 베이스 위치(Cartesian)로 초기 이동
        print(">> 초기 베이스 위치로 이동합니다...")
        base_pose = np.array([350.0, -120.0, 120.0, -90.0, 0.0, -90.0], dtype=np.float64)
        robot.move_l(rc, base_pose, 150.0, 300.0)
        
        if robot.wait_for_move_started(rc, 0.1).type() == rb.ReturnType.Success:
            robot.wait_for_move_finished(rc)
        robot.flush(rc)

        # 3. 현재 관절 각도(Base Joints) 저장 
        # (공간 이동 제약을 풀고, 오직 J6 모터만 한계속도까지 돌리기 위한 기준점)
        sys_data = data_channel.request_data()
        if not sys_data:
            print("데이터 채널을 읽을 수 없습니다.")
            return
        
        base_joints = np.array(sys_data.sdata.jnt_ref, dtype=np.float64)
        print(">> 베이스 자세 관절 각도 저장 완료!")
        print("-" * 50)
        print("[투척 테스트 CLI 모드 활성화]")
        print("  - 입력 예시 1 : home -45   (J6를 -45도로 이동하여 코킹)")
        print("  - 입력 예시 2 : throw 200  (J6를 -90도까지 200deg/s 속도로 발사)")
        print("  - 종료 : exit 또는 quit")
        print("-" * 50)

        # 4. CLI 명령어 대기 루프
        while True:
            cmd_input = input("\n명령어 입력 >> ").strip().lower()
            
            if cmd_input in ['exit', 'quit']:
                print(">> 테스트를 종료합니다.")
                break
                
            parts = cmd_input.split()
            if len(parts) != 2:
                print("!! 오류: '명령어 값' 형태로 입력하세요. (예: home 45)")
                continue
                
            command = parts[0]
            try:
                value = float(parts[1])
            except ValueError:
                print("!! 오류: 값은 숫자여야 합니다.")
                continue

            # ==========================================
            # 명령 1: go home (J6 코킹 자세로 이동)
            # ==========================================
            if command == "home":
                target_joints = base_joints.copy()
                target_joints[5] = value  # 6번 조인트(J6) 각도만 변경
                
                print(f">> [HOME] J6를 {value}도로 이동합니다...")
                robot.move_j(rc, target_joints, 100.0, 200.0) # 안전한 속도로 준비자세 이동
                
                if robot.wait_for_move_started(rc, 0.1).type() == rb.ReturnType.Success:
                    robot.wait_for_move_finished(rc)
                robot.flush(rc)
                print(">> [HOME] 준비 완료!")

            # ==========================================
            # 명령 2: throw (입력된 속도로 J6를 -90도까지 회전)
            # ==========================================
            elif command == "throw":
                target_joints = base_joints.copy()
                target_joints[5] = 60.0  # 지면에 수직한 발사 각도
                
                speed = value
                accel = value * 5.0 # 가속도는 속도의 5배 정도로 넉넉하게 설정 (스냅 폭발력)
                # (주의: 로봇 스펙에 따라 최대 가속도는 제한될 수 있음)
                
                print(f">> [THROW] 발사!! (목표 속도: {speed} deg/s)")

                # --- 데이터 수집 스레드 준비 ---
                collecting = [True] # list로 감싸서 스레드 내부에서 변경 가능하게 함
                time_data = []
                j6_vel_data = []

                def data_logger():
                    start_time = time.time()
                    prev_time = start_time
                    prev_j6_ang = None

                    while collecting[0]:
                        s_data = data_channel.request_data()
                        if s_data:
                            curr_time = time.time()
                            t = curr_time - start_time
                            curr_j6_ang = s_data.sdata.jnt_ang[5] 
                            
                            if prev_j6_ang is not None:
                                dt = curr_time - prev_time
                                if dt > 0:
                                    j6_ang_vel = (curr_j6_ang - prev_j6_ang) / dt
                                    time_data.append(t)
                                    j6_vel_data.append(abs(j6_ang_vel))
                            
                            prev_j6_ang = curr_j6_ang
                            prev_time = curr_time
                        time.sleep(0.005)

                logger_thread = threading.Thread(target=data_logger)
                logger_thread.start()

                # --- 투척 구동 (move_j) ---
                # move_j는 선속도 병목을 무시하고 관절 모터의 한계 속도를 직접 타겟팅합니다.
                robot.move_j(rc, target_joints, speed, accel)
                
                if robot.wait_for_move_started(rc, 0.1).type() == rb.ReturnType.Success:
                    robot.wait_for_move_finished(rc)
                robot.flush(rc)
                
                # --- 데이터 수집 종료 및 그래프 출력 ---
                collecting[0] = False
                logger_thread.join()

                
                rc.error().throw_if_not_empty()
                print(">> [THROW] 투척 동작 및 데이터 수집 완료!")

                # if len(time_data) > 0:
                #     window_size = 10
                #     if len(j6_vel_data) > window_size:
                #         smoothed_vel = np.convolve(j6_vel_data, np.ones(window_size)/window_size, mode='valid')
                #         smoothed_time = time_data[(window_size-1):]
                #     else:
                #         smoothed_vel = j6_vel_data
                #         smoothed_time = time_data

                #     plt.figure(figsize=(8, 4))
                #     plt.plot(time_data, j6_vel_data, label='Raw J6 Vel', color='gray', alpha=0.3)
                #     plt.plot(smoothed_time, smoothed_vel, label='Smoothed J6 Vel', color='darkorange', linewidth=3)
                    
                #     max_vel = np.max(smoothed_vel)
                #     plt.axhline(y=max_vel, color='red', linestyle='--', alpha=0.5)
                #     plt.text(smoothed_time[0], max_vel + 5, f'Peak: {max_vel:.1f} deg/s', color='red', fontweight='bold')

                #     plt.title(f"J6 Throw Velocity (Input Speed: {speed})")
                #     plt.xlabel("Time (s)")
                #     plt.ylabel("Angular Velocity (deg/s)")
                #     plt.grid(True, linestyle='--', alpha=0.7)
                #     plt.legend()
                #     plt.tight_layout()
                #     # 그래프 창을 띄우되, 사용자가 창을 닫아야 다음 명령어 입력이 가능합니다.
                #     plt.show() 

            else:
                print(f"!! 오류: 알 수 없는 명령어입니다. ({command})")

    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    main()