import rbpodo as rb
import numpy as np
import time

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
        target_pose = np.array([300.0, 0.0, 300.0, 180.0, 60.0, 0.0], dtype=np.float64)
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
        
        # # Point 1: 시작점 (손목을 뒤로 -30도 젖힌 상태)
        # pose1 = np.array([300.0, 0.0, 300.0, 180.0, 60.0, 0.0], dtype=np.float64)
        # robot.move_pb_add(rc, pose1, 200.0, rb.BlendingOption.Ratio, 0.5)

        # # Point 2: 중간 가속 구간 (자연스럽게 스윙하며 지나가는 점)
        # pose2 = np.array([500.0, 0.0, 300.0, 180.0, 0.0, 0.0], dtype=np.float64)
        # robot.move_pb_add(rc, pose2, 500.0, rb.BlendingOption.Ratio, 0.5)

        # Point 3: 릴리스 타점 (팔을 뻗으며 손목을 30도로 튕겨줌)
        pose3 = np.array([500.0, 0.0, 500.0, 180.0, -60.0, 0.0], dtype=np.float64)
        robot.move_pb_add(rc, pose3, 1500.0, rb.BlendingOption.Ratio, 0.5)

        # 6. 궤적 구동 실행 (move_pb_run)
        # 가속도를 조금 더 줘서 스냅의 속도를 살려봄
        overall_accel = 1500.0 
        robot.move_pb_run(rc, overall_accel, rb.MovePBOption.Intended)

        # 7. 이동 시작 및 완료 대기
        if robot.wait_for_move_started(rc, 0.1).type() == rb.ReturnType.Success:
            print("이동 시작 확인됨. 완료 대기 중...")
            robot.wait_for_move_finished(rc)
            print("이동 완료!")

        
        # 8. 통신 과정 중 에러가 있었다면 예외 발생
        rc.error().throw_if_not_empty()

    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    main()