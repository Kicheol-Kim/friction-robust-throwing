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
        robot.set_speed_bar(rc, 1.0) 

        # 버퍼에 남아있는 이전 메시지 비우기 (Wait 함수 오류 방지)
        robot.flush(rc)

        # [추가할 부분] 3.5. 초기 준비 자세로 관절 이동 (move_j) - 특이점 회피
        # 로봇이 일자로 펴진 상태(0도)를 벗어나도록 관절을 적당히 구부려줍니다.
        ready_joint = np.array([0.0, 30.0, 90.0, 0.0, 90.0, 0.0], dtype=np.float64)
        print("준비 자세(관절 이동)로 먼저 이동합니다...")
        robot.move_j(rc, ready_joint, 100.0, 200.0) #입력 = 현재모터자세, 목표모터자세, 속도(deg/s), 가속도(deg/s^2)
        
        if robot.wait_for_move_started(rc, 0.1).type() == rb.ReturnType.Success:
            robot.wait_for_move_finished(rc)
            
        robot.flush(rc) # move_j가 끝난 후 버퍼를 비워줌

        # 4. 목표 엔드이펙터 위치 (X, Y, Z, Rx, Ry, Rz)
        # 에러 방지를 위해 반드시 float64 타입으로 명시
        target_pose = np.array([400.0, 0.0, 300.0, 180.0, 0.0, 0.0], dtype=np.float64)

        # 5. 목표 선속도 및 선가속도
        # 투척 전 준비 자세로 갈 때는 안전하게 낮게 설정
        tcp_speed = 150.0  # 150 mm/s
        tcp_accel = 300.0  # 300 mm/s^2

        # 6. 직선 이동 명령 전송 (위치 인자로 전달)
        robot.move_l(rc, target_pose, tcp_speed, tcp_accel)

        if robot.wait_for_move_started(rc, 0.1).type() == rb.ReturnType.Success:
            robot.wait_for_move_finished(rc)

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