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
        robot.set_operation_mode(rc, rb.OperationMode.Simulation)
        
        # 안전을 위해 글로벌 속도를 10% (0.1)로 매우 낮게 설정
        robot.set_speed_bar(rc, 0.1) 

        # 버퍼에 남아있는 이전 메시지 비우기 (Wait 함수 오류 방지)
        robot.flush(rc)

        # 4. Joint Space 이동 명령 (관절 1을 10도로 이동)
        target_joint = np.array([20, 0, 0, 0, 0, 0])
        print("이동 명령 전송 중...")
        robot.move_j(rc, target_joint, 50.0, 100.0)

        # 5. 이동 시작 및 완료 대기
        if robot.wait_for_move_started(rc, 0.1).type() == rb.ReturnType.Success:
            print("이동 시작 확인됨. 완료 대기 중...")
            robot.wait_for_move_finished(rc)
            print("이동 완료!")
        
        # 6. 통신 과정 중 에러가 있었다면 예외 발생
        rc.error().throw_if_not_empty()

    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    main()