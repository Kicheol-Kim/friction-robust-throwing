import tkinter as tk
from core.state_manager import event_queue

class ThrowSimulatorGUI:
    def __init__(self, root, manager, initial_meta):
        self.root = root
        self.manager = manager  # ThrowManager 객체
        self.current_meta = initial_meta
        
        # ... UI 위젯 생성 로직 (이전과 동일) ...
        
        self.root.after(100, self.process_queue)

    def process_queue(self):
        """Manager가 event_queue에 넣은 상태를 화면에 그립니다."""
        while not event_queue.empty():
            event = event_queue.get()
            # ... 상태 라벨 업데이트 또는 에러 팝업창 띄우기 ...
        self.root.after(100, self.process_queue)

    # UI 버튼 콜백 예시
    def cmd_execute_motion(self):
        self._update_metadata_from_ui()
        # 하드웨어 제어는 Manager에게 위임
        self.manager.execute_throw_sequence(self.current_meta)
        
    def cmd_stop_all(self):
        self.manager.stop_all()