import tkinter as tk
from config.config_manager import load_metadata, save_metadata
from core.throw_manager import ThrowManager
from ui.main_window import ThrowSimulatorGUI

def main():
    # 1. 설정 로드
    current_meta = load_metadata()
    
    # 2. 비즈니스 로직(제어기) 생성
    manager = ThrowManager()
    
    # 3. UI 생성 및 제어기 주입 (Dependency Injection)
    root = tk.Tk()
    app = ThrowSimulatorGUI(root, manager, current_meta)
    
    # 4. 종료 시 메타데이터 저장 안전장치
    def on_closing():
        try:
            try:
                app._update_metadata_from_ui()
                save_metadata(app.current_meta)
            finally:
                manager.disconnect_all()
        finally:
            root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()