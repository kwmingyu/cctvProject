import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog, QMessageBox, QProgressBar
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import subprocess

class Worker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path

    def run(self):
        self.progress.emit(10)
        try:
            cmd = f"python main.py \"{self.video_path}\""
            subprocess.run(cmd, shell=True, check=True)
        except Exception as e:
            print("에러 발생:", e)
        self.progress.emit(100)
        self.finished.emit()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("이상행동 감지 데모 프로그램")
        self.setGeometry(400, 200, 400, 200)

        self.label = QLabel("감지할 영상을 선택하세요.")
        self.label.setAlignment(Qt.AlignCenter)

        self.progress = QProgressBar(self)
        self.progress.setValue(0)

        self.select_button = QPushButton("영상 선택")
        self.select_button.clicked.connect(self.select_video)

        self.run_button = QPushButton("분석 시작")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self.run_main)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.select_button)
        layout.addWidget(self.run_button)
        layout.addWidget(self.progress)
        self.setLayout(layout)

        self.video_path = None

    def select_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "영상 파일 선택", "", "Video Files (*.mp4 *.avi)")
        if file_path:
            self.video_path = file_path
            self.label.setText(f"선택된 파일: {os.path.basename(file_path)}")
            self.run_button.setEnabled(True)

    def run_main(self):
        if not self.video_path:
            QMessageBox.warning(self, "오류", "먼저 영상 파일을 선택하세요.")
            return
        self.run_button.setEnabled(False)
        self.worker = Worker(self.video_path)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_finished(self):
        QMessageBox.information(self, "완료", "이상행동 분석이 완료되었습니다!")
        self.run_button.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
