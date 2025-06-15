import sys  # 파이썬 인터프리터 및 시스템 함수 제어를 위한 모듈
import cv2  # OpenCV: 영상 처리 및 컴퓨터 비전 라이브러리
import datetime  # 날짜 및 시간 처리 모듈
import os  # 운영체제 경로 및 파일 처리 모듈
import time  # 시간 지연 및 타이밍 함수 제공 모듈
import numpy as np  # 수치 연산을 위한 NumPy 라이브러리
import torch  # PyTorch: 딥러닝 모델 정의 및 연산을 위한 라이브러리
import joblib  # 모델 및 전처리 스케일러 저장/로드를 위한 라이브러리
from torch import nn  # 신경망 레이어 및 모델 구성 요소
from collections import defaultdict, deque  # 키 기본값 딕셔너리와 고정 길이 큐 자료구조
from PyQt5.QtWidgets import (
    QApplication, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget,
    QComboBox, QTextEdit, QFileDialog, QLineEdit, QStatusBar, QMessageBox, QCheckBox
)  # PyQt5 GUI 구성 요소들
from PyQt5.QtGui import QImage, QPixmap, QIcon  # 이미지 표현 및 아이콘 처리를 위한 클래스
from PyQt5.QtCore import QTimer, Qt  # 타이머 및 정렬/정렬 상수
from ultralytics import YOLO  # Ultralytics YOLOv8 모델 로딩 및 예측


# === 단순 추적기 클래스 정의 ===
class SimpleTracker:
    def __init__(self, max_age=30):
        # next_id: 새 객체에 부여할 추적 ID
        # tracks: 현재 프레임에서 탐지된 객체들의 {ID: (x, y)} 저장
        # last_seen: 각 ID가 마지막으로 갱신된 이후 지난 프레임 수 카운트
        # max_age: 지정 프레임 이상 갱신 없으면 추적 대상에서 제거
        self.next_id = 1
        self.tracks = {}
        self.last_seen = {}
        self.max_age = max_age

    def update(self, detections):
        """
        detections: 리스트 of (bbox, confidence, class)
        bbox: [x1, y1, x2, y2]
        반환: 현재 업데이트된 추적 객체 리스트 [(id, (cx, cy)), ...]
        """
        updated_tracks = {}
        # 각 탐지 결과 순회
        for det in detections:
            bbox, conf, cls = det
            # bbox 중심점 계산
            cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
            matched = False
            # 기존 추적 대상과 거리 비교
            for tid, (px, py) in self.tracks.items():
                # 중심점 간 유클리드 거리 계산
                if np.linalg.norm([cx - px, cy - py]) < 50:
                    # 같은 객체로 판단하여 기존 ID 사용
                    updated_tracks[tid] = (cx, cy)
                    # 마지막 갱신 프레임 카운트 리셋
                    self.last_seen[tid] = 0
                    matched = True
                    break
            if not matched:
                # 새로운 객체라면 새로운 ID 부여
                updated_tracks[self.next_id] = (cx, cy)
                self.last_seen[self.next_id] = 0
                self.next_id += 1

        # 갱신되지 않은 ID들에 대해 age 증가시키고, max_age 초과 시 제거
        to_delete = []
        for tid in self.last_seen:
            if tid not in updated_tracks:
                self.last_seen[tid] += 1
                if self.last_seen[tid] > self.max_age:
                    to_delete.append(tid)
        for tid in to_delete:
            updated_tracks.pop(tid, None)
            self.last_seen.pop(tid, None)

        # 최종 트랙 갱신
        self.tracks = updated_tracks
        # ID와 중심점 좌표 리스트 반환
        return list(self.tracks.items())


# === LSTM AutoEncoder 신경망 정의 ===
class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16):
        super().__init__()
        # 인코더: input_dim 차원 입력을 hidden_dim 차원으로 압축
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        # 디코더: hidden_dim 차원 정보를 다시 input_dim 차원으로 복원
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        # 인코더 통과 후 히든 상태 h 획득
        _, (h, _) = self.encoder(x)
        # 히든 상태를 시퀀스 길이만큼 반복하여 디코더 입력 생성
        repeated = h.repeat(x.size(1), 1, 1).permute(1, 0, 2)
        # 디코더를 통해 복원된 시퀀스 out 반환
        out, _ = self.decoder(repeated)
        return out


# === 클러스터링 및 필터링 함수 정의 ===
def cluster_and_filter(boxes, classes, confs, threshold=80):
    """
    boxes: Nx4 array of [x1,y1,x2,y2]
    classes: N array of class IDs
    confs: N array of confidences
    threshold: 픽셀 거리 임계값
    같은 클래스(주차된 차량 등) 중심점 간 거리가 threshold 이하인 박스들 군집화
    군집마다 confidence 최대값을 가진 박스만 선택하여 반환
    """
    clusters = []
    used = set()
    for i in range(len(boxes)):
        if i in used:
            continue
        cls_i = int(classes[i])
        # 원하는 클래스(0번: 사람 등)만 처리
        if cls_i != 0:
            continue
        x1, y1, x2, y2 = boxes[i]
        cx_i, cy_i = (x1 + x2) / 2, (y1 + y2) / 2
        cluster = [(i, confs[i], boxes[i])]
        used.add(i)
        # 이후 박스들도 같은 클래스이면서 threshold 이내에 있으면 군집에 추가
        for j in range(i + 1, len(boxes)):
            if j in used:
                continue
            cls_j = int(classes[j])
            if cls_j != cls_i:
                continue
            x1j, y1j, x2j, y2j = boxes[j]
            cx_j, cy_j = (x1j + x2j) / 2, (y1j + y2j) / 2
            if np.linalg.norm([cx_i - cx_j, cy_i - cy_j]) < threshold:
                cluster.append((j, confs[j], boxes[j]))
                used.add(j)
        # 군집 중 confidence 최대값을 가진 박스를 선택
        best = max(cluster, key=lambda x: x[1])
        clusters.append(best)
    return clusters


# === PyQt5 기반 스마트 CCTV 애플리케이션 GUI 클래스 정의 ===
class CCTVApp(QWidget):
    def __init__(self):
        super().__init__()
        # 윈도우 타이틀 및 크기/스타일 설정
        self.setWindowTitle("🛡️ 스마트 CCTV 관제 시스템")
        self.setGeometry(200, 100, 1100, 700)
        self.setStyleSheet("background-color: #121212; color: #e0e0e0; font-family: 'Segoe UI';")

        # 영상 표시 크기 지정
        self.video_width, self.video_height = 1440, 960
        self.video_path = None  # 파일 선택 모드를 위한 경로 변수

        # 영상 레이블: 프레임을 보여줄 QLabel
        self.video_label = QLabel()
        self.video_label.setFixedSize(self.video_width, self.video_height)
        self.video_label.setStyleSheet("background-color: #000; border: 2px solid #2e86de; border-radius: 10px;")
        self.video_label.setAlignment(Qt.AlignCenter)

        # 카메라/파일 선택 콤보박스
        self.camera_select = QComboBox()
        self.camera_select.addItems(["카메라 0", "카메라 1", "RTSP 직접 입력", "🎞 영상 파일"])
        self.camera_select.currentIndexChanged.connect(self.toggle_rtsp_input)

        # RTSP URL 입력창
        self.rtsp_input = QLineEdit()
        self.rtsp_input.setPlaceholderText("RTSP 주소를 입력하세요")
        self.rtsp_input.setEnabled(False)  # 기본 비활성화

        # 파일 선택 버튼
        self.video_select_btn = QPushButton("🎞 영상 파일 선택")
        self.video_select_btn.setStyleSheet(
            "QPushButton { background-color: #f39c12; border-radius: 8px; padding: 10px; "
            "color: white; font-weight: bold; } "
            "QPushButton:hover { background-color: #d35400; }"
        )

        self.video_select_btn.clicked.connect(self.select_video_file)

        # 회전 체크박스
        self.rotate_checkbox = QCheckBox("🌀 영상 왼쪽으로 회전")
        self.rotate_checkbox.setStyleSheet("color: white; font-weight: bold; padding: 5px;")

        # 재생/정지/녹화/스냅샷 등 컨트롤 버튼 스타일 정의
        btn_style = "QPushButton { background-color: #3498db; border-radius: 8px; padding: 10px; color: white; font-weight: bold; } QPushButton:hover { background-color: #2980b9; }"
        self.start_btn = QPushButton("▶ 재생"); self.start_btn.setStyleSheet(btn_style)
        self.stop_btn = QPushButton("⏹ 정지"); self.stop_btn.setStyleSheet(btn_style)
        self.record_btn = QPushButton("⏺ 녹화"); self.record_btn.setStyleSheet(btn_style)
        self.snapshot_btn = QPushButton("📸 스냅샷"); self.snapshot_btn.setStyleSheet(btn_style)
        self.select_path_btn = QPushButton("📁 저장 경로 설정"); self.select_path_btn.setStyleSheet(btn_style)
        self.save_log_btn = QPushButton("🧾 로그 저장"); self.save_log_btn.setStyleSheet(btn_style)

        # 저장 경로 표시 라벨 및 로그 텍스트 에디트
        self.path_label = QLabel(f"저장 경로: {os.getcwd()}")
        self.log_text = QTextEdit(); self.log_text.setReadOnly(True); self.log_text.setFixedHeight(580)
        self.status_bar = QStatusBar()

        # 레이아웃 배치: 카메라 선택 + RTSP 입력 + 파일 선택 버튼
        cam_layout = QHBoxLayout()
        cam_layout.addWidget(self.camera_select)
        cam_layout.addWidget(self.rtsp_input)
        cam_layout.addWidget(self.video_select_btn)

        # 제어 버튼 레이아웃
        control = QVBoxLayout()
        control.addLayout(cam_layout)
        control.addWidget(self.rotate_checkbox)
        for b in [self.start_btn, self.stop_btn, self.record_btn, self.snapshot_btn,
                  self.select_path_btn, self.save_log_btn, self.path_label, self.log_text, self.status_bar]:
            control.addWidget(b)

        # 메인 레이아웃: 영상 레이블 + 컨트롤
        layout = QHBoxLayout(); layout.addWidget(self.video_label); layout.addLayout(control)
        self.setLayout(layout)

        # 캡처 및 기록 초기값 설정
        self.cap = None  # OpenCV VideoCapture 객체
        self.writer = None  # 비디오 기록을 위한 VideoWriter
        self.save_path = os.getcwd()  # 녹화 및 스냅샷 저장 경로
        self.is_recording = False  # 녹화 상태 플래그

        # 버튼 클릭 시 연결할 메서드 등록
        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn.clicked.connect(self.stop_camera)
        self.record_btn.clicked.connect(self.toggle_recording)
        self.snapshot_btn.clicked.connect(self.save_snapshot)
        self.select_path_btn.clicked.connect(self.select_save_path)
        self.save_log_btn.clicked.connect(self.save_log)

        # 타이머: 약 33ms 간격으로 update_frame 호출 (약 30fps)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # 디바이스 설정: GPU(cuda) 사용 가능 시 GPU 사용
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 오토인코더 모델 및 스케일러 로드
        self.model = LSTMAutoEncoder().to(self.device)
        model_path = os.path.join(os.path.dirname(__file__), "ae_model.pt")
        self.scaler = joblib.load("scaler.pkl")  # 거리 및 속도 특성 스케일러
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()  # 평가 모드로 전환

        # YOLO 객체 탐지 모델 로드
        self.yolo = YOLO("yolov8n.pt")
        # 단순 추적기, 궤적 버퍼, persistence 카운터 초기화
        self.tracker = SimpleTracker()
        self.track_buffers = defaultdict(lambda: deque(maxlen=30))
        self.persistence = defaultdict(int)
        # 재구성 오차 스코어 임계값
        self.MIN_SCORE = 0.0003
        self.MAX_SCORE = 0.0070

    def toggle_rtsp_input(self, index):
        """콤보박스 인덱스 변경 시 RTSP 입력창 활성화/비활성화"""
        self.rtsp_input.setEnabled(index == 2)
        self.video_select_btn.setEnabled(index == 3)

    def select_video_file(self):
        """파일 대화상자를 열어 영상 파일 선택"""
        path, _ = QFileDialog.getOpenFileName(self, "영상 파일 선택", "", "Video Files (*.mp4 *.avi *.mov)")
        if path:
            self.video_path = path
            self.log(f"영상 파일 선택됨: {path}")

    def start_camera(self):
        """카메라 또는 파일/RTSP 재생 시작"""
        index = self.camera_select.currentIndex()
        if index == 2:
            src = self.rtsp_input.text().strip()
        elif index == 3:
            if not self.video_path:
                QMessageBox.warning(self, "경고", "영상 파일을 먼저 선택해주세요.")
                return
            src = self.video_path
        else:
            src = index

        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "오류", "카메라/영상 열기 실패")
            return
        self.timer.start(33)  # 30fps 재생
        self.log("카메라/영상 재생 시작됨")

    def stop_camera(self):
        """재생 및 녹화 중지"""
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.writer:
            self.writer.release()
            self.writer = None
        self.video_label.clear()
        self.log("재생 정지됨")

    def update_frame(self):
        """타이머 호출로 프레임 갱신 및 객체 탐지/추적/이상 행동 판별"""
        if not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            return

        # 회전 옵션 적용
        if self.rotate_checkbox.isChecked():
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # YOLO 객체 탐지 수행
        results = self.yolo.predict(frame, conf=0.5)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()

        # 클러스터링 및 필터링 후 적합한 탐지만 선택
        clustered = cluster_and_filter(boxes, classes, confs)
        filtered = [(box, conf, int(classes[idx])) for idx, conf, box in clustered]
        input_dets = [([int(x1), int(y1), int(x2), int(y2)], conf, cls) for (x1, y1, x2, y2), conf, cls in filtered]

        # 추적 업데이트
        tracks = self.tracker.update(input_dets)

        # 각 객체에 대해 속도, 거리 계산 및 이상 행동 판별 로직
        for tid, (cx, cy) in tracks:
            buf = self.track_buffers[tid]
            if buf:
                px, py, _, _ = buf[-1]
                dist = np.hypot(cx - px, cy - py)
                speed = dist / 0.033  # 33ms 기준
            else:
                dist = speed = 0
            buf.append((cx, cy, dist, speed))

            label = "Normal"
            color = (0, 255, 0)

            # 충분한 시퀀스 길이가 쌓이면 이상 행동 평가
            if len(buf) >= 30:
                seq = np.array([[b[2], b[3]] for b in buf], dtype=np.float32)
                seq_scaled = self.scaler.transform(seq)
                x = torch.tensor(seq_scaled.reshape(1, 30, 2), dtype=torch.float32).to(self.device)

                dist_seq = seq_scaled[:, 0]
                speed_seq = seq_scaled[:, 1]

                avg_speed = np.mean(speed_seq)
                total_dist = np.sum(dist_seq)
                high_speed_count = np.sum(speed_seq > 0.3)

                # 느리게 움직일 경우 배회(loitering)로 판단
                if avg_speed < 0.1 and total_dist < 0.2:
                    label = "Loitering"
                    color = (0, 165, 255)
                # 빠르게 움직일 경우 달리기(running)로 판단
                elif high_speed_count >= 2:
                    label = "Running"
                    color = (255, 100, 0)

                # AutoEncoder 재구성 오차 기반 이상치 점수 계산
                with torch.no_grad():
                    recon = self.model(x)
                    score = torch.mean((x - recon) ** 2).item()
                    scaled = 100 * (score - self.MIN_SCORE) / (self.MAX_SCORE - self.MIN_SCORE + 1e-6)
            else:
                scaled = 0

            # 연속 임계치 초과 시 이상 행동(Anomaly) 표시
            self.persistence[tid] = self.persistence[tid] + 1 if scaled >= 80 else 0
            if self.persistence[tid] >= 3:
                label = "Anomaly"
                color = (0, 0, 255)

            # 프레임에 레이블 및 박스 그리기
            cv2.putText(frame, f"{label} ({scaled:.1f})", (cx - 50, cy - 20), 0, 0.6, color, 2)
            cv2.rectangle(frame, (cx - 30, cy - 60), (cx + 30, cy + 60), color, 2)

        # 녹화 중일 경우 영상 기록
        if self.is_recording and self.writer:
            self.writer.write(frame)

        # OpenCV BGR -> RGB 변환 후 QImage로 변환하여 QLabel에 표시
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(img).scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))

    def toggle_recording(self):
        """녹화 시작/중지 토글"""
        if not self.cap:
            return
        if not self.is_recording:
            # 파일명에 현재 시각 포함
            now = datetime.datetime.now().strftime("record_%Y%m%d_%H%M%S.avi")
            path = os.path.join(self.save_path, now)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.writer = cv2.VideoWriter(path, fourcc, 20.0, (w, h))
            self.is_recording = True
            self.log(f"녹화 시작: {path}")
            self.status_bar.showMessage("녹화 중...")
        else:
            self.is_recording = False
            if self.writer:
                self.writer.release()
                self.writer = None
            self.log("녹화 중지됨")
            self.status_bar.showMessage("녹화 중지됨")

    def save_snapshot(self):
        """현재 프레임 스냅샷 저장"""
        if not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        name = datetime.datetime.now().strftime("snapshot_%Y%m%d_%H%M%S.jpg")
        path = os.path.join(self.save_path, name)
        cv2.imwrite(path, frame)
        self.log(f"스냅샷 저장: {path}")
        self.status_bar.showMessage("스냅샷 저장 완료", 3000)

    def select_save_path(self):
        """저장 경로 변경 다이얼로그"""
        path = QFileDialog.getExistingDirectory(self, "저장 경로 선택", self.save_path)
        if path:
            self.save_path = path
            self.path_label.setText(f"저장 경로: {self.save_path}")
            self.log(f"저장 경로 변경됨: {self.save_path}")

    def save_log(self):
        """로그 내용을 텍스트 파일로 저장"""
        path, _ = QFileDialog.getSaveFileName(self, "로그 저장", "log.txt", "Text Files (*.txt)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.log_text.toPlainText())
            self.log(f"로그 저장됨: {path}")
            self.status_bar.showMessage("로그 저장 완료", 3000)

    def log(self, msg):
        """로그 메시지를 텍스트 에디트에 추가"""
        now = datetime.datetime.now().strftime("[%H:%M:%S]")
        self.log_text.append(f"{now} {msg}")


# === 메인 함수 ===
if __name__ == "__main__":
    app = QApplication(sys.argv)  # QApplication 초기화
    win = CCTVApp()  # CCTV 애플리케이션 인스턴스 생성
    win.show()  # 윈도우 표시
    sys.exit(app.exec_())  # 이벤트 루프 시작으로 프로그램 실행 