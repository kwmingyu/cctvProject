import sys
import cv2
import datetime
import os
import time
import numpy as np
import torch
import joblib
from torch import nn
from collections import defaultdict, deque
from PyQt5.QtWidgets import (
    QApplication, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget,
    QComboBox, QTextEdit, QFileDialog, QLineEdit, QStatusBar, QMessageBox, QCheckBox
)
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import QTimer, Qt
from ultralytics import YOLO

# === 추적기 ===
class SimpleTracker:
    def __init__(self, max_age=30):
        self.next_id = 1
        self.tracks = {}
        self.last_seen = {}
        self.max_age = max_age

    def update(self, detections):
        updated_tracks = {}
        for det in detections:
            bbox, conf, cls = det
            cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
            matched = False
            for tid, (px, py) in self.tracks.items():
                if np.linalg.norm([cx - px, cy - py]) < 50:
                    updated_tracks[tid] = (cx, cy)
                    self.last_seen[tid] = 0
                    matched = True
                    break
            if not matched:
                updated_tracks[self.next_id] = (cx, cy)
                self.last_seen[self.next_id] = 0
                self.next_id += 1

        to_delete = []
        for tid in self.last_seen:
            if tid not in updated_tracks:
                self.last_seen[tid] += 1
                if self.last_seen[tid] > self.max_age:
                    to_delete.append(tid)
        for tid in to_delete:
            updated_tracks.pop(tid, None)
            self.last_seen.pop(tid, None)

        self.tracks = updated_tracks
        return list(self.tracks.items())

class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        repeated = h.repeat(x.size(1), 1, 1).permute(1, 0, 2)
        out, _ = self.decoder(repeated)
        return out

def cluster_and_filter(boxes, classes, confs, threshold=80):
    clusters = []
    used = set()
    for i in range(len(boxes)):
        if i in used: continue
        cls_i = int(classes[i])
        if cls_i != 0: continue
        x1, y1, x2, y2 = boxes[i]
        cx_i, cy_i = (x1 + x2) / 2, (y1 + y2) / 2
        cluster = [(i, confs[i], boxes[i])]
        used.add(i)
        for j in range(i + 1, len(boxes)):
            if j in used: continue
            cls_j = int(classes[j])
            if cls_j != cls_i: continue
            x1j, y1j, x2j, y2j = boxes[j]
            cx_j, cy_j = (x1j + x2j) / 2, (y1j + y2j) / 2
            if np.linalg.norm([cx_i - cx_j, cy_i - cy_j]) < threshold:
                cluster.append((j, confs[j], boxes[j]))
                used.add(j)
        best = max(cluster, key=lambda x: x[1])
        clusters.append(best)
    return clusters

class CCTVApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🛡️ 스마트 CCTV 관제 시스템")
        self.setGeometry(200, 100, 1100, 700)
        self.setStyleSheet("background-color: #121212; color: #e0e0e0; font-family: 'Segoe UI';")

        self.video_width, self.video_height = 1440, 960
        self.video_path = None

        self.video_label = QLabel()
        self.video_label.setFixedSize(self.video_width, self.video_height)
        self.video_label.setStyleSheet("background-color: #000; border: 2px solid #2e86de; border-radius: 10px;")
        self.video_label.setAlignment(Qt.AlignCenter)

        self.camera_select = QComboBox()
        self.camera_select.addItems(["카메라 0", "카메라 1", "RTSP 직접 입력", "🎞 영상 파일"])
        self.camera_select.currentIndexChanged.connect(self.toggle_rtsp_input)

        self.rtsp_input = QLineEdit()
        self.rtsp_input.setPlaceholderText("RTSP 주소를 입력하세요")
        self.rtsp_input.setEnabled(False)

        self.video_select_btn = QPushButton("🎞 영상 파일 선택")
        self.video_select_btn.setStyleSheet("QPushButton { background-color: #f39c12; border-radius: 8px; padding: 10px; color: white; font-weight: bold; } QPushButton:hover { background-color: #d35400; }")
        self.video_select_btn.clicked.connect(self.select_video_file)

        self.rotate_checkbox = QCheckBox("🌀 영상 왼쪽으로 회전")
        self.rotate_checkbox.setStyleSheet("color: white; font-weight: bold; padding: 5px;")

        btn_style = "QPushButton { background-color: #3498db; border-radius: 8px; padding: 10px; color: white; font-weight: bold; } QPushButton:hover { background-color: #2980b9; }"
        self.start_btn = QPushButton("▶ 재생"); self.start_btn.setStyleSheet(btn_style)
        self.stop_btn = QPushButton("⏹ 정지"); self.stop_btn.setStyleSheet(btn_style)
        self.record_btn = QPushButton("⏺ 녹화"); self.record_btn.setStyleSheet(btn_style)
        self.snapshot_btn = QPushButton("📸 스냅샷"); self.snapshot_btn.setStyleSheet(btn_style)
        self.select_path_btn = QPushButton("📁 저장 경로 설정"); self.select_path_btn.setStyleSheet(btn_style)
        self.save_log_btn = QPushButton("🧾 로그 저장"); self.save_log_btn.setStyleSheet(btn_style)

        self.path_label = QLabel(f"저장 경로: {os.getcwd()}")
        self.log_text = QTextEdit(); self.log_text.setReadOnly(True); self.log_text.setFixedHeight(580)
        self.status_bar = QStatusBar()

        cam_layout = QHBoxLayout()
        cam_layout.addWidget(self.camera_select)
        cam_layout.addWidget(self.rtsp_input)
        cam_layout.addWidget(self.video_select_btn)

        control = QVBoxLayout()
        control.addLayout(cam_layout)
        control.addWidget(self.rotate_checkbox)
        for b in [self.start_btn, self.stop_btn, self.record_btn, self.snapshot_btn, self.select_path_btn, self.save_log_btn, self.path_label, self.log_text, self.status_bar]:
            control.addWidget(b)

        layout = QHBoxLayout(); layout.addWidget(self.video_label); layout.addLayout(control)
        self.setLayout(layout)

        self.cap = None
        self.writer = None
        self.save_path = os.getcwd()
        self.is_recording = False

        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn.clicked.connect(self.stop_camera)
        self.record_btn.clicked.connect(self.toggle_recording)
        self.snapshot_btn.clicked.connect(self.save_snapshot)
        self.select_path_btn.clicked.connect(self.select_save_path)
        self.save_log_btn.clicked.connect(self.save_log)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = LSTMAutoEncoder().to(self.device)
        model_path = os.path.join(os.path.dirname(__file__), "ae_model.pt")
        self.scaler = joblib.load("scaler.pkl")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.yolo = YOLO("yolov8n.pt")
        self.tracker = SimpleTracker()
        self.track_buffers = defaultdict(lambda: deque(maxlen=30))
        self.persistence = defaultdict(int)
        self.MIN_SCORE = 0.0003
        self.MAX_SCORE = 0.0070

    def toggle_rtsp_input(self, index):
        self.rtsp_input.setEnabled(index == 2)
        self.video_select_btn.setEnabled(index == 3)

    def select_video_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "영상 파일 선택", "", "Video Files (*.mp4 *.avi *.mov)")
        if path:
            self.video_path = path
            self.log(f"영상 파일 선택됨: {path}")

    def start_camera(self):
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
        self.timer.start(33)
        self.log("카메라/영상 재생 시작됨")

    def stop_camera(self):
        self.timer.stop()
        if self.cap: self.cap.release(); self.cap = None
        if self.writer: self.writer.release(); self.writer = None
        self.video_label.clear()
        self.log("재생 정지됨")

    def update_frame(self):
        if not self.cap: return
        ret, frame = self.cap.read()
        if not ret: return

        if self.rotate_checkbox.isChecked():
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        results = self.yolo.predict(frame, conf=0.5)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()

        clustered = cluster_and_filter(boxes, classes, confs)
        filtered = [(box, conf, int(classes[idx])) for idx, conf, box in clustered]
        input_dets = [([int(x1), int(y1), int(x2), int(y2)], conf, cls) for (x1, y1, x2, y2), conf, cls in filtered]
        tracks = self.tracker.update(input_dets)

        for tid, (cx, cy) in tracks:
            buf = self.track_buffers[tid]
            if buf:
                px, py, _, _ = buf[-1]
                dist = np.hypot(cx - px, cy - py)
                speed = dist / 0.033
            else:
                dist = speed = 0
            buf.append((cx, cy, dist, speed))

            label = "Normal"
            color = (0, 255, 0)  # Green

            if len(buf) >= 30:
                seq = np.array([[b[2], b[3]] for b in buf], dtype=np.float32)
                seq_scaled = self.scaler.transform(seq)
                x = torch.tensor(seq_scaled.reshape(1, 30, 2), dtype=torch.float32).to(self.device)

                dist_seq = seq_scaled[:, 0]  # 정규화된 distance
                speed_seq = seq_scaled[:, 1]  # 정규화된 speed

                avg_speed = np.mean(speed_seq)
                total_dist = np.sum(dist_seq)
                high_speed_count = np.sum(speed_seq > 0.3)

                if avg_speed < 0.1 and total_dist < 0.2:
                    label = "Loitering"
                    color = (0, 165, 255)  # Orange
                elif high_speed_count >= 2:
                    label = "Running"
                    color = (255, 100, 0)  # Blue-Orange

                with torch.no_grad():
                    recon = self.model(x)
                    score = torch.mean((x - recon) ** 2).item()
                    scaled = 100 * (score - self.MIN_SCORE) / (self.MAX_SCORE - self.MIN_SCORE + 1e-6)
            else:
                scaled = 0

            self.persistence[tid] = self.persistence[tid] + 1 if scaled >= 80 else 0
            if self.persistence[tid] >= 3:
                label = "Anomaly"
                color = (0, 0, 255)  # Red

            cv2.putText(frame, f"{label} ({scaled:.1f})", (cx - 50, cy - 20), 0, 0.6, color, 2)
            cv2.rectangle(frame, (cx - 30, cy - 60), (cx + 30, cy + 60), color, 2)

        if self.is_recording and self.writer:
            self.writer.write(frame)

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
        if not self.cap: return
        if not self.is_recording:
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
            if self.writer: self.writer.release(); self.writer = None
            self.log("녹화 중지됨")
            self.status_bar.showMessage("녹화 중지됨")

    def save_snapshot(self):
        if not self.cap: return
        ret, frame = self.cap.read()
        if not ret: return
        name = datetime.datetime.now().strftime("snapshot_%Y%m%d_%H%M%S.jpg")
        path = os.path.join(self.save_path, name)
        cv2.imwrite(path, frame)
        self.log(f"스냅샷 저장: {path}")
        self.status_bar.showMessage("스냅샷 저장 완료", 3000)

    def select_save_path(self):
        path = QFileDialog.getExistingDirectory(self, "저장 경로 선택", self.save_path)
        if path:
            self.save_path = path
            self.path_label.setText(f"저장 경로: {self.save_path}")
            self.log(f"저장 경로 변경됨: {self.save_path}")

    def save_log(self):
        path, _ = QFileDialog.getSaveFileName(self, "로그 저장", "log.txt", "Text Files (*.txt)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.log_text.toPlainText())
            self.log(f"로그 저장됨: {path}")
            self.status_bar.showMessage("로그 저장 완료", 3000)

    def log(self, msg):
        now = datetime.datetime.now().strftime("[%H:%M:%S]")
        self.log_text.append(f"{now} {msg}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = CCTVApp()
    win.show()
    sys.exit(app.exec_())