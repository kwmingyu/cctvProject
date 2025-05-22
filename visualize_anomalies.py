import numpy as np
import pandas as pd
import cv2
import torch
from torch import nn

# === 설정 ===
VIDEO_PATH = "./video/video.mp4"
X_PATH = "X_train.npy"
FEATURE_CSV = "track_features.csv"
TRACK_CSV = "tracking_output.csv"
SEQUENCE_LENGTH = 30
ANOMALY_THRESHOLD = 0.04
OUTPUT_VIDEO = "anomaly_person_highlighted.avi"
SCORE_SCALE = 100
UPDATE_INTERVAL = 1  # 약 0.5초 간격 (30fps 기준)
PERSISTENCE_THRESHOLD = 3  # 연속 3번 이상 이상 점수 지속 시 이상행동 간주

# === LSTM AutoEncoder 정의 ===
class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMAutoEncoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        repeated = hidden.repeat(x.size(1), 1, 1).permute(1, 0, 2)
        decoded, _ = self.decoder(repeated)
        return decoded

# === 데이터 불러오기 ===
X = np.load(X_PATH)
X_tensor = torch.tensor(X).float()
features_df = pd.read_csv(FEATURE_CSV)
track_df = pd.read_csv(TRACK_CSV)

# === 모델 학습 ===
seq_len = X.shape[1]
feature_dim = X.shape[2]
model = LSTMAutoEncoder(input_dim=feature_dim, hidden_dim=16)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.train()

for epoch in range(5):
    for i in range(0, len(X_tensor), 32):
        batch = X_tensor[i:i+32]
        output = model(batch)
        loss = criterion(output, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# === 이상 점수 계산 ===
model.eval()
with torch.no_grad():
    reconstructed = model(X_tensor)
    raw_scores = torch.mean((X_tensor - reconstructed) ** 2, dim=(1, 2)).numpy()

# === 정규화된 점수 계산 (0~100점으로 scaling) ===
max_score = np.max(raw_scores)
min_score = np.min(raw_scores)
scores = SCORE_SCALE * (raw_scores - min_score) / (max_score - min_score + 1e-6)

# === 이상 시퀀스 인덱스, 점수, 관련 track_id/frame 매핑 ===
anomaly_data = []
idx = 0
for track_id, group in features_df.groupby("track_id"):
    group = group.sort_values("frame").reset_index(drop=True)
    for i in range(len(group) - SEQUENCE_LENGTH + 1):
        if idx >= len(scores):
            break
        if scores[idx] > 0:
            start_frame = group.loc[i, "frame"]
            anomaly_data.append({
                "track_id": track_id,
                "start_frame": int(start_frame),
                "end_frame": int(start_frame + SEQUENCE_LENGTH),
                "score": float(scores[idx])
            })
        idx += 1

# === 영상 시각화 ===
from collections import defaultdict

dangerous_episode_count = defaultdict(int)
dangerous_frame_flags = defaultdict(list)
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))

frame_idx = 0
anomaly_lookup = {(a["track_id"], f): a["score"]
                  for a in anomaly_data
                  for f in range(a["start_frame"], a["end_frame"])}
last_display = {}  # track_id: (score, last_updated_frame)
persistence_count = {}  # track_id: 지속 횟수

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_tracks = track_df[track_df["frame"] == frame_idx]

    for _, row in current_tracks.iterrows():
        tid = row["track_id"]
        cls = row["class"]
        if cls != "person":
            continue
        cx, cy = int(row["x_center"]), int(row["y_center"])

        score = anomaly_lookup.get((tid, frame_idx), None)
        if score is not None:
            prev_score, last_frame = last_display.get(tid, (0, -UPDATE_INTERVAL))
            if frame_idx - last_frame >= UPDATE_INTERVAL:
                last_display[tid] = (score, frame_idx)
            else:
                score = prev_score

            score = float(score)
            count = persistence_count.get(tid, 0)
            if score >= 80:
                count += 1
                dangerous_frame_flags[tid].append(1)
            else:
                if count > 0:
                    count -= 1
                dangerous_frame_flags[tid].append(0)

            if len(dangerous_frame_flags[tid]) >= 30:
                if sum(dangerous_frame_flags[tid][-30:]) >= 10:
                    dangerous_episode_count[tid] += 1
                    dangerous_frame_flags[tid] = []
            persistence_count[tid] = count

            color = (0, 255, 0)
            if dangerous_episode_count[tid] >= PERSISTENCE_THRESHOLD:
                color = (0, 0, 255)
                cv2.putText(frame, "Danger", (cx - 80, cy - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            elif score >= 80:
                color = (0, 0, 255)
            elif score >= 50:
                color = (0, 165, 255)

            cv2.putText(frame, f"Score: {score:.1f}/100", (cx - 50, cy - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            x1, y1 = cx - 30, cy - 60
            x2, y2 = cx + 30, cy + 60
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        else:
            x1, y1 = cx - 30, cy - 60
            x2, y2 = cx + 30, cy + 60
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print("[완료] 이상행동 시각화 영상 저장됨 →", OUTPUT_VIDEO)
