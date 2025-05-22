import numpy as np
import pandas as pd
import cv2
import torch
from torch import nn

# === 설정 ===
VIDEO_PATH = "./video/video.mp4"
X_PATH = "X_train.npy"
FEATURE_CSV = "track_features.csv"
SEQUENCE_LENGTH = 30
ANOMALY_THRESHOLD = 0.04
OUTPUT_VIDEO = "anomaly_highlighted.avi"

# === AutoEncoder 모델 정의 ===
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
df = pd.read_csv(FEATURE_CSV)

# === 모델 초기화 및 학습 ===
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
    scores = torch.mean((X_tensor - reconstructed) ** 2, dim=(1, 2)).numpy()

# === 이상 시퀀스 인덱스 찾기 ===
anomaly_indices = np.where(scores > ANOMALY_THRESHOLD)[0]

# === 해당 시퀀스들의 시작 프레임 찾기 ===
start_frames = []
grouped = df.groupby("track_id")

for track_id, group in grouped:
    group = group.sort_values("frame").reset_index(drop=True)
    for i in range(len(group) - SEQUENCE_LENGTH + 1):
        if len(start_frames) >= len(X):
            break
        start_frames.append(group.loc[i, "frame"])

# === 이상 시퀀스에 해당하는 시작 프레임 목록 ===
anomaly_start_frames = [start_frames[i] for i in anomaly_indices]

# === 영상 처리 ===
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))

frame_index = 0
anomaly_set = set()
for f in anomaly_start_frames:
    anomaly_set.update(range(f, f + SEQUENCE_LENGTH))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_index in anomaly_set:
        cv2.putText(frame, "Anomaly Detected!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.rectangle(frame, (30, 30), (w - 30, h - 30), (0, 0, 255), 5)

    out.write(frame)
    frame_index += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print("[완료] 이상행동 시각화 영상 저장됨 →", OUTPUT_VIDEO)
