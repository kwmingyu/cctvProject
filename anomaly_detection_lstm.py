import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

# === 설정 ===
SEQUENCE_PATH = "X_train.npy"  # 입력 시퀀스 파일 경로
EPOCHS = 10                      # 학습 epoch 수
THRESHOLD = 0.04                # 이상 판단 임계값

# === 데이터 로딩 ===
X = np.load(SEQUENCE_PATH)  # shape: (num_samples, seq_len, feature_dim)
X_tensor = torch.tensor(X).float()

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

# === 모델 초기화 ===
seq_len = X.shape[1]
feature_dim = X.shape[2]
model = LSTMAutoEncoder(input_dim=feature_dim, hidden_dim=16)

# === 손실 함수 및 옵티마이저 ===
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# === 학습 ===
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for i in range(0, len(X_tensor), 32):
        batch = X_tensor[i:i+32]
        output = model(batch)
        loss = criterion(output, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# === 이상 점수 계산 ===
model.eval()
with torch.no_grad():
    reconstructed = model(X_tensor)
    scores = torch.mean((X_tensor - reconstructed) ** 2, dim=(1, 2)).numpy()

# === 이상 여부 판단 ===
anomalies = scores > THRESHOLD
print("\n[이상행동 시퀀스 인덱스]:", np.where(anomalies)[0])

# === 시각화 ===
plt.figure(figsize=(12, 5))
plt.plot(scores, label="Anomaly Score")
plt.axhline(THRESHOLD, color='red', linestyle='--', label=f"Threshold = {THRESHOLD}")
plt.title("Anomaly Scores per Sequence")
plt.xlabel("Sequence Index")
plt.ylabel("Reconstruction Error")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
