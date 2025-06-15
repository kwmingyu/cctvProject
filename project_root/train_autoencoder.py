# train_autoencoder_all_with_scaling.py - 정규화 포함 학습 스크립트

import torch
import numpy as np
from torch import nn
import os
import glob
from sklearn.preprocessing import MinMaxScaler
import joblib

# === LSTM AutoEncoder 모델 정의 ===
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

# === 모든 .npy 시퀀스 파일 불러오기 ===
npy_files = glob.glob("*.npy")
X_all = []

for file in npy_files:
    try:
        data = np.load(file)
        if data.ndim == 3 and data.shape[2] == 2:
            X_all.append(data)
            print(f"✅ 로드됨: {file} → shape: {data.shape}")
        else:
            print(f"⚠️ 건너뜀(형식불일치): {file}")
    except Exception as e:
        print(f"❌ 오류 발생: {file} → {e}")

# === 병합 및 정규화 ===
X = np.concatenate(X_all, axis=0)
X_reshaped = X.reshape(-1, 2)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_reshaped)
X_scaled = X_scaled.reshape(X.shape)

# === 정규화 스케일러 저장 ===
joblib.dump(scaler, "scaler.pkl")
print("✅ 정규화 스케일러 저장 완료 → scaler.pkl")

# === 텐서화 및 모델 학습 ===
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
model = LSTMAutoEncoder()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

model.train()
for epoch in range(10):
    total_loss = 0
    for i in range(0, len(X_tensor), 32):
        batch = X_tensor[i:i+32]
        output = model(batch)
        loss = criterion(output, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss:.6f}")

# === 모델 저장 ===
torch.save(model.state_dict(), "ae_model.pt")
print("✅ 모델 저장 완료 → ae_model.pt")
