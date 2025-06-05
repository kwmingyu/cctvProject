# train_autoencoder_all.py - 현재 폴더의 모든 시퀀스(.npy) 파일 병합 학습

import torch
import numpy as np
from torch import nn
import os
import glob

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

# === 모든 .npy 파일 불러오기 ===
npy_files = glob.glob("*.npy")
X_all = []

for file in npy_files:
    try:
        data = np.load(file)
        if data.ndim == 3 and data.shape[2] == 2:  # (seq, frame, features=2)
            X_all.append(data)
            print(f"✅ 로드됨: {file} → shape: {data.shape}")
        else:
            print(f"⚠️ 건너뜀(형식불일치): {file}")
    except Exception as e:
        print(f"❌ 오류 발생: {file} → {e}")

# === 병합 및 텐서화 ===
X = np.concatenate(X_all, axis=0)
X_tensor = torch.tensor(X).float()

# === 모델 정의 및 학습 ===
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
