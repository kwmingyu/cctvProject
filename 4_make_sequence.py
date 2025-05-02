import pandas as pd
import numpy as np

# 1. CSV 불러오기
df = pd.read_csv("track_features.csv")

# 2. 시퀀스 길이 설정
SEQUENCE_LENGTH = 30

# 3. 시퀀스 저장 리스트 초기화
sequences = []

# 4. track_id별로 시퀀스 생성
for track_id, group in df.groupby("track_id"):
    # 프레임 순으로 정렬
    group = group.sort_values("frame").reset_index(drop=True)

    # 프레임 수가 충분할 때만 시퀀스 생성
    for i in range(len(group) - SEQUENCE_LENGTH + 1):
        sequence = group.iloc[i:i+SEQUENCE_LENGTH][["distance", "speed"]].values
        sequences.append(sequence)

# 5. 시퀀스를 numpy 배열로 변환
X_train = np.array(sequences)

# 6. 시퀀스 저장
np.save("X_train.npy", X_train)

# 7. 정보 출력
print(f"생성된 시퀀스 수: {X_train.shape[0]}")
print(f"시퀀스 shape: {X_train.shape}")
