import pandas as pd
import numpy as np

# 1. CSV 파일 불러오기
df = pd.read_csv("tracking_output.csv")

# 2. 사람(person) 클래스만 필터링
df = df[df["class"] == "person"]

# 3. track_id 별로 이동 특징 계산
results = []

for track_id, group in df.groupby("track_id"):
    group = group.sort_values("frame").reset_index(drop=True)

    # Δx, Δy, distance, dt, speed 계산
    group["dx"] = group["x_center"].diff()
    group["dy"] = group["y_center"].diff()
    group["distance"] = np.sqrt(group["dx"]**2 + group["dy"]**2)
    group["dt"] = group["time"].diff().replace(0, np.nan)
    group["speed"] = group["distance"] / group["dt"]

    # 첫 행은 NaN 발생하므로 0으로 채움
    group.fillna(0, inplace=True)

    results.append(group)

# 4. 모든 track_id 결과 통합
feature_df = pd.concat(results, ignore_index=True)

# 5. 결과 저장
feature_df.to_csv("track_features.csv", index=False)

# 6. (선택) 일부 출력
print(feature_df.head())
