# main.py - 모든 기능 통합 실행 파일

import os
import cv2
import csv
import numpy as np
import pandas as pd
import torch
from torch import nn
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict

# ===== 1. 객체 감지 및 추적 =====
def run_tracking(video_path, output_csv):
    def euclidean(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def cluster_and_filter(boxes, classes, confs, threshold=80):
        clusters = []
        used = set()
        for i in range(len(boxes)):
            if i in used:
                continue
            cls_i = int(classes[i])
            if cls_i != 0:
                continue
            x1, y1, x2, y2 = boxes[i]
            cx_i, cy_i = (x1 + x2) / 2, (y1 + y2) / 2
            cluster = [(i, confs[i], boxes[i])]
            used.add(i)
            for j in range(i + 1, len(boxes)):
                if j in used:
                    continue
                cls_j = int(classes[j])
                if cls_j != cls_i:
                    continue
                x1j, y1j, x2j, y2j = boxes[j]
                cx_j, cy_j = (x1j + x2j) / 2, (y1j + y2j) / 2
                if euclidean((cx_i, cy_i), (cx_j, cy_j)) < threshold:
                    cluster.append((j, confs[j], boxes[j]))
                    used.add(j)
            best = max(cluster, key=lambda x: x[1])
            clusters.append(best)
        return clusters

    model = YOLO("yolov8n.pt")
    tracker = DeepSort(max_age=80, n_init=1)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    with open(output_csv, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time", "track_id", "class", "x_center", "y_center"])

    frame_count = 0
    last_seen = {}
    id_alias_map = {}
    next_alias_id = 1
    DIST_THRESHOLD = 50
    FRAME_GAP = 30

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.3)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()

        clustered_persons = cluster_and_filter(boxes, classes, confs)
        filtered = [(box, conf, int(classes[idx])) for idx, conf, box in clustered_persons]

        for i in range(len(boxes)):
            cls = int(classes[i])
            if cls == 2:
                x1, y1, x2, y2 = boxes[i]
                if x2 - x1 >= 30 and y2 - y1 >= 60:
                    filtered.append((boxes[i], confs[i], cls))

        input_dets = [([x1, y1, x2 - x1, y2 - y1], conf, cls) for (x1, y1, x2, y2), conf, cls in filtered]
        tracks = tracker.update_tracks(input_dets, frame=frame)

        with open(output_csv, "a", newline='') as f:
            writer = csv.writer(f)
            for track in tracks:
                if not track.is_confirmed():
                    continue
                raw_id = track.track_id
                cls = track.det_class
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                if raw_id not in id_alias_map:
                    matched = False
                    for aid, (px, py, lf) in last_seen.items():
                        if frame_count - lf < FRAME_GAP and euclidean((cx, cy), (px, py)) < DIST_THRESHOLD:
                            id_alias_map[raw_id] = aid
                            matched = True
                            break
                    if not matched:
                        id_alias_map[raw_id] = next_alias_id
                        next_alias_id += 1

                alias_id = id_alias_map[raw_id]
                last_seen[alias_id] = (cx, cy, frame_count)

                writer.writerow([frame_count, round(frame_count / fps, 2), alias_id, 'person' if cls == 0 else 'car', cx, cy])
        frame_count += 1
    cap.release()

# ===== 2. 이동 특징 추출 =====
def extract_features(track_csv, output_csv):
    df = pd.read_csv(track_csv)
    df = df[df["class"] == "person"]
    results = []
    for track_id, group in df.groupby("track_id"):
        group = group.sort_values("frame").reset_index(drop=True)
        group["dx"] = group["x_center"].diff()
        group["dy"] = group["y_center"].diff()
        group["distance"] = np.sqrt(group["dx"]**2 + group["dy"]**2)
        group["dt"] = group["time"].diff().replace(0, np.nan)
        group["speed"] = group["distance"] / group["dt"]
        group.fillna(0, inplace=True)
        results.append(group)
    feature_df = pd.concat(results, ignore_index=True)
    feature_df.to_csv(output_csv, index=False)

# ===== 3. 시퀀스 생성 =====
def make_sequences(feature_csv, npy_path, sequence_length=30):
    df = pd.read_csv(feature_csv)
    sequences = []
    for _, group in df.groupby("track_id"):
        group = group.sort_values("frame").reset_index(drop=True)
        for i in range(len(group) - sequence_length + 1):
            seq = group.iloc[i:i + sequence_length][["distance", "speed"]].values
            sequences.append(seq)
    X_train = np.array(sequences)
    np.save(npy_path, X_train)

# ===== 4. 이상행동 학습 및 시각화 =====
def detect_anomalies(video_path, X_path, feature_csv, track_csv, output_video, threshold=0.04):
    class LSTMAutoEncoder(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)
        def forward(self, x):
            _, (h, _) = self.encoder(x)
            repeated = h.repeat(x.size(1), 1, 1).permute(1, 0, 2)
            out, _ = self.decoder(repeated)
            return out

    X = np.load(X_path)
    X_tensor = torch.tensor(X).float()
    model = LSTMAutoEncoder(input_dim=2, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(5):
        for i in range(0, len(X_tensor), 32):
            batch = X_tensor[i:i+32]
            output = model(batch)
            loss = criterion(output, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        recon = model(X_tensor)
        raw_scores = torch.mean((X_tensor - recon) ** 2, dim=(1, 2)).numpy()
    scores = 100 * (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-6)

    features = pd.read_csv(feature_csv)
    tracks = pd.read_csv(track_csv)
    anomaly_data = []
    idx = 0
    for tid, group in features.groupby("track_id"):
        group = group.sort_values("frame").reset_index(drop=True)
        for i in range(len(group) - 30 + 1):
            if idx >= len(scores): break
            if scores[idx] > 0:
                f = group.loc[i, "frame"]
                anomaly_data.append({"track_id": tid, "start_frame": int(f), "end_frame": int(f + 30), "score": float(scores[idx])})
            idx += 1

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))

    lookup = {(a["track_id"], f): a["score"] for a in anomaly_data for f in range(a["start_frame"], a["end_frame"])}
    frame_idx = 0
    last_display, persistence, flags = {}, defaultdict(int), defaultdict(list)
    while True:
        ret, frame = cap.read()
        if not ret: break
        for _, row in tracks[tracks["frame"] == frame_idx].iterrows():
            if row["class"] != "person": continue
            tid, cx, cy = row["track_id"], int(row["x_center"]), int(row["y_center"])
            score = lookup.get((tid, frame_idx), None)
            if score is not None:
                prev, last = last_display.get(tid, (0, -1))
                if frame_idx - last >= 1:
                    last_display[tid] = (score, frame_idx)
                else:
                    score = prev
                color = (0, 255, 0)
                if score >= 80:
                    flags[tid].append(1)
                    persistence[tid] += 1
                else:
                    flags[tid].append(0)
                    persistence[tid] = max(0, persistence[tid] - 1)
                if len(flags[tid]) >= 30 and sum(flags[tid][-30:]) >= 10:
                    color = (0, 0, 255)
                    flags[tid] = []
                elif score >= 80:
                    color = (0, 0, 255)
                elif score >= 50:
                    color = (0, 165, 255)
                cv2.putText(frame, f"Score: {score:.1f}/100", (cx-50, cy-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.rectangle(frame, (cx-30, cy-60), (cx+30, cy+60), color, 2)
            else:
                cv2.rectangle(frame, (cx-30, cy-60), (cx+30, cy+60), (0, 255, 0), 1)
        out.write(frame)
        frame_idx += 1
    cap.release()
    out.release()
    print(f"[완료] 저장됨 → {output_video}")

# ===== 메인 실행 =====
if __name__ == "__main__":
    VIDEO_PATH = "./video/video.mp4"
    TRACK_CSV = "tracking_output.csv"
    FEATURE_CSV = "track_features.csv"
    X_PATH = "X_train.npy"
    OUTPUT_VIDEO = "anomaly_result.avi"

    run_tracking(VIDEO_PATH, TRACK_CSV)
    extract_features(TRACK_CSV, FEATURE_CSV)
    make_sequences(FEATURE_CSV, X_PATH)
    detect_anomalies(VIDEO_PATH, X_PATH, FEATURE_CSV, TRACK_CSV, OUTPUT_VIDEO)
