# make_sequences_batch.py -  시퀀스 생성

import os
import cv2
import csv
import numpy as np
import pandas as pd
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

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


        results = model.predict(frame, conf=0.3, classes=[0])[0]
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

# ===== 메인 실행 =====
if __name__ == "__main__":
    import glob

    video_dir = "./video"
    video_paths = glob.glob(os.path.join(video_dir, "*.mp4"))

    for path in video_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        track_csv = f"tracking_output_{name}.csv"
        feature_csv = f"track_features_{name}.csv"
        x_path = f"X_train_{name}.npy"

        print(f"[처리 중] {name}")
        run_tracking(path, track_csv)
        extract_features(track_csv, feature_csv)
        make_sequences(feature_csv, x_path)
        print(f"[완료] 시퀀스 저장됨 → {x_path}")
