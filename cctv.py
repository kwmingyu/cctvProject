from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import csv
import numpy as np

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area != 0 else 0

def cluster_and_filter(boxes, classes, confs, threshold=80):
    clusters = []
    used = set()
    for i in range(len(boxes)):
        if i in used:
            continue
        cls_i = int(classes[i])
        if cls_i != 0:
            continue  # only cluster person class
        x1, y1, x2, y2 = boxes[i]
        cx_i, cy_i = (x1 + x2) / 2, (y1 + y2) / 2
        cluster = [(i, confs[i], boxes[i])]
        used.add(i)
        for j in range(i+1, len(boxes)):
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
        if cluster:
            # keep only the box with highest confidence
            best = max(cluster, key=lambda x: x[1])
            clusters.append(best)
    return clusters

model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=80, n_init=1)

video_path = "./video/video.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None

csv_file = "tracking_output.csv"
with open(csv_file, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "time", "track_id", "class", "x_center", "y_center"])

frame_count = 0
last_seen = {}
id_alias_map = {}
next_alias_id = 1

DIST_THRESHOLD = 50
FRAME_GAP = 30
MIN_WIDTH = 30
MIN_HEIGHT = 60

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame.shape[1] < 720:
        frame = cv2.resize(frame, (1280, 720))

    if out is None:
        h, w = frame.shape[:2]
        out = cv2.VideoWriter("output_annotated.avi", fourcc, fps, (w, h))

    results = model.predict(frame, conf=0.3)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()

    # 클러스터링 중복 제거 적용
    clustered_persons = cluster_and_filter(boxes, classes, confs)
    filtered = []
    for idx, conf, box in clustered_persons:
        cls = int(classes[idx])
        filtered.append((box, conf, cls))

    # 차량은 따로 추가
    for i in range(len(boxes)):
        cls_i = int(classes[i])
        if cls_i == 2:  # car only
            x1, y1, x2, y2 = boxes[i]
            w, h = x2 - x1, y2 - y1
            if w >= MIN_WIDTH and h >= MIN_HEIGHT:
                filtered.append((boxes[i], confs[i], cls_i))

    input_dets = []
    for box, conf, cls in filtered:
        x1, y1, x2, y2 = box
        input_dets.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

    tracks = tracker.update_tracks(input_dets, frame=frame)

    with open(csv_file, "a", newline='') as f:
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
                for alias_id, (px, py, last_f) in last_seen.items():
                    if frame_count - last_f < FRAME_GAP and euclidean((cx, cy), (px, py)) < DIST_THRESHOLD:
                        id_alias_map[raw_id] = alias_id
                        matched = True
                        break
                if not matched:
                    id_alias_map[raw_id] = next_alias_id
                    next_alias_id += 1

            alias_id = id_alias_map[raw_id]
            last_seen[alias_id] = (cx, cy, frame_count)

            label = f"{'person' if cls == 0 else 'car'}{alias_id}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            writer.writerow([
                frame_count,
                round(frame_count / fps, 2),
                alias_id,
                'person' if cls == 0 else 'car',
                cx,
                cy
            ])

    cv2.imshow("Tracking", frame)
    out.write(frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
