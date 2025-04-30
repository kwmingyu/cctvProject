from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import csv
import numpy as np

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# 모델 및 추적기 초기화
model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=80, n_init=1)

video_path = "./video/video.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None

# CSV 초기화
csv_file = "tracking_output.csv"
with open(csv_file, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "time", "track_id", "class", "x_center", "y_center"])

frame_count = 0

# ID 병합용 상태 저장
last_seen = {}  # {alias_id: (x, y, last_frame)}
id_alias_map = {}  # {raw_id: alias_id}
next_alias_id = 1

DIST_THRESHOLD = 50  # 중심 좌표 거리
FRAME_GAP = 30       # 최대 ID 연속 허용 프레임 간격

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame.shape[1] < 720:
        frame = cv2.resize(frame, (1280, 720))

    if out is None:
        h, w = frame.shape[:2]
        out = cv2.VideoWriter("output_annotated.avi", fourcc, fps, (w, h))

    detections = model.predict(frame, conf=0.3)[0]
    boxes = detections.boxes.xyxy.cpu().numpy()
    classes = detections.boxes.cls.cpu().numpy()
    confs = detections.boxes.conf.cpu().numpy()

    input_dets = []
    for i in range(len(boxes)):
        cls = int(classes[i])
        if cls in [0, 2]:  # person, car
            x1, y1, x2, y2 = boxes[i]
            w, h = x2 - x1, y2 - y1
            input_dets.append(([x1, y1, w, h], confs[i], cls))

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
            now = (cx, cy)

            # ID 병합 로직
            if raw_id not in id_alias_map:
                matched = False
                for alias_id, (px, py, last_f) in last_seen.items():
                    if frame_count - last_f < FRAME_GAP and euclidean(now, (px, py)) < DIST_THRESHOLD:
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
