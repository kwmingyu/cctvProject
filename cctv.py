from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import csv

# 모델 및 추적기 초기화
model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=80, n_init=1)

# 비디오 설정
video_path = "./video/video.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# 출력 영상 설정 (선택사항)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None

# CSV 초기화
csv_file = "tracking_output.csv"
with open(csv_file, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "time", "track_id", "class", "x_center", "y_center"])

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame.shape[1] < 720:
        frame = cv2.resize(frame, (1280, 720))

    if out is None:
        h, w = frame.shape[:2]
        out = cv2.VideoWriter("output_annotated.avi", fourcc, fps, (w, h))

    # YOLO 예측
    detections = model.predict(frame, conf=0.3)[0]
    boxes = detections.boxes.xyxy.cpu().numpy()
    classes = detections.boxes.cls.cpu().numpy()
    confs = detections.boxes.conf.cpu().numpy()

    # DeepSORT 입력
    input_dets = []
    for i in range(len(boxes)):
        cls = int(classes[i])
        if cls in [0, 2]:  # person, car
            x1, y1, x2, y2 = boxes[i]
            w, h = x2 - x1, y2 - y1
            input_dets.append(([x1, y1, w, h], confs[i], cls))

    tracks = tracker.update_tracks(input_dets, frame=frame)

    # 트래킹 및 화면 출력
    with open(csv_file, "a", newline='') as f:
        writer = csv.writer(f)
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb, cls = track.to_ltrb(), track.det_class
            x1, y1, x2, y2 = map(int, ltrb)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            label = f"{'person' if cls == 0 else 'car'}{track_id}"

            # 사각형 + 라벨 출력
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # CSV 저장
            writer.writerow([
                frame_count,
                round(frame_count / fps, 2),
                track_id,
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
