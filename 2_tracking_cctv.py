from ultralytics import YOLO  # YOLOv8 객체 감지 모델 로드
from deep_sort_realtime.deepsort_tracker import DeepSort  # 객체 추적기 DeepSORT 사용
import cv2  # OpenCV: 영상 처리
import csv  # 감지 결과 CSV 저장
import numpy as np  # 수학 계산용

# 두 점 사이의 거리 계산 함수 (중심좌표 거리 비교용)
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# IoU 계산 함수 (박스 겹침 비율 계산)
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

# 클러스터링 기반 중복 제거 함수 (사람 객체만)
def cluster_and_filter(boxes, classes, confs, threshold=80):
    clusters = []  # 결과 저장용
    used = set()   # 중복 검사용 인덱스
    for i in range(len(boxes)):
        if i in used:
            continue
        cls_i = int(classes[i])
        if cls_i != 0:  # 사람(class=0)만 처리
            continue
        x1, y1, x2, y2 = boxes[i]
        cx_i, cy_i = (x1 + x2) / 2, (y1 + y2) / 2
        cluster = [(i, confs[i], boxes[i])]  # 첫 박스 클러스터 시작
        used.add(i)
        for j in range(i+1, len(boxes)):
            if j in used:
                continue
            cls_j = int(classes[j])
            if cls_j != cls_i:
                continue
            x1j, y1j, x2j, y2j = boxes[j]
            cx_j, cy_j = (x1j + x2j) / 2, (y1j + y2j) / 2
            # 중심 좌표 거리 기준으로 그룹핑
            if euclidean((cx_i, cy_i), (cx_j, cy_j)) < threshold:
                cluster.append((j, confs[j], boxes[j]))
                used.add(j)
        if cluster:
            # 클러스터 내에서 conf 높은 박스 하나만 남김
            best = max(cluster, key=lambda x: x[1])
            clusters.append(best)
    return clusters

# 모델 및 추적기 초기화
model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=80, n_init=1)

# 비디오 로드
video_path = "./video/video.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# 결과 영상 저장용 설정
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None

# CSV 파일 초기화
csv_file = "tracking_output.csv"
with open(csv_file, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "time", "track_id", "class", "x_center", "y_center"])

# 초기 상태 변수들
frame_count = 0
last_seen = {}         # alias_id별 마지막 위치 저장
id_alias_map = {}      # raw_id → alias_id 매핑
next_alias_id = 1      # alias_id 증가용 카운터

# 하이퍼파라미터
DIST_THRESHOLD = 50
FRAME_GAP = 30
MIN_WIDTH = 30
MIN_HEIGHT = 60

# 메인 루프: 프레임 단위로 감지 및 추적
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 해상도가 작을 경우 리사이즈
    if frame.shape[1] < 720:
        frame = cv2.resize(frame, (1280, 720))

    # 출력 영상 초기화
    if out is None:
        h, w = frame.shape[:2]
        out = cv2.VideoWriter("output_annotated.avi", fourcc, fps, (w, h))

    # YOLO 감지 수행
    results = model.predict(frame, conf=0.3)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()

    # 클러스터링 기반 사람 중복 제거
    clustered_persons = cluster_and_filter(boxes, classes, confs)
    filtered = []
    for idx, conf, box in clustered_persons:
        cls = int(classes[idx])
        filtered.append((box, conf, cls))

    # 차량(car)은 그대로 필터만 적용해서 추가
    for i in range(len(boxes)):
        cls_i = int(classes[i])
        if cls_i == 2:  # car only
            x1, y1, x2, y2 = boxes[i]
            w, h = x2 - x1, y2 - y1
            if w >= MIN_WIDTH and h >= MIN_HEIGHT:
                filtered.append((boxes[i], confs[i], cls_i))

    # DeepSORT가 받을 입력 데이터 형식으로 변환
    input_dets = []
    for box, conf, cls in filtered:
        x1, y1, x2, y2 = box
        input_dets.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

    # 객체 추적 실행
    tracks = tracker.update_tracks(input_dets, frame=frame)

    # 결과 저장
    with open(csv_file, "a", newline='') as f:
        writer = csv.writer(f)
        for track in tracks:
            if not track.is_confirmed():
                continue

            raw_id = track.track_id
            cls = track.det_class
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # ID 병합 처리
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

            # 시각화 출력
            label = f"{'person' if cls == 0 else 'car'}{alias_id}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # CSV 저장
            writer.writerow([
                frame_count,
                round(frame_count / fps, 2),
                alias_id,
                'person' if cls == 0 else 'car',
                cx,
                cy
            ])

    # 화면 출력 및 영상 저장
    cv2.imshow("Tracking", frame)
    out.write(frame)
    frame_count += 1

    # 종료 키
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 정리
cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
