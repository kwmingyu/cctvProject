import os  # 운영체제 경로 처리 및 파일 작업을 위한 모듈
import cv2  # OpenCV: 영상 읽기 및 프레임 처리용 라이브러리
import csv  # CSV 파일 입출력용 모듈
import numpy as np  # 수치 계산을 위한 NumPy 라이브러리
import pandas as pd  # 데이터프레임 형태의 데이터 처리용 라이브러리
from ultralytics import YOLO  # Ultralytics YOLOv8 모델 로딩 및 예측
from deep_sort_realtime.deepsort_tracker import DeepSort  # DeepSORT 추적기 클래스

# === 비디오 트래킹 결과를 CSV로 저장하는 함수 정의 ===
def run_tracking(video_path, output_csv):
    """
    video_path: 입력 비디오 파일 경로
    output_csv: 트래킹 결과를 기록할 CSV 파일 경로
    """
    # 유클리드 거리 계산 함수
    def euclidean(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    # 박스 좌표 클러스터링 및 필터링 함수
    def cluster_and_filter(boxes, classes, confs, threshold=80):
        clusters = []
        used = set()  # 이미 클러스터에 포함된 인덱스 추적
        for i in range(len(boxes)):
            if i in used:
                continue
            cls_i = int(classes[i])
            # 관심 클래스(예: 사람 cls=0)만 처리
            if cls_i != 0:
                continue
            x1, y1, x2, y2 = boxes[i]
            cx_i, cy_i = (x1 + x2) / 2, (y1 + y2) / 2
            cluster = [(i, confs[i], boxes[i])]
            used.add(i)
            # 다른 박스와 거리가 threshold 이하인 것들 클러스터링
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
            # 클러스터 내에서 confidence 최대의 박스 선택
            best = max(cluster, key=lambda x: x[1])
            clusters.append(best)
        return clusters

    # YOLO 모델과 DeepSORT tracker 초기화
    model = YOLO("yolov8n.pt")
    tracker = DeepSort(max_age=80, n_init=1)
    cap = cv2.VideoCapture(video_path)  # 비디오 캡처 객체 생성
    fps = cap.get(cv2.CAP_PROP_FPS)  # 비디오 FPS 획득

    # 결과 CSV 파일 헤더 작성
    with open(output_csv, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time", "track_id", "class", "x_center", "y_center"])

    frame_count = 0  # 프레임 카운터 초기화
    last_seen = {}  # alias_id 별 마지막 중심점 및 프레임 기록
    id_alias_map = {}  # raw tracker ID -> alias ID 매핑
    next_alias_id = 1  # 새로운 alias ID 증가값
    DIST_THRESHOLD = 50  # alias 매칭용 거리 임계값
    FRAME_GAP = 30  # alias 매칭용 프레임 차이 임계값

    # 비디오 프레임 순회
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO로 객체 탐지 (사람 클래스만)
        results = model.predict(frame, conf=0.3, classes=[0])[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()

        # 사람 박스 클러스터링 및 필터링
        clustered_persons = cluster_and_filter(boxes, classes, confs)
        filtered = [(box, conf, int(classes[idx])) for idx, conf, box in clustered_persons]

        # 추가로 자동차 등 다른 클래스 필터링 예시 (cls==2 조건)
        for i in range(len(boxes)):
            cls = int(classes[i])
            if cls == 2:
                x1, y1, x2, y2 = boxes[i]
                # 크기 조건 충족 시 포함
                if x2 - x1 >= 30 and y2 - y1 >= 60:
                    filtered.append((boxes[i], confs[i], cls))

        # DeepSORT 입력 형식: [x, y, w, h]
        input_dets = [([x1, y1, x2 - x1, y2 - y1], conf, cls)
                      for (x1, y1, x2, y2), conf, cls in filtered]
        tracks = tracker.update_tracks(input_dets, frame=frame)

        # CSV에 결과 기록
        with open(output_csv, "a", newline='') as f:
            writer = csv.writer(f)
            for track in tracks:
                if not track.is_confirmed():
                    continue
                raw_id = track.track_id
                cls = track.det_class
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # alias ID 매핑 로직
                if raw_id not in id_alias_map:
                    matched = False
                    for aid, (px, py, lf) in last_seen.items():
                        # 프레임 차이 및 거리 기준 매칭
                        if frame_count - lf < FRAME_GAP and euclidean((cx, cy), (px, py)) < DIST_THRESHOLD:
                            id_alias_map[raw_id] = aid
                            matched = True
                            break
                    if not matched:
                        id_alias_map[raw_id] = next_alias_id
                        next_alias_id += 1

                alias_id = id_alias_map[raw_id]
                last_seen[alias_id] = (cx, cy, frame_count)

                # 프레임, 시간, alias ID, 클래스명, 중심 좌표 기록
                writer.writerow([
                    frame_count,
                    round(frame_count / fps, 2),
                    alias_id,
                    'person' if cls == 0 else 'car',
                    cx,
                    cy
                ])
        frame_count += 1
    cap.release()  # 캡처 객체 해제


# === 트래킹 CSV로부터 행동 특성(거리, 속도) 계산 및 CSV 저장 함수 정의 ===
def extract_features(track_csv, output_csv):
    # CSV 읽어와서 사람 클래스만 필터링
    df = pd.read_csv(track_csv)
    df = df[df["class"] == "person"]
    results = []
    # 트랙별로 그룹화하여 거리 및 속도 계산
    for track_id, group in df.groupby("track_id"):
        group = group.sort_values("frame").reset_index(drop=True)
        group["dx"] = group["x_center"].diff()
        group["dy"] = group["y_center"].diff()
        group["distance"] = np.sqrt(group["dx"]**2 + group["dy"]**2)
        group["dt"] = group["time"].diff().replace(0, np.nan)
        group["speed"] = group["distance"] / group["dt"]
        group.fillna(0, inplace=True)
        results.append(group)
    # 모든 그룹을 합쳐서 CSV로 저장
    feature_df = pd.concat(results, ignore_index=True)
    feature_df.to_csv(output_csv, index=False)


# === 특성 CSV로부터 고정 길이 시퀀스 생성 및 NPY 파일 저장 함수 정의 ===
def make_sequences(feature_csv, npy_path, sequence_length=30):
    # 특성 CSV 읽기
    df = pd.read_csv(feature_csv)
    sequences = []
    # 트랙별로 시퀀스 슬라이딩 윈도우 생성
    for _, group in df.groupby("track_id"):
        group = group.sort_values("frame").reset_index(drop=True)
        for i in range(len(group) - sequence_length + 1):
            seq = group.iloc[i:i + sequence_length][["distance", "speed"]].values
            sequences.append(seq)
    # NumPy 배열로 변환 후 NPY 파일 저장
    X_train = np.array(sequences)
    np.save(npy_path, X_train)


# ===== 스크립트 직접 실행 시 전체 비디오 디렉토리 처리 =====
if __name__ == "__main__":
    import glob  # 파일 패턴 매칭 모듈

    video_dir = "./video"  # 비디오 폴더 경로
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
