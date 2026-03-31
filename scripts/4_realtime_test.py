"""
4_realtime_test.py - 실시간 카메라 테스트 스크립트
웹캠 또는 핸드폰을 연결하여 실시간으로 농산물을 인식합니다
"""

import cv2
import time
import torch
from ultralytics import YOLO

# ===== 설정 =====
MODEL_PATH  = "./runs/vegetables_v1/weights/best.pt"
CAMERA_ID   = 0      # 0: 내장 웹캠, 1: 외부 카메라 / IP 카메라면 URL 입력
CONF_THRESH = 0.5    # 신뢰도 임계값
# ================

# 클래스별 색상 (BGR)
COLORS = [
    (0, 255, 0),   (0, 200, 100), (0, 100, 255), (255, 200, 0), (200, 0, 255),
    (255, 0, 100), (100, 255, 0), (0, 255, 200), (255, 100, 0), (200, 100, 0),
    (0, 0, 255),   (255, 0, 0),   (0, 255, 255), (255, 255, 0), (100, 0, 255),
]

CLASS_NAMES_KR = {
    "lettuce": "상추", "cabbage": "배추", "tomato": "토마토",
    "potato": "감자", "sweet_potato": "고구마", "carrot": "당근",
    "onion": "양파", "cucumber": "오이", "pumpkin": "호박",
    "pepper": "고추", "garlic": "마늘", "green_onion": "파",
    "broccoli": "브로콜리", "spinach": "시금치", "radish": "무",
}


def run_camera():
    if not __import__("os").path.exists(MODEL_PATH):
        print(f"[오류] 모델 파일 없음: {MODEL_PATH}")
        print("먼저 2_train_model.py 를 실행하여 모델을 학습하세요.")
        return

    model = YOLO(MODEL_PATH)
    cap   = cv2.VideoCapture(CAMERA_ID)

    if not cap.isOpened():
        print(f"[오류] 카메라를 열 수 없습니다. CAMERA_ID={CAMERA_ID} 확인하세요.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("=" * 50)
    print(" 실시간 농산물 인식 시작 ")
    print(" 종료: 'q' 키 누르기 ")
    print("=" * 50)

    fps_list = []

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv8 추론
        results = model(frame, conf=CONF_THRESH, verbose=False)

        detected_items = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cls_id = int(box.cls)
                conf   = float(box.conf)
                name   = model.names[cls_id]
                kr     = CLASS_NAMES_KR.get(name, name)
                color  = COLORS[cls_id % len(COLORS)]

                # 박스 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # 라벨 배경
                label = f"{kr} {conf:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w + 6, y1), color, -1)
                cv2.putText(frame, label, (x1 + 3, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                detected_items.append(f"{kr}({conf:.0%})")

        # FPS 계산
        fps = 1.0 / (time.time() - t0)
        fps_list.append(fps)
        if len(fps_list) > 30:
            fps_list.pop(0)
        avg_fps = sum(fps_list) / len(fps_list)

        # 정보 표시
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if detected_items:
            items_str = " | ".join(detected_items)
            cv2.putText(frame, f"감지: {items_str}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            print(f"\r인식: {items_str}  ", end="", flush=True)
        else:
            cv2.putText(frame, "감지된 농산물 없음", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("전통시장 농산물 인식 - 'q' 종료", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n\n카메라 종료")


def test_single_image(image_path: str):
    """단일 이미지 테스트"""
    if not __import__("os").path.exists(MODEL_PATH):
        print(f"[오류] 모델 파일 없음: {MODEL_PATH}")
        return

    model = YOLO(MODEL_PATH)
    results = model(image_path, conf=CONF_THRESH)

    for r in results:
        print(f"\n{image_path} 인식 결과:")
        for box in r.boxes:
            cls_id = int(box.cls)
            name = model.names[cls_id]
            kr   = CLASS_NAMES_KR.get(name, name)
            conf = float(box.conf)
            print(f"  - {kr}: {conf:.1%}")

    # 결과 이미지 저장
    save_path = image_path.replace(".", "_result.")
    r.save(filename=save_path)
    print(f"결과 저장: {save_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # 이미지 파일 경로가 인자로 주어진 경우
        test_single_image(sys.argv[1])
    else:
        # 기본: 실시간 카메라
        run_camera()
