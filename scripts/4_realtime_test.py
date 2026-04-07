"""
4_realtime_test.py - 실시간 카메라 테스트 스크립트
웹캠 또는 핸드폰을 연결하여 실시간으로 농산물을 인식합니다
"""

import cv2
import time
import torch
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from ultralytics import YOLO

# ===== 설정 =====
MODEL_PATH  = "./runs/vegetables_26classes2/weights/best.pt"
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
    "avocado": "아보카도", "beans": "콩", "beet": "비트",
    "bell pepper": "피망", "broccoli": "브로콜리", "brus capusta": "방울양배추",
    "cabbage": "배추", "carrot": "당근", "cayliflower": "콜리플라워",
    "celery": "샐러리", "corn": "옥수수", "cucumber": "오이",
    "eggplant": "가지", "fasol": "강낭콩", "garlic": "마늘",
    "hot pepper": "매운고추", "onion": "양파", "peas": "완두콩",
    "potato": "감자", "pumpkin": "호박", "rediska": "래디시(무)",
    "redka": "무", "salad": "상추", "squash-patisson": "패티슨호박",
    "tomato": "토마토", "vegetable marrow": "애호박",
}


def run_camera():
    if not __import__("os").path.exists(MODEL_PATH):
        print(f"[오류] 모델 파일 없음: {MODEL_PATH}")
        print("먼저 2_train_model.py 를 실행하여 모델을 학습하세요.")
        return

    model = YOLO(MODEL_PATH)
    
    # 카메라 자동 감지 (0~3 순회)
    cap = None
    for cam_idx in range(4):
        # cv2.CAP_DSHOW는 Windows에서 카메라 지연이나 오류를 줄여줍니다.
        temp_cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        if temp_cap.isOpened():
            cap = temp_cap
            print(f"[*] 카메라가 정상적으로 연결되었습니다! (CAMERA_ID: {cam_idx})")
            break
    
    if cap is None:
        print("[오류] 연결된 카메라를 찾을 수 없습니다. 외부 카메라 선 연결 여부나 다른 앱 사용 여부를 확인해주세요.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("=" * 50)
    print(" 실시간 농산물 인식 시작 ")
    print(" 종료: 'q' 키 누르기 ")
    print("=" * 50)

    # 화면에 그릴 한글 폰트 로드 (윈도우 기본 맑은 고딕)
    try:
        font = ImageFont.truetype("malgun.ttf", 20)
    except:
        font = ImageFont.load_default()

    fps_list = []

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv8 추론
        results = model(frame, conf=CONF_THRESH, verbose=False)
        detected_items = []

        # 한글 표시를 위해 OpenCV 이미지를 PIL 이미지로 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_img)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cls_id = int(box.cls)
                conf   = float(box.conf)
                name   = model.names[cls_id]
                kr     = CLASS_NAMES_KR.get(name, name)
                
                # BGR 색상을 RGB로 변환
                cbgr = COLORS[cls_id % len(COLORS)]
                crgb = (cbgr[2], cbgr[1], cbgr[0])

                # 바운딩 박스 그리기
                draw.rectangle((x1, y1, x2, y2), outline=crgb, width=3)

                # 라벨 텍스트 크기 구하기 및 그리기
                label = f"{kr} {conf:.2f}"
                bbox = draw.textbbox((0, 0), label, font=font)
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                
                draw.rectangle((x1, y1 - h - 10, x1 + w + 10, y1), fill=crgb)
                draw.text((x1 + 5, y1 - h - 6), label, font=font, fill=(255, 255, 255))
                
                detected_items.append(f"{kr}({conf:.0%})")

        # 화면 하단 감지 요약 텍스트
        if detected_items:
            items_str = " | ".join(detected_items)
            draw.text((10, 45), f"감지: {items_str}", font=font, fill=(255, 255, 0))
            print(f"\r인식: {items_str}  ", end="", flush=True)
        else:
            draw.text((10, 45), "감지된 농산물 없음", font=font, fill=(200, 200, 200))
            print(f"\r감지된 농산물 없음  ", end="", flush=True)

        # 다시 OpenCV BGR 이미지로 변환
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # FPS 표시 (영어이므로 기존 cv2.putText 사용)
        fps = 1.0 / (time.time() - t0)
        fps_list.append(fps)
        if len(fps_list) > 30: fps_list.pop(0)
        avg_fps = sum(fps_list) / len(fps_list)
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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
