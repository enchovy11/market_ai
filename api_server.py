# GrandMarket 농산물 AI 서버 연동 예시
# FastAPI에서 YOLOv8 모델로 이미지 인식 후 자동 등록

import os
import io
import cv2
import json
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image

app = FastAPI(title="전통시장 농산물 인식 API")

# 모델 로드 (서버 시작 시 1회만)
MODEL_PATH = "./runs/vegetables_v1/weights/best.pt"
model = None

CLASS_NAMES_KR = {
    "lettuce": "상추", "cabbage": "배추", "tomato": "토마토",
    "potato": "감자", "sweet_potato": "고구마", "carrot": "당근",
    "onion": "양파", "cucumber": "오이", "pumpkin": "호박",
    "pepper": "고추", "garlic": "마늘", "green_onion": "파",
    "broccoli": "브로콜리", "spinach": "시금치", "radish": "무",
}


@app.on_event("startup")
async def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        print(f"모델 로드 완료: {MODEL_PATH}")
    else:
        print(f"[Warning] 모델 없음: {MODEL_PATH}")
        print("학습 후 사용하세요: python scripts/2_train_model.py")


@app.get("/")
async def root():
    return {
        "service": "전통시장 농산물 인식 API",
        "status": "running",
        "model_loaded": model is not None,
        "endpoints": {
            "POST /recognize": "이미지 업로드 후 농산물 인식",
            "GET /classes": "지원 클래스 목록",
        }
    }


@app.get("/classes")
async def get_classes():
    """지원하는 농산물 클래스 목록"""
    return {"classes": CLASS_NAMES_KR}


@app.post("/recognize")
async def recognize_vegetable(
    file: UploadFile = File(...),
    conf_threshold: float = 0.5
):
    """
    이미지를 업로드하면 농산물을 자동 인식하여 결과 반환
    - file: 이미지 파일 (jpg, png 등)
    - conf_threshold: 신뢰도 임계값 (기본 0.5)
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="모델이 아직 학습되지 않았습니다. 먼저 2_train_model.py를 실행하세요."
        )

    # 파일 형식 확인
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")

    try:
        # 이미지 읽기
        contents = await file.read()
        img_array = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="이미지를 읽을 수 없습니다.")

        # AI 추론
        results = model(img, conf=conf_threshold, verbose=False)

        detected = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls)
                conf   = float(box.conf)
                name   = model.names[cls_id]
                kr     = CLASS_NAMES_KR.get(name, name)
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detected.append({
                    "id":          cls_id,
                    "name_en":     name,
                    "name_kr":     kr,        # 한국어 이름
                    "confidence":  round(conf, 4),
                    "bbox":        [round(x1), round(y1), round(x2), round(y2)],
                    "image_size":  [img.shape[1], img.shape[0]],  # [width, height]
                })

        # 가장 높은 신뢰도 상품 (자동 등록용)
        top_product = None
        if detected:
            top_product = max(detected, key=lambda x: x["confidence"])

        return {
            "success":     True,
            "detected":    detected,          # 모든 감지 결과
            "top_product": top_product,       # 가장 높은 신뢰도 상품 (자동 등록 추천)
            "count":       len(detected),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"처리 중 오류: {str(e)}")


# 직접 실행 시
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
