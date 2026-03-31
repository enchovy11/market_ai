"""
5_export_model.py - 학습된 모델을 다양한 형식으로 내보내기
모바일 앱 또는 서버 배포용으로 변환합니다
"""

import os
from ultralytics import YOLO

# ===== 설정 =====
MODEL_PATH  = "./runs/vegetables_v1/weights/best.pt"
EXPORT_DIR  = "./models"
IMG_SIZE    = 640
# ================


def export_all():
    if not os.path.exists(MODEL_PATH):
        print(f"[오류] 모델 파일 없음: {MODEL_PATH}")
        return

    os.makedirs(EXPORT_DIR, exist_ok=True)
    model = YOLO(MODEL_PATH)

    print("=" * 60)
    print(" 모델 내보내기 ")
    print("=" * 60)

    # 1. ONNX (서버/범용 배포)
    print("\n[1/3] ONNX 형식으로 내보내기 (서버 배포용)...")
    try:
        model.export(
            format="onnx",
            imgsz=IMG_SIZE,
            dynamic=True,   # 동적 입력 크기 지원
            simplify=True,
        )
        print("    완료: best.onnx")
    except Exception as e:
        print(f"    실패: {e}")

    # 2. TFLite (안드로이드 앱용)
    print("\n[2/3] TFLite 형식으로 내보내기 (안드로이드용)...")
    try:
        model.export(format="tflite", imgsz=IMG_SIZE)
        print("    완료: best.tflite")
    except Exception as e:
        print(f"    실패 (TensorFlow 미설치): {e}")
        print("    설치: pip install tensorflow")

    # 3. TorchScript (PyTorch 서버용)
    print("\n[3/3] TorchScript 형식으로 내보내기...")
    try:
        model.export(format="torchscript", imgsz=IMG_SIZE)
        print("    완료: best.torchscript")
    except Exception as e:
        print(f"    실패: {e}")

    print("\n" + "=" * 60)
    print(" 내보내기 완료 ")
    print("=" * 60)
    print()
    print("용도별 추천:")
    print("  FastAPI 서버 배포: best.pt 또는 best.onnx")
    print("  안드로이드 앱:     best.tflite")
    print("  iOS 앱:            Core ML 변환 필요 (Mac에서 변환)")
    print()
    print("FastAPI 연동은 api_server.py 참고")


if __name__ == "__main__":
    export_all()
