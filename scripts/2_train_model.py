"""
2_train_model.py - YOLOv8 모델 학습 스크립트
농산물 이미지로 YOLOv8 모델을 파인튜닝합니다
"""

import os
import torch
from pathlib import Path
from ultralytics import YOLO

# ===== 학습 설정 =====
MODEL_SIZE   = "yolov8n"   # n(빠름), s, m, l, x(정확) 중 선택
EPOCHS       = 100         # 학습 반복 수 (데이터 적으면 50, 많으면 150)
IMG_SIZE     = 640         # 입력 이미지 크기
BATCH_SIZE   = 8           # 배치 사이즈 (VRAM 8GB → 8로 조정)
PATIENCE     = 20          # 조기 종료 (성능 개선 없으면 N epoch 후 중단)
DEVICE       = "0" if torch.cuda.is_available() else "cpu"  # GPU 자동 감지
DATASET_YAML = "./datasets/vegetables.yaml"
PROJECT_DIR  = "./runs"
RUN_NAME     = "vegetables_26classes"
# =====================


def check_dataset():
    """데이터셋 존재 확인"""
    train_dir = "./datasets/vegetables/images/train"
    val_dir   = "./datasets/vegetables/images/val"

    train_count = len(list(Path(train_dir).glob("*"))) if os.path.exists(train_dir) else 0
    val_count   = len(list(Path(val_dir).glob("*")))   if os.path.exists(val_dir)   else 0

    print(f"학습 이미지: {train_count}개")
    print(f"검증 이미지: {val_count}개")

    if train_count == 0:
        print("\n[오류] 학습 데이터가 없습니다!")
        print("먼저 1_prepare_data.py 를 실행하세요.")
        return False
    if val_count == 0:
        print("\n[오류] 검증 데이터가 없습니다!")
        return False
    return True


def train():
    print("=" * 60)
    print(" 전통시장 농산물 YOLOv8 모델 학습 시작 ")
    print("=" * 60)

    # GPU 확인
    if torch.cuda.is_available():
        print(f"\nGPU 사용: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\nCPU 사용 (GPU 없음 - 학습이 느릴 수 있습니다)")
        print("Google Colab 사용을 권장합니다: https://colab.research.google.com")

    # 데이터셋 확인
    print()
    if not check_dataset():
        return

    # 사전학습 모델 로드 (전이학습 - COCO 데이터셋으로 이미 학습됨)
    pretrained = f"{MODEL_SIZE}.pt"
    print(f"\n사전학습 모델 로드: {pretrained}")
    model = YOLO(pretrained)

    print(f"\n학습 파라미터:")
    print(f"  모델: {MODEL_SIZE}")
    print(f"  에폭: {EPOCHS}")
    print(f"  이미지 크기: {IMG_SIZE}")
    print(f"  배치 사이즈: {BATCH_SIZE}")
    print(f"  디바이스: {DEVICE}")
    print()

    # 학습 시작
    results = model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        device=DEVICE,
        project=PROJECT_DIR,
        name=RUN_NAME,
        save=True,
        plots=True,           # 학습 그래프 저장
        augment=True,         # 데이터 증강
        cache=False,          # 메모리 캐싱 (RAM 16GB 이상이면 True)
        workers=0,            # Windows 환경 멈춤 방지를 위해 0으로 설정
        verbose=True,
    )

    print("\n" + "=" * 60)
    print(" 학습 완료! ")
    print("=" * 60)
    print(f"최고 성능 모델 저장 위치:")
    print(f"  {PROJECT_DIR}/{RUN_NAME}/weights/best.pt")
    print()
    print("다음 단계: 3_evaluate_model.py 실행")

    return results


if __name__ == "__main__":
    train()
