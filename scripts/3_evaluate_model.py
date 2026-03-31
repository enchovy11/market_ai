"""
3_evaluate_model.py - 학습된 모델 성능 평가 스크립트
mAP, Precision, Recall 등 지표를 확인합니다
"""

import os
import cv2
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from ultralytics import YOLO

# ===== 설정 =====
MODEL_PATH   = "./runs/vegetables_v1/weights/best.pt"
DATASET_YAML = "./datasets/vegetables.yaml"
TEST_IMG_DIR = "./datasets/vegetables/images/test"
RESULT_DIR   = "./runs/evaluation"
CONF_THRESH  = 0.5  # 신뢰도 임계값
# ================

# 클래스 한국어 이름
CLASS_NAMES_KR = {
    "lettuce":      "상추",
    "cabbage":      "배추",
    "tomato":       "토마토",
    "potato":       "감자",
    "sweet_potato": "고구마",
    "carrot":       "당근",
    "onion":        "양파",
    "cucumber":     "오이",
    "pumpkin":      "호박",
    "pepper":       "고추",
    "garlic":       "마늘",
    "green_onion":  "파",
    "broccoli":     "브로콜리",
    "spinach":      "시금치",
    "radish":       "무",
}


def evaluate():
    if not os.path.exists(MODEL_PATH):
        print(f"[오류] 모델 파일 없음: {MODEL_PATH}")
        print("먼저 2_train_model.py 를 실행하세요.")
        return

    os.makedirs(RESULT_DIR, exist_ok=True)

    print("=" * 60)
    print(" 모델 성능 평가 ")
    print("=" * 60)

    model = YOLO(MODEL_PATH)

    # 전체 검증셋으로 mAP 평가
    print("\n전체 검증셋 평가 중...")
    metrics = model.val(data=DATASET_YAML, conf=CONF_THRESH)

    print("\n" + "=" * 60)
    print(" 평가 결과 ")
    print("=" * 60)
    print(f"  mAP@0.5:      {metrics.box.map50:.4f}  (0.5 이상이면 양호)")
    print(f"  mAP@0.5:0.95:  {metrics.box.map:.4f}")
    print(f"  Precision:     {metrics.box.mp:.4f}  (맞다고 한 것 중 실제 맞은 비율)")
    print(f"  Recall:        {metrics.box.mr:.4f}  (실제 물체 중 탐지한 비율)")

    # 클래스별 성능
    print("\n클래스별 mAP:")
    if hasattr(metrics.box, "maps"):
        for i, ap in enumerate(metrics.box.maps):
            class_name = model.names[i]
            kr_name = CLASS_NAMES_KR.get(class_name, class_name)
            bar = "█" * int(ap * 20)
            print(f"  {kr_name:8s} ({class_name:12s}): {ap:.3f} |{bar}")


def predict_samples(n_samples=6):
    """테스트 이미지로 예측 결과 시각화"""
    if not os.path.exists(MODEL_PATH):
        return

    test_images = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        test_images.extend(glob.glob(os.path.join(TEST_IMG_DIR, ext)))

    if not test_images:
        print("\n[Info] 테스트 이미지 없음 - 시각화 건너뜀")
        return

    model = YOLO(MODEL_PATH)
    n_samples = min(n_samples, len(test_images))
    sample_images = test_images[:n_samples]

    print(f"\n{n_samples}개 샘플 이미지 예측 결과 저장 중...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("농산물 인식 결과", fontsize=16, fontweight="bold")
    axes = axes.flatten()

    for i, img_path in enumerate(sample_images):
        results = model(img_path, conf=CONF_THRESH, verbose=False)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ax = axes[i]
        ax.imshow(img)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls)
                conf   = float(box.conf)
                name   = model.names[cls_id]
                kr     = CLASS_NAMES_KR.get(name, name)

                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor="lime", facecolor="none"
                )
                ax.add_patch(rect)
                ax.text(
                    x1, y1 - 5, f"{kr} {conf:.2f}",
                    color="lime", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7)
                )

        ax.set_title(Path(img_path).name, fontsize=8)
        ax.axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    save_path = os.path.join(RESULT_DIR, "sample_predictions.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"결과 이미지 저장: {save_path}")
    plt.show()


if __name__ == "__main__":
    evaluate()
    predict_samples()
    print("\n평가 완료!")
    print("다음 단계: 4_realtime_test.py 로 실시간 카메라 테스트")
