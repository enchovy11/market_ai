"""
1_prepare_data.py - 학습 데이터 준비 스크립트
AI Hub 또는 직접 촬영한 데이터를 YOLOv8 형식으로 정리합니다
"""

import os
import shutil
import random
import glob
from pathlib import Path

# ===== 설정 =====
SOURCE_IMAGES_DIR = "./raw_images"   # 원본 이미지 폴더
DATASET_DIR = "./datasets/vegetables"
TRAIN_RATIO = 0.7  # 학습 70%
VAL_RATIO   = 0.2  # 검증 20%
TEST_RATIO  = 0.1  # 테스트 10%
# ================


def split_dataset(source_dir: str, dataset_dir: str):
    """이미지와 라벨을 train/val/test로 분할"""

    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

    # 모든 이미지 파일 수집
    image_files = []
    for ext in image_extensions:
        image_files.extend(
            glob.glob(os.path.join(source_dir, "**", f"*{ext}"), recursive=True)
        )

    if not image_files:
        print(f"[Warning] {source_dir} 에 이미지가 없습니다.")
        print("raw_images 폴더에 이미지를 넣어주세요.")
        return

    print(f"총 {len(image_files)}개 이미지 발견")

    # 섞기
    random.shuffle(image_files)

    # 분할
    n = len(image_files)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    splits = {
        "train": image_files[:train_end],
        "val":   image_files[train_end:val_end],
        "test":  image_files[val_end:],
    }

    for split_name, files in splits.items():
        img_dir = os.path.join(dataset_dir, "images", split_name)
        lbl_dir = os.path.join(dataset_dir, "labels", split_name)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for img_path in files:
            img_name = Path(img_path).name
            shutil.copy2(img_path, os.path.join(img_dir, img_name))

            # 대응하는 라벨 파일 복사
            label_path = Path(img_path).with_suffix(".txt")
            if label_path.exists():
                shutil.copy2(label_path, os.path.join(lbl_dir, label_path.name))

        print(f"  {split_name}: {len(files)}개")

    print("\n데이터셋 준비 완료!")


def print_guide():
    """사용 안내"""
    print("=" * 60)
    print(" 전통시장 농산물 AI - 데이터 준비 가이드 ")
    print("=" * 60)
    print()
    print("[방법 1] AI Hub 데이터 사용 (무료, 대용량)")
    print("   1. https://aihub.or.kr 접속 로그인")
    print("   2. '농산물' 또는 '채소' 검색")
    print("   3. 다운로드 후 raw_images/ 폴더에 넣기")
    print()
    print("[방법 2] Roboflow 사용 (라벨링 포함)")
    print("   1. https://roboflow.com 접속")
    print("   2. 프로젝트 생성 > 이미지 업로드 > 라벨링")
    print("   3. Export: YOLOv8 형식으로 다운로드")
    print("   4. datasets/vegetables/ 에 덮어쓰기")
    print()
    print("[방법 3] 직접 촬영")
    print("   1. raw_images/ 폴더에 사진 넣기")
    print("   2. 라벨링 도구(LabelImg) 설치:")
    print("      pip install labelImg")
    print("      labelImg")
    print("   3. YOLO 형식으로 라벨 저장")
    print()
    print("클래스 ID 매핑:")
    classes = [
        "0:상추", "1:배추", "2:토마토", "3:감자", "4:고구마",
        "5:당근", "6:양파", "7:오이", "8:호박", "9:고추",
        "10:마늘", "11:파", "12:브로콜리", "13:시금치", "14:무"
    ]
    for c in classes:
        print(f"   {c}")


if __name__ == "__main__":
    print_guide()

    raw_dir = "./raw_images"
    os.makedirs(raw_dir, exist_ok=True)

    files_in_raw = []
    if os.path.exists(raw_dir):
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            files_in_raw.extend(glob.glob(os.path.join(raw_dir, "**", f"*{ext}"), recursive=True))

    if files_in_raw:
        print(f"\n{len(files_in_raw)}개 이미지 발견 - 데이터셋 분할 시작...")
        split_dataset(raw_dir, DATASET_DIR)
    else:
        print("\nraw_images/ 폴더를 생성했습니다.")
        print("위 가이드를 참고하여 이미지를 넣은 후 다시 실행하세요.")
