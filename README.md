# 전통시장 농산물 AI 인식 시스템 🥬🍅🥔

핸드폰 카메라로 농산물을 자동 인식하여 전통시장 상품을 등록하는 AI 모델 프로젝트

## 📁 프로젝트 구조

```
market_ai/
├── datasets/
│   ├── vegetables.yaml         # 클래스 설정 파일
│   └── vegetables/
│       ├── images/
│       │   ├── train/          # 학습 이미지 (이곳에 준비)
│       │   ├── val/            # 검증 이미지
│       │   └── test/           # 테스트 이미지
│       └── labels/
│           ├── train/          # 학습 라벨 (.txt)
│           ├── val/
│           └── test/
├── scripts/
│   ├── 1_prepare_data.py       # STEP 1: 데이터 준비 및 분할
│   ├── 2_train_model.py        # STEP 2: YOLOv8 모델 학습
│   ├── 3_evaluate_model.py     # STEP 3: 성능 평가
│   ├── 4_realtime_test.py      # STEP 4: 실시간 카메라 테스트
│   └── 5_export_model.py       # STEP 5: 모바일/서버용 내보내기
├── api_server.py               # FastAPI 서버 (GrandMarket 연동)
├── requirements.txt            # 의존성 패키지
└── README.md
```

## 🚀 빠른 시작

### 1단계: 환경 설치

```bash
cd market_ai
pip install -r requirements.txt
```

### 2단계: 데이터 준비

**방법 A - AI Hub (무료, 추천)**
1. https://aihub.or.kr 접속 → 회원가입
2. `농산물 이미지` 검색 → 신청 및 다운로드
3. `raw_images/` 폴더에 이미지+라벨 파일 저장

**방법 B - Roboflow (라벨링 GUI 포함)**
1. https://roboflow.com 접속 → 무료 계정 생성
2. 프로젝트 생성 → 이미지 업로드 → 웹에서 라벨링
3. Export → **YOLOv8 형식** 다운로드
4. `datasets/vegetables/` 에 덮어쓰기

**방법 C - 직접 사진 촬영 후 라벨링**
```bash
pip install labelImg
labelImg  # GUI 라벨링 도구 실행
# YOLO 형식으로 저장할 것!
```

```bash
python scripts/1_prepare_data.py
```

### 3단계: GPU 학습 (로컬 GPU 있을 때)

```bash
python scripts/2_train_model.py
# 학습 완료 후: runs/vegetables_v1/weights/best.pt
```

### 3단계 대안: Google Colab으로 무료 GPU 학습

GPU가 없으면 Google Colab 사용 (무료 T4 GPU):
1. https://colab.research.google.com 접속
2. `colab_train.ipynb` 파일 업로드
3. 런타임 → GPU 변경 → 실행

### 4단계: 성능 평가

```bash
python scripts/3_evaluate_model.py
# mAP@0.5: 0.85 이상이면 우수
```

### 5단계: 실시간 카메라 테스트

```bash
python scripts/4_realtime_test.py
# 'q' 키로 종료
```

### 6단계: 모델 내보내기

```bash
python scripts/5_export_model.py
# best.onnx, best.tflite 생성
```

### 7단계: FastAPI 서버 실행

```bash
pip install fastapi uvicorn
python api_server.py
# http://localhost:8000/docs 에서 API 테스트
```

## 🥬 지원 클래스 (15종)

| ID | 영어 | 한국어 |
|----|------|--------|
| 0  | lettuce | 상추 |
| 1  | cabbage | 배추 |
| 2  | tomato | 토마토 |
| 3  | potato | 감자 |
| 4  | sweet_potato | 고구마 |
| 5  | carrot | 당근 |
| 6  | onion | 양파 |
| 7  | cucumber | 오이 |
| 8  | pumpkin | 호박 |
| 9  | pepper | 고추 |
| 10 | garlic | 마늘 |
| 11 | green_onion | 파 |
| 12 | broccoli | 브로콜리 |
| 13 | spinach | 시금치 |
| 14 | radish | 무 |

> `datasets/vegetables.yaml`에서 클래스 추가/수정 가능

## 📱 핸드폰 연동 방법

### REST API 방식 (권장)
```
핸드폰 카메라 촬영
      ↓ POST /recognize
FastAPI 서버 (api_server.py)
      ↓ 인식 결과 JSON
GrandMarket 앱 자동 등록
```

### API 사용 예시 (Flutter/앱)
```dart
// 이미지 촬영 후 서버로 전송
final response = await http.post(
  Uri.parse('http://서버IP:8000/recognize'),
  body: {'file': imageBytes},
);
final result = jsonDecode(response.body);
final topProduct = result['top_product']['name_kr'];  // "토마토"
```

## ⚙️ 학습 설정 조정

`scripts/2_train_model.py` 상단의 설정 변경:

```python
MODEL_SIZE  = "yolov8n"   # n=빠름, s, m, l, x=정확
EPOCHS      = 100         # 데이터 부족 시 50, 충분 시 150
BATCH_SIZE  = 16          # VRAM 4GB=8, 8GB=16, 16GB=32
```

## 📊 기대 성능

| 데이터 수 | 예상 mAP@0.5 | 학습 시간(GPU) |
|-----------|-------------|---------------|
| 클래스당 100장 | 0.60~0.70 | 30분 |
| 클래스당 300장 | 0.75~0.85 | 1~2시간 |
| 클래스당 500장+ | 0.85~0.95 | 2~4시간 |
