# AIX-deep-learning
---

### 다양한 환경 조건에서의 YOLO를 활용한 객체 인식 성능 비교 연구

#### 컴퓨터소프트웨어학부 박지홍 wlghd0126@naver.com
#### 미래자동차공학과 심승환 paulmsa@naver.com
#### 데이터사이언스학과 최다예 allyes1227@gmail.com

---
## Title: 
Addressing domain imbalance in weather conditions for autonomous driving object detection : A comparative study of data balancing strategies

---

# 1. Proposal

### Motivation : Why are you doing this?

자율주행 시스템의 객체 검출 성능은 학습 데이터의 분포에 크게 영향을 받습니다. 특히 BDD100K와 같은 대규모 자율주행 데이터셋은 날씨와 시간대에 따른 심각한 도메인 불균형 문제를 가지고 있습니다.

핵심 문제:
+ Clear 조건의 데이터가 전체의 약 75%를 차지
+ Adverse weather(비,눈) 조건은 각각 10% 미만

연구 질문:
1. 도메인 불균형(날씨 조건별)이 객체 검출 성능에 얼마나 영향을 미치는가?
2. Real balanced data vs Synthetic augmentation의 성능 차이는?

### What do you want to see at the end?

+ 정량적 분석
  + 6개 날씨 조건별 mAP 비교 (Day/Night x Clear/Rain/Snow)
  + 불균형 데이터의 성능 저하 정도 측정
  + Augmentation의 효과 정량화

 
---

# 2. Datasets

## Overview
+ 데이터셋명: BDD100K (Berkeley DeepDrive 100K)
+ 출처 : UC Berkeley, 100,000개의 주행 영상 프레임
+ 이미지 해상도 : 1280x720 pixels
+ 주석 형식: Bounding box annotations (10 classes)

## Object Classes
+ 차량 : car, truck, bus, train
+ 이륜차 : motorcycle, bicycle
+ 보행자 : person, rider
+ 교통시설 : traffic light, traffic sign

## Weather & Time conditions
+ Clear
+ Rainy
+ Snowy
+ Foggy
+ Day
+ Night

## Train set 분포 분석
- Clear : 74.8%
- Adverse weather : 25.2%



## 데이터 구성

| Group | 구성 | 총 이미지 수 |
|-------|------|-------------|
| **Group 1: Real Imbalanced** | day-clear: 3,600 / day-rain: 600 / day-snow: 600 / night-clear: 6,000 / night-rain: 600 / night-snow: 600 | 12,000장 |
| **Group 2: Real Balanced** | 각 조건별 2,000장씩 균등 분배 | 12,000장 |
| **Group 3: Imbalanced + Augmentation** | Group 1 기반 + Albumentations weather augmentation (p=0.5): RandomRain, RandomSnow | 12,000장 |
| **Group 4: Balanced + Augmentation** | 각 조건별 2,000장 (day-rain: 600 real + 1,400 aug / day-snow: 600 real + 1,400 aug / night-rain: 600 real + 1,400 aug / night-snow: 600 real + 1,400 aug) | 12,000장 |


## 데이터 전처리
+ 아래는 BDD100K 데이터셋을 YOLOv11n 모델 형태에 맞게 변환하는 과정입니다.
  
### BDD100K 데이터셋 다운로드
+ 공식 다운로드 링크
  + http://bdd-data.berkeley.edu/download.html
  <img width="600" src="https://github.com/user-attachments/assets/a101e81c-80c8-4474-9c47-fa2862b275ae" />

+ BDD-100K 페이지에서 **100K Images**, **Labels**를 다운로드합니다.

**주의사항**
10K Images에는라벨 누락 이미지가 많아 학습 오류가 발생합니다.
따라서 **100K Images + Labels 다운로드를 권장**합니다.

### YOLO 학습을 위한 Dataset 구조 만들기
<img width="550" src="https://github.com/user-attachments/assets/774dcd85-18a2-4b44-8e94-014c474b0d11" />

+ BDD100K는 JSON 파일 형태 기반이므로, YOLO 라벨(txt) 형식으로의 변환이 필요합니다.
+ YOLO 라벨 형식은 다음과 같습니다:
  ```
  class x_center y_center width height
  ```
<img width="500" src="https://github.com/user-attachments/assets/a3ac8eae-9235-410d-a504-20540278a70b" />


### 개발 환경 설정
+ 가상환경 생성
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
+ 의존성 설치
  ```bash
  pip install -r requirements.txt
  ```

### 데이터 필터링 (Subset 생성성)
+ config.yaml 기반으로 목적에 맞는 데이터만 필터링합니다.
  + config.yaml 파일을 수정
    ```bash
    python3 make_bdd_subset.py --config config.yaml
    ```
  + 출력 경로
    ```
    bdd_to_yolo/bdd100k_subset_filtered/
    ```
### BBD100K → YOLO 포맷 변환
```bash
cd bdd_to_yolo
python3 converter.py
```

### YOLO 라벨 시각화 (옵션)
+ convert 과정 확인
  ```bash
  python3 viz_yolo_labels.py
  ```
  + `yolo_viz/` 폴더가 생성됩니다.
  
<img width="600" src="https://github.com/user-attachments/assets/01a175e7-33d6-4978-909b-9f3d2825b1d9" />
<img width="600" src="https://github.com/user-attachments/assets/be31c1f9-a7cf-4cd7-8d7b-b147ca044204" />
    
### Data Augmentation (Group 4)
+ 변환이 완료된 YOLO 데이터셋을 읽어와 증강을 수행하며, 이미지와 라벨을 학습에 바로 사용할 수 있는 구조로 자동 저장합니다.
+ 설정 변경
  + make_aug_image_4.py 파일을 열어 다음 옵션을 수정하여 증강 모드를 설정할 수 있습니다.
    ```python
      # 옵션 예시
      aug_type = "rain"      # "rain", "snow", 또는 None (단순 복사)
      n_aug_per_image = 2.3  # 이미지당 생성할 증강 개수 (기본 2장 + 30% 확률로 1장 추가)
      ```
+ 증강 스크립트 실행
  ```bash
  python3 make_aug_image_4.py
  ```
+ 출력 경로
  + 증강된 데이터는 아래 경로에 저장되며, YOLO 학습에 즉시 사용 가능합니다.
    ```
    ./bdd100k_augmented/
        ├── images/
        └── labels/
    ```
    





# 3. Methodology

### 3.1 Model Selection

본 연구에서는 **YOLOv11-nano**을 기본 객체 검출 모델로 선택하였습니다.

**선택 이유:**
- Real-time 추론이 가능한 one-stage detector
- 자율주행 환경에서 요구되는 빠른 처리 속도
- 최신 아키텍처로 baseline 성능 우수
- Ultralytics 프레임워크를 통한 손쉬운 실험 재현성

**모델 구성:**
- Backbone: CSPDarknet 기반 feature extractor
- Neck: PANet (Path Aggregation Network)
- Head: Decoupled head for classification and regression
- Input size: 640×640 (원본 1280×720에서 리사이즈)
   
### 3.2 Data Augmentation Strategy

**RandomRain Parameter:**
```python
RandomRain(
    slant_range=[-15, 15],
    drop_length="60",
    drop_width=1,
    drop_color=[200, 200, 200],
    blur_value=7,
    brightness_coefficient=0.5,
    rain_type="heavy"
)
```
**RandomSnow Parameter:**
```python
RandomSnow(
brightness_coeff=4,
snow_point_range=[0.3, 0.7],
method="bleach"
)
```
<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/c0ea5d58-f042-477e-816f-ed29b8944652" width="400" />
      <br><b>day-rain<br/>RandomRain</b>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/d66b0559-b12b-4a10-a22a-4d12ffa4db20" width="400" />
      <br><b>night-rain<br/>RandomRain</b>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/a746a969-3d13-4336-9d5f-e78048454ee8" width="400" />
      <br><b>day-snow<br/>RandomSnow</b>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/b4ee22b8-72db-4c05-9490-461809a0afb6" width="400" />
      <br><b>night-snow<br/>RandomSnow</b>
    </td>
  <tr>
<table>

### 3.3 Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 60 |
| Batch size | 16 |
| Image size | 640×640 |

**학습 환경:**
- GPU: NVIDIA RTX 5060 (8GB)
- Framework: PyTorch 2.0 + Ultralytics

### 3.4 Train
+ 다음과 같이 실행합니다.
  + Group 1, Group 2, Group 4
    ```
    python3 yolo_noaug_train.py
    ```
  + Group 3
    ```
    python3 yolo_aug_train.py
    ```

# 4. Evaluation & Analysis

### Evaluation Metrics
- **Precision (P)**: 검출된 객체 중 정답 객체의 비율
- **Recall (R)**: 정답 객체 중 검출된 객체의 비율
- **mAP50**: IoU threshold 0.5에서의 mean Average Precision
- **mAP50-95**: IoU threshold 0.5~0.95에서의 평균 mAP

### Test Dataset
- 각 날씨/시간 조건별 200장씩 균등 분배
- 총 1,200장 (6개 조건 × 200장)
- 학습 데이터와 중복되지 않는 이미지로 구성

---

## 4.1 Experimental Results

### 4.1.1 Overall Performance Comparison (mAP50)

| Subset | Group 1 | Group 2 | Group 3 | Group 4 |
|--------|---------|---------|---------|---------|
| | Real Imbalanced | Real Balanced | Imbal. + Aug | Bal. + Aug |
| day_clear | 0.457 | **0.473** | 0.409 | 0.384 |
| day_rain | 0.427 | **0.466** | 0.405 | 0.438 |
| day_snow | 0.433 | **0.457** | 0.412 | 0.416 |
| night_clear | **0.372** | **0.372** | 0.359 | 0.336 |
| night_rain | **0.476** | 0.455 | 0.457 | 0.408 |
| night_snow | **0.393** | 0.378 | 0.376 | 0.353 |
| **Average** | 0.426 | **0.434** | 0.403 | 0.389 |

### 4.1.2 Detailed Metrics by Group

#### Group 1: Real Imbalanced

| Subset | P | R | mAP50 | mAP50-95 |
|--------|-------|-------|-------|----------|
| day_clear | 0.579 | 0.440 | 0.457 | 0.223 |
| day_rain | 0.672 | 0.382 | 0.427 | 0.247 |
| day_snow | 0.481 | 0.445 | 0.433 | 0.230 |
| night_clear | 0.456 | 0.369 | 0.372 | 0.203 |
| night_rain | 0.665 | 0.404 | 0.476 | 0.266 |
| night_snow | 0.656 | 0.347 | 0.393 | 0.223 |

#### Group 2: Real Balanced

| Subset | P | R | mAP50 | mAP50-95 |
|--------|-------|-------|-------|----------|
| day_clear | 0.620 | 0.419 | 0.473 | 0.233 |
| day_rain | 0.544 | 0.419 | 0.466 | 0.261 |
| day_snow | 0.730 | 0.392 | 0.457 | 0.245 |
| night_clear | 0.645 | 0.319 | 0.372 | 0.198 |
| night_rain | 0.644 | 0.405 | 0.455 | 0.246 |
| night_snow | 0.541 | 0.359 | 0.378 | 0.211 |

#### Group 3: Imbalanced + Augmentation

| Subset | P | R | mAP50 | mAP50-95 |
|--------|-------|-------|-------|----------|
| day_clear | 0.510 | 0.396 | 0.409 | 0.211 |
| day_rain | 0.672 | 0.360 | 0.405 | 0.226 |
| day_snow | 0.612 | 0.347 | 0.412 | 0.215 |
| night_clear | 0.497 | 0.318 | 0.359 | 0.195 |
| night_rain | 0.572 | 0.404 | 0.457 | 0.259 |
| night_snow | 0.568 | 0.326 | 0.376 | 0.200 |

#### Group 4: Balanced + Augmentation

| Subset | P | R | mAP50 | mAP50-95 |
|--------|-------|-------|-------|----------|
| day_clear | 0.523 | 0.374 | 0.384 | 0.205 |
| day_rain | 0.654 | 0.355 | 0.438 | 0.233 |
| day_snow | 0.591 | 0.368 | 0.416 | 0.206 |
| night_clear | 0.438 | 0.327 | 0.336 | 0.171 |
| night_rain | 0.484 | 0.440 | 0.408 | 0.218 |
| night_snow | 0.428 | 0.364 | 0.353 | 0.195 |

---

## 4.2 Analysis

### 4.2.1 Robustness 

Real Balanced (Group 2)가 평균 mAP50 0.434로 가장 높은 성능을 기록하였습니다. 이는 Real Imbalanced (Group 1)의 0.426 대비 **1.9% 향상**된 수치입니다. 하지만 이 수치는 유의미한 차이를 보이지 않으며, 학습한 모델이 domain shift에 robust함을 나타냅니다.

다음은 조건별 성능 향상 분석입니다.

**조건별 성능 향상 분석:**
- day_clear: +3.5% (0.457 → 0.473)
- day_rain: +9.1% (0.427 → 0.466)
- day_snow: +5.5% (0.433 → 0.457)
- night_clear: 0.0% (0.372 → 0.372)

adverse weather 조건(rain, snow)에서 balanced 데이터의 효과가 크게 나타났습니다. 이는 소수 클래스의 데이터를 충분히 확보하는 것이 해당 조건에서의 검출 성능 향상에 직접적인 영향을 미친다는 것을 시사합니다.

### 4.2.2 Synthetic Augmentation의 부정적 효과

예상과 달리, weather augmentation을 적용한 Group 3, 4가 augmentation을 적용하지 않은 그룹보다 **낮은 성능**을 보였습니다.

**성능 저하율:**
- Group 1 → Group 3: -5.4% (0.426 → 0.403)
- Group 2 → Group 4: -10.4% (0.434 → 0.389)

**가능한 원인 분석:**

1. **Domain Gap**: Albumentations의 RandomRain, RandomSnow가 생성하는 합성 날씨 효과가 BDD100K의 실제 날씨 패턴과 시각적으로 상이함

2. **Night 조건에서의 비현실성**: 야간 이미지에 합성 비/눈 효과 적용 시 빛 반사, 조명 효과 등이 비현실적으로 표현됨

3. **Augmentation 강도**: p=0.5의 적용 확률이 과도하여 모델이 실제 날씨 패턴 학습에 혼란을 겪음

4. **Feature Corruption**: 합성 효과가 원본 이미지의 중요한 특징(edge, texture)을 손상시킴

### 4.2.3 Night 조건의 특수성

모든 그룹에서 night 조건의 성능이 day 조건보다 낮게 나타났습니다.

**Day vs Night 평균 성능 비교 (mAP50):**

| Group | Day Average | Night Average | Gap |
|-------|-------------|---------------|-----|
| Group 1 | 0.439 | 0.414 | -5.7% |
| Group 2 | 0.465 | 0.402 | -13.5% |
| Group 3 | 0.409 | 0.397 | -2.9% |
| Group 4 | 0.413 | 0.366 | -11.4% |

Night 조건에서의 성능 저하는 저조도 환경에서의 객체 검출이 본질적으로 어려운 문제임을 보여줍니다. 특히 Real Balanced (Group 2)에서 day-night gap이 가장 크게 나타났는데, 이는 day 조건에서의 성능 향상이 night 조건으로 전이되지 않음을 의미합니다.

### 4.2.4 Anomaly: Night Rain의 높은 성능

전체 그룹에서 night_rain이 가장 높은 mAP50을 기록한 것은 예상 외의 결과입니다.

**가능한 해석:**
1. 야간 비 환경에서 젖은 노면의 빛 반사로 인해 객체 경계가 더 선명하게 드러남
2. 테스트셋 200장이 상대적으로 검출이 용이한 장면들로 구성되었을 가능성
3. 비로 인해 원거리 객체가 가려져 근거리 객체 위주로 평가됨

이 현상은 추가적인 정성적 분석(qualitative analysis)이 필요하며, 향후 연구에서 테스트셋 구성의 다양성 검증이 요구됩니다.

---

## 4.3 Key Findings Summary

| Finding | Evidence |
|---------|----------|
| Real balanced data가 가장 효과적 | Group 2 평균 mAP50 0.434 (최고) |
| Real data > Synthetic augmentation | Group 1,2 > Group 3,4 (전 조건) |
| Weather augmentation이 오히려 성능 저하 | Aug 적용 시 평균 5~10% 하락 |
| Balanced + Aug가 최악의 조합 | Group 4 평균 mAP50 0.389 (최저) |

---

# 5. Related Work

## 5.1 Object Detection Models

### YOLO Series
- **Redmon et al. (2016)**: "You Only Look Once" 
- **Ultralytics YOLOv11 (2024)**: 최신 YOLO 아키텍처로 속도와 정확도의 균형 제공
- 본 연구에서는 YOLOv11n을 사용하여 빠른 실험 반복과 합리적인 성능 확보

### 참고 문헌
- Ultralytics YOLO Documentation: https://docs.ultralytics.com/

## 5.2 Autonomous Driving Datasets

### BDD100K
- **Yu et al. (2020)**: "BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning"
- 100,000개의 주행 영상에서 추출한 다양한 환경 조건의 이미지 제공
- 날씨(clear, rainy, snowy, foggy), 시간대(day, night, dawn/dusk), 장면(city, highway, residential) 등 풍부한 메타데이터 포함


## 5.3 Data Augmentation for Object Detection

### Traditional Augmentation
- Geometric transformations (flip, rotation, scaling)
- Color jittering, brightness/contrast adjustment
- Mosaic augmentation (YOLOv4+)

### Weather-specific Augmentation
- **Albumentations Library**: RandomRain, RandomSnow 날씨 효과 합성
- **Domain Randomization**: 합성 데이터를 통한 모델 일반화

### 참고 라이브러리
- Albumentations: https://albumentations.ai/

## 5.4 Tools and Libraries Used

| Tool/Library | Purpose | Reference |
|--------------|---------|-----------|
| YOLOv11 | Object detection model | Ultralytics |
| BDD100K | Dataset | UC Berkeley |
| Albumentations | Data augmentation | albumentations.ai |
| PyTorch | Deep learning framework | pytorch.org |
| OpenCV | Image processing | opencv.org |
| Python 3.10 | Programming language | python.org |

---

# 6. Conclusion: Discussion

## 6.1 Summary of Findings

본 연구는 자율주행 객체 검출에서 날씨 조건별 도메인 불균형 문제를 해결하기 위한 데이터 밸런싱 전략을 비교 분석하였다. BDD100K 데이터셋을 활용하여 4가지 실험 그룹을 구성하고, 6개 날씨/시간 조건에서 YOLOv11n 모델의 검출 성능을 평가하였다.

**주요 발견:**

1. **Yolov11n의 Robustness**: 균등하게 분배된 실제 데이터(Group 2)가 평균 mAP50 0.434로 최고 성능을 달성하였다. 이는 불균형 데이터(Group 1) 대비 1.9% 향상된 수치이다. 하지만 이 수치는 유의미한 차이를 보이지 않으므로 학습 모델은 domain shift에 robust하다는 것을 알 수있다.

2. **Synthetic Augmentation의 한계**: Albumentations의 weather augmentation은 오히려 성능을 저하시켰다. 이는 합성 날씨 효과가 실제 날씨 패턴의 복잡성을 충분히 모사하지 못함을 시사한다.

3. **Real Data의 대체 불가성**: 어떤 augmentation 전략도 실제 균형 데이터의 성능에 도달하지 못했다. 이는 adverse weather 조건에서의 성능 향상을 위해서는 실제 데이터 수집이 필수적임을 의미한다.

## 6.2 Research Questions Revisited

### RQ1: 도메인 불균형이 객체 검출 성능에 얼마나 영향을 미치는가?

도메인 불균형은 특히 소수 클래스(rain, snow)의 성능에 영향을 미쳤다. Group 1(불균형)에서 Group 2(균형)로 전환 시, day_rain에서 9.1%, day_snow에서 5.5%의 성능 향상이 관찰되었다. 이는 도메인의 균형이 소수 조건에서의 검출 성능 향상에 직접적인 영향을 미친다는 것을 입증한다.
하지만 이 모델에서는 Group1과 Group2간의 mAP가 크게 유의미한 차이가 나지 않음을 알 수 있었다.

### RQ2: Real balanced data vs Synthetic augmentation의 성능 차이는?

Real balanced data가 synthetic augmentation보다 일관되게 우수한 성능을 보였다. Group 2(Real Balanced)는 Group 4(Balanced + Aug)보다 평균 11.6% 높은 mAP50을 달성하였다. 단순 합성 데이터는 실제 데이터와의 괴리가 있으므로 학습에 이용하기 힘들다는 것을 알 수 있다.

## 6.3 Implications

### 실용적 시사점

1. **데이터 수집 전략**: 자율주행 시스템 개발 시 악천후 조건의 실제 데이터 수집에 투자해야 함
2. **Augmentation 신중 적용**: 단순 합성 날씨 효과는 도움이 되지 않으며, 더 정교한 domain adaptation 기법이 필요함
3. **평가 다양화**: 전체 성능뿐 아니라 조건별 성능을 개별 평가하여 약점 파악 필요

### 학술적 기여

1. 환경 조건별 도메인 불균형 문제에 대한 체계적 실험 설계 제시
2. Real data와 synthetic augmentation의 정량적 비교 근거 제공
3. 단순 Weather augmentation의 한계점 실증적 확인

## 6.4 Limitations

1. **모델 다양성 부족**: YOLOv11n 단일 모델로만 실험하여 다른 아키텍처에서의 일반화 검증 필요
2. **Augmentation 기법 제한**: Albumentations의 기본 날씨 효과만 사용하였으며, GAN 기반 등 고급 기법 미적용
3. **테스트셋 크기**: 조건별 200장으로 통계적 변동성이 존재할 수 있음
4. **Night_rain 이상치**: 예상과 다른 결과에 대한 심층 분석 필요

## 6.5 Future Work

1. **고급 Domain Adaptation**: CycleGAN, UNIT 등을 활용한 더 현실적인 날씨 변환
2. **Multi-model Validation**: Faster R-CNN, DETR 등 다양한 검출 모델에서의 검증
3. **Loss Function 개선**: Focal Loss, Class-balanced Loss 등과의 조합 실험
4. **Larger Scale Experiment**: 더 많은 데이터와 다양한 테스트셋으로 통계적 신뢰성 확보
5. **Qualitative Analysis**: Night_rain 등 이상 결과에 대한 시각적 분석 수행

## 6.6 Final Remarks

본 연구는 자율주행 객체 검출에서 도메인 불균형 문제의 중요성을 확인하고, 이를 해결하기 위한 전략들의 효과를 실증적으로 비교하였다. 연구 결과, **실제 균형 데이터의 확보**가 합성 데이터를 통한 augmentation보다 효과적이며, 단순 weather augmentation은 오히려 성능 저하를 유발할 수 있음을 발견하였다. 이는 자율주행 시스템의 안전성 확보를 위해 다양한 환경 조건에서의 실제 데이터 수집이 필수적임을 시사한다.

## 6.7 Contributors

박지홍 : Albumentations parameter 분석, table 분석 후 보고서 작성

심승환 : 데이터 전처리, 데이터 학습 및 테스트, 결과 보고

최다예 : 코드 작성, Youtube recording, 보고서 검토

## 6.8 Youtube link

