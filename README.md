# AIX-deep-learning
---

### 다양한 환경 조건에서의 YOLO를 활용한 객체 인식 성능 향상 연구

#### 컴퓨터소프트웨어학부 2022075069 박지홍
#### 미래자동차공학과 2022020437 심승환
#### 데이터사이언스학과 9535620251 최다예

---
## Title: 
Addressing class imbalance in weather conditions for autonomous driving object detection : A comparative study of data balancing strategies

---

# 1. 제안 (Proposal)

### Motivation : Why are you doing this?

자율주행 시스템의 객체 검출 성능은 학습 데이터의 분포에 크게 영향을 받습니다. 특히 BDD100K와 같은 대규모 자율주행 데이터셋은 날씨와 시간대에 따른 심각한 클래스 불균형 문제를 가지고 있습니다.

핵심 문제:
+ Clear 조건의 데이터가 전체의 약 75%를 차지
+ Adverse weather(비,눈) 조건은 각각 10% 미만

연구 질문:
1. 데이터 불균형이 날씨 조건별 검출 성능에 얼마나 영향을 미치는가?
2. Real balanced data vs Synthetic augmentation의 성능 차이는?

### What do you want to see at the end?

+ 정량적 분석
  + 6개 날씨 조건별 mAP 비교 (Day/Night x Clear/Rain/Snow)
  + 불균형 데이터의 성능 저하 정도 측정
  + Augmentation의 효과 정량화

 
---

# 2. 데이터셋 (Datasets)

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
| **Group 3: Imbalanced + Augmentation** | Group 1 기반 + Albumentations weather augmentation (p=0.6): RandomRain, RandomSnow | 12,000장 |
| **Group 4: Balanced + Augmentation** | 각 조건별 2,000장 (day-rain: 600 real + 1,400 aug / day-snow: 600 real + 1,400 aug / night-rain: 600 real + 1,400 aug / night-snow: 600 real + 1,400 aug) | 12,000장 |


# 3. 방법론 (Methodology)

### 3.1 Model Selection

본 연구에서는 **YOLOv11**을 기본 객체 검출 모델로 선택하였습니다.

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
  </


### 3.3 Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 100 |
| Batch size | 16 |
| Optimizer | AdamW |
| Learning rate | 0.001 (with cosine annealing) |
| Weight decay | 0.0005 |
| Image size | 640×640 |
| Warmup epochs | 3 |

**학습 환경:**
- GPU: NVIDIA RTX 4090 (24GB)
- Framework: PyTorch 2.0 + Ultralytics


