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
### Group 1 : Real Imbalanced (12,000장)

+ day-clear: 3,600

+ day-rain : 600

+ day-snow : 600

+ night-clear : 6,000

+ night-rain : 600

+ night-snow : 600

### Group 2 : Real Balanced (12,000장)

+ 각 조건 별 2,000장씩

### Group 3 : Imbalanced + Augmentation (12,000장)

Base data : Group 1과 동일
+ Albumentations weather augmentation (p=0.6)
  - RandomRain
  - RandomSnow

### Group 4 : Balanced + Augmentation (12,000장)

+ 각 조건 별 2,000장씩
  - day-rain : 600 real + 1400 augmentation
  - day-snow : 600 real + 1400 augmentation
  - night-rain : 600 real + 1400 augmentation
  - night-snow : 600 real + 1400 augmentation
  

---

# 3. 방법론 (Methodology)

### 모델 구조
  + YOLOv11 Nano
  + 선택 이유 :
      + 경량 모델로 빠른 실험 가능
   
### Data Augmentation

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


