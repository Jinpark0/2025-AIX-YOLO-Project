# AIX-deep-learning
---

### 다양한 환경 조건에서의 YOLO를 활용한 객체 인식 성능 향상 연구

#### 컴퓨터소프트웨어학부 2022075069 박지홍
#### 미래자동차공학과 2022020437 심승환
#### 데이터사이언스학과 9535620251 최다예

---
## Title: 
Real vs Synthetic Weather Augmentation: A Comparative Study for Robust Object Detection in Autonomous Driving

---

### 1. 제안 (Proposal)

### Motivation : Why are you doing this?

자율주행 시스템은 다양한 기상 조건(맑음, 비, 눈, 안개)과 시간대(주간, 야간)에서 안정적으로 작동해야 합니다. 하지만 실제로는 특정 조건에서만 학습된 객체 검출 모델이 다른 환경에서 성능이 크게 저하되는 domain shift 문제가 발생합니다.
이 문제를 해결하기 위한 두 가지 접근법이 있습니다.
1. 실제 다양한 기상 조건의 데이터를 수집하여 학습 - 비용과 시간이 많이 소요됨
2. Synthetic weather augmentation을 적용하여 학습 - 저비용이지만 실제 데이터와의 차이 존재

본 연구는 BDD100K 데이터셋을 활용하여 이 두 접근법을 정량적으로 비교함으로써, 제한된 예산과 시간 내에서 robust한 객체 검출 모델을 개발하기 위한 실질적인 가이드라인을 제시하고자 합니다.

### What do you want to see at the end?

1. 정량적 성능 비교
  + 3가지 학습 전략 (Pure baseline, Real diverse data, Synthetic augmentation)의 mAP 비교
  + 각 기상 조건별 테스트셋(daytime&clear, rain, snow, fog, night)에서의 성능 분석

2. Weather-specific insight
  + 어떤 날씨 조건에서 synthetic augmentation이 효과적인가?
 
---

### 2. 데이터셋 (Datasets)

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
+ Daytime
+ Night

## Data Split Strategy
Training Sets
  
