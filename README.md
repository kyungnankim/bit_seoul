# bit_seoul

2020-11-9
note
집가서 설치하기 설치 후 스크린샷 보내기

 cuda 10.1 243_426 비주얼 스튜디오 삭제
conda env list 
pip install tensorflow-gpu==2.3.0
python

import tensorflow as tf
pip install keras==2.4.3
python
kaggle
1. kaggle
2. dacon
3. git hub →bit_soul /잔디밭

레이어 요오드 모델링 모델
y = ax+ b

최적의 weight 가중치
y = w x + b
데이터정제
loss x를 1로 맞추기
파이썬 파라미터 튜닝

2020-11-10
note
y = wx + b
mse
batch size 기본값 / 적절값 
신경망
https://needjarvis.tistory.com/230
default
분류 : 
작업관리자 gpu cuda
mse  mae acc
99 만들기
잘 만들어진 모델의 가중치만 빼와서 다른데이터정리
Tuning
rmse 평균제곱근 오차
https://m.blog.naver.com/PostView.nhn?blogId=owl6615&logNo=221537580561&proxyReferer=https:%2F%2Fwww.google.com%2F
https://ebbnflow.tistory.com/120?category=738689
R2 제곱
회계지표 x
머신스스로 검증 데이터
검증
test predict validation
train_test_split

데이터 자르기, shuffle
column
스칼라1 벡터1 행렬2 텐서3
스칼라 : 하나의 숫자 의미
벡터 : 숫자(스칼라)의 배열
행렬 : 2차원의 배열
스칼라 소문자의 이탈릭체 n
벡터 소문자의 강조 이탈릭체 x
행렬 대문자의 강조 이탈릭체 X
keras12_mlp.py 까지
2020-11-11
note
행무시 열우선

특성 피처 컬럼 열 =동일얘기
슬라이싱 많이 해보기
슬라이싱 :  [시작번호:끝번호] 를 사용하여 문자열을 뽑는 것\

주가 환율 채권을 넣어 주가만 나오면 되는 결과물
keras16_ensemble4 까지 
2020-11-12
early_stopping 중요
CNN
데이터 전처리하는날
데이터 전처리란?
실제 업무나 활동에서 발생하는 데이터는 분석이나 머신러닝에 적합하지 않은 경우가 많습니다. 의미 없는 값이 포함되거나 실수로 오타가 발생하는 등 수많은 변수로 인해 데이터의 품질이 떨어지기 때문입니다. ‘Garbage in, Garbage out’, 준비된 데이터가 왜곡되어 있다면 분석결과 역시 왜곡되는 것이 당연하겠지요. 이를 방지하기 위하여 데이터의 품질을 올리는 일련의 과정들을 데이터 전처리라고 합니다. 많은 데이터 분석가들이 가장 많은 시간을 투자하고 고민을 거듭하는 데이터 전처리에 대하여 알아보도록 합시다.
데이터 전처리 기법은 어떤 것이 있을까?
– 데이터 정제
불완전한 데이터는 채우고, 모순된 데이터는 수정하여 이치에 맞게 다듬는 작업들을 말합니다.
– 데이터 통합
다양하게 나뉘어져 있는 여러 데이터베이스, 파일들을 분석하기 좋게 합쳐서 다듬는 작업을 말합니다.
– 데이터 축소
가지고 있는 모든 데이터를 무작정 통합하기만 하는 것 역시 능사는 아닙니다. 오히려 과도한 분량의 데이터는 분석이나 머신러닝에 효율을 떨어트리기 때문입니다. 이를 위해 일부 데이터만 샘플링하거나 분석 대상 데이터의 차원을 줄이는 작업을 데이터 축소라고 합니다.
– 데이터 변환
간단하게는 평균값을 구하여 사용하는 것부터 로그를 씌우는 등 데이터를 정규화 또는 집단화하는 작업들을 말합니다.
note
Artificial Intelligence(인공지능 :AI) : 인간의 지능이 갖고 있는 기능을 갖춘 컴퓨터 시스템
인간의 지능을 기계 등에 인공적으로 구현한 것
Machine Learing
ANN
DNN
CNN
RNN : 

LSTM :  input output shape reshape
특별한 종류의 RNN
장기 의존성의 문제를 해결할 수 있다.
단일 뉴럴 네트워크 레이어를 가지는 대신, 4개의 상호작용 가능한 방식 구조 지님
무조건 
행 열 몇개씩 자르는지 이게 lstm의 shape


GRU : 
참고사이트 : 
https://m.blog.naver.com/PostView.nhn?blogId=shakey7&logNo=221406221628&proxyReferer=https:%2F%2Fwww.google.com%2F
https://ebbnflow.tistory.com/119
https://ratsgo.github.io/natural%20language%20processing/2017/03/09/rnnlstm/
https://3months.tistory.com/424
https://www.w3resource.com/numpy/manipulation/reshape.php
데이터 전처리하는날
2020-11-13
homework
리스트(List), 튜플(Tuple), 딕셔너리(dictionary), 집합(Set)
dictionary(key,value)
list()
history
early_stopping
주말과제 : 
---
VS code
AN

---
note


차원
input_shape
Dense
2
(1, ) 스칼라
LSTM
3
(?, ?)
CNN
4
(? ,?, ?)
연산이 잘되고 터지는 사태를 막기위해 데이터 전처리 진행
데이터 전처리
min-man

x
y
train
fit
x
val
 transform
x
test
 transform
x
predict
 transform
값처리x
Robust Scaler
Standard Scaler
MAXABS  Scaler
2020-11-16
homework
리스트(List), 튜플(Tuple), 딕셔너리(dictionary), 집합(Set)

---
VS code

---
note
keras35_cnn1.py

다중분류

이중분류 mnist

DNN - Deep의 개념 
-> Node, Layer (신경망)
-> y = wx + b
-> w는 모든 layer에 존재한다
-> 최적의 weight를 구하기 위해 최소의 loss(0의 근사치로)
-> optimizer 'adam', 평균 85%

summary를 통해 parameter를 계산
-> 3x4 = 16, 4x3 = 15
-> bias의 존재 (1)

함수형 모델
-> 모델끼리 만들어서 연결을해서 묶어준다(앙상블)
-> concatenate (대문자, 소문자 문법 다름 주의!)

LSTM
-> 순차 데이터(ex - 시간순서가 있는 데이터)
-> TimeSeries(시계열)
-> 행, 열, 몇개씩 자르는지
-> input_shape(행무시! = 열, 자르는 갯수)

CNN
-> 차원 4차원
-> 이미지의 갯수 x 가로 x 세로 x 픽셀
-> 행무시! = 가로 x 세로 x 픽셀(3차원)

LSTM을 두개로 여러개로 묶을 경우
-> 데이터에 따라 다르다
-> 좋다 - 정답x,  안좋다 - 정답x
-> 돌려보고 확인해야 한다
-> return_sequence(차원을 넘겨줄때 사용)

early_stopping
-> monitor = 'loss'
-> patience = 100
-> mode = 'min'

history
-> hist = fit
-> 시각화
-> matplotlib
-> tensorboard

model save, load

데이터 전처리
-> scaler(minmax, standard, robust, maxabs)
-> 85% Data 전처리
-> standard_scaler - 가운데지점이 0

ex) 101, 102, 103, 104, 105, 107, 299 라는 데이터로 결과를 추출한다고 가정
-> 299는 이상한 데이터이다
-> 이상치라고 한다
-> 이상치 제거
-> 깔끔한 데이터셋을 구성할 수 있다
-> but 299라는 데이터가 존재하지만 이상치라고 제거했을 경우에는 데이터 조작이 되기 때문에 주의해야 한다

외우기!
      ㅣ      X      ㅣ    Y    ㅣ
---------------------------------
train ㅣfit/transformㅣ    X    ㅣ
test  ㅣ transform   ㅣ    X    ㅣ
val   ㅣ transform   ㅣ    X    ㅣ
pred  ㅣ transform   ㅣ    X    ㅣ

ai developer 지향, data scientist 지양

---------------------------------------------------------------------

CNN - Convolution(복합적인)

Conv2D

Maxpooling2D
-> 가장 특성치가 높은것만 남기는 것

Flatten
-> 현재까지 내려왔던 것을 일자로 펴주는 기능 - 이차원으로 변경
-> 4차원으로 구성된 layer를 다음 Dense층과 연결시키기 위해 사용

Mnist
-> 간단한 컴퓨터 비전 데이터셋, 이는 손으로 쓰여진 숫자 1~9까지의 이미지로 구성되어 있습니다.

One-Hot Encoding
-> 원-핫 인코딩은 단어 집합의 크기를 벡터의 차원으로 하고, 표현하고 싶은 단어의 인덱스에 1의 값을 부여하고,
   다른 인덱스에는 0을 부여하는 단어의 벡터 표현 방식
-> keras, sklearn 2가지가 있다
-> keras - from tensorflow.keras.utils import to_categorycal
-> sklearn - from sklearn.preprocessing import OneHotEncoder

다중분류
OneHotEncoding, model - activation='softmax', compile - loss='categorical_crossentropy'

CNN, LSTM activation default
-> CNN은 activation default = 'relu'이다 
-> LSTM activation default ='tanh'이다 - 탄젠트
2020-11-17 
homework
다중분류 이진분류 희귀모델 복습
다중분류cnn
회귀모델 dnn
 내일 이진분류 잡기술
---
VS code
AN

---
note
오전 다중분류 categorical 
mnist cifar10 fashion cifar100
--------------------------------------------------------
오후 이진분류 회귀모델
droptout 과적합 잡기 1번
val
과적합
분류는 딱 결정되는게 있어야 하므로
회기이다.  load_boston
2020-11-18
homework
다중분류 이진분류 희귀모델 복습
다중분류cnn
회귀모델 dnn
 내일 이진분류 잡기술
---
VS code
AN

---
note
keras45_iris_1_cnn.py
keras46_cancer_1_dnn.py
keras47_gpu_test.py
keras48_ModelChekPoint.py
오전 다중분류 categorical 

keras45_iris_1_cnn.py
