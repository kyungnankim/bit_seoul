import numpy as np

#1. 데이터 
x = np.array([1,2,3,4,5]) 
y = np.array([1,2,3,4,5]) 

# y = ax + b 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 

# 2. 모델 구성 
model = Sequential() # 순차적
model.add(Dense(3, input_dim=1)) # 한 개 입력 
model.add(Dense(5)) # 다음층에 5개 Jod가 생성 
model.add(Dense(3)) 
model.add(Dense(1)) # 결과 Weight 
# Jod의 갯수 layer의 깊이 AI Developer가 결정 
# Hyper Parameter Tuning
# 1-3-5-3-1 구조 
# Dense = DNN 

# 3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam', metrics=['acc']) 
   # mse(Mean Square Error) = 손실함수는 정답에 대한 오류를 숫자로 나타내는 것
   # 예측값과 정답값의 차이값을 제곱하여, 그 값들을 전부 더하고, 개수로 나누어 평균을 낸 것
   # square라는 제곱의 영문명으로 보면 말 그대로 차이를 면적으로 나타낸 것
   # 제곱을 하기에 값이 높아져서?부풀어져 x, mae와 마찬가지로 방향성이 상실되는것은 마찬가지
   # 오답에 가까울수록 큰 값이 나옴. 반대로 정답에 가까울수록 작은 값이 나옴
   # Optimizer(최적화) = adam ,Metrics = 평가 지표 
model.fit(x, y, epochs=10, batch_size=1) 
   # epochs = 훈련 횟수 
   # batch_size = 몇 개 샘플로 가중치를 갱신할 것인지 지정. 32 or 64 개 샘플이 가장 최적화

# 4. 평가, 예측 
loss, acc = model.evaluate(x, y, batch_size=1) 
print("loss : ", loss) 
print("acc : ", acc)