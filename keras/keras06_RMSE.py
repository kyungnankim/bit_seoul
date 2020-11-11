import numpy as np

# 1.  데이터 
x_train = np.array([1,2,3,4,5,6,7,8,9,10]) 
y_train = np.array([1,2,3,4,5,6,7,8,9,10])  
x_test = np.array([11,12,13,14,15]) #데이터를 추가한게 아니라 평가용으로 테스트 한 것 뿐이다.
y_test = np.array([11,12,13,14,15]) #모의고사
x_pred = np.array([16,17,18]) #수능
#데이터 갯수는 문제가 아님
# y = ax + b 
# train-훈련시킬데이터 
# test-평가할데이터 
# pred-예측할데이터(y을 알고싶기 때문에 y값이 없음) 암묵적인 약속
# 2. 모델 구성 

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 

model = Sequential() # 순차적
model.add(Dense(1000, input_dim=1)) # 한 개 입력 
model.add(Dense(700))
model.add(Dense(40)) # 다음층에 5개 Jod가 생성 
model.add(Dense(1)) # 결과 Weight 
# Jod의 갯수 layer의 깊이 AI Developer가 결정 
# Hyper Parameter Tuning
# 1-3-5-3-1 구조 
# Dense = DNN 

# 3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam', metrics=['mae']) 
   # mse(Mean Square Error) = 손실함수는 정답에 대한 오류를 숫자로 나타내는 것
   # 예측값과 정답값의 차이값을 제곱하여, 그 값들을 전부 더하고, 개수로 나누어 평균을 낸 것
   # square라는 제곱의 영문명으로 보면 말 그대로 차이를 면적으로 나타낸 것
   # 제곱을 하기에 값이 높아져서?부풀어져 x, mae와 마찬가지로 방향성이 상실되는것은 마찬가지
   # 오답에 가까울수록 큰 값이 나옴. 반대로 정답에 가까울수록 작은 값이 나옴
   # Optimizer(최적화) = adam ,Metrics = 평가 지표 
#model.fit(x, y, epochs=100, batch_size=1) 
model.fit(x_train, y_train, epochs=1) 
   # epochs = 훈련 횟수 
   # batch_size = 몇 개 샘플로 가중치를 갱신할 것인지 지정. 32 or 64 개 샘플이 가장 최적화

# 4. 평가, 예측 
#평가
#loss, acc = model.evaluate(x, y, batch_size=1) 
#loss, acc = model.evaluate(x, y) 
loss = model.evaluate(x_test, y_test) 

print("loss : ", loss) 
#print("acc : ", acc)

#예측
#model.predict(x) #내가 훈련시킨 값이 나온다.
y_predict = model.predict(x_test) # 평가지표 나올 때 predict를 통과해서 원래 값이랑 비교 평가
print("결과물 : \n : ",y_predict)

# 실습, 결과물 오차 수집, 미세조절

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))