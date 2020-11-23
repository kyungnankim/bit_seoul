#다중분류
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

#머신러닝 : 데이터 → 모델 → 컴파일 → 평가예측

#1.데이터
x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data = [0,0,0,1]

# x = datasets.data
# y = datasets.target

#####2. 모델
model = LinearSVC()

#####3. 컴파일, 훈련
model.fit(x_data, y_data)

#####4. 평가, 예측
y_predict = model.predict(x_data)
print(x_data, "의 예측결과", y_predict)

# acc1 = model.score(y_data, y_predict)
# print('acc_score : ', acc1)
acc2 = accuracy_score(y_data, y_predict)
print('acc_score : ', acc2)