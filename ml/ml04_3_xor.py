#다중분류
# 인공지능 겨울
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score

#머신러닝 : 데이터 → 모델 → 컴파일 → 평가예측

#1.데이터
x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data = [0,1,1,0]

# x = datasets.data
# y = datasets.target

#####2. 모델
# model = LinearSVC()
model = SVC()

#####3. 컴파일, 훈련
model.fit(x_data, y_data)
#####4. 평가, 예측
y_predict = model.predict(x_data)
print(x_data, "의 예측결과", y_predict)

acc1 = model.score(x_data, y_data)
print('acc_score : ', acc1)
acc2 = accuracy_score(y_data, y_predict)
print('acc_score : ', acc2)