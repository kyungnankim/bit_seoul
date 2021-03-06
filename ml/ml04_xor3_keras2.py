#다중분류
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#머신러닝 : 데이터 → 모델 → 컴파일 → 평가예측

#1.데이터
x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data = [0,1,1,0]

#####2. 모델
model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(7, activation='relu'))
# model.add(Dense(3, activation='sigmoid'))
# model.add(Dense(5, activation='sigmoid'))
model.add(Dense(1,activation='sigmoid'))

#####3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['acc'])
model.fit(x_data, y_data, batch_size=1,epochs=100)

#####4. 평가, 예측
y_predict = model.predict(x_data)
print(x_data, "의 예측결과", y_predict)

acc1 = model.evaluate(x_data, y_data)
print('model.evaluate : ', acc1)
# acc2 = accuracy_score(y_data, y_predict)
# print('acc_score : ', acc2)