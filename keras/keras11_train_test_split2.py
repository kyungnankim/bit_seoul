import numpy as np

#1. 데이터
x = np.array(range(1,21))
y = np.array(range(31,51))

#2. 
from sklearn.model_selection import train_test_split

x1_train, x_test, y1_train, y_test, = train_test_split(x, y, train_size=0.7) 
x2_train, x_val, y2_train, y_val = train_test_split(x1_train, y1_train, train_size=0.7)
#3. 컴파일, 훈련
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

print("x1_train : ", x1_train)
print("x2_train : ", x2_train)
print("x_test : ", x_test)
print("x_val : ", x_val)

print(x_test.shape)


#4. 평가, 예측
# loss = model.evaluate(x_test, y_test)
# print("loss: ", loss)

# y_pred = model.predict(x_test)
# print("result : \n", y_pred)

#RMSE
# from sklearn.metrics import mean_squared_error
# def RMSE(y_test, y_pred):
#     return np.sqrt(mean_squared_error(y_test, y_pred))
# print("RMSE : ", RMSE(y_test, y_pred))

# #R2
# from sklearn.metrics import r2_score
# r2 = r2_score(y_test, y_pred)
# print("R2: ", r2)