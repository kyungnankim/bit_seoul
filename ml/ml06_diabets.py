#다중분류
import numpy as np
from sklearn.datasets import load_iris, load_diabetes,load_boston,load_breast_cancer,load_wine
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
# Classifier :  분류, Regressor : 회귀모델 단 logit Regression 는 분류
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
#RandomForest : 상당히 중요하다!

#머신러닝 : 데이터 → 모델 → 컴파일 → 평가예측
#acc_score = 분류. r2 = 회귀
#1.데이터
x, y= load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66,shuffle=True, train_size=0.8)

scale=StandardScaler()
scale.fit(x_train)
x_train=scale.transform(x_train)
x_test=scale.transform(x_test)

# from tensorflow.keras.utils import to_categorical
# y_train=to_categorical(y_train)
# y_test=to_categorical(y_test)

#####2. 모델
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = KNeighborsRegressor()
model = RandomForestClassifier()
# model = RandomForestRegressor()

####3. 훈련
model.fit(x_train, y_train)

#####4. 평가, 예측
score = model.score(x_test, y_test)
print("model.score : ", score)

#accuracy_score를 넣어서 비교할 것
#회귀모델일 경우  r2_score로 코딩해서 score와 비교할 것
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("accuract_Score : ", acc)

print(y_test[:10],'의 예측결과','\n',y_predict)
# # 실습, 결과물 오차 수집, 미세조절

# from sklearn.metrics import mean_squared_error
# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))
# print("RMSE : ", RMSE(y_test, y_predict))

# from sklearn.metrics import r2_score
# r2 = r2_score(y_test, y_predict)
# print("R2 : ",r2)
''' 이게 무조건 좋은게 아니라 iris_dataset 에 대해서 좋은 모델이지 무조건 좋은 모델이 아니다.!
model = LinearSVC()
model.score :  0.011235955056179775
accuract_Score :  0.011235955056179775
 [ 91. 232. 281. 134.  97.  69.  53. 109. 230. 101. 302. 182. 152.  51.
 178. 104. 170. 200.  48. 242. 281.  88. 152. 232.  89. 199.  85. 155.
 113. 192. 296. 281. 220. 257. 152. 200.  88.  63. 168.  90. 230. 259.
  42.  72. 101. 283. 110. 217.  55. 243.  42. 206.  80. 274.  40. 175.
 241. 272. 183.  63. 257. 152.  89.  39. 109. 281. 141. 281. 233. 200.
  64. 137. 221. 151. 270.  94.  91.  39. 101.  91. 270.  69.  55. 144.
  63.  53. 127. 110.  72.]

model = SVC()
model.score :  0.0
accuract_Score :  0.0
[235. 150. 124.  78. 168. 253.  97. 102. 249. 142.] 의 예 
측결과
 [ 91. 220.  91.  90.  90.  91.  53.  90. 220. 200. 200. 200. 200.  90.
  53.  53.  63. 200. 170. 200. 281.  53. 200.  91.  90. 200. 200. 281.
 200. 220.  90. 281. 220. 200. 200. 200.  53.  90. 220. 200.  91. 220.
  53.  90. 200. 200.  53. 220.  72. 220. 200. 200. 200. 220. 200. 200.
 220. 200. 200. 200. 200. 200.  90. 200.  90. 281.  90. 281. 220. 200.
 200. 200. 200. 200. 200. 220.  91.  90. 200.  91. 200.  53. 200.  91.
 200.  53.  90. 281.  72.]

model = KNeighborsClassifier()
model.score :  0.0
accuract_Score :  0.0
[235. 150. 124.  78. 168. 253.  97. 102. 249. 142.] 의 예 
측결과
 [ 91. 129.  77.  64.  60.  63.  42.  53.  67.  74.  60.  
88.  90.  31.
 118.  65.  59.  68.  87.  78. 220.  42.  78.  91.  31.  53.  61.  66.
  72. 121. 143. 131.  78.  52.  93.  93.  42.  60. 129.  85.  83.  42.
  42.  60.  85.  65.  44. 139.  43. 230.  44.  83.  66. 274.  40.  95.
 197.  78.  93.  47. 167.  97.  49.  52. 109.  66.  60. 215. 100.  59.
  49.  58. 125.  53.  64.  84.  77.  97.  55. 122.  48.  47.  31.  91.
  49.  44.  31. 141.  72.]

model = KNeighborsRegressor()


model = RandomForestClassifier()
model.score :  0.011235955056179775
accuract_Score :  0.011235955056179775
[235. 150. 124.  78. 168. 253.  97. 102. 249. 142.] 의 예측결과
 [ 69. 208. 156. 170.  97. 138.  53. 214. 229. 158.  60. 121. 152. 125.
 118. 104.  87. 144.  87. 242. 281. 111.  99. 110.  63. 127.  77. 262.
  96. 248. 296. 281. 275. 280. 184.  53.  53.  48. 168. 121. 259. 131.
 138.  72.  85. 265.  88. 122.  77. 245.  44.  91. 102. 310.  74. 237.
 152. 265.  96.  80. 257. 150.  49.  52. 109. 217.  60. 275. 220.  68.
 179.  61.  97. 129. 252. 181.  77.  60.  94. 163. 270.  50.  78.  91.
  49. 200.  88. 281. 158.]

model = RandomForestRegressor()

'''