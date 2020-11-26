#분류 select 모델 11
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')
#RandomForest : 상당히 중요하다!
#머신러닝 : 데이터 → 모델 → 컴파일 → 평가예측
#acc_score = 분류. r2 = 회귀

#1.데이터
iris = pd.read_csv('./data/csv/iris_ys.csv', header=0)
x = iris.iloc[:, :4]
y = iris.iloc[:, -1]

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=66)


kfold=KFold(n_splits=5, shuffle=True)
allAlgorithms=all_estimators(type_filter='regressor')
for (name, algorithm) in allAlgorithms :
    try :
        model=algorithm()
        scores=cross_val_score(model, x_train, y_train, cv=kfold)
        print(name, '의 정답률 : ', scores)
    except :
        pass
'''
(150, 4) (150,)
scores [1.         1.         0.875      1.         0.95833333]
'''

# ####3. 훈련
# model.fit(x_train, y_train)

# # 4. 평가, 예측
# from sklearn.metrics import accuracy_score, r2_score
# score=model.score(x_test, y_test)

# # accuracy_score를 넣어서 비교할 것
# # 회귀모델일 경우 r2_score와 비교할 것

# y_predict=model.predict(x_test)
# acc=accuracy_score(y_test, y_predict)
# # r2=r2_score(y_test, y_predict)

# print('score :', score)
# print('acc :', acc)
# # print('r2 : ', r2)

# print(y_test[:10], '의 예측 결과 ', y_predict[:10])
