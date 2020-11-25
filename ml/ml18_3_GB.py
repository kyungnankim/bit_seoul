#Tree구조의 모델들 성능이 다른 모델에 비해 좋음. keras보다 잘 나올 수도 있음.
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor, GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer=load_breast_cancer()
x=cancer.data
y=cancer.target
x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8)

# model=DecisionTreeClassifier(max_depth=4)
# model=RandomForestClassifier(max_depth=4)
model=GradientBoostingClassifier(max_depth=4)
model.fit(x_train, y_train)

acc=model.score(x_test, y_test)

print("acc : ", acc)
print(model.feature_importances_) #컬럼 중요도
'''
acc :  0.9473684210526315
[3.34549921e-05 3.14872538e-02 1.15913367e-03 6.09901400e-03
 4.63116546e-04 1.91316530e-04 2.95922144e-04 7.95507149e-02
 1.98654350e-03 4.63285250e-04 2.46274359e-03 1.08208504e-05
 2.06760816e-05 1.09219006e-02 2.34194071e-03 1.53631267e-04
 5.94414738e-04 3.87038392e-03 1.25446918e-04 5.91403413e-04
 5.90423937e-01 3.00103102e-02 3.50360513e-02 2.95460854e-02
 2.03461045e-03 9.53747125e-04 5.14508367e-02 1.12773416e-01
 3.94140711e-04 4.55374819e-03]
'''

import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_cancer(model) :
    n_features=cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importances_cancer(model)
plt.show()