import pandas as pd
from sklearn.datasets import load_iris, load_diabetes,load_boston,load_breast_cancer,load_wine
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm import LinearSVC, SVC
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, MaxPooling1D, Dropout, Conv1D, Flatten
# Classifier :  분류, Regressor : 회귀모델 단 logit Regression 는 분류
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils import all_estimators
import warnings

warnings.filterwarnings('ignore')

iris = pd.read_csv('./data/csv/iris_ys.csv',header=0, index_col=0)

x =iris.iloc[:,0:4]
y =iris.iloc[:, 4]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66,shuffle=True, train_size=0.2)
allAlgorithms = all_estimators(type_filter='classifier')

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률:', accuracy_score(y_test, y_pred))
    except:
        pass

# algorithms = all_estimators(type_filter='classifier')

# for (name, algorithms) in allAlgorithms:
#   model = algorithms()

#   model.fit(x_train,y_train)
#   y_pred = model.predict(x_test)
#   print(name,'의 정답률 : ', accuracy_score(y_test,y_pred))

#   import sklearn
#   print(sklearn.__version__)