# 1.데이터
# 1.1 load_data
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils import all_estimators
import warnings

warnings.filterwarnings('ignore')

iris = pd.read_csv('./data/csv/boston_house_prices.csv', header=0, index_col=0, encoding='CP949', sep=',')
iris = iris[1:]
print(iris)
x = iris.iloc[:,:-1]
y = iris.iloc[:,-1:]

# 1.2 train_test_split
from sklearn.model_selection import train_test_split 
x_train,x_test, y_train,y_test = train_test_split(
    x,y, random_state=44, shuffle=True, test_size=0.2)


allAlgorithms = all_estimators(type_filter='regressor')

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률:', r2_score(y_test, y_pred))
    except:
        pass
'''
ARDRegression 의 정답률: 0.7413660842736123
AdaBoostRegressor 의 정답률: 0.8346412647550048
BaggingRegressor 의 정답률: 0.9058540582054116
BayesianRidge 의 정답률: 0.7397243134288032
CCA 의 정답률: 0.7145358120880195
DecisionTreeRegressor 의 정답률: 0.8085923139165242
DummyRegressor 의 정답률: -0.0007982049217318821
ElasticNet 의 정답률: 0.6952835513419808
ElasticNetCV 의 정답률: 0.6863712064842076
ExtraTreeRegressor 의 정답률: 0.7962903445827698
ExtraTreesRegressor 의 정답률: 0.8931378827596826
GammaRegressor 의 정답률: -0.0007982049217318821
GaussianProcessRegressor 의 정답률: -5.586473869478007
GradientBoostingRegressor 의 정답률: 0.8990123005378842
HistGradientBoostingRegressor 의 정답률: 0.8843141840898427
HuberRegressor 의 정답률: 0.765516461537028
KernelRidge 의 정답률: 0.7635967087108912
Lars 의 정답률: 0.7440140846099284
LarsCV 의 정답률: 0.7499770153318335
Lasso 의 정답률: 0.683233856987759
LassoCV 의 정답률: 0.7121285098074346
LassoLars 의 정답률: -0.0007982049217318821
LassoLarsCV 의 정답률: 0.7477692079348519
LassoLarsIC 의 정답률: 0.7447915470841701
LinearRegression 의 정답률: 0.7444253077310311
LinearSVR 의 정답률: 0.6484298289720596
MLPRegressor 의 정답률: 0.5219237539665682
MultiTaskElasticNet 의 정답률: 0.6952835513419808
MultiTaskElasticNetCV 의 정답률: 0.6863712064842078
MultiTaskLasso 의 정답률: 0.6832338569877592
MultiTaskLassoCV 의 정답률: 0.7121285098074348
NuSVR 의 정답률: 0.32492104048309933
OrthogonalMatchingPursuit 의 정답률: 0.5661769106723642
OrthogonalMatchingPursuitCV 의 정답률: 0.7377665753906504
PLSCanonical 의 정답률: -1.30051983252021
PLSRegression 의 정답률: 0.7600229995900804
PassiveAggressiveRegressor 의 정답률: 0.22578230429203228
PoissonRegressor 의 정답률: 0.7903831388798964
RANSACRegressor 의 정답률: 0.6965985887085426
RandomForestRegressor 의 정답률: 0.8821979125143224
Ridge 의 정답률: 0.746533704898842
RidgeCV 의 정답률: 0.7452747014482557
SGDRegressor 의 정답률: -1.2564290516649873e+26
SVR 의 정답률: 0.2867592174963418
TheilSenRegressor 의 정답률: 0.7666081569221769
TransformedTargetRegressor 의 정답률: 0.7444253077310311
TweedieRegressor 의 정답률: 0.6899090088434408
Lasso Ridge
'''