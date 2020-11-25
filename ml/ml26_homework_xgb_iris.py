
from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler

x,y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=44)

params = [
    {"nini__n_estimators":[100,200,300], "nini__learning_rate":[0.1,0.3,0.01,0.001], 
    "nini__max_depth":[4,5,6]},
    {"nini__n_estimators":[90,100,110], "nini__learning_rate":[0.1,0.01,0.001], 
    "nini__max_depth":[4,5,6], "nini__colsample_bytree":[0.6,0.9,1]},
    {"nini__n_estimators":[9,110], "nini__learning_rate":[0.1,0.001,0.5], 
    "nini__max_depth":[4,5,6], "nini__colsample_bytree":[0.6,0.9,1], 
    "nini__colsample_bylevel":[0.6,0.7,0.9]}
]
n_jobs = -1

# pipe = Pipeline([("scaler", StandardScaler()),('nini', XGBClassifier())])
# pipe = Pipeline([("scaler", MinMaxScaler()),('nini', XGBClassifier())])
# pipe = Pipeline([("scaler", RobustScaler()),('nini', XGBClassifier())])
pipe = Pipeline([("scaler", MaxAbsScaler()),('nini', XGBClassifier())])

model = RandomizedSearchCV(pipe, params, cv=5, n_jobs=n_jobs, verbose=2)

model.fit(x_train, y_train)

print("최적의 매개변수 : ", model.best_estimator_)
print("최적 하이퍼 파라미터 : ", model.best_params_)
print("최고 정확도 : {0:.4f}".format(model.best_score_))

acc = model.score(x_test,y_test)
print("acc : ", acc)
