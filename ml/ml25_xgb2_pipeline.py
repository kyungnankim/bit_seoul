
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.pipeline import Pipeline, make_pipeline

x,y = load_boston(return_X_y=True)

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

pipe = Pipeline([("scaler", MaxAbsScaler()),('nini', XGBRegressor())])

model = RandomizedSearchCV(pipe, params, cv=5, n_jobs=n_jobs, verbose=2)
#3. 훈련
model.fit(x_train, y_train) 

#4. 평가, 예측
print("최적의 매개변수 : ", model.best_estimator_)
y_predict=model.predict(x_test)
print('최종정답률 : ', r2_score(y_test, y_predict))
print('최적의 매개변수 : ', model.best_estimator_) #(150, 4) (150,) acc :  1.0
print('최적의 매개변수 : ', model.best_params_) #(150, 4) (150,) acc :  1.0
