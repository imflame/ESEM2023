import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn import metrics
import math
import random
from sklearn.model_selection import GridSearchCV
data = pd.read_csv(r"E:\mysplconqueror\dataset_input\x264_AllNumeric.csv")
x = math.ceil(len(data)/len(data.columns))-1
if(x>5):
    x=5
result_array=np.zeros((x,4))
for m in range(x):#样本量从n到x*n
    explained_variance_score = 0
    mean_absolute_error = 0
    root_mean_squared_error = 0
    r2_score = 0
    for n in range(30):#每一个样本量跑30个experiment
        c = random.sample(range(0, len(data) - 1), (m+1) * len(data.columns))
        S = data.iloc[c]  # S为随机取样的x*N个样本
        x_Training_set = S.iloc[:, 0:(len(S.columns) - 1)]
        y_Training_set = S.iloc[:, len(S.columns) - 1]
        Testing_index=np.setdiff1d(np.array(range(len(data))),c)
        x_Test_set = data.iloc[Testing_index, 0:(len(S.columns) - 1)]
        y_Test_set = data.iloc[Testing_index, (len(S.columns) - 1)]
        x_array = np.array(x_Training_set)
        y_array = np.array(y_Training_set)
        x_test_array = np.array(x_Test_set)
        y_test_array = np.array(y_Test_set)
        turned_parameters=[{'kernel':['linear'],'C':[0.01,0.1,0.5,1,5,10,50,100,500,1000],'epsilon':[0.001,0.01,0.1,0.5,1]},
                           {'kernel':['rbf'],'gamma':[0.001,0.005,0.007,0.01,0.05,0.07,0.1,0.5,0.7,1],'C':[0.01,0.1,0.5,1,5,10,50,100,500,1000],'epsilon':[0.001,0.01,0.1,0.5,1]},
                           {'kernel':['poly'],'gamma':[0.001,0.005,0.007,0.01,0.05,0.07,0.1,0.5,0.7,1],'C':[0.01,0.1,0.5,1,5,10,50,100,500,1000],'epsilon':[0.001,0.01,0.1,0.5,1]}]
        svr=GridSearchCV(SVR(),turned_parameters,cv=5,scoring='neg_mean_absolute_error')
        #svr_lr=SVR(kernel='linear',C=1)
        svr.fit(x_array,y_array)
        print("best parameter:",svr.best_params_,"best score:",svr.best_score_)
        predict=svr.predict(x_test_array)
        explained_variance_score = explained_variance_score + metrics.explained_variance_score(y_test_array, predict)
        mean_absolute_error = mean_absolute_error + metrics.mean_absolute_error(y_test_array, predict)
        root_mean_squared_error = root_mean_squared_error + metrics.mean_squared_error(y_test_array, predict) ** 0.5
        r2_score = r2_score + metrics.r2_score(y_test_array, predict)
        print("sample:" + str(m + 1), "experiment:" + str(n + 1))
        print(metrics.explained_variance_score(y_test_array, predict))
        print(metrics.mean_absolute_error(y_test_array, predict))
        print(metrics.mean_squared_error(y_test_array, predict) ** 0.5)
        print(metrics.r2_score(y_test_array, predict))
    result_array[m, 0] = explained_variance_score / 30
    result_array[m, 1] = mean_absolute_error / 30
    result_array[m, 2] = root_mean_squared_error / 30
    result_array[m, 3] = r2_score / 30
print(result_array)
pd.DataFrame(result_array).to_csv("result_x264_SVR.csv")