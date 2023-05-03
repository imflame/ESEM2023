import random

import numpy as np
import pandas as pd
import math
#此程序目的是将数据处理为有label和feature的文件来处理，由于原程序没有随机取样的方法，需要自己写
#所以步骤如下：1，读入总数据2，随机取样3，分解为label和feature文件
#取样规则：每次取x*n数据，67%用于训练，33%用于验证，总的数据集剩余的所有则用于测试
def makedata(x):
    data = pd.read_csv(r"F:\桌面\GANPerf\x264_AllNumeric.csv")
    #x = 20
    n = len(data.columns)
    c = random.sample(range(0, len(data) - 1), x * n)
    data_notest = data.iloc[c]
    train_sample_num = math.floor(x * n * 0.67)
    vaild_sample_num = x * n - train_sample_num
    data_train = data_notest.iloc[0:train_sample_num]
    data_valid = data_notest.iloc[train_sample_num:len(data_notest)]
    test_index = np.setdiff1d(np.array(range(len(data))), c)
    data_test = data.iloc[test_index]
    data_train_features = data_train.iloc[:, 0:n - 1]
    data_train_label = data_train.iloc[:, n - 1]
    data_valid_features = data_valid.iloc[:, 0:n - 1]
    data_valid_label = data_valid.iloc[:, n - 1]
    data_test_features = data_test.iloc[:, 0:n - 1]
    data_test_label = data_test.iloc[:, n - 1]
    data_train_features.to_csv(r"F:\桌面\GANPerf\Database_input\features\train_features.csv", index=None)
    data_valid_features.to_csv(r"F:\桌面\GANPerf\Database_input\features\valid_features.csv", index=None)
    data_test_features.to_csv(r"F:\桌面\GANPerf\Database_input\features\test_features.csv", index=None)
    data_train_label.to_csv(r"F:\桌面\GANPerf\Database_input\label\train_label.csv", index=None, header=['label'])
    data_valid_label.to_csv(r"F:\桌面\GANPerf\Database_input\label\valid_label.csv", index=None, header=['label'])
    data_test_label.to_csv(r"F:\桌面\GANPerf\Database_input\label\test_label.csv", index=None, header=['label'])
    print("data_divide success!")
