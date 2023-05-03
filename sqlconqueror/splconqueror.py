import torch
from torch import nn
from torch import optim
import numpy as np
import math
import pandas as pd
import random
from sklearn import metrics


#2023.3.13
#定义数据,从csv中读取features和labels
#采用和DECART相同的取样方式，n为数据集的特征个数
#取数据时要随机取用n,2n.3n,4n,5n等多种样本大小，其中训练集：测试集=7：3，进行反复的实验，剩余的所有样本作为测试集使用
#需要考虑的是，去到几n的问题,即抽样停止准则
#这边我们暂时规定，保持先截止到5n，看看误差再说
#数据集的名称有如下：Apache,BDBC,BDBJ,Dune,hipacc,hsmgp,javagc,LLVM,sac,SQL,x264
data = pd.read_csv(r"E:\mysplconqueror\dataset_input\Dune_AllNumeric.csv")
x = math.ceil(len(data)/len(data.columns))-1
if x>=20:
    x=20
#result_df=pd.DataFrame(np.arange(20).reshape(5,4),columns=["explained_variance_score","mean_absolute_error","root_mean_squared_error","r2_score"])
result_array=np.zeros((x,4))
device=torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
#验证有无无穷数据或者缺失值记录

for m in range(x):#样本量从n到x*n
    explained_variance_score = 0
    mean_absolute_error = 0
    root_mean_squared_error = 0
    r2_score = 0
    for n in range(30):#每一个样本量跑30个experiment
        c = random.sample(range(0, len(data) - 1), (m + 1) * len(data.columns))
        S = data.iloc[c]  # S为随机取样的x*N个样本
        x_Training_set = S.iloc[:, 0:(len(S.columns) - 1)]
        y_Training_set = S.iloc[:, len(S.columns) - 1]
        Testing_index = np.setdiff1d(np.array(range(len(data))), c)
        x_Test_set = data.iloc[Testing_index, 0:(len(S.columns) - 1)]
        y_Test_set = data.iloc[Testing_index, (len(S.columns) - 1)]
        x_array = np.array(x_Training_set)
        y_array = np.array(y_Training_set)
        # x_tensor=torch.from_numpy(x_array)
        # y_tensor=torch.from_numpy(y_array)
        x_test_array = np.array(x_Test_set)
        y_test_array = np.array(y_Test_set)
        # x_test_tensor=torch.from_numpy(x_test_array)
        train_x_data, train_y_data = torch.Tensor(x_array), torch.Tensor(y_array)
        test_x_data = torch.Tensor(x_test_array)
        test_y_data = torch.Tensor(y_test_array)
        #print(x_array.isnull().any())
        print(np.isnan(x_array).any())
        print(np.isfinite(x_array).all())
        print(np.isinf(x_array).all())
        class LinerModel(nn.Module):
            def __init__(self):
                super(LinerModel, self).__init__()
                self.Linear = nn.Linear(len(x_Training_set.columns), 1)

            def forward(self, x_tensor):
                out = self.Linear(x_tensor)
                return out
        train_x_data,train_y_data=train_x_data.to(device),train_y_data.to(device)
        test_x_data,test_y_data=test_x_data.to(device),test_y_data.to(device)
        model = LinerModel().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=1e-3)
        for i in range(40000):
            out = model(train_x_data)
            out = out.squeeze(-1)
            loss = criterion(train_y_data, out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 10000 == 0:
                print('Epoch[{}/{}],loss:{:.6f}'.format(i, 40000, loss.data))
        model.eval()
        predict = model(test_x_data)
        predict=predict.cpu().detach().numpy()
        predict = predict.squeeze(-1)
        #predict = predict.detach().numpy()
        explained_variance_score = explained_variance_score + metrics.explained_variance_score(y_test_array, predict)
        mean_absolute_error = mean_absolute_error + metrics.mean_absolute_error(y_test_array, predict)
        root_mean_squared_error = root_mean_squared_error + metrics.mean_squared_error(y_test_array, predict)**0.5
        r2_score = r2_score+metrics.r2_score(y_test_array, predict)
        print("sample:"+str(m+1),"experiment:"+str(n+1))
        print(metrics.explained_variance_score(y_test_array, predict))
        print(metrics.mean_absolute_error(y_test_array, predict))
        print(metrics.mean_squared_error(y_test_array, predict)**0.5)
        print(metrics.r2_score(y_test_array, predict))
    result_array[m,0] = explained_variance_score / 30
    result_array[m,1] = mean_absolute_error / 30
    result_array[m,2] = root_mean_squared_error / 30
    result_array[m,3] = r2_score / 30
print(result_array)
pd.DataFrame(result_array).to_csv("result_Dune_sqlconqueror.csv")
