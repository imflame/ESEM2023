#此程序从csv文件中读入数据然后使用pca处理后存入另一个csv文件中
#步骤：1，读入数据 2，计算应当保留的信息为99%时的剩余特征数量n 3，使用pca压缩到n维 4，存入一个csv文件中
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
#filename=["Apache","BDBC","BDBJ","Dune","hipacc","hsmgp","javagc","LLVM","sac","SQL","x264"] ,"hipacc","hsmgp","javagc","sac"
filename=["Dune","hipacc","hsmgp","javagc","sac"]
for i in range(5):
    filepath="E:/mysplconqueror/dataset_input/"+filename[i]+"_AllNumeric.csv"
    data = pd.read_csv(filepath)
    perf=data.iloc[:, (len(data.columns) - 1)]
    data = data.iloc[:, 0:(len(data.columns) - 1)]
    #对数据类型的软件需要使用Scaled Label Encoding（缩放标签编码），即减去最小值，除以最大值最小值之差，如果最大最小值相同则不做处理
    data_array = np.array(data)
    for j in range(data_array.shape[1]):
        datamin=np.min(data_array[:,j-1])
        datamax = np.max(data_array[:,j-1])
        if(datamax!=datamin):
            data_array[:, j - 1] = (data_array[:, j - 1] - datamin) / (datamax-datamin)
    print(data_array)
    pca = PCA(n_components=len(data.columns))
    newdata = pca.fit_transform(data_array)
    olddata = pca.inverse_transform(newdata)
    datainfo_percent = pca.explained_variance_ratio_
    info = sorted(datainfo_percent)
    k = 0
    sum_least_info = 0
    while (sum_least_info <= 0.05):
        sum_least_info = sum_least_info + info[k]
        k = k + 1
    final_pca = PCA(n_components=len(data.columns) - k + 1)
    finaldata = final_pca.fit_transform(data_array)
    df_data = pd.DataFrame(finaldata)
    df_data["PREF"]=perf
    df_data.to_csv(filename[i]+"_AllNumeric_PCA.csv",index=None)
    #perf.to_csv(filename[i]+"_AllNumeric_PCA.csv",mode='a',header="PERF",index=None)
print("PCA数据处理结束")

