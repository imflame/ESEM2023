from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

#读入不同模型的结果：
#splconqueror
spl_beforePCA_pd=pd.read_csv(r"F:\桌面\Wilcoxon_signed_rank_test\results\sqlconqueror\noPCA\result_x264_sqlconqueror.csv")
spl_afterPCA_pd=pd.read_csv(r"F:\桌面\Wilcoxon_signed_rank_test\results\sqlconqueror\PCA\result_x264_sqlconqueror_PCA.csv")
spl_beforePCA_array=np.array(spl_beforePCA_pd)
spl_afterPCA_array=np.array(spl_afterPCA_pd)
spl_beforePCA_explained_variance_score=spl_beforePCA_array[:,1]
spl_beforePCA_mean_absolute_error=spl_beforePCA_array[:,2]
spl_beforePCA_root_mean_squared_error=spl_beforePCA_array[:,3]
spl_beforePCA_r2_score=spl_beforePCA_array[:,4]
k1=len(spl_beforePCA_explained_variance_score)
spl_afterPCA_explained_variance_score=spl_afterPCA_array[0:k1,1]
spl_afterPCA_mean_absolute_error=spl_afterPCA_array[0:k1,2]
spl_afterPCA_root_mean_squared_error=spl_afterPCA_array[0:k1,3]
spl_afterPCA_r2_score=spl_afterPCA_array[0:k1,4]
spl_x=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
# spl_beforePCA_explained_variance_score=np.delete(spl_beforePCA_explained_variance_score,[0])
# spl_beforePCA_mean_absolute_error=np.delete(spl_beforePCA_mean_absolute_error,[0])
# spl_beforePCA_root_mean_squared_error=np.delete(spl_beforePCA_root_mean_squared_error,[0])
# spl_beforePCA_r2_score=np.delete(spl_beforePCA_r2_score,[0])
# spl_afterPCA_explained_variance_score=np.delete(spl_afterPCA_explained_variance_score,[0])
# spl_afterPCA_mean_absolute_error=np.delete(spl_afterPCA_mean_absolute_error,[0])
# spl_afterPCA_root_mean_squared_error=np.delete(spl_afterPCA_root_mean_squared_error,[0])
# spl_afterPCA_r2_score=np.delete(spl_afterPCA_r2_score,[0])
#DECART
DECART_beforePCA_explained_variance_score=[0]*20
DECART_beforePCA_mean_absolute_error=[0]*20
DECART_beforePCA_root_mean_squared_error=[0]*20
DECART_beforePCA_r2_score=[0]*20
DECART_afterPCA_explained_variance_score=[0]*20
DECART_afterPCA_mean_absolute_error=[0]*20
DECART_afterPCA_root_mean_squared_error=[0]*20
DECART_afterPCA_r2_score=[0]*20
data_bfPCA=pd.read_csv(r"F:\桌面\毕设相关\DECART\decart_test\data\results_noPCA\CART_hipacc_Details_crossvalidation_gridsearch.csv")
data_afPCA=pd.read_csv(r"F:\桌面\毕设相关\DECART\decart_test\data\results\CART_hipacc_Details_crossvalidation_gridsearch.csv")
for k in range(20):#每一个循环取30个样本，取平均得出结果，下标为(k-1)*30到k+30
    DECART_beforePCA_explained_variance_score[k]=(DECART_beforePCA_explained_variance_score[k]+data_bfPCA.iloc[k*30:(k+1)*30-1,11].sum())/30
    DECART_beforePCA_mean_absolute_error[k]=(DECART_beforePCA_mean_absolute_error[k]+data_bfPCA.iloc[k*30:(k+1)*30-1,12].sum())/30
    DECART_beforePCA_root_mean_squared_error[k] = (DECART_beforePCA_root_mean_squared_error[k] + data_bfPCA.iloc[k*30:(k+1)*30-1,13].sum())/30
    DECART_beforePCA_r2_score[k] = (DECART_beforePCA_r2_score[k] + data_bfPCA.iloc[k*30:(k+1)*30-1,14].sum())/30

    DECART_afterPCA_explained_variance_score[k] = (DECART_afterPCA_explained_variance_score[k] + data_afPCA.iloc[k * 30:(k + 1) * 30 - 1,11].sum())/30
    DECART_afterPCA_mean_absolute_error[k] = (DECART_afterPCA_mean_absolute_error[k] + data_afPCA.iloc[k * 30:(k + 1) * 30 - 1, 12].sum())/30
    DECART_afterPCA_root_mean_squared_error[k] = (DECART_afterPCA_root_mean_squared_error[k] + data_afPCA.iloc[k * 30:(k + 1) * 30 - 1,13].sum())/30
    DECART_afterPCA_r2_score[k] = (DECART_afterPCA_r2_score[k] + data_afPCA.iloc[k * 30:(k + 1) * 30 - 1, 14].sum())/30
DECART_x=[2,3,4,5,6,7,8,9]
# DECART_beforePCA_explained_variance_score=np.delete(DECART_beforePCA_explained_variance_score,[0])
# DECART_beforePCA_mean_absolute_error=np.delete(DECART_beforePCA_mean_absolute_error,[0])
# DECART_beforePCA_root_mean_squared_error=np.delete(DECART_beforePCA_root_mean_squared_error,[0])
# DECART_beforePCA_r2_score=np.delete(DECART_beforePCA_r2_score,[0])
# DECART_afterPCA_explained_variance_score=np.delete(DECART_afterPCA_explained_variance_score,[0])
# DECART_afterPCA_mean_absolute_error=np.delete(DECART_afterPCA_mean_absolute_error,[0])
# DECART_afterPCA_root_mean_squared_error=np.delete(DECART_afterPCA_root_mean_squared_error,[0])
# DECART_afterPCA_r2_score=np.delete(DECART_afterPCA_r2_score,[0])
#DeepPerf
DeepPerf_beforePCA_pd=pd.read_csv(r"F:\桌面\Wilcoxon_signed_rank_test\results\DeepPerf\noPCA\result_hipacc_DeepPerf.csv")
DeepPerf_afterPCA_pd=pd.read_csv(r"F:\桌面\Wilcoxon_signed_rank_test\results\DeepPerf\PCA\result_hipacc_DeepPerf_PCA.csv")
DeepPerf_beforePCA_array=np.array(DeepPerf_beforePCA_pd)
DeepPerf_afterPCA_array=np.array(DeepPerf_afterPCA_pd)
DeepPerf_beforePCA_explained_variance_score=DeepPerf_beforePCA_array[:,2]
DeepPerf_beforePCA_mean_absolute_error=DeepPerf_beforePCA_array[:,3]
DeepPerf_beforePCA_root_mean_squared_error=DeepPerf_beforePCA_array[:,4]
DeepPerf_beforePCA_r2_score=DeepPerf_beforePCA_array[:,5]
k2=len(DeepPerf_beforePCA_explained_variance_score)
DeepPerf_afterPCA_explained_variance_score=DeepPerf_afterPCA_array[0:k2,2]
DeepPerf_afterPCA_mean_absolute_error=DeepPerf_afterPCA_array[0:k2,3]
DeepPerf_afterPCA_root_mean_squared_error=DeepPerf_afterPCA_array[0:k2,4]
DeepPerf_afterPCA_r2_score=DeepPerf_afterPCA_array[0:k2,5]
DeepPerf_x=[2,3,4,5]
# DeepPerf_beforePCA_explained_variance_score=np.delete(DeepPerf_beforePCA_explained_variance_score,[0])
# DeepPerf_beforePCA_mean_absolute_error=np.delete(DeepPerf_beforePCA_mean_absolute_error,[0])
# DeepPerf_beforePCA_root_mean_squared_error=np.delete(DeepPerf_beforePCA_root_mean_squared_error,[0])
# DeepPerf_beforePCA_r2_score=np.delete(DeepPerf_beforePCA_r2_score,[0])
# DeepPerf_afterPCA_explained_variance_score=np.delete(DeepPerf_afterPCA_explained_variance_score,[0])
# DeepPerf_afterPCA_mean_absolute_error=np.delete(DeepPerf_afterPCA_mean_absolute_error,[0])
# DeepPerf_afterPCA_root_mean_squared_error=np.delete(DeepPerf_afterPCA_root_mean_squared_error,[0])
# DeepPerf_afterPCA_r2_score=np.delete(DeepPerf_afterPCA_r2_score,[0])
#SVR
SVR_beforePCA_pd=pd.read_csv(r"F:\桌面\Wilcoxon_signed_rank_test\results\SVR\noPCA\result_x264_SVR.csv")
SVR_afterPCA_pd=pd.read_csv(r"F:\桌面\Wilcoxon_signed_rank_test\results\SVR\PCA\result_x264_SVR_PCA.csv")
SVR_beforePCA_array=np.array(SVR_beforePCA_pd)
SVR_afterPCA_array=np.array(SVR_afterPCA_pd)
SVR_beforePCA_explained_variance_score=SVR_beforePCA_array[:,1]
SVR_beforePCA_mean_absolute_error=SVR_beforePCA_array[:,2]
SVR_beforePCA_root_mean_squared_error=SVR_beforePCA_array[:,3]
SVR_beforePCA_r2_score=SVR_beforePCA_array[:,4]
k3=len(SVR_beforePCA_explained_variance_score)
SVR_afterPCA_explained_variance_score=spl_afterPCA_array[0:k3,1]
SVR_afterPCA_mean_absolute_error=SVR_afterPCA_array[0:k3,2]
SVR_afterPCA_root_mean_squared_error=SVR_afterPCA_array[0:k3,3]
SVR_afterPCA_r2_score=SVR_afterPCA_array[0:k3,4]
SVR_x=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
# SVR_beforePCA_explained_variance_score=np.delete(SVR_beforePCA_explained_variance_score,[0])
# SVR_beforePCA_mean_absolute_error=np.delete(SVR_beforePCA_mean_absolute_error,[0])
# SVR_beforePCA_root_mean_squared_error=np.delete(SVR_beforePCA_root_mean_squared_error,[0])
# SVR_beforePCA_r2_score=np.delete(SVR_beforePCA_r2_score,[0])
# SVR_afterPCA_explained_variance_score=np.delete(SVR_afterPCA_explained_variance_score,[0])
# SVR_afterPCA_mean_absolute_error=np.delete(SVR_afterPCA_mean_absolute_error,[0])
# SVR_afterPCA_root_mean_squared_error=np.delete(SVR_afterPCA_root_mean_squared_error,[0])
# SVR_afterPCA_r2_score=np.delete(SVR_afterPCA_r2_score,[0])
#Perf-AL
Perf_AL_beforePCA_explained_variance_score=np.zeros(20)
Perf_AL_beforePCA_mean_absolute_error=np.zeros(20)
Perf_AL_beforePCA_root_mean_squared_error=np.zeros(20)
Perf_AL_beforePCA_r2_score=np.zeros(20)
Perf_AL_afterPCA_explained_variance_score=np.zeros(20)
Perf_AL_afterPCA_mean_absolute_error=np.zeros(20)
Perf_AL_afterPCA_root_mean_squared_error=np.zeros(20)
Perf_AL_afterPCA_r2_score=np.zeros(20)
for j in range(9):
    num_bf=30
    num_af=30
    for i in range(30):
        Perf_AL_beforePCA_pd = pd.read_csv(r"F:\桌面\GANPerf\results\hipacc\x="+str(j+1)+r"\record" + str(i) + ".csv")
        Perf_AL_afterPCA_pd = pd.read_csv(r"F:\桌面\GANPerf\results\hipacc_PCA\x="+str(j+1)+r"\record" + str(i) + ".csv")
        maxid_bf = Perf_AL_beforePCA_pd[['r2_score']].idxmax()
        maxid_af = Perf_AL_afterPCA_pd[['r2_score']].idxmax()
        #beforePCA_pd=np.array(beforePCA_pd)
        #afterPCA_pd = np.array(afterPCA_pd)
        if(Perf_AL_beforePCA_pd.iloc[maxid_bf[0], 12]<0):
            num_bf=num_bf-1
        else:
            Perf_AL_beforePCA_explained_variance_score[j] = Perf_AL_beforePCA_explained_variance_score[j] + Perf_AL_beforePCA_pd.iloc[
                maxid_bf, 9]
            Perf_AL_beforePCA_mean_absolute_error[j] = Perf_AL_beforePCA_mean_absolute_error[j] + Perf_AL_beforePCA_pd.iloc[
                maxid_bf, 10]
            Perf_AL_beforePCA_root_mean_squared_error[j] = Perf_AL_beforePCA_root_mean_squared_error[j] + Perf_AL_beforePCA_pd.iloc[
                maxid_bf, 11]
            Perf_AL_beforePCA_r2_score[j] = Perf_AL_beforePCA_r2_score[j] + Perf_AL_beforePCA_pd.iloc[
                maxid_bf, 12]


        if (Perf_AL_afterPCA_pd.iloc[maxid_af[0], 12] < 0):
            num_af = num_af - 1
        else:
            Perf_AL_afterPCA_explained_variance_score[j] = Perf_AL_afterPCA_explained_variance_score[j] + Perf_AL_afterPCA_pd.iloc[
                maxid_af, 9]
            Perf_AL_afterPCA_mean_absolute_error[j] = Perf_AL_afterPCA_mean_absolute_error[j] + Perf_AL_afterPCA_pd.iloc[
                maxid_af, 10]
            Perf_AL_afterPCA_root_mean_squared_error[j] = Perf_AL_afterPCA_root_mean_squared_error[j] + Perf_AL_afterPCA_pd.iloc[
                maxid_af, 11]
            Perf_AL_afterPCA_r2_score[j] = Perf_AL_afterPCA_r2_score[j] + Perf_AL_afterPCA_pd.iloc[
                maxid_af, 12]
    if (num_af != 0):
        Perf_AL_afterPCA_explained_variance_score[j] = Perf_AL_afterPCA_explained_variance_score[j] / num_af
        Perf_AL_afterPCA_mean_absolute_error[j] = Perf_AL_afterPCA_mean_absolute_error[j] / num_af
        Perf_AL_afterPCA_root_mean_squared_error[j] = Perf_AL_afterPCA_root_mean_squared_error[j] / num_af
        Perf_AL_afterPCA_r2_score[j] = Perf_AL_afterPCA_r2_score[j] / num_af
    else:
        Perf_AL_afterPCA_explained_variance_score[j]=0
        Perf_AL_afterPCA_mean_absolute_error[j]=0
        Perf_AL_afterPCA_root_mean_squared_error[j]=0
        Perf_AL_afterPCA_r2_score[j]=0
    if (num_bf != 0):
        Perf_AL_beforePCA_explained_variance_score[j] = Perf_AL_beforePCA_explained_variance_score[j] / num_bf
        Perf_AL_beforePCA_mean_absolute_error[j] = Perf_AL_beforePCA_mean_absolute_error[j] / num_bf
        Perf_AL_beforePCA_root_mean_squared_error[j] = Perf_AL_beforePCA_root_mean_squared_error[j] / num_bf
        Perf_AL_beforePCA_r2_score[j] = Perf_AL_beforePCA_r2_score[j] / num_bf
    else:
        Perf_AL_beforePCA_explained_variance_score[j] = 0
        Perf_AL_beforePCA_mean_absolute_error[j] = 0
        Perf_AL_beforePCA_root_mean_squared_error[j] = 0
        Perf_AL_beforePCA_r2_score[j] = 0
Perf_AL_x=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
#Perf_AL_beforePCA_explained_variance_score=np.delete(Perf_AL_beforePCA_explained_variance_score,[0])
#Perf_AL_beforePCA_mean_absolute_error=np.delete(Perf_AL_beforePCA_mean_absolute_error,[0])
#Perf_AL_beforePCA_root_mean_squared_error=np.delete(Perf_AL_beforePCA_root_mean_squared_error,[0])
#Perf_AL_beforePCA_r2_score=np.delete(Perf_AL_beforePCA_r2_score,[0])
#Perf_AL_afterPCA_explained_variance_score=np.delete(Perf_AL_afterPCA_explained_variance_score,[0])
#Perf_AL_afterPCA_mean_absolute_error=np.delete(Perf_AL_afterPCA_mean_absolute_error,[0])
#Perf_AL_afterPCA_root_mean_squared_error=np.delete(Perf_AL_afterPCA_root_mean_squared_error,[0])
#Perf_AL_afterPCA_r2_score=np.delete(Perf_AL_afterPCA_r2_score,[0])
#进行差值箱线图的绘制

# plt.figure(1)
# labels=['SPLConqueror\n(8.20E-05)','SVR\n(0.0625)','DECART\n(0.004)','DeepPerf\n(0.0625)','Perf_AL\n(1.91E-06)']
# num=len(DeepPerf_afterPCA_root_mean_squared_error)
# plt.ylim(-10,50)
# plt.boxplot([spl_afterPCA_root_mean_squared_error[0:num]-spl_beforePCA_root_mean_squared_error[0:num],
#             SVR_afterPCA_root_mean_squared_error[0:num]-SVR_beforePCA_root_mean_squared_error[0:num],
#             np.array(DECART_afterPCA_root_mean_squared_error[0:num])-np.array(DECART_beforePCA_root_mean_squared_error[0:num]),
#             DeepPerf_afterPCA_root_mean_squared_error[0:num]-DeepPerf_beforePCA_root_mean_squared_error[0:num],
#             Perf_AL_afterPCA_root_mean_squared_error[0:num]-Perf_AL_beforePCA_root_mean_squared_error[0:num]],
#             meanline=True,showmeans=True,labels=labels,
#             medianprops={'color':'red','linewidth':'1.5'},
#             meanprops={'color':'blue','linewidth':'1.5'},
#             flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 10})
# plt.ylabel("RMSE",fontsize=10)
# plt.legend()
# plt.suptitle('RMSE of hsmgp')
# plt.savefig("data_analysis/RMSE_of_hsmgp.png",dpi=300)

plt.figure(1)
labels=['DECART\n(0.570)','DeepPerf\n(0.01562)','Perf_AL\n(1.91E-06)']
num=len(DeepPerf_afterPCA_root_mean_squared_error)
plt.ylim(-100,750)
DECART_afterPCA_root_mean_squared_error=np.array(DECART_afterPCA_root_mean_squared_error)
DECART_beforePCA_root_mean_squared_error=np.array(DECART_beforePCA_root_mean_squared_error)
plt.boxplot([DECART_afterPCA_root_mean_squared_error.take([0,5,8])-DECART_beforePCA_root_mean_squared_error.take([0,5,8]),
            DeepPerf_afterPCA_root_mean_squared_error[0:num]-DeepPerf_beforePCA_root_mean_squared_error[0:num],
            Perf_AL_afterPCA_root_mean_squared_error.take([0,5,10,15,19])-Perf_AL_beforePCA_root_mean_squared_error.take([0,5,10,15,19])],
            meanline=True,showmeans=True,labels=labels,
            medianprops={'color':'red','linewidth':'1.5'},
            meanprops={'color':'blue','linewidth':'1.5'},
            flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 10})
plt.ylabel("RMSE",fontsize=10)
plt.legend()
plt.suptitle('RMSE of hsmgp')
plt.savefig("data_analysis/RMSE_of_hsmgp.png",dpi=300)

#plt.figure(2)
#labels=['SPLConqueror','SVR','DECART','DeepPerf','Perf_AL']
#num=len(DeepPerf_afterPCA_r2_score)
# plt.ylim(-0.4,0.6)
# plt.boxplot([spl_afterPCA_r2_score[0:num]-spl_beforePCA_r2_score[0:num],
#             SVR_afterPCA_r2_score[0:num]-SVR_beforePCA_r2_score[0:num],
#             np.array(DECART_afterPCA_r2_score[0:num])-np.array(DECART_beforePCA_r2_score[0:num]),
#             DeepPerf_afterPCA_r2_score[0:num]-DeepPerf_beforePCA_r2_score[0:num],
#             Perf_AL_afterPCA_r2_score[0:num]-Perf_AL_beforePCA_r2_score[0:num]],
#             meanline=True,showmeans=True,labels=labels,
#             medianprops={'color':'red','linewidth':'1.5'},
#             meanprops={'color':'blue','linewidth':'1.5'},
#             flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 10})
# plt.ylabel("R2",fontsize=10)
# plt.legend()
# plt.suptitle('R2 of SQL')
# plt.savefig("data_analysis/R2_of_SQL.png",dpi=300)
#进行折线图的作图
# plt.figure(1)
# #x=np.arange(k)
# plt.plot(spl_x,spl_beforePCA_explained_variance_score,label='spl_bfPCA')
# plt.plot(spl_x,spl_afterPCA_explained_variance_score,label='spl_afPCA')
# plt.plot(SVR_x,SVR_beforePCA_explained_variance_score,label='SVR_bfPCA')
# plt.plot(SVR_x,SVR_afterPCA_explained_variance_score,label='SVR_afPCA')
# plt.plot(DECART_x,DECART_beforePCA_explained_variance_score,label='DECART_bfPCA')
# plt.plot(DECART_x,DECART_afterPCA_explained_variance_score,label='DECART_afPCA')
# plt.plot(DeepPerf_x,DeepPerf_beforePCA_explained_variance_score,label='DeepPerf_bfPCA')
# plt.plot(DeepPerf_x,DeepPerf_afterPCA_explained_variance_score,label='DeepPerf_afPCA')
# plt.plot(Perf_AL_x,Perf_AL_beforePCA_explained_variance_score,label='Perf_AL_bfPCA')
# plt.plot(Perf_AL_x,Perf_AL_afterPCA_explained_variance_score,label='Perf_AL_afPCA')
# plt.ylabel("explained_variance_score",fontsize=10)
# plt.legend()
# plt.suptitle('BDBC of explained')
# plt.savefig("data_analysis/BDBC_of_expliained.png",dpi=300)
#
#
# plt.figure(2)
# #x=np.arange(k)
# plt.plot(spl_x,spl_beforePCA_mean_absolute_error,label='spl_bfPCA')
# plt.plot(spl_x,spl_afterPCA_mean_absolute_error,label='spl_afPCA')
# plt.plot(SVR_x,SVR_beforePCA_mean_absolute_error,label='SVR_bfPCA')
# plt.plot(SVR_x,SVR_afterPCA_mean_absolute_error,label='SVR_afPCA')
# plt.plot(DECART_x,DECART_beforePCA_mean_absolute_error,label='DECART_bfPCA')
# plt.plot(DECART_x,DECART_afterPCA_mean_absolute_error,label='DECART_afPCA')
# plt.plot(DeepPerf_x,DeepPerf_beforePCA_mean_absolute_error,label='DeepPerf_bfPCA')
# plt.plot(DeepPerf_x,DeepPerf_afterPCA_mean_absolute_error,label='DeepPerf_afPCA')
# plt.plot(Perf_AL_x,Perf_AL_beforePCA_mean_absolute_error,label='Perf_AL_bfPCA')
# plt.plot(Perf_AL_x,Perf_AL_afterPCA_mean_absolute_error,label='Perf_AL_afPCA')
# plt.ylabel("mean_absolute_error",fontsize=10,labelpad=-200)
# plt.legend()
# plt.suptitle('BDBC of MAE')
# plt.savefig("data_analysis/BDBC_of_MAE.png",dpi=300)
#
# plt.figure(3)
# #x=np.arange(k)
# plt.plot(spl_x,spl_beforePCA_root_mean_squared_error,label='spl_bfPCA')
# plt.plot(spl_x,spl_afterPCA_root_mean_squared_error,label='spl_afPCA')
# plt.plot(SVR_x,SVR_beforePCA_root_mean_squared_error,label='SVR_bfPCA')
# plt.plot(SVR_x,SVR_afterPCA_root_mean_squared_error,label='SVR_afPCA')
# plt.plot(DECART_x,DECART_beforePCA_root_mean_squared_error,label='DECART_bfPCA')
# plt.plot(DECART_x,DECART_afterPCA_root_mean_squared_error,label='DECART_afPCA')
# plt.plot(DeepPerf_x,DeepPerf_beforePCA_root_mean_squared_error,label='DeepPerf_bfPCA')
# plt.plot(DeepPerf_x,DeepPerf_afterPCA_root_mean_squared_error,label='DeepPerf_afPCA')
# plt.plot(Perf_AL_x,Perf_AL_beforePCA_root_mean_squared_error,label='Perf_AL_bfPCA')
# plt.plot(Perf_AL_x,Perf_AL_afterPCA_root_mean_squared_error,label='Perf_AL_afPCA')
# plt.ylabel("root_mean_squared_error",fontsize=10)
# plt.legend()
# plt.suptitle('BDBC of RMSE')
# plt.savefig("data_analysis/BDBC_of_RMSE.png",dpi=300)
#
# plt.figure(4)
# #x=np.arange(k)
# plt.plot(spl_x,spl_beforePCA_r2_score,label='spl_bfPCA')
# plt.plot(spl_x,spl_afterPCA_r2_score,label='spl_afPCA')
# plt.plot(SVR_x,SVR_beforePCA_r2_score,label='SVR_bfPCA')
# plt.plot(SVR_x,SVR_afterPCA_r2_score,label='SVR_afPCA')
# plt.plot(DECART_x,DECART_beforePCA_r2_score,label='DECART_bfPCA')
# plt.plot(DECART_x,DECART_afterPCA_r2_score,label='DECART_afPCA')
# plt.plot(DeepPerf_x,DeepPerf_beforePCA_r2_score,label='DeepPerf_bfPCA')
# plt.plot(DeepPerf_x,DeepPerf_afterPCA_r2_score,label='DeepPerf_afPCA')
# plt.plot(Perf_AL_x,Perf_AL_beforePCA_r2_score,label='Perf_AL_bfPCA')
# plt.plot(Perf_AL_x,Perf_AL_afterPCA_r2_score,label='Perf_AL_afPCA')
# plt.ylabel("r2_score",fontsize=10,labelpad=-208)
# plt.legend()
# plt.suptitle('BDBC of r2')
# plt.savefig("data_analysis/BDBC_of_r2.png",dpi=300)

