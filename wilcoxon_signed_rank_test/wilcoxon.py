from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

#数据分析专用程序，分析结果：对PCA作用的分析（1）wilcoxon评分，判定PCA前后是否有显著区别（2）每个指标的箱型图比较，用来分析整个训练何时趋于稳定（3）曲线图直观判定优化程度
#对不同方法之间的比较：放一起做箱型图比较模型训练完整的速度，做曲线图直观得出不同方法的优化程度
beforePCA_pd=pd.read_csv(r"F:\桌面\Wilcoxon_signed_rank_test\results\SVR\noPCA\result_x264_SVR.csv")
afterPCA_pd=pd.read_csv(r"F:\桌面\Wilcoxon_signed_rank_test\results\SVR\PCA\result_x264_SVR_PCA.csv")
beforePCA_array=np.array(beforePCA_pd)
afterPCA_array=np.array(afterPCA_pd)
beforePCA_explained_variance_score=beforePCA_array[:,1]
beforePCA_mean_absolute_error=beforePCA_array[:,2]
beforePCA_root_mean_squared_error=beforePCA_array[:,3]
beforePCA_r2_score=beforePCA_array[:,4]
k=len(beforePCA_explained_variance_score)
afterPCA_explained_variance_score=afterPCA_array[0:k,1]
afterPCA_mean_absolute_error=afterPCA_array[0:k,2]
afterPCA_root_mean_squared_error=afterPCA_array[0:k,3]
afterPCA_r2_score=afterPCA_array[0:k,4]
explained_variance_score_statistic,explained_variance_score_pvalue=stats.wilcoxon(beforePCA_explained_variance_score,afterPCA_explained_variance_score[0:5])
mean_absolute_error_statistic,mean_absolute_error_pvalue=stats.wilcoxon(beforePCA_mean_absolute_error,afterPCA_mean_absolute_error[0:5])
root_mean_squared_error_statistic,root_mean_squared_error_pvalue=stats.wilcoxon(beforePCA_root_mean_squared_error,afterPCA_root_mean_squared_error[0:5])
r2_score_statistic,r2_score_pvalue=stats.wilcoxon(beforePCA_r2_score,afterPCA_r2_score[0:5])
#一般来讲，p值小于0.05才能说明有显著不同
list1=[explained_variance_score_statistic,mean_absolute_error_statistic,root_mean_squared_error_statistic,r2_score_statistic]
list2=[explained_variance_score_pvalue,mean_absolute_error_pvalue,root_mean_squared_error_pvalue,r2_score_pvalue]
result=np.array([list1,list2])
df=pd.DataFrame(result)
df.columns=["explained_variance_score","mean_absolute_error","root_mean_squared_error","r2_score"]
df.index=["statistic","pvalue"]
df.to_csv("data_analysis/wilcoxon_result_x264_SVR.csv")
#绘制箱线图和曲线图
#PCA的横向比较（对同一个模型，对pca处理前后的结果比较，一个数据集四个指标，画四个箱线图
labels='bfPCA','afPCA'
plt.figure(1)
plt.grid(True)
plt.subplot(221)
plt.boxplot([beforePCA_explained_variance_score[2:len(beforePCA_explained_variance_score)],afterPCA_explained_variance_score[2:len(afterPCA_explained_variance_score)]],
            meanline=True,showmeans=True,labels=labels,
            medianprops={'color':'red','linewidth':'1.5'},
            meanprops={'color':'blue','linewidth':'1.5'},
            flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 10})
plt.ylabel("explained_variance_score",fontsize=10)

plt.subplot(222)
plt.boxplot([beforePCA_mean_absolute_error,afterPCA_mean_absolute_error],
            meanline=True,showmeans=True,labels=labels,
            medianprops={'color':'red','linewidth':'1.5'},
            meanprops={'color':'blue','linewidth':'1.5'},
            flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 10})
plt.ylabel("explained_mean_absolute_error",fontsize=10,labelpad=-200)

plt.subplot(223)
plt.boxplot([beforePCA_root_mean_squared_error,afterPCA_root_mean_squared_error],
            meanline=True,showmeans=True,labels=labels,
            medianprops={'color':'red','linewidth':'1.5'},
            meanprops={'color':'blue','linewidth':'1.5'},
            flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 10})
plt.ylabel("root_mean_squared_error",fontsize=10)

plt.subplot(224)
plt.boxplot([beforePCA_r2_score[1:len(beforePCA_explained_variance_score)],afterPCA_r2_score[1:len(afterPCA_explained_variance_score)]],
            meanline=True,showmeans=True,labels=labels,
            medianprops={'color':'red','linewidth':'1.5'},
            meanprops={'color':'blue','linewidth':'1.5'},
            flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 10})
plt.ylabel("r2_score",fontsize=10,labelpad=-200)
plt.suptitle('x264 of SVR')
plt.savefig("data_analysis/x264_of_SVR_1.png",dpi=300)


plt.figure(2)

plt.subplot(221)
x=np.arange(k)
#x=[1,6,11,16,21,26,31]
y1=beforePCA_explained_variance_score
y2=afterPCA_explained_variance_score
plt.plot(x,y1,label='bfPCA')
plt.plot(x,y2,label='afPCA')
plt.ylabel("explained_variance_score",fontsize=10)
plt.legend()

plt.subplot(222)
x=np.arange(k)
#x=[1,6,11,16,21,26,31]
y1=beforePCA_mean_absolute_error
y2=afterPCA_mean_absolute_error
plt.plot(x,y1,label='bfPCA')
plt.plot(x,y2,label='afPCA')
plt.ylabel("mean_absolute_error",fontsize=10,labelpad=-200)
plt.legend()

plt.subplot(223)
x=np.arange(k)
#x=[1,6,11,16,21,26,31]
y1=beforePCA_root_mean_squared_error
y2=afterPCA_root_mean_squared_error
plt.plot(x,y1,label='bfPCA')
plt.plot(x,y2,label='afPCA')
plt.ylabel("root_mean_squared_error",fontsize=10)
plt.legend()

plt.subplot(224)
x=np.arange(k)
#x=[1,6,11,16,21,26,31]
y1=beforePCA_r2_score
y2=afterPCA_r2_score
plt.plot(x,y1,label='bfPCA')
plt.plot(x,y2,label='afPCA')
plt.ylabel("r2_score",fontsize=10,labelpad=-208)
plt.legend()
plt.suptitle('x264 of SVR')
plt.savefig("data_analysis/x264_of_SVR_2.png",dpi=300)

plt.figure(3)
x=np.arange(k)
#x=[1,6,11,16,21,26,31]
y3=(afterPCA_explained_variance_score-beforePCA_explained_variance_score)/beforePCA_explained_variance_score
y4=-(afterPCA_mean_absolute_error-beforePCA_mean_absolute_error)/beforePCA_mean_absolute_error
y5=-(afterPCA_root_mean_squared_error-beforePCA_root_mean_squared_error)/beforePCA_root_mean_squared_error
y6=(afterPCA_r2_score-beforePCA_r2_score)/beforePCA_r2_score
plt.plot(x, y3, label='explained')
plt.plot(x, y4, label='MAE')
plt.plot(x, y5, label='RMSE')
plt.plot(x, y6, label='R2')
plt.ylabel("Relative optimization rate",fontsize=10)
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=3))
ax = plt.gca()    # 得到图像的Axes对象
ax.spines['right'].set_color('none')   # 将图像右边的轴设为透明
ax.spines['top'].set_color('none')     # 将图像上面的轴设为透明
ax.xaxis.set_ticks_position('bottom')    # 将x轴刻度设在下面的坐标轴上
ax.yaxis.set_ticks_position('left')         # 将y轴刻度设在左边的坐标轴上
ax.spines['bottom'].set_position(('data', 0))   # 将两个坐标轴的位置设在数据点原点
ax.spines['left'].set_position(('data', 0))
plt.legend()
plt.suptitle('x264 of SVR')
plt.savefig("data_analysis/x264_of_SVR_3.png", dpi=300)

