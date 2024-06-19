本文件从70个PD患者14个活动的原始数据文件提取特征，生成242维度的特征文件

1、get_fea(data)
对原始信号进行标准化和滤波的预处理，而后提取特征。
'''
	Describe
	选择指定活动进行拼接
	-----------------
	Parameters
	-----------------
	data:原始特征文件
	windowsize：窗口大小
	f_s: 采样频率
	overlapping：重叠大小
	ACC_XEA(0)	加速度融合轴均值
	ACC_XAX(1)	加速度融合轴最大值
	ACC_XIN(2)	加速度融合轴最小值
	ACC_XST(3)	加速度融合轴标准差
	ACC_XRS(4)	加速度融合轴均方根
	ACC_XVR(5)	加速度融合轴方差
	ACC_ASS(6)	加速度融合轴熵
	ACC_XEAx(7)	加速度x轴均值
	ACC_XAXx(8)	加速度x轴最大值
	ACC_XINx(9)	加速度x轴最小值
	ACC_XSTx(10)	加速度x轴标准差
	ACC_XRSx(11)	加速度x轴均方根
	ACC_XVRx(12)	加速度x轴方差
	ACC_XSS(13)	加速度x轴熵
	ACC_XEAy(14)	加速度y轴均值
	ACC_XAXy(15)	加速度y轴最大值
	ACC_XINy(16)	加速度y轴最小值
	ACC_XSTy(17)	加速度y轴标准差
	ACC_XRSy(18)	加速度y轴均方根
	ACC_XVRy(19)	加速度y轴方差
	ACC_YSS(20)	加速度y轴熵
	ACC_XEAz(21)	加速度z轴均值
	ACC_XAXz(22)	加速度z轴最大值
	ACC_XINz(23)	加速度z轴最小值
	ACC_XSTz(24)	加速度z轴标准差
	ACC_XRSz(25)	加速度z轴均方根
	ACC_XVRz(26)	加速度z轴方差
	ACC_ZSS(27)	加速度z轴熵
	ACC_ACO(28)	加速度xy轴相关性
	ACC_BCO(29)	加速度yz轴相关性
	ACC_CCO(30)	加速度xz轴相关性
	ACC_DCO(31)	加速度xa轴相关性
	ACC_ECO(32)	加速度ya轴相关性
	ACC_FCO(33)	加速度za轴相关性
	ACC_ta1(34)	加速度T轴最大值
	ACC_ta2(35)	加速度t轴最小值
	ACC_ta3(36)	加速度t轴均值
	ACC_ta4(37)	加速度t轴方差
	ACC_qq(38)	加速度t轴均方根
	ACC_piana(39)	加速度t轴偏度
	ACC_fenga(40)	加速度t轴峰度
	ACC_acfx1(41)	加速度x轴自相关系数最大值
	ACC_acfx3(42)	加速度x轴自相关系数最小值
	ACC_acfx4(43)	加速度x轴自相关系数方差
	ACC_acfy1(44)	加速度y轴自相关系数最大值
	ACC_acfy3(45)	加速度y轴自相关系数最小值
	ACC_acfy4(46)	加速度y轴自相关系数方差
	ACC_acfz1(47)	加速度z轴自相关系数最大值
	ACC_acfz3(48)	加速度z轴自相关系数最小值
	ACC_acfz4(49)	加速度z轴自相关系数方差
	ACC_acfa1(50)	加速度融合轴自相关系数最大值
	ACC_acfa3(51)	加速度融合轴自相关系数最小值
	ACC_acfa4(52)	加速度融合轴自相关系数方差
	ACC_auto(53-61)	X,Y,Z及融合轴自相关系数峰值
	ACC_fft_peak_x1(61-80)	X,Y,Z及融合轴FFT变换后前五个峰值大小
	ACC_psd_peak(81-100)	X,Y,Z及融合轴PSD变换后前五个峰值大小
	ACC_autocoor_peak(56)	X,Y,Z及融合轴自相关系数前五个峰值大小
       
       Return
       -----------------
       Feat2：提取的特征     '''

执行流程：
1、特征提取：
	输入原始病人数据的路径
	filefullpath="./datasets/raw/person{}/{}_session{}_wristr.csv".format(person,person,activity) #病人数据的根目录为./datasets/raw/
	输出为基于加速度计、陀螺仪原始数据的两个特征文件
	加速度特征文件：./result/preprocess/feature_70/wristr_feature_acc.csv
	陀螺仪特征文件：./result/preprocess/feature_70/wristr_feature_gyro.csv

2、拼接加速度和陀螺仪的特征：
	输入特征文件：
	加速度特征文件：./result/preprocess/feature_70/wristr_feature_acc.csv
	陀螺仪特征文件：./result/preprocess/feature_70/wristr_feature_gyro.csv
	输出文件：
	./result/preprocess/feature_70/final_wristr_fea_acc_gyro_70pd_200hz.csv

3、添加分期标签：
	输入:
	特征文件: ./result/preprocess/feature_70/final_wristr_fea_acc_gyro_70pd_200hz.csv
	标签文件：./datasets/label_shimmer3.xlsx
	输出：
	./result/preprocess/feature_70/242D_70PD_windo100__wristr_acc_gyro.csv
     
