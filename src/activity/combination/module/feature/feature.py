import pandas as pd
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.fftpack
from scipy.signal import find_peaks
import numpy as np
from scipy.special import entr
from scipy.fftpack import fft,fftshift
from statsmodels.tsa import stattools
from statsmodels.graphics.tsaplots import *
from scipy.signal import welch
import string
from sklearn.preprocessing import StandardScaler
import os
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_curve
# scaler = StandardScaler()a
# x_train = scaler.fit_transform(x_np)

N = 100
windowsize = N
f_s = 200
fs =f_s #filter fs
overlapping=2
label_map={"stndg":0,"wlkgs":1,"wlkgc":2,"strsu":3,"strsd":4,"wlkgp":5,"ftnr":6,"ftnl":7,"ramr":8,"raml":9,"ststd":10,"drawg":11,"typng":12}

def get_psd_values(y_values, N, f_s):
    f_values, psd_values = welch(y_values, fs=f_s)
    return f_values, psd_values

def get_fft_values(y_values, N, f_s): #N为采样点数，f_s是采样频率，返回f_values希望的频率区间, fft_values真实幅值
    f_values = np.linspace(0.0, f_s/2.0, N//2)
    fft_values_ = fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[len(result)//2:]
 
def get_autocorr_values(y_values, N, f_s):
    autocorr_values = autocorr(y_values)
    x_values = np.array([ 1.0*jj/f_s for jj in range(0, N)])
    return x_values, autocorr_values

def butter_highpass(highcut, fs, order):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype="highpass")
    return b, a

def butter_highpass_filter(data, cutOff, fs, order=3):
    b, a = butter_highpass(cutOff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

#定义一个函数，输入是一个dataframe，输出也是一行提出的特征，
def get_fea(data):    
#     data = pd.read_csv(Folder_Path + '\\' + file_list[f])
    
    label = data.iloc[:,16]
    x = data.iloc[:,4]
    y = data.iloc[:,5]
    z = data.iloc[:,6]
    ACCW2 = data.iloc[:,7]
    i = 0
    j = 0
    #windowsize = 100
    #smv
    XEA = np.zeros(len(data),dtype=float)##MEAN
    XAX = np.zeros(len(data),dtype=float)##MAX
    XIN = np.zeros(len(data),dtype=float)##MIN
    XST = np.zeros(len(data),dtype=float)##STD标准差
    XRS = np.zeros(len(data),dtype=float)##RMS
    XVR = np.zeros(len(data),dtype=float)##VAR方差
    #x轴
    XEAx = np.zeros(len(data),dtype=float)##MEAN
    XAXx = np.zeros(len(data),dtype=float)##MAX
    XINx = np.zeros(len(data),dtype=float)##MIN
    XSTx = np.zeros(len(data),dtype=float)##STD标准差
    XRSx = np.zeros(len(data),dtype=float)##RMS
    XVRx = np.zeros(len(data),dtype=float)##VAR方差
    #y轴
    XEAy = np.zeros(len(data),dtype=float)##MEAN
    XAXy = np.zeros(len(data),dtype=float)##MAX
    XINy = np.zeros(len(data),dtype=float)##MIN
    XSTy = np.zeros(len(data),dtype=float)##STD标准差
    XRSy = np.zeros(len(data),dtype=float)##RMS
    XVRy = np.zeros(len(data),dtype=float)##VAR方差
    #z轴
    XEAz = np.zeros(len(data),dtype=float)##MEAN
    XAXz = np.zeros(len(data),dtype=float)##MAX
    XINz = np.zeros(len(data),dtype=float)##MIN
    XSTz = np.zeros(len(data),dtype=float)##STD标准差
    XRSz = np.zeros(len(data),dtype=float)##RMS
    XVRz = np.zeros(len(data),dtype=float)##VAR方差
    #相关性
    ACO = np.zeros(len(data),dtype=float)##correlation
    BCO = np.zeros(len(data),dtype=float)
    CCO = np.zeros(len(data),dtype=float)
    DCO = np.zeros(len(data),dtype=float)
    ECO = np.zeros(len(data),dtype=float)
    FCO = np.zeros(len(data),dtype=float)
    ##频域的熵
    XSS = np.zeros(len(data),dtype=float)
    YSS = np.zeros(len(data),dtype=float)
    ZSS = np.zeros(len(data),dtype=float)
    ASS = np.zeros(len(data),dtype=float)
    #tilt angle
#    ta = np.zeros(len(data),dtype=float)
    #skew偏度和kurt峰度
    piana = np.zeros(len(data),dtype=float)
    fenga = np.zeros(len(data),dtype=float)
    #幅度标准差
    qq = np.zeros(len(data),dtype=float)
    #自相关系数
    acfx1 = np.zeros(len(data),dtype=float)
    acfx3 = np.zeros(len(data),dtype=float)
    acfx4 = np.zeros(len(data),dtype=float)
    acfy1 = np.zeros(len(data),dtype=float)
    acfy3 = np.zeros(len(data),dtype=float)
    acfy4 = np.zeros(len(data),dtype=float)
    acfz1 = np.zeros(len(data),dtype=float)
    acfz3 = np.zeros(len(data),dtype=float)
    acfz4 = np.zeros(len(data),dtype=float)
    acfa1 = np.zeros(len(data),dtype=float)
    acfa3 = np.zeros(len(data),dtype=float)
    acfa4 = np.zeros(len(data),dtype=float)
    ta1 = np.zeros(len(data),dtype=float)
    ta2 = np.zeros(len(data),dtype=float)
    ta3 = np.zeros(len(data),dtype=float)
    ta4 = np.zeros(len(data),dtype=float)
    autox1 = np.zeros(len(data),dtype=float)
    autox2 = np.zeros(len(data),dtype=float)
    autoy1 = np.zeros(len(data),dtype=float)
    autoy2 = np.zeros(len(data),dtype=float)
    autoz1 = np.zeros(len(data),dtype=float)
    autoz2 = np.zeros(len(data),dtype=float)
    autoa1 = np.zeros(len(data),dtype=float)
    autoa2 = np.zeros(len(data),dtype=float)
    fft_peak_x = np.zeros([len(data),5],dtype=float)
    fft_peak_y = np.zeros([len(data),5],dtype=float)
    fft_peak_z = np.zeros([len(data),5],dtype=float)
    fft_peak_a = np.zeros([len(data),5],dtype=float)
    psd_peak_x = np.zeros([len(data),5],dtype=float)
    psd_peak_y = np.zeros([len(data),5],dtype=float)
    psd_peak_z = np.zeros([len(data),5],dtype=float)
    psd_peak_a = np.zeros([len(data),5],dtype=float)
    autocorr_peak_x = np.zeros([len(data),5],dtype=float)
    autocorr_peak_y = np.zeros([len(data),5],dtype=float)
    autocorr_peak_z = np.zeros([len(data),5],dtype=float)
    autocorr_peak_a = np.zeros([len(data),5],dtype=float)
    
    f1x = np.zeros(len(data),dtype=float)
    f1y = np.zeros(len(data),dtype=float)
    #label
    label2 = np.zeros(len(data),dtype=float)
    label2[0:len(data) - windowsize] = label[0:len(data) - windowsize]
    print("len_data-windowsize",len(data)-windowsize)
    
    #T轴
    #A轴
    #PSD轴
    #fft轴
    while (i < len(data) - windowsize):
            #smv
            XEA[j] = ACCW2[i:i + windowsize].mean()
            XAX[j] = ACCW2[i:i + windowsize].max()
            XIN[j] = ACCW2[i:i + windowsize].min()
            XST[j] = ACCW2[i:i + windowsize].std()
            XRS[j] = np.sqrt((np.square(ACCW2[i:i + windowsize]).mean()))
            XVR[j] = ACCW2[i:i + windowsize].var()
            ASS[j] = entr(abs(ACCW2[i:i + windowsize])).sum(axis=0)/np.log(10)
            #x
            XEAx[j] = x[i:i + windowsize].mean()
            XAXx[j] = x[i:i + windowsize].max()
            XINx[j] = x[i:i + windowsize].min()
            XSTx[j] = x[i:i + windowsize].std()
            XRSx[j] = np.sqrt((np.square(x[i:i + windowsize]).mean()))
            XVRx[j] = x[i:i + windowsize].var()
            XSS[j] = entr(abs(x[i:i + windowsize])).sum(axis=0)/np.log(10)
            #y
            XEAy[j] = y[i:i + windowsize].mean()
            XAXy[j] = y[i:i + windowsize].max()
            XINy[j] = y[i:i + windowsize].min()
            XSTy[j] = y[i:i + windowsize].std()
            XRSy[j] = np.sqrt((np.square(y[i:i + windowsize]).mean()))
            XVRy[j] = y[i:i + windowsize].var()
            YSS[j] = entr(abs(y[i:i + windowsize])).sum(axis=0)/np.log(10)
            #z
            XEAz[j] = z[i:i + windowsize].mean()
            XAXz[j] = z[i:i + windowsize].max()
            XINz[j] = z[i:i + windowsize].min()
            XSTz[j] = z[i:i + windowsize].std()
            XRSz[j] = np.sqrt((np.square(z[i:i + windowsize]).mean()))
            XVRz[j] = z[i:i + windowsize].var()
            ZSS[j] = entr(abs(z[i:i + windowsize])).sum(axis=0)/np.log(10)
            #相关性
            ACOR = np.corrcoef(x[i:i + windowsize],y[i:i + windowsize])
            ACO[j] = ACOR[0,1]
            BCOR = np.corrcoef(x[i:i + windowsize],z[i:i + windowsize])
            BCO[j] = BCOR[0,1]
            CCOR = np.corrcoef(x[i:i + windowsize],ACCW2[i:i + windowsize])
            CCO[j] = CCOR[0,1]
            DCOR = np.corrcoef(y[i:i + windowsize],z[i:i + windowsize])
            DCO[j] = DCOR[0,1]
            ECOR = np.corrcoef(y[i:i + windowsize],ACCW2[i:i + windowsize])
            ECO[j] = ECOR[0,1]
            FCOR = np.corrcoef(z[i:i + windowsize],ACCW2[i:i + windowsize])
            FCO[j] = FCOR[0,1]
            data1 = x[i:i+windowsize]
            data2 = y[i:i+windowsize]
            data3 = z[i:i+windowsize]
            data4 = ACCW2[i:i+windowsize]
            data1 = data1.values
            data2 = data2.values
            data3 = data3.values
            data4 = data4.values
            
            #fft_x
            f_values, fft_values = get_fft_values(data1, N, f_s)
            signal_min = np.nanpercentile(fft_values,5)
            signal_max = np.nanpercentile(fft_values, 95)
            mph = signal_min + (signal_max - signal_min)/len(fft_values)#set minimum peak height
            peaks2, _ = find_peaks(fft_values, prominence = mph) 
            peak_save = fft_values[peaks2].argsort()[::-1][:5]
            #index_peakmax = fft_values[peaks2].argsort()[-1]
            #主峰
            #f1x[j] = f_values[peaks2[index_peakmax]]
            #f1y[j] = fft_values[peaks2[index_peakmax]]
            temp_arr =  f_values[peaks2[peak_save]] + fft_values[peaks2[peak_save]]
            fft_peak_x[j,:] = np.pad(temp_arr, (0, 5-len(temp_arr)), 'constant', constant_values=0)  ## (0, pad_len), 前面填充0个0，后面填充pad_len个0
            #print("fft_peak:",fft_peak)
            ##fft_y
            f_values, fft_values = get_fft_values(data2, N, f_s)
            signal_min = np.nanpercentile(fft_values, 5)
            signal_max = np.nanpercentile(fft_values, 95)
            mph = signal_min + (signal_max - signal_min)/len(fft_values)#set minimum peak height
            peaks2, _ = find_peaks(fft_values, prominence = mph) 
            peak_save = fft_values[peaks2].argsort()[::-1][:5]
            temp_arr =  f_values[peaks2[peak_save]] + fft_values[peaks2[peak_save]]
            fft_peak_y[j,:] = np.pad(temp_arr, (0, 5-len(temp_arr)), 'constant', constant_values=0)  ## (0, pad_len), 前面填充0个0，后面填充pad_len个0
            ##fft_z
            f_values, fft_values = get_fft_values(data3, N, f_s)
            signal_min = np.nanpercentile(fft_values, 5)
            signal_max = np.nanpercentile(fft_values, 95)
            mph = signal_min + (signal_max - signal_min)/len(fft_values)#set minimum peak height
            peaks2, _ = find_peaks(fft_values, prominence = mph) 
            peak_save = fft_values[peaks2].argsort()[::-1][:5]
            temp_arr =  f_values[peaks2[peak_save]] + fft_values[peaks2[peak_save]]
            fft_peak_z[j,:] = np.pad(temp_arr, (0, 5-len(temp_arr)), 'constant', constant_values=0)  ## (0, pad_len), 前面填充0个0，后面填充pad_len个0
            ##fft_a
            f_values, fft_values = get_fft_values(data4, N, f_s)
            signal_min = np.nanpercentile(fft_values, 5)
            signal_max = np.nanpercentile(fft_values, 95)
            mph = signal_min + (signal_max - signal_min)/len(fft_values)#set minimum peak height
            peaks2, _ = find_peaks(fft_values, prominence = mph) 
            peak_save = fft_values[peaks2].argsort()[::-1][:5]
            temp_arr =  f_values[peaks2[peak_save]] + fft_values[peaks2[peak_save]]
            fft_peak_a[j,:] = np.pad(temp_arr, (0, 5-len(temp_arr)), 'constant', constant_values=0)  ## (0, pad_len), 前面填充0个0，后面填充pad_len个0

            #psd_x
            p_values, psd_values = get_psd_values(data1, N, f_s)
            signal_min = np.nanpercentile(psd_values, 5)
            signal_max = np.nanpercentile(psd_values, 95)
            mph = signal_min + (signal_max - signal_min)/len(psd_values)#set minimum peak height
            peaks3, _ = find_peaks(psd_values,height = mph) 
            peak_save = psd_values[peaks3].argsort()[::-1][:5]
            temp_arr = p_values[peaks3[peak_save]] + psd_values[peaks3[peak_save]]
            psd_peak_x[j,:] = np.pad(temp_arr, (0, 5-len(temp_arr)), 'constant', constant_values=0)  ## (0, pad_len), 前面填充0个0，后面填充pad_len个0
            #psd_y
            p_values, psd_values = get_psd_values(data2, N, f_s)
            signal_min = np.nanpercentile(psd_values, 5)
            signal_max = np.nanpercentile(psd_values, 95)
            mph = signal_min + (signal_max - signal_min)/len(psd_values)#set minimum peak height
            peaks3, _ = find_peaks(psd_values,height = mph) 
            peak_save = psd_values[peaks3].argsort()[::-1][:5]
            temp_arr = p_values[peaks3[peak_save]] + psd_values[peaks3[peak_save]]
            psd_peak_y[j,:] = np.pad(temp_arr, (0, 5-len(temp_arr)), 'constant', constant_values=0)  ## (0, pad_len), 前面填充0个0，后面填充pad_len个0
            #psd_z
            p_values, psd_values = get_psd_values(data3, N, f_s)
            signal_min = np.nanpercentile(psd_values, 5)
            signal_max = np.nanpercentile(psd_values, 95)
            mph = signal_min + (signal_max - signal_min)/len(psd_values)#set minimum peak height
            peaks3, _ = find_peaks(psd_values,height = mph) 
            peak_save = psd_values[peaks3].argsort()[::-1][:5]
            temp_arr = p_values[peaks3[peak_save]] + psd_values[peaks3[peak_save]]
            psd_peak_z[j,:] = np.pad(temp_arr, (0, 5-len(temp_arr)), 'constant', constant_values=0)  ## (0, pad_len), 前面填充0个0，后面填充pad_len个0
            #psd_a
            p_values, psd_values = get_psd_values(data4, N, f_s)
            signal_min = np.nanpercentile(psd_values, 5)
            signal_max = np.nanpercentile(psd_values, 95)
            mph = signal_min + (signal_max - signal_min)/len(psd_values)#set minimum peak height
            peaks3, _ = find_peaks(psd_values,height = mph) 
            peak_save = psd_values[peaks3].argsort()[::-1][:5]
            temp_arr = p_values[peaks3[peak_save]] + psd_values[peaks3[peak_save]]
            psd_peak_a[j,:] = np.pad(temp_arr, (0, 5-len(temp_arr)), 'constant', constant_values=0)  ## (0, pad_len), 前面填充0个0，后面填充pad_len个0
                            
            #autocorr_x
            a_values, autocorr_values = get_autocorr_values(data1, N, f_s)
            signal_min = np.nanpercentile(autocorr_values, 5)
            signal_max = np.nanpercentile(autocorr_values, 95)
            mph = signal_min + (signal_max - signal_min)/len(autocorr_values)#set minimum peak height
            peaks4, _ = find_peaks(autocorr_values,height = mph) 
            peak_save = autocorr_values[peaks4].argsort()[::-1][:5]
            #index_peakmax = autocorr_values[peaks4].argsort()[-1] 
            
            temp_arr = a_values[peaks4[peak_save]] + autocorr_values[peaks4[peak_save]]
            autocorr_peak_x[j,:] = np.pad(temp_arr, (0, 5-len(temp_arr)), 'constant', constant_values=0)  ## (0, pad_len), 前面填充0个0，后面填充pad_len个0
            #autocorr_y
            a_values, autocorr_values = get_autocorr_values(data2, N, f_s)
            signal_min = np.nanpercentile(autocorr_values, 5)
            signal_max = np.nanpercentile(autocorr_values, 95)
            mph = signal_min + (signal_max - signal_min)/len(autocorr_values)#set minimum peak height
            peaks4, _ = find_peaks(autocorr_values,height = mph) 
            peak_save = autocorr_values[peaks4].argsort()[::-1][:5]
            #index_peakmax = autocorr_values[peaks4].argsort()[-1]
            
            temp_arr = a_values[peaks4[peak_save]] + autocorr_values[peaks4[peak_save]]
            autocorr_peak_y[j,:] = np.pad(temp_arr, (0, 5-len(temp_arr)), 'constant', constant_values=0)  ## (0, pad_len), 前面填充0个0，后面填充pad_len个0
            #autocorr_z
            a_values, autocorr_values = get_autocorr_values(data3, N, f_s)
            signal_min = np.nanpercentile(autocorr_values, 5)
            signal_max = np.nanpercentile(autocorr_values, 95)
            mph = signal_min + (signal_max - signal_min)/len(autocorr_values)#set minimum peak height
            peaks4, _ = find_peaks(autocorr_values,height = mph) 
            peak_save = autocorr_values[peaks4].argsort()[::-1][:5]
            #index_peakmax = autocorr_values[peaks4].argsort()[-1]
            
            temp_arr = a_values[peaks4[peak_save]] + autocorr_values[peaks4[peak_save]]
            autocorr_peak_z[j,:] = np.pad(temp_arr, (0, 5-len(temp_arr)), 'constant', constant_values=0)  ## (0, pad_len), 前面填充0个0，后面填充pad_len个0
            #autocorr_a
            a_values, autocorr_values = get_autocorr_values(data4, N, f_s)
            signal_min = np.nanpercentile(autocorr_values, 5)
            signal_max = np.nanpercentile(autocorr_values, 95)
            mph = signal_min + (signal_max - signal_min)/len(autocorr_values)#set minimum peak height
            peaks4, _ = find_peaks(autocorr_values,height = mph) 
            peak_save = autocorr_values[peaks4].argsort()[::-1][:5]
            temp_arr = a_values[peaks4[peak_save]] + autocorr_values[peaks4[peak_save]]
            autocorr_peak_a[j,:] = np.pad(temp_arr, (0, 5-len(temp_arr)), 'constant', constant_values=0)  ## (0, pad_len), 前面填充0个0，后面填充pad_len个0

      
            ta = np.arccos(z[i:i + windowsize])
            #幅度标准差
            qq[j] = (XST[j]**2+XSTx[j]**2+XSTy[j]**2+XSTz[j]**2)**0.5
            #偏度和峰度
            seriesa = pd.Series(ACCW2[i:i + windowsize])
            piana[j] = seriesa.skew()
            fenga[j] = seriesa.kurt()
            #自相关系数
            acfx=stattools.acf(x[i:i + windowsize])
            acfy=stattools.acf(y[i:i + windowsize]) 
            acfz=stattools.acf(z[i:i + windowsize]) 
            acfa=stattools.acf(ACCW2[i:i + windowsize])
            acfx1[j] = acfx.mean()
            #acfx2[j] = acfx.max()
            acfx3[j] = acfx.min()
            acfx4[j] = acfx.std()
            acfy1[j] = acfy.mean()
            #acfy2[j] = acfy.max()
            acfy3[j] = acfy.min()
            acfy4[j] = acfy.std()
            acfz1[j] = acfz.mean()
            #acfz2[j] = acfz.max()
            acfz3[j] = acfz.min()
            acfz4[j] = acfz.std()
            acfa1[j] = acfa.mean()
            #acfa2[j] = acfa.max()
            acfa3[j] = acfa.min()
            acfa4[j] = acfa.std()
            ta1[j] = ta.mean()
            ta2[j] = ta.max()
            ta3[j] = ta.min()
            ta4[j] = ta.std()
            f_values, fft_values = get_autocorr_values(x[i:i + windowsize],N, f_s)
            autox1[j] = fft_values.max()
            autox2[j] = fft_values.min()
            f_values, fft_values = get_autocorr_values(y[i:i + windowsize], N, f_s)
            autoy1[j] = fft_values.max()
            autoy2[j] = fft_values.min()
            f_values, fft_values = get_autocorr_values(z[i:i + windowsize], N, f_s)
            autoz1[j] = fft_values.max()
            autoz2[j] = fft_values.min()
            f_values, fft_values = get_autocorr_values(ACCW2[i:i + windowsize], N, f_s)
            autoa1[j] = fft_values.max()
            autoa2[j] = fft_values.min()
            i = i + windowsize // overlapping - 1
            j = j + 1
            
            
    feata = np.c_[XEA,XAX,XIN,XST,XRS,XVR,ASS,XEAx,XAXx,XINx,XSTx,XRSx,XVRx,XSS,XEAy,XAXy,XINy,XSTy,XRSy,XVRy,YSS,XEAz,XAXz,XINz,XSTz,XRSz,XVRz,ZSS,ACO,BCO,CCO,DCO,ECO,FCO]##只有前j行有效 34
    #tilt angle,qq,偏度，峰度，自相关系数27个特征加一个label
    featb = np.c_[ta1,ta2,ta3,ta4,qq,piana,fenga,acfx1,acfx3,acfx4,acfy1,acfy3,acfy4,acfz1,acfz3,acfz4,acfa1,acfa3,acfa4,autox1,autox2,autoy1,autoy2,autoz1,autoz2,autoa1,autoa2,fft_peak_x,fft_peak_y,fft_peak_z,fft_peak_a,psd_peak_x,psd_peak_y,psd_peak_z,psd_peak_a,autocorr_peak_x,autocorr_peak_y,autocorr_peak_z,autocorr_peak_a]  #36+1

    Feat = np.c_[feata,featb,label2]
    Feat2 = np.zeros((j,Feat.shape[1]))#后一个参数为特征种类加一 28 38 16 45
    Feat2[0:j,:] = Feat[0:j,:]
    Feat2 = pd.DataFrame(Feat2)
    return Feat2

