from scipy.signal import butter, lfilter
import numpy as np
from scipy.signal import welch
from scipy.signal import find_peaks
from scipy.fftpack import fft,fftshift
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import simps
import pandas as pd
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.fftpack
from scipy.signal import find_peaks
import numpy as np
from scipy.special import entr
from scipy.fftpack import fft, fftshift
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

from scipy.signal import butter, lfilter
import numpy as np
from scipy.signal import welch
from scipy.signal import find_peaks
from scipy.fftpack import fft,fftshift
from scipy.special import entr
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import simps
import scipy.integrate as integrate
from scipy.integrate import simpson
import math
import numpy as np
from numpy import array, sign, zeros
from scipy.interpolate import interp1d
import scipy.signal


def get_psd_values(y_values, N, fs):
    f_values, psd_values = welch(y_values, fs)
    return f_values, psd_values

def infor(data):
    a = pd.value_counts(data) / len(data)
    return sum(np.log2(a) * a * (-1))

def get_fft_values(y_values, N, fs):  
    f_values = np.linspace(0.0, fs / 2.0, N // 2)
    fft_values_ = fft(y_values)
    fft_values = 2.0 / N * np.abs(fft_values_[0:N // 2])
    return f_values, fft_values


def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[len(result) // 2:]


def get_autocorr_values(y_values, N, fs):
    autocorr_values = autocorr(y_values)
    x_values = np.array([1.0 * jj / fs for jj in range(0, N)])
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


def fft_peak_xy(data,N,fs,peak_num=5):
    f_values, fft_values = get_fft_values(data, N, fs)
    signal_min = np.nanpercentile(fft_values, 5)
    signal_max = np.nanpercentile(fft_values, 95)
    mph = signal_min + (signal_max - signal_min) / len(fft_values)  # set minimum peak height
    peaks, _ = find_peaks(fft_values, prominence=mph)
    peak_save = fft_values[peaks].argsort()[::-1][:peak_num]
    temp_arr = f_values[peaks[peak_save]] + fft_values[peaks[peak_save]] ## 峰值前5的极值点横纵坐标之和
    fft_peak_xy = np.pad(temp_arr, (0, peak_num - len(temp_arr)), 'constant', constant_values=0)  ## (0, pad_len), 前面填充0个0，后面填充pad_len个0
    return fft_peak_xy


def psd_peak_xy(data,N,fs,peak_num=5):
    p_values, psd_values = get_psd_values(data, N, fs)
    signal_min = np.nanpercentile(psd_values, 5)
    signal_max = np.nanpercentile(psd_values, 95)
    mph = signal_min + (signal_max - signal_min) / len(psd_values)  # set minimum peak height
    peaks3, _ = find_peaks(psd_values, height=mph)
    peak_save = psd_values[peaks3].argsort()[::-1][:peak_num]
    temp_arr = p_values[peaks3[peak_save]] + psd_values[peaks3[peak_save]]
    psd_peak_xy = np.pad(temp_arr, (0, peak_num - len(temp_arr)), 'constant',
                              constant_values=0)  ## (0, pad_len), 前面填充0个0，后面填充pad_len个0
    return psd_peak_xy


def auto_peak_xy(data,N,fs,peak_num=5):
    a_values, autocorr_values = get_autocorr_values(data, N, fs)
    signal_min = np.nanpercentile(autocorr_values, 5)
    signal_max = np.nanpercentile(autocorr_values, 95)
    mph = signal_min + (signal_max - signal_min) / len(autocorr_values)  # set minimum peak height
    peaks4, _ = find_peaks(autocorr_values, height=mph)
    peak_save = autocorr_values[peaks4].argsort()[::-1][:peak_num]
    temp_arr = a_values[peaks4[peak_save]] + autocorr_values[peaks4[peak_save]]
    autocorr_peak_xy = np.pad(temp_arr, (0, peak_num - len(temp_arr)), 'constant',
                                   constant_values=0)  ## (0, pad_len), 前面填充0个0，后面填充pad_len个0
    return autocorr_peak_xy


import math
import numpy as np
from numpy import array, sign, zeros
from scipy.interpolate import interp1d
import scipy.signal


# 输入信号序列即可(list)
def envelope_extraction(signal):
    s = signal.astype(float)
    q_u = np.zeros(s.shape)
    q_l = np.zeros(s.shape)

    # 在插值值前加上第一个值。这将强制模型对上包络和下包络模型使用相同的起点。
    # Prepend the first value of (s) to the interpolating values. This forces the model to use the same starting point for both the upper and lower envelope models.
    u_x = [0, ]  # 上包络的x序列
    u_y = [s[0], ]  # 上包络的y序列

    l_x = [0, ]  # 下包络的x序列
    l_y = [s[0], ]  # 下包络的y序列

    # 检测波峰和波谷，并分别标记它们在u_x,u_y,l_x,l_中的位置。
    # Detect peaks and troughs and mark their location in u_x,u_y,l_x,l_y respectively.

    for k in range(1, len(s) - 1):
        if (sign(s[k] - s[k - 1]) == 1) and (sign(s[k] - s[k + 1]) == 1):
            u_x.append(k)
            u_y.append(s[k])

        if (sign(s[k] - s[k - 1]) == -1) and ((sign(s[k] - s[k + 1])) == -1):
            l_x.append(k)
            l_y.append(s[k])

    u_x.append(len(s) - 1)  # 上包络与原始数据切点x
    u_y.append(s[-1])  # 对应的值

    l_x.append(len(s) - 1)  # 下包络与原始数据切点x
    l_y.append(s[-1])  # 对应的值

    # u_x,l_y是不连续的，以下代码把包络转为和输入数据相同大小的数组[便于后续处理，如滤波]
    upper_envelope_y = np.zeros(len(signal))
    lower_envelope_y = np.zeros(len(signal))

    upper_envelope_y[0] = u_y[0]  # 边界值处理
    upper_envelope_y[-1] = u_y[-1]
    lower_envelope_y[0] = l_y[0]  # 边界值处理
    lower_envelope_y[-1] = l_y[-1]

    # 上包络
    last_idx, next_idx = 0, 0
    k, b = general_equation(u_x[0], u_y[0], u_x[1], u_y[1])  # 初始的k,b
    for e in range(1, len(upper_envelope_y) - 1):

        if e not in u_x:
            v = k * e + b
            upper_envelope_y[e] = v
        else:
            idx = u_x.index(e)
            upper_envelope_y[e] = u_y[idx]
            last_idx = u_x.index(e)
            next_idx = u_x.index(e) + 1
            # 求连续两个点之间的直线方程
            k, b = general_equation(u_x[last_idx], u_y[last_idx], u_x[next_idx], u_y[next_idx])

            # 下包络
    last_idx, next_idx = 0, 0
    k, b = general_equation(l_x[0], l_y[0], l_x[1], l_y[1])  # 初始的k,b
    for e in range(1, len(lower_envelope_y) - 1):

        if e not in l_x:
            v = k * e + b
            lower_envelope_y[e] = v
        else:
            idx = l_x.index(e)
            lower_envelope_y[e] = l_y[idx]
            last_idx = l_x.index(e)
            next_idx = l_x.index(e) + 1
            # 求连续两个切点之间的直线方程
            k, b = general_equation(l_x[last_idx], l_y[last_idx], l_x[next_idx], l_y[next_idx])

    return upper_envelope_y, lower_envelope_y


def general_equation(first_x, first_y, second_x, second_y):
    # 斜截式 y = kx + b
    A = second_y - first_y
    B = first_x - second_x
    C = second_x * first_y - first_x * second_y
    k = -1 * A / B
    b = -1 * C / B
    return k, b

def mAmp(data):  
    L = np.size(data, 0) #计算data的样本数量大小
    upper_envolope, low_envolope = envelope_extraction(data)
    mAmp = (upper_envolope-low_envolope)/L*1.0
    return mAmp

def sampEn(data, N, r): 
    L = len(data)
    B = 0.0
    A = 0.0
    # Split time series and save all templates of length m
    xmi = np.array([data[i: i + N] for i in range(L - N)])
    xmj = np.array([data[i: i + N] for i in range(L - N + 1)])
    # Save all matches minus the self-match, compute B
    B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])
    # Similar for computing A
    N += 1
    xm = np.array([data[i: i + N] for i in range(L - N + 1)])
    A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])
    # Return SampEn
    return -np.log(A / B)

def DF(X, Y, Z, N, fs, peak_num=5): 
    _, fft_peak_x_main = fft_domain(X,N,fs,peak_num=5)
    _, fft_peak_y_main = fft_domain(Y,N,fs,peak_num=5)
    _, fft_peak_z_main = fft_domain(Z,N,fs,peak_num=5)
    return max(fft_peak_x_main, fft_peak_y_main, fft_peak_z_main)

from scipy.integrate import simps

def PSDEnergy_XYZ(X, Y, Z, N, fs): 
    X_p, X_psd = get_psd_values(X, N, fs)
    Y_p, Y_psd = get_psd_values(Y, N, fs)
    Z_p, Z_psd = get_psd_values(Z, N, fs)
    x_energy = simps(X_psd, X_p, dx=0.001)
    y_energy = simps(Y_psd, Y_p, dx=0.001)
    z_energy = simps(Z_psd, Z_p, dx=0.001)
    return x_energy+y_energy+z_energy

import scipy.integrate as integrate

def spectrumConcentration(X, Y, Z, N, fs): 
    DF_x, _ = fft_domain(X, N, fs, peak_num=5)
    DF_y, _ = fft_domain(Y, N, fs, peak_num=5)
    DF_z, _ = fft_domain(Z, N, fs, peak_num=5)
    X_spectrumDistribution = integrate.quad(get_fft_values(X, N, fs), DF_x - 0.4, DF_x + 0.4)
    Y_spectrumDistribution = integrate.quad(get_fft_values(Y, N, fs), DF_y - 0.4, DF_y + 0.4)
    Z_spectrumDistribution = integrate.quad(get_fft_values(Z, N, fs), DF_z - 0.4, DF_z + 0.4)
    spectrumDistribution = X_spectrumDistribution + Y_spectrumDistribution + Z_spectrumDistribution
    return spectrumDistribution / PSDEnergy_XYZ(X, Y, Z, N, fs) *1.0

def base(data):
    damp = mAmp(data)
    dmean = data.mean()
    dmax = data.max()
    dstd = data.std()
    dvar = data.var()
    # entropy
    dentr = entr(abs(data)).sum(axis=0) / np.log(10)
    # log_energy
    log_energy_value = np.log10(data ** 2).sum(axis=0)
    # SMA
    time = np.arange(data.shape[0])
    signal_magnitude_area = simpson(data,time)
    # Interquartile range (interq)
    per25 = np.nanpercentile(data, 25)
    per75 = np.nanpercentile(data, 75)
    interq = per75 - per25
    # 偏度
    seriesdata = pd.Series(data)
    skew = seriesdata.skew()
    # 峰度
    kurt = seriesdata.kurt()
    return damp, dmean, dmax, dstd, dvar, dentr, log_energy_value, signal_magnitude_area, interq, skew, kurt

def time_domain(data):
    damp, dmean, dmax, dstd, dvar, dentr, log_energy_value, signal_magnitude_area, interq, skew, kurt = base(data)
    drms = np.sqrt((np.square(data).mean()))  # rms
    # 主峰次峰横坐标间隔
    signal_min = np.nanpercentile(data, 5)
    signal_max = np.nanpercentile(data, 95)
    mph = signal_min + (signal_max - signal_min) / len(data)  # set minimum peak height
    peaks, _ = find_peaks(data, prominence=mph)

    if (len(peaks) == 0):
        index_peaksub = 0
        index_peakmax = 0
    elif (len(peaks) == 1):
        index_peakmax = data[peaks].argsort()[-1]
        index_peaksub = index_peakmax
    else:
        index_peakmax = data[peaks].argsort()[-1]
        index_peaksub = data[peaks].argsort()[-2]
    dif_peak_X = index_peaksub - index_peakmax
    # 波峰因数Crest factor(cft)
    cftor = data[index_peakmax] / drms * 1.0
    return  damp, dmean, dmax, dstd, dvar, dentr, log_energy_value, signal_magnitude_area, interq, skew, kurt, drms, dif_peak_X, cftor

def corrcoef(x,y,z,a):
    xy_cor = np.corrcoef(x, y)
    xz_cor = np.corrcoef(x, z)
    xa_cor = np.corrcoef(x, a)
    yz_cor = np.corrcoef(y, z)
    ya_cor = np.corrcoef(y, a)
    za_cor = np.corrcoef(z, a)
    return xy_cor[0, 1], xz_cor[0, 1], xa_cor[0, 1], yz_cor[0, 1], ya_cor[0, 1], za_cor[0, 1]

def fft_domain(data,N,fs):  ##频域图峰值细节
    f_values, fft_values = get_fft_values(data, N, fs)
    damp, dmean, dmax, dstd, dvar, dentr, log_energy_value, signal_magnitude_area, interq, skew, kurt = base(fft_values)
    drms = np.sqrt((np.square(fft_values).mean()))  # rms

    signal_min = np.nanpercentile(fft_values, 5)
    signal_max = np.nanpercentile(fft_values, 95)
    mph = signal_min + (signal_max - signal_min) / len(fft_values)  # set minimum peak height
    peaks, _ = find_peaks(fft_values, prominence=mph)
    peak_save = fft_values[peaks].argsort()[::-1][:2]
    peak_x = f_values[peaks[peak_save]]   ## 峰值前5的极值点横纵坐标之和
    peak_y = fft_values[peaks[peak_save]]
    peak_x = np.pad(peak_x, (0, 2 - len(peak_x)), 'constant', constant_values=0)
    peak_y = np.pad(peak_y, (0, 2 - len(peak_y)), 'constant', constant_values=0)
    # 主峰
    peak_main_X = peak_x[0]
    peak_main_Y = peak_y[0]
    # 次峰
    peak_sub_X = peak_x[1]
    peak_sub_Y = peak_y[1]
    #主峰次峰差距
    dif_peak_X = peak_sub_X - peak_main_X
    dif_peak_Y =peak_main_Y - peak_sub_Y
    # 波峰因数Crest factor(cft or)
    cftor = peak_main_Y / drms * 1.0
    return damp, dmean, dmax, dstd, dvar, dentr, log_energy_value, signal_magnitude_area, interq, skew, kurt, peak_main_X, peak_main_Y, peak_sub_X, peak_sub_Y, dif_peak_X, drms, dif_peak_Y, cftor

def DF(X, Y, Z, N, fs): ##多轴主峰最大amplitude
    fft_peak_x_main = fft_domain(X,N,fs)
    fft_peak_y_main = fft_domain(Y,N,fs)
    fft_peak_z_main = fft_domain(Z,N,fs)
    return max(fft_peak_x_main[12], fft_peak_y_main[12], fft_peak_z_main[12])

def psd_domain(data,N,fs):  ##频域图峰值细节
    p_values, psd_values = get_psd_values(data, N, fs)
    damp, dmean, dmax, dstd, dvar, dentr, log_energy_value, signal_magnitude_area, interq, skew, kurt = base(psd_values)
    drms = np.sqrt((np.square(psd_values).mean()))  # rms

    signal_min = np.nanpercentile(psd_values, 5)
    signal_max = np.nanpercentile(psd_values, 95)
    mph = signal_min + (signal_max - signal_min) / len(psd_values)  # set minimum peak height
    peaks, _ = find_peaks(psd_values, prominence=mph)

    peak_save = psd_values[peaks].argsort()[::-1][:2]
    peak_x = p_values[peaks[peak_save]]   ## 峰值前5的极值点横纵坐标之和
    peak_y = psd_values[peaks[peak_save]]
    peak_x = np.pad(peak_x, (0, 2 - len(peak_x)), 'constant', constant_values=0)
    peak_y = np.pad(peak_y, (0, 2 - len(peak_y)), 'constant', constant_values=0)
    # 主峰
    peak_main_X = peak_x[0]
    peak_main_Y = peak_y[0]
    # 次峰
    peak_sub_X = peak_x[1]
    peak_sub_Y = peak_y[1]
    #主峰次峰差距
    dif_peak_X = peak_sub_X - peak_main_X
    dif_peak_Y =peak_main_Y - peak_sub_Y
    # 波峰因数Crest factor(cft or)
    cftor = peak_main_Y / drms * 1.0
    return damp, dmean, dmax, dstd, dvar, dentr, log_energy_value, signal_magnitude_area, interq, skew, kurt, drms, peak_main_X, peak_main_Y, peak_sub_X, peak_sub_Y, dif_peak_X, dif_peak_Y, cftor

def PSDEnergy_XYZ(X, Y, Z, N, fs): ##多轴功率谱密度能量值的和
    X_p, X_psd = get_psd_values(X, N, fs)
    Y_p, Y_psd = get_psd_values(Y, N, fs)
    Z_p, Z_psd = get_psd_values(Z, N, fs)
    x_energy = simps(X_psd, X_p, dx=0.001)
    y_energy = simps(Y_psd, Y_p, dx=0.001)
    z_energy = simps(Z_psd, Z_p, dx=0.001)
    return x_energy+y_energy+z_energy

def autocorr_domain(data,N,fs):  ##自相关图峰值细节
    a_values, autocorr_values = get_autocorr_values(data, N, fs)
    damp, dmean, dmax, dstd, dvar, dentr, log_energy_value, signal_magnitude_area, interq, skew, kurt = base(autocorr_values)
    drms = np.sqrt((np.square(autocorr_values).mean()))  # rms

    signal_min = np.nanpercentile(autocorr_values, 5)
    signal_max = np.nanpercentile(autocorr_values, 95)
    mph = signal_min + (signal_max - signal_min) / len(autocorr_values)  # set minimum peak height
    peaks, _ = find_peaks(autocorr_values, prominence=mph)

    peak_save = autocorr_values[peaks].argsort()[::-1][:2]
    peak_x = a_values[peaks[peak_save]]   ## 峰值前5的极值点横纵坐标之和
    peak_y = autocorr_values[peaks[peak_save]]
    peak_x = np.pad(peak_x, (0, 2 - len(peak_x)), 'constant', constant_values=0)
    peak_y = np.pad(peak_y, (0, 2 - len(peak_y)), 'constant', constant_values=0)
    # 主峰
    peak_main_X = peak_x[0]
    peak_main_Y = peak_y[0]
    # 次峰
    peak_sub_X = peak_x[1]
    peak_sub_Y = peak_y[1]
    #主峰次峰差距
    dif_peak_X = peak_sub_X - peak_main_X
    dif_peak_Y =peak_main_Y - peak_sub_Y
    # 波峰因数Crest factor(cft or)
    cftor = peak_main_Y / drms * 1.0
    return damp, dmean, dmax, dstd, dvar, dentr, log_energy_value, signal_magnitude_area, interq, skew, kurt, drms, peak_main_X, peak_main_Y, peak_sub_X, peak_sub_Y, dif_peak_X, dif_peak_Y, cftor

def featureExtract(x, y, z, ACCW2, windowsize, overlapping, frequency):
    N = windowsize
    fs = frequency
    f_s = frequency
    data = ACCW2
    i = 0
    j = 0
    
    xmean = x.mean()
    xvar = x.var()
    ymean = y.mean()
    yvar = y.var()
    zmean = z.mean()
    zvar = z.var()
    amean = ACCW2.mean()
    avar = ACCW2.var()

    #peak_detail
    z_filter = np.array(z).flatten()
    signal_min = np.nanpercentile(z_filter,5)
    signal_max = np.nanpercentile(z_filter, 97)
    mph = signal_max + (signal_max - signal_min)/len(z_filter)#set minimum peak height
    peaks_t, _ = find_peaks(z_filter,prominence=mph,distance=120) 
    peak_num=len(peaks_t) ##z轴peak数量
    t_value = np.arange(len(z_filter))
    t_peakmax = np.argsort(z_filter[peaks_t])[-1]
    sampley=sampEn(t_value[peaks_t],3,500)#z轴peak y样本熵
    samplex=sampEn(z_filter[peaks_t],3,1)#z轴peak x样本熵
    infory=infor(t_value[peaks_t])#z轴peak x信息熵
    inforx=infor(z_filter[peaks_t])#z轴peak y信息熵
    t_peakmax_Y = z_filter[peaks_t[t_peakmax]]
    t_peak_y = z_filter[peaks_t]
    dyski_num=len(t_peak_y[(t_peak_y< t_peakmax_Y-mph)])##z轴异常peak数量
    a_values, autocorr_values = get_autocorr_values(z_filter, N, fs)
    peaks4, _ = find_peaks(autocorr_values)
    auto_peak_num=len(peaks4)
    index_peakmax = np.argsort(autocorr_values[peaks4])[-1]
    auto_y = autocorr_values[peaks4[index_peakmax]] #全局自相关系数
    

    peaks_normal = np.zeros([len(data), 1], dtype=float)
    fea_sampley = np.zeros([len(data), 1], dtype=float)
    fea_samplex = np.zeros([len(data), 1], dtype=float)
    fea_infory = np.zeros([len(data), 1], dtype=float)
    fea_inforx = np.zeros([len(data), 1], dtype=float)
    peaks_abnormal = np.zeros([len(data), 1], dtype=float)
    fea_autoy = np.zeros([len(data), 1], dtype=float)
    meandif = np.zeros([len(data), 1], dtype=float)
    vardif = np.zeros([len(data), 1], dtype=float)
    fea_auto_num = np.zeros([len(data), 1], dtype=float)
    time_domain1 = np.zeros([len(data), 14], dtype=float)
    time_domain2 = np.zeros([len(data), 14], dtype=float)
    time_domain3 = np.zeros([len(data), 14], dtype=float)
    time_domain4 = np.zeros([len(data), 14], dtype=float)
    time_axiscof = np.zeros([len(data), 6], dtype=float)
    fre_domain1 = np.zeros([len(data), 19], dtype=float)
    fre_domain2 = np.zeros([len(data), 19], dtype=float)
    fre_domain3 = np.zeros([len(data), 19], dtype=float)
    fre_domain4 = np.zeros([len(data), 19], dtype=float)
    fre_df = np.zeros(len(data), dtype=float)
    fft_peak_a = np.zeros([len(data), 5], dtype=float)
    psd_domain1 = np.zeros([len(data), 19], dtype=float)
    psd_domain2 = np.zeros([len(data), 19], dtype=float)
    psd_domain3 = np.zeros([len(data), 19], dtype=float)
    psd_domain4 = np.zeros([len(data), 19], dtype=float)
    PSDEnergy_XYZ = np.zeros(len(data), dtype=float)
    spectrumConcentration = np.zeros(len(data), dtype=float)
    psd_peak_a = np.zeros([len(data), 5], dtype=float)
    autoco_domain1 = np.zeros([len(data), 19], dtype=float)
    autoco_domain2 = np.zeros([len(data), 19], dtype=float)
    autoco_domain3 = np.zeros([len(data), 19], dtype=float)
    autoco_domain4 = np.zeros([len(data), 19], dtype=float)
    autocorr_peak_a = np.zeros([len(data), 5], dtype=float)
    print("len_data-windowsize", len(data) - windowsize)


    while (i < len(data) - windowsize):
        data1 = x[i:i + windowsize]
        data2 = y[i:i + windowsize]
        data3 = z[i:i + windowsize]
        data4 = ACCW2[i:i + windowsize]
        data1 = data1.values  # dataframe转numpy数组
        data2 = data2.values
        data3 = data3.values
        data4 = data4.values
        #***************************features(与data4无关)*******************
        peaks_normal[j,:] = peak_num #z轴0.2-2滤波后的波峰个数
        peaks_abnormal[j,:] = dyski_num  #z轴异常波峰个数
        fea_sampley[j,:] = sampley
        fea_samplex[j,:] = samplex
        fea_infory[j,:] = infory
        fea_inforx[j,:] = inforx
        fea_autoy[j,:] = auto_y
        fea_auto_num[j,:]=auto_peak_num
        meandif[j,:] = meandif[j,:]+abs(data1.mean()-xmean)+abs(data2.mean()-ymean)+abs(data3.mean()-zmean)
        vardif[j,:] = vardif[j,:]+abs(data1.var()-xvar)+abs(data2.var()-yvar)+abs(data3.var()-zvar)
        
        #***************************************features******************************
        time_domain1[j,:] = time_domain(data1)#14
        time_domain2[j,:] = time_domain(data2)
        time_domain3[j,:] = time_domain(data3)
        time_domain4[j,:] = time_domain(data4)
        time_axiscof[j, :] = corrcoef(data1,data2,data3,data4)
        fre_domain1[j,:] = fft_domain(data1, N, fs)#19
        fre_domain2[j,:] = fft_domain(data2, N, fs)
        fre_domain3[j,:] = fft_domain(data3, N, fs)
        fre_domain4[j,:] = fft_domain(data4, N, fs)
        fre_df[j] = DF(data1,data2,data3,N,fs)
        fft_peak_a[j, :] = fft_peak_xy(data4, N, fs, peak_num=5)
        psd_domain1[j,:] = psd_domain(data1, N, fs)#19
        psd_domain2[j,:] = psd_domain(data2, N, fs)
        psd_domain3[j,:] = psd_domain(data3, N, fs)
        psd_domain4[j,:] = psd_domain(data4, N, fs)
        PSDEnergy_XYZ[j] = PSDEnergy_XYZ(data1,data2,data3,N,fs)
        spectrumConcentration[j] = spectrumConcentration(data1,data2,data3,N,fs)
        psd_peak_a[j,:]  = psd_peak_xy(data4, N, fs, peak_num=5)

        data1234 = np.c_[data1,data2,data3,data4]
        data1234 = StandardScaler().fit_transform(data1234)#19
        data1 = data1234[:,0]
        data2 = data1234[:,1]
        data3 = data1234[:,2]
        data4 = data1234[:,3]
        autoco_domain1[j,:] = autocorr_domain(data1, N, fs)
        autoco_domain2[j,:] = autocorr_domain(data2, N, fs)
        autoco_domain3[j,:] = autocorr_domain(data3, N, fs)
        autoco_domain4[j,:] = autocorr_domain(data4, N, fs)
        autocorr_peak_a[j,:]  = auto_peak_xy(data4, N, fs, peak_num=5)

        i = i + windowsize // overlapping - 1
        j = j + 1
    
    fea_whole = np.c_[peaks_normal, fea_sampley, fea_samplex, fea_infory, fea_inforx, peaks_abnormal,fea_autoy,fea_auto_num,meandif,vardif]
    f1 = np.c_[time_axiscof,fft_peak_a,psd_peak_a,autocorr_peak_a,fre_df,PSDEnergy_XYZ,spectrumConcentration]
    # 20，25，26，24
    fx = np.c_[time_domain1,fre_domain1,psd_domain1, autoco_domain1] 
    fy = np.c_[time_domain2,fre_domain2,psd_domain2, autoco_domain2] 
    fz = np.c_[time_domain3,fre_domain3,psd_domain3, autoco_domain3] 
    fa = np.c_[time_domain4,fre_domain4,psd_domain4, autoco_domain4] 
    
    Feat = np.c_[fea_whole,f1, fx, fy, fz, fa]

    print(Feat.shape)
    print(Feat)
    Feat2 = np.zeros((j, Feat.shape[1]))  # 后一个参数为特征种类加一 28 38 16 45
    Feat2[0:j, :] = Feat[0:j, :]
    Feat2 = pd.DataFrame(Feat2)
    return Feat2

def FeatureExtractWithProcess(pd_num, activity_num, data_path, side, fea_column, window_size, overlapping_rate, frequency):
    Feature  = pd.DataFrame()
    for pdn in range(1,pd_num+1,1):
        for acn in range(1,activity_num+1,1):
            #select one side data  
            filefullpath=data_path+"person{}/{}_session{}_{}.csv".format(pdn,pdn,acn,side)
            if not os.path.exists(filefullpath):
                continue
            data = pd.read_csv(filefullpath,header=0)
            data = data.drop(0)
            
            new_column_labels = {"Accel_WR_X_CAL": "wr_acc_x", "Accel_WR_Y_CAL": "wr_acc_y", "Accel_WR_Z_CAL": "wr_acc_z", "Gyro_X_CAL": "gyro_x", "Gyro_Y_CAL": "gyro_y", "Gyro_Z_CAL": "gyro_z"}
            data = data.rename(columns=new_column_labels)
            
            for col in ["wr_acc_x","wr_acc_y","wr_acc_z"]: 
                data[col] = data[col].astype('float64')
            for col in ["gyro_x","gyro_y","gyro_z"]:     
                data[col] = data[col].astype('float64')
            data["acca"]=np.sqrt(data["wr_acc_x"]*data["wr_acc_x"]+data["wr_acc_y"]*data["wr_acc_y"]+data["wr_acc_z"]*data["wr_acc_z"])
            data["gyroa"]=np.sqrt(data["gyro_x"]*data["gyro_x"]+data["gyro_y"]*data["gyro_y"]+data["gyro_z"]*data["gyro_z"])
            
            accdata=data[["wr_acc_x","wr_acc_y","wr_acc_z","acca"]]
            gyrodata=data[["gyro_x","gyro_y","gyro_z","gyroa"]]
            
            accdata = accdata.values
            gyrodata = gyrodata.values
            #输入需要为numpy数组
            accdata = StandardScaler().fit_transform(accdata)
            gyrodata = StandardScaler().fit_transform(gyrodata)
            databand_acc = accdata.copy()
            databand_gyro = gyrodata.copy()
            for k in range(0,4): 
                databand_acc[:,k] = butter_bandpass_filter(accdata[:,k],0.3, 17, 200, order = 3)
                databand_gyro[:,k] = butter_bandpass_filter(gyrodata[:,k],0.3, 17, 200, order = 3)
            
            databand_gyro[:,2] = butter_bandpass_filter(accdata[:,2],0.3, 3, 200, order = 3)
            databand_gyro[:,2] = butter_bandpass_filter(gyrodata[:,2],0.3, 3, 200, order = 3)
            databand_acc = pd.DataFrame(databand_acc)
            databand_gyro = pd.DataFrame(databand_gyro)
            databand = pd.concat([databand_acc,databand_gyro],axis = 1)
            
            datax = databand.iloc[:,0]
            datay = databand.iloc[:,1]
            dataz = databand.iloc[:,2]
            acca = databand.iloc[:,3]
            
            feature1 = featureExtract(datax, datay, dataz, acca, window_size, overlapping_rate, frequency)

            feature1.columns = fea_column

            feature1["PatientID"]=pdn
            feature1["activity_label"]=acn
            print("pdn{}.format()",pdn)
            print("acn{}.format()",acn)
            if((pdn==1)&(acn==1)):
                Feature  = feature1
            else:
                Feature = pd.concat([Feature,feature1],axis=0)
                
    return Feature