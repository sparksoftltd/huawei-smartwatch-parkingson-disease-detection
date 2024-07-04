
from sklearn.preprocessing import StandardScaler
import os
from scipy.signal import butter, lfilter
from scipy.signal import welch
from scipy.signal import find_peaks
from scipy.fftpack import fft,fftshift
from scipy.special import entr
import pandas as pd
from scipy.integrate import simps
from scipy.integrate import simpson

def sampEn(data, N, r): #多窗口样本熵
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

def infor(data):
    a = pd.value_counts(data) / len(data)
    return sum(np.log2(a) * a * (-1))


def get_psd_values(y_values, N, fs):
    f_values, psd_values = welch(y_values, fs)
    return f_values, psd_values


def get_fft_values(y_values, N, fs):  # N为采样点数，f_s是采样频率，返回f_values希望的频率区间, fft_values真实幅值
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
    temp_arr = f_values[peaks[peak_save]] + fft_values[peaks[peak_save]]
    fft_peak_xy = np.pad(temp_arr, (0, peak_num - len(temp_arr)), 'constant', constant_values=0)
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
                              constant_values=0)
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
                                   constant_values=0)
    return autocorr_peak_xy


import math
import numpy as np
from numpy import array, sign, zeros
from scipy.interpolate import interp1d
import scipy.signal

def envelope_extraction(signal):
    s = signal.astype(float )
    q_u = np.zeros(s.shape)
    q_l =  np.zeros(s.shape)

    #Prepend the first value of (s) to the interpolating values. This forces the model to use the same starting point for both the upper and lower envelope models.
    u_x = [0,]
    u_y = [s[0],]

    l_x = [0,]
    l_y = [s[0],]

    #Detect peaks and troughs and mark their location in u_x,u_y,l_x,l_y respectively.

    for k in range(1,len(s)-1):
        if (sign(s[k]-s[k-1])==1) and (sign(s[k]-s[k+1])==1):
            u_x.append(k)
            u_y.append(s[k])

        if (sign(s[k]-s[k-1])==-1) and ((sign(s[k]-s[k+1]))==-1):
            l_x.append(k)
            l_y.append(s[k])

    u_x.append(len(s) - 1)
    u_y.append(s[-1])

    l_x.append(len(s) - 1)
    l_y.append(s[-1])

    upper_envelope_y = np.zeros(len(signal))
    lower_envelope_y = np.zeros(len(signal))

    upper_envelope_y[0] = u_y[0]
    upper_envelope_y[-1] = u_y[-1]
    lower_envelope_y[0] = l_y[0]
    lower_envelope_y[-1] = l_y[-1]

    k, b = general_equation(u_x[0], u_y[0], u_x[1], u_y[1])
    for e in range(1, len(upper_envelope_y) - 1):

        if e not in u_x:
            v = k * e + b
            upper_envelope_y[e] = v
        else:
            idx = u_x.index(e)
            upper_envelope_y[e] = u_y[idx]
            last_idx = u_x.index(e)
            next_idx = u_x.index(e) + 1
            k, b = general_equation(u_x[last_idx], u_y[last_idx], u_x[next_idx], u_y[next_idx])

    last_idx, next_idx = 0, 0
    k, b = general_equation(l_x[0], l_y[0], l_x[1], l_y[1])
    for e in range(1, len(lower_envelope_y) - 1):

        if e not in l_x:
            v = k * e + b
            lower_envelope_y[e] = v
        else:
            idx = l_x.index(e)
            lower_envelope_y[e] = l_y[idx]
            last_idx = l_x.index(e)
            next_idx = l_x.index(e) + 1
            k, b = general_equation(l_x[last_idx], l_y[last_idx], l_x[next_idx], l_y[next_idx])

    return upper_envelope_y, lower_envelope_y


def general_equation(first_x, first_y, second_x, second_y):
    A = second_y - first_y
    B = first_x - second_x
    C = second_x * first_y - first_x * second_y
    k = -1 * A / B
    b = -1 * C / B
    return k, b

def mAmp(data):
    L = np.size(data, 0)
    upper_envolope, low_envolope = envelope_extraction(data)
    mAmp = np.sum(upper_envolope-low_envolope)/L*1.0
    return mAmp

def sampEn(data, N, r):
    L = len(data)
    B = 0.0
    A = 0.0
    xmi = np.array([data[i: i + N] for i in range(L - N)])
    xmj = np.array([data[i: i + N] for i in range(L - N + 1)])
    B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])
    N += 1
    xm = np.array([data[i: i + N] for i in range(L - N + 1)])
    A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])
    return -np.log(A / B)

def base(data):
    damp = mAmp(data)
    dmean = data.mean()
    dmax = data.max()
    dstd = data.std()
    dvar = data.var()
    # entropy
    dentr = entr(abs(data)).sum(axis=0) / np.log(10)
    log_energy_value = np.log10(data ** 2).sum(axis=0)
    time = np.arange(data.shape[0])
    signal_magnitude_area = simpson(data,time)
    per25 = np.nanpercentile(data, 25)
    per75 = np.nanpercentile(data, 75)
    interq = per75 - per25
    seriesdata = pd.Series(data)
    skew = seriesdata.skew()
    kurt = seriesdata.kurt()
    return damp, dmean, dmax, dstd, dvar, dentr, log_energy_value, signal_magnitude_area, interq, skew, kurt

def time_domain(data):
    damp, dmean, dmax, dstd, dvar, dentr, log_energy_value, signal_magnitude_area, interq, skew, kurt = base(data)
    drms = np.sqrt((np.square(data).mean()))
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

def fft_domain(data,N,fs):
    f_values, fft_values = get_fft_values(data, N, fs)
    damp, dmean, dmax, dstd, dvar, dentr, log_energy_value, signal_magnitude_area, interq, skew, kurt = base(fft_values)
    drms = np.sqrt((np.square(fft_values).mean()))
    signal_min = np.nanpercentile(fft_values, 5)
    signal_max = np.nanpercentile(fft_values, 95)
    mph = signal_min + (signal_max - signal_min) / len(fft_values)  # set minimum peak height
    peaks, _ = find_peaks(fft_values, prominence=mph)
    peak_save = fft_values[peaks].argsort()[::-1][:2]
    peak_x = f_values[peaks[peak_save]]
    peak_y = fft_values[peaks[peak_save]]
    peak_x = np.pad(peak_x, (0, 2 - len(peak_x)), 'constant', constant_values=0)
    peak_y = np.pad(peak_y, (0, 2 - len(peak_y)), 'constant', constant_values=0)
    peak_main_X = peak_x[0]
    peak_main_Y = peak_y[0]
    peak_sub_X = peak_x[1]
    peak_sub_Y = peak_y[1]
    dif_peak_X = peak_sub_X - peak_main_X
    dif_peak_Y =peak_main_Y - peak_sub_Y
    cftor = peak_main_Y / drms * 1.0
    return damp, dmean, dmax, dstd, dvar, dentr, log_energy_value, signal_magnitude_area, interq, skew, kurt, peak_main_X, peak_main_Y, peak_sub_X, peak_sub_Y, dif_peak_X, drms, dif_peak_Y, cftor

def DF(X, Y, Z, N, fs):
    fft_peak_x_main = fft_domain(X,N,fs)
    fft_peak_y_main = fft_domain(Y,N,fs)
    fft_peak_z_main = fft_domain(Z,N,fs)
    return max(fft_peak_x_main[12], fft_peak_y_main[12], fft_peak_z_main[12])

def psd_domain(data,N,fs):
    p_values, psd_values = get_psd_values(data, N, fs)
    damp, dmean, dmax, dstd, dvar, dentr, log_energy_value, signal_magnitude_area, interq, skew, kurt = base(psd_values)
    drms = np.sqrt((np.square(psd_values).mean()))  # rms

    signal_min = np.nanpercentile(psd_values, 5)
    signal_max = np.nanpercentile(psd_values, 95)
    mph = signal_min + (signal_max - signal_min) / len(psd_values)  # set minimum peak height
    peaks, _ = find_peaks(psd_values, prominence=mph)

    peak_save = psd_values[peaks].argsort()[::-1][:2]
    peak_x = p_values[peaks[peak_save]]
    peak_y = psd_values[peaks[peak_save]]
    peak_x = np.pad(peak_x, (0, 2 - len(peak_x)), 'constant', constant_values=0)
    peak_y = np.pad(peak_y, (0, 2 - len(peak_y)), 'constant', constant_values=0)
    peak_main_X = peak_x[0]
    peak_main_Y = peak_y[0]
    peak_sub_X = peak_x[1]
    peak_sub_Y = peak_y[1]
    dif_peak_X = peak_sub_X - peak_main_X
    dif_peak_Y =peak_main_Y - peak_sub_Y
    cftor = peak_main_Y / drms * 1.0
    return damp, dmean, dmax, dstd, dvar, dentr, log_energy_value, signal_magnitude_area, interq, skew, kurt, drms, peak_main_X, peak_main_Y, peak_sub_X, peak_sub_Y, dif_peak_X, dif_peak_Y, cftor

def psdEnergy_XYZ(X, Y, Z, N, fs):
    X_p, X_psd = get_psd_values(X, N, fs)
    Y_p, Y_psd = get_psd_values(Y, N, fs)
    Z_p, Z_psd = get_psd_values(Z, N, fs)
    x_energy = simps(X_psd, X_p, dx=0.001)
    y_energy = simps(Y_psd, Y_p, dx=0.001)
    z_energy = simps(Z_psd, Z_p, dx=0.001)
    return x_energy+y_energy+z_energy

def spectrumConcentration(X, Y, Z, N, fs):
    DF_x = fft_domain(X, N, fs)
    DF_y = fft_domain(Y, N, fs)
    DF_z = fft_domain(Z, N, fs)
    spectrumDistribution = DF_x[12] + DF_y[12] + DF_z[12]
    return spectrumDistribution / psdEnergy_XYZ(X, Y, Z, N, fs) *1.0

def autocorr_domain(data,N,fs):
    a_values, autocorr_values = get_autocorr_values(data, N, fs)
    damp, dmean, dmax, dstd, dvar, dentr, log_energy_value, signal_magnitude_area, interq, skew, kurt = base(autocorr_values)
    drms = np.sqrt((np.square(autocorr_values).mean()))  # rms
    signal_min = np.nanpercentile(autocorr_values, 5)
    signal_max = np.nanpercentile(autocorr_values, 95)
    mph = signal_min + (signal_max - signal_min) / len(autocorr_values)  # set minimum peak height
    peaks, _ = find_peaks(autocorr_values, prominence=mph)
    peak_save = autocorr_values[peaks].argsort()[::-1][:2]
    peak_x = a_values[peaks[peak_save]]
    peak_y = autocorr_values[peaks[peak_save]]
    peak_x = np.pad(peak_x, (0, 2 - len(peak_x)), 'constant', constant_values=0)
    peak_y = np.pad(peak_y, (0, 2 - len(peak_y)), 'constant', constant_values=0)
    peak_main_X = peak_x[0]
    peak_main_Y = peak_y[0]
    peak_sub_X = peak_x[1]
    peak_sub_Y = peak_y[1]
    dif_peak_X = peak_sub_X - peak_main_X
    dif_peak_Y =peak_main_Y - peak_sub_Y
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

    # peak_detail
    z_filter = np.array(z).flatten()
    signal_min = np.nanpercentile(z_filter, 5)
    signal_max = np.nanpercentile(z_filter, 97)
    mph = signal_max + (signal_max - signal_min) / len(z_filter)  # set minimum peak height
    peaks_t, _ = find_peaks(z_filter, prominence=mph, distance=120)
    peak_num = len(peaks_t)
    t_value = np.arange(len(z_filter))
    t_peakmax = np.argsort(z_filter[peaks_t])[-1]
    sampley = sampEn(t_value[peaks_t], 3, 500)
    samplex = sampEn(z_filter[peaks_t], 3, 1)
    infory = infor(t_value[peaks_t])
    inforx = infor(z_filter[peaks_t])
    t_peakmax_Y = z_filter[peaks_t[t_peakmax]]
    t_peak_y = z_filter[peaks_t]
    dyski_num = len(t_peak_y[(t_peak_y < t_peakmax_Y - mph)])
    a_values, autocorr_values = get_autocorr_values(z_filter, N, fs)
    peaks4, _ = find_peaks(autocorr_values)
    auto_peak_num = len(peaks4)
    index_peakmax = np.argsort(autocorr_values[peaks4])[-1]
    auto_y = autocorr_values[peaks4[index_peakmax]]  # 全局自相关系数

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
    SpectrumConcentration = np.zeros(len(data), dtype=float)
    psd_peak_a = np.zeros([len(data), 5], dtype=float)
    autoco_domain1 = np.zeros([len(data), 19], dtype=float)
    autoco_domain2 = np.zeros([len(data), 19], dtype=float)
    autoco_domain3 = np.zeros([len(data), 19], dtype=float)
    autoco_domain4 = np.zeros([len(data), 19], dtype=float)
    autocorr_peak_a = np.zeros([len(data), 5], dtype=float)
    print("len_data-windowsize", len(data) - windowsize)

    while (i < int(len(data) - windowsize)):
        data1 = x[int(i):int(i + windowsize)]
        data2 = y[int(i):int(i + windowsize)]
        data3 = z[int(i):int(i + windowsize)]
        data4 = ACCW2[int(i):int(i + windowsize)]
        data1 = data1.values  # dataframe转numpy数组
        data2 = data2.values
        data3 = data3.values
        data4 = data4.values
        # ***************************features(NON DATA4)*******************
        peaks_normal[j, :] = peak_num  # z axis 0.2-2 filtering
        peaks_abnormal[j, :] = dyski_num  # z axis abnormal
        fea_sampley[j, :] = sampley
        fea_samplex[j, :] = samplex
        fea_infory[j, :] = infory
        fea_inforx[j, :] = inforx
        fea_autoy[j, :] = auto_y
        fea_auto_num[j, :] = auto_peak_num
        meandif[j, :] = meandif[j, :] + abs(data1.mean() - xmean) + abs(data2.mean() - ymean) + abs(
            data3.mean() - zmean)
        vardif[j, :] = vardif[j, :] + abs(data1.var() - xvar) + abs(data2.var() - yvar) + abs(data3.var() - zvar)

        # ***************************************features******************************
        time_domain1[j, :] = time_domain(data1)  # 14
        time_domain2[j, :] = time_domain(data2)
        time_domain3[j, :] = time_domain(data3)
        time_domain4[j, :] = time_domain(data4)
        time_axiscof[j, :] = corrcoef(data1, data2, data3, data4)
        fre_domain1[j, :] = fft_domain(data1, N, fs)  # 19
        fre_domain2[j, :] = fft_domain(data2, N, fs)
        fre_domain3[j, :] = fft_domain(data3, N, fs)
        fre_domain4[j, :] = fft_domain(data4, N, fs)
        fre_df[j] = DF(data1, data2, data3, N, fs)
        fft_peak_a[j, :] = fft_peak_xy(data4, N, fs, peak_num=5)
        psd_domain1[j, :] = psd_domain(data1, N, fs)  # 19
        psd_domain2[j, :] = psd_domain(data2, N, fs)
        psd_domain3[j, :] = psd_domain(data3, N, fs)
        psd_domain4[j, :] = psd_domain(data4, N, fs)
        PSDEnergy_XYZ[j] = psdEnergy_XYZ(data1, data2, data3, N, fs)
        SpectrumConcentration[j] = spectrumConcentration(data1, data2, data3, N, fs)
        psd_peak_a[j, :] = psd_peak_xy(data4, N, fs, peak_num=5)

        data1234 = np.c_[data1, data2, data3, data4]
        data1234 = StandardScaler().fit_transform(data1234)  # 19
        data1 = data1234[:, 0]
        data2 = data1234[:, 1]
        data3 = data1234[:, 2]
        data4 = data1234[:, 3]
        autoco_domain1[j, :] = autocorr_domain(data1, N, fs)
        autoco_domain2[j, :] = autocorr_domain(data2, N, fs)
        autoco_domain3[j, :] = autocorr_domain(data3, N, fs)
        autoco_domain4[j, :] = autocorr_domain(data4, N, fs)
        autocorr_peak_a[j, :] = auto_peak_xy(data4, N, fs, peak_num=5)

        i = i + windowsize // overlapping - 1
        j = j + 1

    fea_whole = np.c_[
        peaks_normal, fea_sampley, fea_samplex, fea_infory, fea_inforx, peaks_abnormal, fea_autoy, fea_auto_num, meandif, vardif]
    f1 = np.c_[time_axiscof, fft_peak_a, psd_peak_a, autocorr_peak_a, fre_df, PSDEnergy_XYZ, SpectrumConcentration]
    # 20，25，26，24
    fx = np.c_[time_domain1, fre_domain1, psd_domain1, autoco_domain1]
    fy = np.c_[time_domain2, fre_domain2, psd_domain2, autoco_domain2]
    fz = np.c_[time_domain3, fre_domain3, psd_domain3, autoco_domain3]
    fa = np.c_[time_domain4, fre_domain4, psd_domain4, autoco_domain4]

    Feat = np.c_[fea_whole, f1, fx, fy, fz, fa]

    print(Feat.shape)
    print(Feat)
    Feat2 = np.zeros((j, Feat.shape[1]))  # feature number 28 38 16 45
    Feat2[0:j, :] = Feat[0:j, :]
    Feat2 = pd.DataFrame(Feat2)
    return Feat2


def FeatureExtractWithProcess(pd_num, activity_num, data_path, side, window_size, overlapping_rate,
                              frequency, std_mod):
    Feature = pd.DataFrame()
    for pdn in range(1, pd_num + 1, 1):
        for acn in range(1, activity_num + 1, 1):
            # select one side data
            filefullpath = data_path + "person{}/{}_session{}_{}.csv".format(pdn, pdn, acn, side)
            if not os.path.exists(filefullpath):
                continue
            data = pd.read_csv(filefullpath, header=0)
            data = data.drop(0)

            new_column_labels = {"Accel_WR_X_CAL": "wr_acc_x", "Accel_WR_Y_CAL": "wr_acc_y",
                                 "Accel_WR_Z_CAL": "wr_acc_z", "Gyro_X_CAL": "gyro_x", "Gyro_Y_CAL": "gyro_y",
                                 "Gyro_Z_CAL": "gyro_z"}

            data = data.rename(columns=new_column_labels)

            for col in ["wr_acc_x", "wr_acc_y", "wr_acc_z"]:
                data[col] = data[col].astype('float64')
            for col in ["gyro_x", "gyro_y", "gyro_z"]:
                data[col] = data[col].astype('float64')
            data["acca"] = np.sqrt(
                data["wr_acc_x"] * data["wr_acc_x"] + data["wr_acc_y"] * data["wr_acc_y"] + data["wr_acc_z"] * data[
                    "wr_acc_z"])
            data["gyroa"] = np.sqrt(
                data["gyro_x"] * data["gyro_x"] + data["gyro_y"] * data["gyro_y"] + data["gyro_z"] * data["gyro_z"])

            accdata = data[["wr_acc_x", "wr_acc_y", "wr_acc_z", "acca"]]
            gyrodata = data[["gyro_x", "gyro_y", "gyro_z", "gyroa"]]

            accdata = accdata.values
            gyrodata = gyrodata.values
            # numpy input
            accdata = StandardScaler().fit_transform(accdata)
            gyrodata = StandardScaler().fit_transform(gyrodata)
            databand_acc = accdata.copy()
            databand_gyro = gyrodata.copy()
            for k in range(0, 4):
                databand_acc[:, k] = butter_bandpass_filter(accdata[:, k], 0.3, 17, 200, order=3)
                databand_gyro[:, k] = butter_bandpass_filter(gyrodata[:, k], 0.3, 17, 200, order=3)

            databand_gyro[:, 2] = butter_bandpass_filter(accdata[:, 2], 0.3, 3, 200, order=3)
            databand_gyro[:, 2] = butter_bandpass_filter(gyrodata[:, 2], 0.3, 3, 200, order=3)
            databand_acc = pd.DataFrame(databand_acc)
            databand_gyro = pd.DataFrame(databand_gyro)
            databand = pd.concat([databand_acc, databand_gyro], axis=1)

            datax = databand.iloc[:, 0]
            datay = databand.iloc[:, 1]
            dataz = databand.iloc[:, 2]
            acca = databand.iloc[:, 3]

            feature1 = featureExtract(datax, datay, dataz, acca, window_size, overlapping_rate, frequency)

            feature1["PatientID"] = pdn
            feature1["activity_label"] = acn
            print("pdn{}.format()", pdn)
            print("acn{}.format()", acn)
            if ((pdn == 1) & (acn == 1)):
                Feature = feature1
            else:
                Feature = pd.concat([Feature, feature1], axis=0)

    if(std_mod == True):
        for acn in range(1, activity_num + 1, 1):
            feature_select = Feature[Feature['activity_label'] == acn]
            # 需要排除的列
            exclude_columns = ['PatientID', 'activity_label']
            # 标准化除去指定列之外的所有列
            scaler = StandardScaler()
            columns_to_scale = [col for col in feature_select.columns if col not in exclude_columns]
            X_data = feature_select[columns_to_scale]
            X_label = feature_select[exclude_columns]
            # 标准化
            X_data = (X_data - X_data.mean()) / (X_data.std())
            feature_data = pd.concat([X_data, X_label], axis=1)
            feature_data = feature_data.reset_index(drop=True)
    return feature_data