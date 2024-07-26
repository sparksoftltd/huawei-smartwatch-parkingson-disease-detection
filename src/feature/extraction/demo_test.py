#%%feature extraction from PD
import pandas as pd
from utils import pd_utils
from utils import utils_v1
#%
# get feature
#data_path1= "/Users/macpro/PycharmProjects/huawei/datasets/pd1/"
data_path= "../../../input/activity/step_1_feature_extraction/raw/"
Feature2 = pd.DataFrame()

fea_column = ['acc_peaks_normal', 'acc_peaks_abnormal', 'acc_fea_autoy', 'acc_fea_auto_num', 'acc_t_xyCor', 'acc_t_xzCor', 'acc_t_xaCor', 'acc_t_yzCor', 'acc_t_yaCor', 'acc_t_zaCor', 'acc_f_peakXY1', 'acc_f_peakXY2', 'acc_f_peakXY3', 'acc_f_peakXY4', 'acc_f_peakXY5', 'acc_p_peakXY1', 'acc_p_peakXY2', 'acc_p_peakXY3', 'acc_p_peakXY4', 'acc_p_peakXY5', 'acc_a_peakXY1', 'acc_a_peakXY2', 'acc_a_peakXY3', 'acc_a_peakXY4', 'acc_a_peakXY5', 'acc_t_amp_x', 'acc_t_mean_x', 'acc_t_max_x', 'acc_t_std_x', 'acc_t_var_x', 'acc_t_entr_x', 'acc_t_lgEnergy_x', 'acc_t_sma_x', 'acc_t_interq_x', 'acc_t_skew_x', 'acc_t_kurt_x', 'acc_t_rms_x', 'acc_f_amp_x', 'acc_f_mean_x', 'acc_f_max_x', 'acc_f_std_x', 'acc_f_var_x', 'acc_f_entr_x', 'acc_f_lgEnergy_x', 'acc_f_sma_x', 'acc_f_interq_x', 'acc_f_skew_x', 'acc_f_kurt_x', 'acc_f_rms_x', 'acc_p_amp_x', 'acc_p_mean_x', 'acc_p_max_x', 'acc_p_std_x', 'acc_p_var_x', 'acc_p_entr_x', 'acc_p_lgEnergy_x', 'acc_p_sma_x', 'acc_p_interq_x', 'acc_p_skew_x', 'acc_p_kurt_x', 'acc_p_rms_x', 'acc_a_amp_x', 'acc_a_mean_x', 'acc_a_max_x', 'acc_a_std_x', 'acc_a_var_x', 'acc_a_entr_x', 'acc_a_lgEnergy_x', 'acc_a_sma_x', 'acc_a_interq_x', 'acc_a_skew_x', 'acc_a_kurt_x', 'acc_a_rms_x', 'acc_a_mainY_x', 'acc_a_subY_x', 'acc_a_cftor_x', 'acc_t_amp_y', 'acc_t_mean_y', 'acc_t_max_y', 'acc_t_std_y', 'acc_t_var_y', 'acc_t_entr_y', 'acc_t_lgEnergy_y', 'acc_t_sma_y', 'acc_t_interq_y', 'acc_t_skew_y', 'acc_t_kurt_y', 'acc_t_rms_y', 'acc_f_amp_y', 'acc_f_mean_y', 'acc_f_max_y', 'acc_f_std_y', 'acc_f_var_y', 'acc_f_entr_y', 'acc_f_lgEnergy_y', 'acc_f_sma_y', 'acc_f_interq_y', 'acc_f_skew_y', 'acc_f_kurt_y', 'acc_f_rms_y', 'acc_p_amp_y', 'acc_p_mean_y', 'acc_p_max_y', 'acc_p_std_y', 'acc_p_var_y', 'acc_p_entr_y', 'acc_p_lgEnergy_y', 'acc_p_sma_y', 'acc_p_interq_y', 'acc_p_skew_y', 'acc_p_kurt_y', 'acc_p_rms_y', 'acc_a_amp_y', 'acc_a_mean_y', 'acc_a_max_y', 'acc_a_std_y', 'acc_a_var_y', 'acc_a_entr_y', 'acc_a_lgEnergy_y', 'acc_a_sma_y', 'acc_a_interq_y', 'acc_a_skew_y', 'acc_a_kurt_y', 'acc_a_rms_y', 'acc_a_mainY_y', 'acc_a_subY_y', 'acc_a_cftor_y', 'acc_t_amp_z', 'acc_t_mean_z', 'acc_t_max_z', 'acc_t_std_z', 'acc_t_var_z', 'acc_t_entr_z', 'acc_t_lgEnergy_z', 'acc_t_sma_z', 'acc_t_interq_z', 'acc_t_skew_z', 'acc_t_kurt_z', 'acc_t_rms_z', 'acc_f_amp_z', 'acc_f_mean_z', 'acc_f_max_z', 'acc_f_std_z', 'acc_f_var_z', 'acc_f_entr_z', 'acc_f_lgEnergy_z', 'acc_f_sma_z', 'acc_f_interq_z', 'acc_f_skew_z', 'acc_f_kurt_z', 'acc_f_rms_z', 'acc_p_amp_z', 'acc_p_mean_z', 'acc_p_max_z', 'acc_p_std_z', 'acc_p_var_z', 'acc_p_entr_z', 'acc_p_lgEnergy_z', 'acc_p_sma_z', 'acc_p_interq_z', 'acc_p_skew_z', 'acc_p_kurt_z', 'acc_p_rms_z', 'acc_a_amp_z', 'acc_a_mean_z', 'acc_a_max_z', 'acc_a_std_z', 'acc_a_var_z', 'acc_a_entr_z', 'acc_a_lgEnergy_z', 'acc_a_sma_z', 'acc_a_interq_z', 'acc_a_skew_z', 'acc_a_kurt_z', 'acc_a_rms_z', 'acc_a_mainY_z', 'acc_a_subY_z', 'acc_a_cftor_z', 'acc_t_amp_a', 'acc_t_mean_a', 'acc_t_max_a', 'acc_t_std_a', 'acc_t_var_a', 'acc_t_entr_a', 'acc_t_lgEnergy_a', 'acc_t_sma_a', 'acc_t_interq_a', 'acc_t_skew_a', 'acc_t_kurt_a', 'acc_t_rms_a', 'acc_f_amp_a', 'acc_f_mean_a', 'acc_f_max_a', 'acc_f_std_a', 'acc_f_var_a', 'acc_f_entr_a', 'acc_f_lgEnergy_a', 'acc_f_sma_a', 'acc_f_interq_a', 'acc_f_skew_a', 'acc_f_kurt_a', 'acc_f_rms_a', 'acc_p_amp_a', 'acc_p_mean_a', 'acc_p_max_a', 'acc_p_std_a', 'acc_p_var_a', 'acc_p_entr_a', 'acc_p_lgEnergy_a', 'acc_p_sma_a', 'acc_p_interq_a', 'acc_p_skew_a', 'acc_p_kurt_a', 'acc_p_rms_a', 'acc_a_amp_a', 'acc_a_mean_a', 'acc_a_max_a', 'acc_a_std_a', 'acc_a_var_a', 'acc_a_entr_a', 'acc_a_lgEnergy_a', 'acc_a_sma_a', 'acc_a_interq_a', 'acc_a_skew_a', 'acc_a_kurt_a', 'acc_a_rms_a', 'acc_a_mainY_a', 'acc_a_subY_a', 'acc_a_cftor_a']

label_map = {"ft": 0, "coa": 1, "alter": 2, "hr-r": 3, "hr-l": 4, "fn-l": 5, "fn-r": 6, "standh": 7, "wa": 8,
             "ac": 9, "drink": 10, "pick": 11, "sit": 12, "stand":13}
window_size=300
overlapping_rate = 0.5  #0.5 overlap, overlap = 1 / overlapping_rate
frequency = 200
pd_num = 10
activity_num = 3 #要提取的活动数目
side_l = "wristl"
side_r = "wristr"
std_mod = True  #控制特征是否标准化
Feature = utils_v1.FeatureExtractWithProcess1(pd_num, activity_num, data_path, side_r, fea_column, window_size, overlapping_rate, frequency)
#Feature2 = pd_utils.FeatureExtractWithProcess(pd_num, activity_num, data_path, side_r, window_size, overlapping_rate, frequency, std_mod)
print(Feature)
Feature.to_csv(data_path + "feature_right_side_activity{}.csv".format(activity_num))