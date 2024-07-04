#%%feature extraction from PD
import pandas as pd
from utils import pd_utils
#%
# get feature
data_path= "/Users/macpro/PycharmProjects/huawei/datasets/pd1/"
# severity_data = pd.read_excel('/Volumes/T7 Shield/PD_data_summary.xlsx', sheet_name="Patients information")
# severity_data = severity_data[["PatientID","Affected_Side","Severity_Level"]]
Feature2 = pd.DataFrame()

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
# for activity_num in range(1,15,1):
Feature2 = pd_utils.FeatureExtractWithProcess(pd_num, activity_num, data_path, side_r, window_size, overlapping_rate, frequency, std_mod)
print(Feature2)
Feature2.to_csv(data_path + "feature_right_side_activity{}.csv".format(activity_num))