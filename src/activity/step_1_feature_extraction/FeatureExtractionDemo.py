import os
import pandas as pd

from utils import pd_utils

# 指定文件路径和文件名
data_dir_path = r"../../../input/activity/step_1_feature_extraction/raw/"
feature_name_file = 'feature_name.csv'

# 组合完整路径
feature_name_path = os.path.join(data_dir_path, feature_name_file)

# 读取CSV文件
df_read = pd.read_csv(feature_name_path)
# 分别读取三个传感器
acc_list = df_read['acc'].tolist()
gyro_list = df_read['gyro'].tolist()
mag_list = df_read['mag'].tolist()
# 将三个列表拼接起来
fea_column = acc_list + gyro_list + mag_list
# ==============================================新增=============================================================

# 特征提取的参数设置
window_size = 300
overlapping_rate = 0.5
frequency = 200
pd_num = 135
activity_num = 16
side_l = "wristl"
side_r = "wristr"

Feature = pd_utils.FeatureExtractWithProcess(pd_num, activity_num, data_dir_path, side_r, fea_column, window_size,
                                             overlapping_rate, frequency)

# 读取标签信息
label_data = pd.read_csv(data_dir_path + "Information_Sheet_Version_1.csv")
label_table = label_data.loc[:, ["PatientID", "Severity_Level"]]

# 将病人序号列数据转换为int
Feature['PatientID'] = Feature['PatientID'].astype(int)
label_table['PatientID'] = label_table['PatientID'].astype(int)

# 为特征数据添加label，Feature中存放数据，label_table存放标签，通过PatientID索引进行merge
feature_label = pd.merge(Feature, label_table, on='PatientID')

data_output_path = r'../../../output/activity/step_1_feature_extraction/'
# 将特征保存为csv文件
feature_label.to_csv(data_output_path + side_r + "_acc_gyro_mag_feature_label.csv", index=False)


