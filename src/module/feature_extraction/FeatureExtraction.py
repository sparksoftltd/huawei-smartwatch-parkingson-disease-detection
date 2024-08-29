import os
import pandas as pd
from utils import pd_utils


class FeatureExtraction:
    def __init__(self, data_dir_path, pd_num, activity_num, side_r, side_l, window_size=300, overlapping_rate=0.5,
                 frequency=200):
        self.data_dir_path = data_dir_path
        self.pd_num = pd_num
        self.activity_num = activity_num
        self.side_r = side_r
        self.side_l = side_l
        self.window_size = window_size
        self.overlapping_rate = overlapping_rate
        self.frequency = frequency
        self.fea_column = self._load_feature_names()

    def _load_feature_names(self):
        """加载并合并传感器特征列名"""
        feature_name_file = 'feature_name.csv'
        feature_name_path = os.path.join(self.data_dir_path, feature_name_file)
        df_read = pd.read_csv(feature_name_path)

        # 分别读取三个传感器的列名
        acc_list = df_read['acc'].tolist()
        gyro_list = df_read['gyro'].tolist()
        mag_list = df_read['mag'].tolist()

        # 将三个列表拼接起来
        fea_column = acc_list + gyro_list + mag_list
        return fea_column

    def extract_features(self):
        """提取特征并合并标签"""
        Feature = pd_utils.FeatureExtractWithProcess(self.pd_num, self.activity_num, self.data_dir_path, self.side_r,
                                                     self.fea_column, self.window_size, self.overlapping_rate,
                                                     self.frequency)

        # 读取标签信息
        label_data = pd.read_csv(os.path.join(self.data_dir_path, "Information_Sheet_Version_1.csv"))
        label_table = label_data.loc[:, ["PatientID", "Severity_Level"]]

        # 将病人序号列数据转换为int
        Feature['PatientID'] = Feature['PatientID'].astype(int)
        label_table['PatientID'] = label_table['PatientID'].astype(int)

        # 为特征数据添加标签，通过 PatientID 进行 merge
        feature_label = pd.merge(Feature, label_table, on='PatientID')

        return feature_label

    def save_features(self, feature_label, output_path):
        """保存特征为 CSV 文件"""
        output_file_path = os.path.join(output_path, f"{self.side_r}_acc_gyro_mag_feature_label.csv")
        feature_label.to_csv(output_file_path, index=False)
        print(f"特征保存到: {output_file_path}")


if __name__ == '__main__':
    # 初始化参数
    back_to_root = "../../.."
    data_dir_path = r"input/feature_extraction/raw/"
    pd_num = 135
    activity_num = 16
    side_r = "wristr"
    side_l = "wristl"
    window_size = 300
    overlapping_rate = 0.5
    frequency = 200

    # 创建 FeatureExtraction 对象
    feature_extraction = FeatureExtraction(os.path.join(back_to_root, data_dir_path), pd_num, activity_num, side_r,
                                           side_l, window_size,
                                           overlapping_rate, frequency)

    # 提取特征并合并标签
    feature_label = feature_extraction.extract_features()

    # 保存特征到 CSV 文件
    output_path = r'output/feature_extraction/'
    feature_extraction.save_features(feature_label, os.path.join(back_to_root, output_path))

#
#
