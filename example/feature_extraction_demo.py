import os
import sys
# get src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))

# put src into sys.path
sys.path.append(project_root)

from src.module.feature_extraction import FeatureExtraction


# 初始化参数
back_to_root = "../"
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