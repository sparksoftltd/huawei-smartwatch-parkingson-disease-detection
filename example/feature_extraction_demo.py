import os
import sys

# get project path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
project_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
# put src into sys.path
sys.path.append(project_root)
sys.path.append(project_src)
from src.module.feature_extraction import FeatureExtraction


# Initial parameters
back_to_root = project_root
data_dir_path = r"input/feature_extraction/raw/"
pd_num = 135
activity_num = 16
side_r = "wristr"
side_l = "wristl"
window_size = 300
overlapping_rate = 0.5
frequency = 200

# create FeatureExtraction instance
feature_extraction = FeatureExtraction(os.path.join(back_to_root, data_dir_path), pd_num, activity_num, side_r,
                                       side_l, window_size,
                                       overlapping_rate, frequency)

# label
feature_label = feature_extraction.extract_features()

# save
output_path = r'output/feature_extraction/'
feature_extraction.save_features(feature_label, os.path.join(back_to_root, output_path))
