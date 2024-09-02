import os
import sys

# get src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))

# put src into sys.path
sys.path.append(project_root)

from src.module.feature_selection import FeatureSelection

back_to_root = "../"

for i in range(1, 16 + 1):
    # instance feature selection
    fs = FeatureSelection(activity_id=i, back_to_root=back_to_root, sensors=['acc'], seed=0)

    # single activity feature selection
    fs.single_activity_feature_selection()

    # Analysis the performance for different num of features
    fs.single_activity_best_num_features()

    # save activity data
    fs.save_important_feature()
    print(f"Saved activity data {i}")
