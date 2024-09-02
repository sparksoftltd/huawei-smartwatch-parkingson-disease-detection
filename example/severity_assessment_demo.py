import os
import sys

# get src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))

# put src into sys.path
sys.path.append(project_root)

from src.module.severity_aeeseement import SeverityAssessment


_back_to_root = "../"
# # 单一活动测试
activity_id = [11]
_data_path = "output/feature_selection"
_data_name = f"activity_{activity_id[0]}.csv"
sa = SeverityAssessment(_back_to_root, _data_path, _data_name, activity_id, 'lgbm')
sa.assessment()
