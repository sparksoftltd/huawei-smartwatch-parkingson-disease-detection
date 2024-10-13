import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# get project path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
# put src into sys.path
sys.path.append(project_root)


data_dir_path = os.path.join(project_root, "output/feature_selection/")
output_dir_path = os.path.join(project_root, "output/feature_selection/doctor_label")

train_output_path = os.path.join(output_dir_path, "train")
valid_output_path = os.path.join(output_dir_path, "valid")
test_output_path = os.path.join(output_dir_path, "test")

# 确保输出文件夹存在
os.makedirs(train_output_path, exist_ok=True)
os.makedirs(valid_output_path, exist_ok=True)
os.makedirs(test_output_path, exist_ok=True)

# 标签文件路径
label_path = os.path.join(project_root, "input/feature_extraction/raw/Information_Sheet_Version_1_Doctor.csv")
# 读取 label 文件
label_df = pd.read_csv(label_path)


def split_dataset(data_path: str, label_path: str, train_ratio: float = 0.8):
    # 读取数据
    data = pd.read_csv(data_path)
    label = pd.read_csv(label_path)

    # 筛选出 Doctor_Level 不为 -1 的数据作为测试集
    test_label = label[label['Doctor_Level'] != -1]

    # 筛选出 Doctor_Level 为 -1 的数据
    remaining_label = label[label['Doctor_Level'] == -1]

    # 根据 PatientID 来进行分组，按患者为单位进行划分
    patient_ids = remaining_label['PatientID'].unique()  # 获取所有PatientID

    # 使用 train_test_split 来划分 train 和 valid
    train_patient_ids, valid_patient_ids = train_test_split(patient_ids, train_size=train_ratio, random_state=42)
    test_patient_ids = test_label['PatientID'].unique()

    # 根据划分好的患者ID将数据分为 train 和 valid, test
    train_data = data[data['PatientID'].isin(train_patient_ids)]
    valid_data = data[data['PatientID'].isin(valid_patient_ids)]
    test_data = data[data['PatientID'].isin(test_patient_ids)]

    # 打印每个数据集的大小
    print(f"训练集大小: {train_data.shape} ")
    print(f"验证集大小: {valid_data.shape} ")
    print(f"测试集大小: {test_data.shape} ")

    return train_data, valid_data, test_data


# 处理 activity_1.csv 到 activity_16.csv
for i in range(1, 17):
    # 构造文件名
    activity_file = f"activity_{i}.csv"
    data_path = os.path.join(data_dir_path, activity_file)

    # 调用分割函数
    train_data, valid_data, test_data = split_dataset(data_path, label_path)

    # 修改 test_data 中的 Severity_Level，根据 PatientID 从 label_df 查找 Doctor_Level
    test_data = test_data.merge(label_df[['PatientID', 'Doctor_Level']], on='PatientID', how='left')
    test_data['Severity_Level'] = test_data['Doctor_Level']  # 用 Doctor_Level 更新 Severity_Level
    test_data.drop(columns=['Doctor_Level'], inplace=True)  # 删除临时的 Doctor_Level 列

    # 保存分割后的数据
    train_data.to_csv(os.path.join(train_output_path, f"train_activity_{i}.csv"), index=False)
    valid_data.to_csv(os.path.join(valid_output_path, f"valid_activity_{i}.csv"), index=False)
    test_data.to_csv(os.path.join(test_output_path, f"test_activity_{i}.csv"), index=False)

    print(f"End of processing for activity {i}")


