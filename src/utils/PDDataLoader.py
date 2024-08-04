import os
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler


class PDDataLoader:
    def __init__(self, activity_id, data_path, fold_groups_path, severity_mapping=None):
        if severity_mapping is None:
            severity_mapping = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 3}  # 映射关系
        self.data_path = data_path  # 手工特征文件路径
        self.activity_id = activity_id  # 从手工特征文件中选取指定activity_id对应的数据
        self.severity_mapping = severity_mapping  # 映射关系
        self.PD_data = self.load_and_preprocess_data()  # 预处理数据文件
        self.feature_name = self.PD_data.columns[:-3]

        self.fold_groups = self.process_fold_groups(fold_groups_path)
        self.bag_data_dict, self.patient_ids, self.fold_groups = self.group_data(self.fold_groups)  # 组织数据

    def process_fold_groups(self, csv_file_path):
        try:
            data = pd.read_csv(csv_file_path)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return []
        data['activity_label'] = data['activity_label'].astype(str)
        filtered_data = data[data['activity_label'] == str(self.activity_id)]
        fold_groups = []
        for i in range(1, 6):
            group_data = []
            for col_type in ['HC', 'Hy1', 'Hy2', 'Hy3', 'Hy4']:
                column_name = f'Group{i}_{col_type}'
                group_data.extend(json.loads(filtered_data.iloc[0][column_name]))
            fold_groups.append(group_data)
        return fold_groups

    def load_and_preprocess_data(self):
        data = pd.read_csv(self.data_path)
        try:
            activity_id_num = int(self.activity_id)
            if 1 <= activity_id_num <= 16:
                data = data.loc[data['activity_label'] == activity_id_num]
            else:
                raise AssertionError(f"activity_id {self.activity_id} is not in the range 1 to 16.")
        except ValueError:
            pass
        print(f"Loading data of activity_id: {self.activity_id}")
        data['Severity_Level'] = data['Severity_Level'].map(self.severity_mapping)
        data = data.dropna()
        numerical_columns = data.columns[:-3]
        scaler = StandardScaler()
        data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
        return data

    def create_train_test_split(self, fold_num, test_ids):
        train_ids = []
        for num, fold_ids in enumerate(self.fold_groups):
            if num != fold_num:
                train_ids.extend(fold_ids)
        train_X = [self.bag_data_dict[pid]["bag_data"] for pid in train_ids]
        train_X = np.vstack(train_X)
        train_Y = [self.bag_data_dict[pid]["bag_data_instance_label"] for pid in train_ids]
        train_Y = np.hstack(train_Y)

        test_X_ls = [self.bag_data_dict[pid]["bag_data"] for pid in test_ids]
        test_Y_ls = [self.bag_data_dict[pid]["bag_data_label"] for pid in test_ids]

        return train_X, train_Y, test_X_ls, test_Y_ls, train_ids, test_ids

    def group_data(self, fold_groups):
        grouped = self.PD_data.groupby(['PatientID', 'Severity_Level'])
        bag_data_dict = dict()
        patient_ids = []
        for (patient_id, _), group in grouped:
            try:
                activity_id_num = int(self.activity_id)
                if 1 <= activity_id_num <= 16:
                    bag_data = np.array(group.iloc[:, :-3])
                else:
                    raise AssertionError(f"activity_id {self.activity_id} is not in the range 1 to 16.")
            except ValueError:
                bag_data = np.array(group.iloc[:, :-2])
            bag_data_instance_label = (np.array(group.loc[:, 'Severity_Level']))
            patient_ids.append(patient_id)
            if patient_id not in bag_data_dict.keys():
                bag_data_dict[patient_id] = {"pid": patient_id, "bag_data": bag_data,
                                             "bag_data_label": bag_data_instance_label[0],
                                             "bag_data_instance_label": bag_data_instance_label
                                             }
        return bag_data_dict, patient_ids, fold_groups


if __name__ == '__main__':
    # 以活动3为例
    activity_id = 3
    data_path = "../../output/activity/step_2_select_sensors"
    data_name = "acc_data.csv"
    fold_groups_path = "../../input/activity/step_3_output_feature_importance"
    fold_groups_name = "fold_groups_new_with_combinations.csv"
    severity_mapping = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 3}

    classifier = PDDataLoader(activity_id, os.path.join(data_path, data_name),
                              os.path.join(fold_groups_path, fold_groups_name), severity_mapping=severity_mapping)



