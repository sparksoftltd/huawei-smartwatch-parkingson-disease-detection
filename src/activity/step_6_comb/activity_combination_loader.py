import os
import pandas as pd


class ActivityCombLoader:
    def __init__(self, base_path, activity_ids, combination_mode='horizontal'):
        """
        初始化类，定义数据路径、活动ID列表和组合方式。

        :param base_path: 活动数据的基本路径。
        :param activity_ids: 需要拼接的活动ID列表。
        :param combination_mode: 拼接方式，'horizontal'表示横向，'vertical'表示纵向。
        """
        self.base_path = base_path
        self.activity_ids = activity_ids
        self.combination_mode = combination_mode
        self.data_frames = {}
        self.common_patient_ids = None
        self._load_data()

    def _load_data(self):
        """加载所有指定的活动数据，并提取特征列和patient_id，severity_level。"""
        for activity_id in self.activity_ids:
            file_path = os.path.join(self.base_path, f'acc_data_activity_{activity_id}.csv')
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                # 检查数据是否包含所需的列
                if all(col in df.columns for col in ['PatientID', 'Severity_Level', 'activity_label']):
                    # 根据组合方式决定是否修改特征列名
                    if self.combination_mode == 'horizontal':
                        # 修改特征列名，为列名添加activity_id后缀
                        features = df.iloc[:, :-3].add_suffix(f'_{activity_id}')
                    else:
                        # 保持原有特征列名
                        features = df.iloc[:, :-3]

                    patient_ids = df['PatientID']
                    severity_levels = df['Severity_Level']

                    # 存储数据帧到字典中
                    self.data_frames[activity_id] = (features, patient_ids, severity_levels)

                    # 维护共同存在的 patient_id 列表
                    if self.common_patient_ids is None:
                        self.common_patient_ids = set(patient_ids)
                    else:
                        self.common_patient_ids.intersection_update(patient_ids)
                else:
                    print(f"文件 {file_path} 中缺少必要的列，跳过此活动。")
            else:
                print(f"文件 {file_path} 不存在，跳过此活动。")
        print(f"该活动组合可用人数({len(self.common_patient_ids)}):{self.common_patient_ids}")

        # 转换为列表形式，便于后续操作
        self.common_patient_ids = list(self.common_patient_ids)

    def merge_weighted(self, weights):
        """加权拼接数据，仅拼接共同的特征列，并添加 PatientID 和 Severity_Level。

        :param weights: 一个包含每个活动数据对应权重的列表，长度应与活动ID数量相同，值在0-1之间。
        """
        if not self.data_frames or not self.common_patient_ids:
            return None

        # 确定所有活动间的共同特征列
        common_columns = None
        for activity_id in self.activity_ids:
            features, _, _ = self.data_frames[activity_id]
            if common_columns is None:
                common_columns = set(features.columns)
            else:
                common_columns.intersection_update(features.columns)

        # 确保存在共同特征列
        if not common_columns:
            raise ValueError("没有找到共同的特征列，无法进行加权拼接。")

        # 转换为列表形式
        common_columns = list(common_columns)

        # 初始化空数据框，用于存放最终拼接结果
        merged_features_list = []
        patient_ids_list = []
        severity_levels_list = []

        # 遍历每个共同的 PatientID
        for patient_id in self.common_patient_ids:
            # 使用列表表达式找到当前病人在所有活动中的最大样本长度
            max_sample_length = max(
                len(self.data_frames[activity_id][0][self.data_frames[activity_id][1] == patient_id])
                for activity_id in self.activity_ids
            )

            total_weight = 0
            patient_severity = None
            weighted_features = pd.DataFrame(0, index=range(max_sample_length), columns=common_columns)

            for activity_id, weight in zip(self.activity_ids, weights):
                features, patient_ids, severity_levels = self.data_frames[activity_id]
                # 获取当前 PatientID 对应的行数据，并重置索引
                patient_features = features[patient_ids == patient_id][common_columns].reset_index(drop=True)

                # 如果当前活动样本数少于最大样本长度，使用均值填补
                if len(patient_features) < max_sample_length:
                    fill_values = patient_features.mean()
                    patient_features = patient_features.reindex(range(max_sample_length), fill_value=None)
                    patient_features = patient_features.fillna(fill_values)

                weighted_features += patient_features * weight
                total_weight += weight
                patient_severity = severity_levels[patient_ids == patient_id].values[0]  # 保留唯一的 severity_level

            # 归一化加权特征
            weighted_features /= total_weight

            merged_features_list.append(weighted_features)
            patient_ids_list.append(pd.Series([patient_id] * max_sample_length, name='PatientID'))
            severity_levels_list.append(pd.Series([patient_severity] * max_sample_length, name='Severity_Level'))

        # 拼接所有 PatientID 的数据
        if merged_features_list:
            final_merged_features = pd.concat(merged_features_list, axis=0).reset_index(drop=True)
            final_patient_ids = pd.concat(patient_ids_list, axis=0).reset_index(drop=True)
            final_severity_levels = pd.concat(severity_levels_list, axis=0).reset_index(drop=True)
        else:
            final_merged_features = pd.DataFrame()
            final_patient_ids = pd.Series(name='PatientID')
            final_severity_levels = pd.Series(name='Severity_Level')

        # 合并特征数据和 PatientID、Severity_Level
        final_merged = pd.concat([final_merged_features, final_patient_ids, final_severity_levels], axis=1)

        return final_merged

    def merge_vertical(self):
        """纵向拼接数据，仅拼接共同的特征列，并添加 PatientID 和 Severity_Level。"""
        if not self.data_frames or not self.common_patient_ids:
            return None

        # 确定所有活动间的共同特征列
        common_columns = None
        for activity_id in self.activity_ids:
            features, _, _ = self.data_frames[activity_id]
            if common_columns is None:
                common_columns = set(features.columns)
            else:
                common_columns.intersection_update(features.columns)

        # 确保存在共同特征列
        if not common_columns:
            raise ValueError("没有找到共同的特征列，无法进行纵向拼接。")

        # 转换为列表形式
        common_columns = list(common_columns)

        # 初始化空数据框，用于存放最终拼接结果
        merged_features_list = []

        # 遍历每个活动数据
        for activity_id in self.activity_ids:
            features, patient_ids, severity_levels = self.data_frames[activity_id]
            # 只保留共同的特征列
            common_features = features[common_columns].copy()

            # 将 PatientID 和 Severity_Level 列添加到特征数据中
            common_features.loc[:, 'PatientID'] = patient_ids.values
            common_features.loc[:, 'Severity_Level'] = severity_levels.values

            # 添加到合并列表
            merged_features_list.append(common_features)

        # 将所有活动的数据纵向拼接
        final_merged = pd.concat(merged_features_list, axis=0).reset_index(drop=True)

        return final_merged

    def merge_horizontal(self):
        """横向拼接数据，确保根据 PatientID 对齐，并在拼接完成后填充缺失值。"""
        if not self.data_frames or not self.common_patient_ids:
            return None

        # 初始化空数据框，用于存放最终拼接结果
        merged_features_list = []
        patient_ids_list = []
        severity_levels_list = []

        # 遍历每个共同的 PatientID
        for patient_id in self.common_patient_ids:
            patient_data = []
            patient_severity = None
            for activity_id in self.activity_ids:
                features, patient_ids, severity_levels = self.data_frames[activity_id]
                # 获取当前 PatientID 对应的行数据
                patient_features = features[patient_ids == patient_id].copy()
                if patient_features.empty:
                    continue

                patient_data.append(patient_features)
                patient_severity = severity_levels[patient_ids == patient_id].values[0]  # 保留唯一的 severity_level

            # 将该 PatientID 的所有活动数据横向拼接
            if patient_data:
                merged_features = pd.concat(patient_data, axis=1)

                # 在拼接完成后填充缺失值
                merged_features = merged_features.apply(lambda col: col.fillna(col.mean()), axis=0)

                # 将 patient_id 和 patient_severity 广播到与 merged_features 相同的长度
                patient_id_series = pd.Series([patient_id] * merged_features.shape[0], name='PatientID')
                patient_severity_series = pd.Series([patient_severity] * merged_features.shape[0],
                                                    name='Severity_Level')

                merged_features_list.append(merged_features)
                patient_ids_list.append(patient_id_series)
                severity_levels_list.append(patient_severity_series)

        # 拼接所有 PatientID 的数据
        if merged_features_list:
            final_merged_features = pd.concat(merged_features_list, axis=0).reset_index(drop=True)
            final_patient_ids = pd.concat(patient_ids_list, axis=0).reset_index(drop=True)
            final_severity_levels = pd.concat(severity_levels_list, axis=0).reset_index(drop=True)
        else:
            final_merged_features = pd.DataFrame()
            final_patient_ids = pd.Series(name='PatientID')
            final_severity_levels = pd.Series(name='Severity_Level')

        # 合并特征数据和广播后的 PatientID、Severity_Level
        final_merged = pd.concat([final_merged_features, final_patient_ids, final_severity_levels], axis=1)

        return final_merged

    def save_merged_data(self, merged_data, output_dir):
        """保存拼接后的数据到指定路径，并将活动ID和拼接方式包含在文件名中。"""
        # 生成包含活动 ID 和拼接方式的文件名
        activity_ids_str = "_".join(map(str, self.activity_ids))
        file_name = f"merged_activities_{activity_ids_str}_{self.combination_mode}.csv"
        output_path = os.path.join(output_dir, file_name)

        # 保存数据到生成的文件名中
        merged_data.to_csv(output_path, index=False)
        print(f"文件已保存到: {output_path}")

    def combine_and_save(self, output_dir, weights=None):
        """根据组合模式执行相应的拼接并保存数据。

        :param output_dir: 数据保存的目标目录。
        :param weights: 当组合模式为'weighted'时，需要提供权重列表。
        """
        if self.combination_mode == 'weighted':
            if weights is None:
                raise ValueError("组合模式为 'weighted' 时必须提供权重列表。")
            merged_data = self.merge_weighted(weights)
        elif self.combination_mode == 'horizontal':
            merged_data = self.merge_horizontal()
        elif self.combination_mode == 'vertical':
            merged_data = self.merge_vertical()
        else:
            raise ValueError(f"未知的组合模式: {self.combination_mode}")

        self.save_merged_data(merged_data, output_dir)


if __name__ == '__main__':
    # 定义活动数据的路径
    base_path = "../../../output/activity/step_4_feature_selection"
    # 定义需要拼接的活动ID
    activity_ids = [14, 15, 16]
    # 定义权重
    weights = [0.5, 0.3, 0.2]

    # 初始化类，并指定拼接方式为加权拼接
    # horizontal
    # vertical
    # weighted
    loader = ActivityCombLoader(base_path, activity_ids, combination_mode='vertical')
    # 统一调用拼接并保存结果
    loader.combine_and_save("../../../output/activity/step_6_comb", weights=weights)

