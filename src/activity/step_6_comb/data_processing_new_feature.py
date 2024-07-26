import os.path
import pandas as pd
import re
import itertools



# 类 ActivityCombinationPreprocessor 处理不同方案的活动数据组合
# 它支持水平、垂直、融合等不同方案，融合时根据给定的权重计算

# 模块 ActivityCombinationPreprocessor 说明
# 这个模块用于处理活动数据组合，支持不同的组合方案，包括水平、垂直和融合等
# 融合方案根据权重进行计算
# - data_path: 字符串，CSV 数据文件的路径
# - sequence: 列表，活动 ID 的序列，要求每个元素是整数

# - scheme: 整数，组合方案类型，0 表示垂直拼接，1 表示水平拼接，2 表示融合
#   scheme=0 垂直拼接，数据按照垂直方向，将活动数据按sequence次序垂直排列，先排列sequence[i]的全部数据，再排列sequence[i+1]的全部数据。
#   scheme=1 水平拼接，将按照sequence次序水平拼接，因为不同活动的执行时间不一样，可能出现水平上的NaN值。
#   scheme=2 数据融合，将活动数据按sequence次序，按照sequence[i]对应的活动数据*wights[i]+sequence[i+1]*wights[i+1]融合全部数据，
#   注意当遇到受试者确实sequence中指定活动序号的数据时，放弃该受试者的数据,不参与数据融合。

# - samples: 整数，非负，指定每个活动截取的窗口数
# - weights: 列表，数值型，指定融合时的权重。scheme 为 2 时必须提供，表示按照累加sequence[i]序号活动数据*weights[i]融合。
# 在入口程序目录下生成 m命名为：活动拼接次序按+号连接_拼接方式_comb.csv 形如：1+2_horizontal_comb.csv，活动1+活动2水平拼接的csv文件


class ActivityCombinationPreprocessor:
    def __init__(self, data_path, sequence, scheme, samples=None, weights=None):
        # 构造函数初始化必要的参数
        self._sequence = sequence  # 活动 ID 的序列
        self._samples = samples  # 每个活动截取多少个窗口（实例）
        self._scheme = scheme  # 方案类型，0表示横向拼接，1表示纵向拼接，2表示按照累加sequence[i]序号活动数据*weights[i]融合
        self._weights = weights  # 权重，当scheme=2时需要指定每个活动数据累加时的权重
        self._data_path = os.path.normpath(data_path)  # 规范化路径数据文件的路径
        self._data = self._load_data()  # 按照data_path加载数据
        # self.window_size, self.overlapping = self.extract_ws_ol()

    def extract_ws_ol(self):
        # 编译正则表达式来匹配 ws 和 ol 后的数字
        ws_pattern = re.compile(r"_ws(\d+)")
        # ol_pattern = re.compile(r"_ol(\d+)")
        ol_pattern = re.compile(r"_ol(-?\d+\.?\d*)")

        # 使用正则表达式查找数字
        ws_match = ws_pattern.search(self._data_path)
        ol_match = ol_pattern.search(self._data_path)

        # 提取窗口大小和重叠
        window_size = int(ws_match.group(1)) if ws_match else None
        overlapping = float(ol_match.group(1)) if ol_match else None

        return window_size, overlapping

    # 加载 CSV 数据并进行数据验证
    def _load_data(self):
        try:
            self._validate_arguments()  # 验证参数有效性

            data = pd.read_csv(self._data_path)
            # ============更改名称======================================
            # 更改列名称
            data = data.rename(columns={
                'activity_label': 'Activity_id',
                'PatientID': 'subject_id',
                'Severity_Level': 'severity_level'
            })
            # ============更改名称======================================
            # 按列转换为数值
            data['Activity_id'] = data['Activity_id'].astype(int)
            data['subject_id'] = data['subject_id'].astype(int)
            data['severity_level'] = data['severity_level'].astype(int)

            self._validate_unique_subject(
                data)  # 验证subject_id 的唯一性,保证数据中subject_id没有重复，导致同一个subject_id数据对应多个severity_level

            return data
        except FileNotFoundError:
            raise Exception(f"Data file not found at path: {self._data_path}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    # 检查每组 subject_id的数据，只有一个 severity_level，也就是验证了subject_id的唯一性
    @staticmethod
    def _validate_unique_subject(data):
        grouped = data.groupby('subject_id')  # 按 subject_id 分组
        unique_severity_counts = grouped['severity_level'].nunique()  # 获取唯一数量
        duplicate_subject_ids = unique_severity_counts[unique_severity_counts != 1]  # 找到多于1的subject_id

        if not duplicate_subject_ids.empty:
            duplicated_subjects = ", ".join(map(str, duplicate_subject_ids.index.tolist()))
            raise ValueError(
                f"Data contains subject_ids with multiple severity levels: {duplicated_subjects}"
            )

    # 验证构造函数传入的参数
    def _validate_arguments(self):

        if not isinstance(self._sequence, list):
            raise TypeError("sequence must be a list.")

        if not all(isinstance(item, int) for item in self._sequence):
            raise ValueError("All elements in sequence must be integers.")

        if self._samples is not None:
            if not isinstance(self._samples, int):
                raise TypeError("samples must be an integer.")
            if self._samples < 0:
                raise ValueError("samples must be a non-negative integer.")

        if not isinstance(self._scheme, int):
            raise TypeError("scheme must be an integer.")

        if self._scheme not in {0, 1, 2}:
            raise ValueError("scheme must be 0, 1, or 2.")

        if self._scheme == 2:
            if self._weights is None:
                raise ValueError("weights must be provided for scheme 2.")
            else:
                if not isinstance(self._weights, list):
                    raise TypeError("sequence must be a list.")

                if not all(isinstance(item, (int, float)) for item in self._weights):
                    raise ValueError("All elements in weights must be numbers.")
                if len(self._sequence) != len(self._weights):
                    raise AssertionError("In scheme 2, sequence length must match weights length.")

    # 保存处理后的数据到 CSV
    def _save_data(self, data):
        # filename = {
        #     0: f"{'+'.join(map(str, self._sequence))}_{self._samples}samples_ws{self.window_size}_ol{self.overlapping}_vertical_comb.csv",
        #     1: f"{'+'.join(map(str, self._sequence))}_{self._samples}samples_ws{self.window_size}_ol{self.overlapping}_horizontal_comb.csv",
        #     2: f"{'+'.join(map(str, self._sequence))}_{self._samples}samples_ws{self.window_size}_ol{self.overlapping}_fused_comb.csv"
        # }.get(self._scheme, None)
        filename = {
            0: f"{'+'.join(map(str, self._sequence))}_{self._samples}_v.csv",
            1: f"{'+'.join(map(str, self._sequence))}_{self._samples}_h.csv",
            2: f"{'+'.join(map(str, self._sequence))}_{self._samples}_f.csv"
        }.get(self._scheme, None)

        # PatientID
        # Severity_Level
        # 更改指定列的列名
        data = data.rename(columns={"subject_id": "PatientID", "severity_level": "Severity_Level"})
        # file_path = fr"../../../output/activity/step_6_activity_combination/{filename}"
        file_path = fr"../../../output/activity/step_6_comb/{filename}"
        # data.to_csv(filename, index=False)
        data.to_csv(file_path, index=False)

    # 根据组合字典提供的方案，选择活动组合方案处理
    def run(self):
        processed_data = (
            self._merge_features()
            if self._scheme == 2
            else self._combine_features()
        )

        self._save_data(processed_data)

    # 横向或纵向拼接，组合数据的核心逻辑1
    def _combine_features(self):
        all_combined_data = pd.DataFrame()  # 存储组合后全部受试者的数据
        grouped_data = self._data.groupby('subject_id')  # 按 subject_id 分组
        for subject_id, group in grouped_data:
            personal_combined_data = self._build_personal_combined_data(group, subject_id)
            all_combined_data = pd.concat([all_combined_data, personal_combined_data], axis=0, ignore_index=True)
        print('\n', all_combined_data)
        # 返回组合后全部受试者数据
        return all_combined_data

    # 横向或纵向拼接，组合数据的核心逻辑2（按照每个人的不同活动拼接数据）
    def _build_personal_combined_data(self, group, subject_id):
        personal_combined_data = pd.DataFrame()  # 存储个人组合数据
        unique_values = {
            "Unique_subject_id": subject_id,  # 获取该组数据的subject_id值，因为是按照subject_id分组的数据，所以该值唯一
            "Unique_severity_level": group["severity_level"].unique()[0]  # 获取该组数据的标签值，因为是按照subject_id分组的数据，所以该值唯一
        }
        for activity_id in self._sequence:  # 按照self._sequence中指定的活动筛选数据进行活动组合
            if self._samples is not None:  # 每个活动是否指定了截取的窗口（实例）数量
                # 截取self._samples数量的部分行（实例），且不要后三列，后三列为Activity_id、subject_id、severity_level
                activity_data = group[group['Activity_id'] == activity_id].iloc[:self._samples, :-3].reset_index(
                    drop=True)
            else:
                # 截取所有行（实例）
                activity_data = group[group['Activity_id'] == activity_id].iloc[:, :-3].reset_index(
                    drop=True)
                if activity_data.empty:
                    return pd.DataFrame()
            # 拼接方向为self._scheme， 0表示纵向垂直拼接， 1表示横向水平拼接
            personal_combined_data = pd.concat([personal_combined_data, activity_data], axis=self._scheme,
                                               ignore_index=True)
        # 组合后的个人数据需要添加上subject_id(组合后的数据来自于哪个受试者，id是受试者的唯一标识)和severity_level(标签)
        personal_combined_data['subject_id'] = unique_values["Unique_subject_id"]
        personal_combined_data['severity_level'] = unique_values["Unique_severity_level"]
        # 返回个人组合数据
        return personal_combined_data

    # 数据融合方案的核心逻辑1
    def _merge_features(self):
        all_fused_data = pd.DataFrame()  # 存储融合后的全部受试者数据
        grouped_data = self._data.groupby('subject_id')  # 按 subject_id 分组
        for subject_id, group in grouped_data:
            personal_fused_data = self._build_personal_fused_data(group, subject_id)
            all_fused_data = pd.concat([all_fused_data, personal_fused_data], axis=0, ignore_index=True)
        all_fused_data = all_fused_data.dropna()  # 去除缺失值
        print('\n', all_fused_data)
        return all_fused_data

    def _build_personal_fused_data(self, group, subject_id):
        personal_fused_data = pd.DataFrame()  # 存储个人融合数据
        unique_values = {
            "Unique_subject_id": subject_id,
            "Unique_severity_level": group["severity_level"].unique()[0]
        }
        for activity_id, weight in zip(self._sequence, self._weights):
            if self._samples is not None:
                # 截取self._samples数量的部分行（实例），且不要后三列，后三列为Activity_id、subject_id、severity_level
                activity_data = group[group['Activity_id'] == activity_id].iloc[:self._samples, :-3].reset_index(
                    drop=True)
            else:
                activity_data = group[group['Activity_id'] == activity_id].iloc[:, :-3].reset_index(
                    drop=True)
            if activity_data.empty:
                return pd.DataFrame()
                # break  # 如果为空，停止循环，表示如果该受试者进行数据融合时，出现部分活动数据确实，放弃执行该受试者的数据融合。
            if personal_fused_data.empty:
                personal_fused_data = weight * activity_data  # 第一次执行时初始化融合
            else:
                personal_fused_data = personal_fused_data.add(weight * activity_data, fill_value=None)
        personal_fused_data['subject_id'] = unique_values["Unique_subject_id"]
        personal_fused_data['severity_level'] = unique_values["Unique_severity_level"]
        # 返回个人融合数据
        return personal_fused_data


if __name__ == '__main__':
    # data_param1 = r'0602_3_ws400_ol0.5_important_all_20.csv'  # 18
    # data_param1 = r'0602_3_ws500_ol0.5_important_all_20.csv'  # 14
    data_param1 = r"../../../output/activity/step_4_feature_selection/acc_data_important_all_20.csv"  # 14
    combinations = list(itertools.combinations([9, 11], 2))
    for sequence in combinations:
        activityCombinationPreprocessor = ActivityCombinationPreprocessor(data_param1,
                                                                          sequence=list(sequence), scheme=1,
                                                                          samples=None)
        activityCombinationPreprocessor.run()
    # ==================================================================================
    # combinations = list(itertools.combinations([2, 3, 4, 5, 9, 11], 5))
    # for sequence in combinations:
    #     acc_dict = {1: 0.4296, 2: 0.5111, 3: 0.5333, 4: 0.5481, 5: 0.4963, 6: 0.4584, 7: 0.4667, 8: 0.3943,
    #                 9: 0.5071, 10: 0.4948, 11: 0.5700, 12: 0.5239, 13: 0.4680, 14: 0.5266, 15: 0.4903, 16: 0.5128}
    #     weights = [acc_dict[activity_id_num] for activity_id_num in list(sequence)]
    #     activityCombinationPreprocessor = ActivityCombinationPreprocessor(data_param1,
    #                                                                       sequence=list(sequence), scheme=2,
    #                                                                       samples=None, weights=weights)
    #     activityCombinationPreprocessor.run()
    # ===================================================================================
    # # 1.纵向拼接示例，data_path='./impoetantall20.csv'表示从哪个文件路径中读取数据,sequence=[1,2]表示拼接的活动是每个受试者的1和2活动序号数据，samples=24表示每个活动截取24个窗口拼接，如果该活动没有24个窗口则截取最大长度
    # activityCombinationPreprocessor = ActivityCombinationPreprocessor(r'./impoetantall20.csv',
    #                                                                   [1, 2], 0, samples=24)
    # # samples为选填默认为None，自动获取每个活动的所有窗口
    # # activityCombinationPreprocessor = ActivityCombinationPreprocessor(r'./impoetantall20.csv',
    #                                                                   # [1, 2])
    # activityCombinationPreprocessor.run()  # 运行处理程序
    #
    # # 2.水平拼接示例，data_path='./impoetantall20.csv'表示从哪个文件路径中读取数据,sequence=[1,2]表示拼接的活动是每个受试者的1和2活动序号数据，samples=24表示每个活动截取24个窗口拼接，如果该活动没有24个窗口则截取最大长度
    # # 可能拼接后的数据会出现NaN值，因为活动时间长度不一样，活动2的数据可能多与活动21，所以导致水平拼接后，在一行数据中，活动2部分有数据，活动1部分为NaN值。
    # activityCombinationPreprocessor = ActivityCombinationPreprocessor(r'./impoetantall20.csv',
    #                                                                   [1, 2], 1, samples=24)
    # # samples为选填默认为None，自动获取每个活动的所有窗口
    # # activityCombinationPreprocessor = ActivityCombinationPreprocessor(r'./impoetantall20.csv',
    # #                                                                   [1, 2], 1)
    # activityCombinationPreprocessor.run()  # 运行处理程序
    #
    # # 3.数据融合示例，data_path='./impoetantall20.csv'表示从哪个文件路径中读取数据,sequence=[1,2]表示参与融合的活动是每个受试者的1和2活动序号数据，samples=24表示每个活动截取24个窗口拼接，
    # # weights=[0.5, 0.5]表示对活动1的数据乘以权重0.9，对活动2数据乘以权重0.1融合后数据维度不发生改变，数值变换为=活动1*weights[0]+活动2*weights[1]。
    # # 注意：如果进行数据融合时，在data_path指定文件中该受试者没有活动1和活动2，将会放弃该受试者的数据融合，从而融合后数据会缺失该受试者的数据。
    # activityCombinationPreprocessor = ActivityCombinationPreprocessor(r'./impoetantall20.csv',
    #                                                                   [1, 2], 2, samples=24, weights=[0.9, 0.1])
    # # samples为选填默认为None，自动获取每个活动的所有窗口
    # # activityCombinationPreprocessor = ActivityCombinationPreprocessor(r'./impoetantall20.csv',
    # #                                                                   [1, 2], 2, weights=[0.9, 0.1])
    # activityCombinationPreprocessor.run()  # 运行处理程序
    # ===================================================================================
