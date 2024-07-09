import pandas as pd
import datetime
import re
import os

# def format_sequence(sequence):
#     sequence = sorted(sequence)
#     # sequence =
#     if not sequence:
#         return ""
#
#     # 初始化结果字符串和追踪连续数字的起始和结束变量
#     result = ""
#     start = sequence[0]
#     end = start
#
#     for i in range(1, len(sequence)):
#         if sequence[i] == end + 1:
#             end = sequence[i]
#         else:
#             # 如果找到连续的部分，则将它们格式化为"x_y"形式，否则只写数字
#             if end > start:
#                 result += f"{start}_{end}+"
#             else:
#                 result += f"{start}+"
#             # 重新设置start和end为当前数字
#             start = sequence[i]
#             end = start
#
#     # 添加最后一段数字或数字范围
#     if end > start:
#         result += f"{start}_{end}"
#     else:
#         result += f"{start}"
#
#     return result
#
#
# def extract_ws_ol(filename):
#     # 编译正则表达式来匹配 ws 和 ol 后的数字
#     ws_pattern = re.compile(r"_ws(\d+)")
#     # ol_pattern = re.compile(r"_ol(\d+)")
#     # ol_pattern = re.compile(r"-?\d+\.\d+")
#     ol_pattern = re.compile(r"_ol(-?\d+\.?\d*)")
#
#     # 使用正则表达式查找数字
#     ws_match = ws_pattern.search(filename)
#     ol_match = ol_pattern.search(filename)
#
#     # 提取窗口大小和重叠
#     window_size = int(ws_match.group(1)) if ws_match else None
#     overlapping = float(ol_match.group(1)) if ol_match else None
#
#     return window_size, overlapping


"""
按照指定动作选取指定的column
"""
def feature_selection(data_path, sequence, choosefeaturenum):
    temp = pd.DataFrame()
    # data = pd.read_csv(r"242D_HC_PD_window300_wristr_acc_gro_200hz.csv")
    data = pd.read_csv(data_path)
    # window_size =
    # overlapping =
    # window_size, overlapping = extract_ws_ol(data_path)

    for se in sequence:
        print('正在提取动作{}的重要特征'.format(se))  #
        # tempdata = pd.read_csv(r"../../datasets/featureimportant/feature_important{}_60.csv".format(se))
        # tempdata = pd.read_csv(rf"{se}_shap_importance_0618.csv".format(se))
        # shap_summary.to_csv(os.path.join(r'../../../output/activity/step_3_output_feature_importance' ,
        #     f'{os.path.basename(self.data_path)}_{self.activity_id}_shap_importance.csv'), index=False
        # )
        tempdata = pd.read_csv(os.path.join(r'../../../output/activity/step_3_output_feature_importance',
            f'{os.path.basename(data_path)}_{se}_shap_importance.csv'))

        importantfeaturelist = [str(i) for i in tempdata['Feature'].values.tolist()]
        importantfeaturelist = importantfeaturelist[:choosefeaturenum]
        print(importantfeaturelist)
        # 已知列索引如何获取对应的列名
        # importantfeaturelist = list(map(int, importantfeaturelist))  # 获取指定的列索引
        # importantfeaturelist = data.columns[importantfeaturelist].tolist()
        print(importantfeaturelist)
        # importantfeaturelist.append('Activity_id')
        # importantfeaturelist.append('subject_id')
        # importantfeaturelist.append('severity_level')
        importantfeaturelist.append('PatientID')
        importantfeaturelist.append('activity_label')
        importantfeaturelist.append('Severity_Level')
        # 选取指定的动作
        cdata = data.loc[data['activity_label'] == se][importantfeaturelist]
        # 生成新的列名
        new_columns = [str(i) for i in range(0, choosefeaturenum)] + ['PatientID', 'activity_label', 'Severity_Level']
        # 重命名列名
        cdata.columns = new_columns
        temp = pd.concat([temp, cdata], ignore_index=True)
    print('特征挑选完成')
    print('正在保存特征')
    data = temp
    # 获取当前日期
    # now = datetime.datetime.now()
    # 格式化月和日
    # month_day = now.strftime("%m%d")  # 月和日格式为"MMdd"，例如"0509"
    # formatted_sequence = format_sequence(sequence)

    # 构造文件名
    filename = rf"../../../output/activity/step_4_feature_selection/{os.path.splitext(os.path.basename(data_path))[0]}_important_all_{choosefeaturenum}.csv"
    data.to_csv(f"{filename}", index=False)
    print('特征保存成功')


if __name__ == '__main__':
    data_path_param1 = r"../../../output/activity/step_2_select_sensors/acc_data.csv"
    sequence_param2 = list(range(1, 17))
    choosefeaturenum_param3 = 20

    feature_selection(data_path_param1, sequence_param2, choosefeaturenum_param3)
