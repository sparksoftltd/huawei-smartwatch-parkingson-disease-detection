import os.path

import pandas as pd


def select_data(column_names, data_csv_path=r'../../../output/activity/step_1_feature_extraction/wristr_acc_gyro_mag_feature_label.csv',
                feature_name_csv_path='../../../input/activity/step_1_feature_extraction/raw/feature_name.csv'):
    # 读取CSV文件
    df_read = pd.read_csv(feature_name_csv_path)
    data_read = pd.read_csv(data_csv_path)
    column_name_ls = []
    for column_name in column_names:
        column_name_ls = column_name_ls + df_read[column_name].tolist()
    column_name_ls.extend(['PatientID', 'activity_label', 'Severity_Level'])
    selected_data = data_read[column_name_ls]
    output_name = '_'.join(column_names) + '_data.csv'
    output_name_path = os.path.join(r'../../../output/activity/step_2_select_sensors', f'{output_name}')
    selected_data.to_csv(output_name_path, index=False)


if __name__ == '__main__':
    select_data(['acc',
                 ])
