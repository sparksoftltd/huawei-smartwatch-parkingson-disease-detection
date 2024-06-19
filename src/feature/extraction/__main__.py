import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from utils import pd_utils

def main() -> None:
    """
    Main function.
    """
    logger.info(
        f"Current module: {get_module_name()}, tzname: {datetime.now().astimezone().tzname()}, localzone: {get_localzone()}"
    )
    logger.info(
        f"Starting {get_module_name()} using Python {platform.python_version()} on "
        f"{platform.node()} with PID {os.getpid()} ({Path(__file__).parent})"
    )

    data_path= "../../../data/clean_raw_data/PD/"   #  手动修改个人路径

    # 声明需要提取的特征
    fea_column = ['acc_peaks_normal', 'acc_peaks_abnormal', 'acc_fea_autoy', 'acc_fea_auto_num', 'acc_t_xyCor', 'acc_t_xzCor', 'acc_t_xaCor', 'acc_t_yzCor', 'acc_t_yaCor', 'acc_t_zaCor', 'acc_f_peakXY1', 'acc_f_peakXY2', 'acc_p_peakXY1', 'acc_p_peakXY2', 'acc_a_peakXY1', 'acc_a_peakXY2', 'acc_t_amp_x', 'acc_t_mean_x', 'acc_t_max_x', 'acc_t_std_x', 'acc_t_var_x', 'acc_t_entr_x', 'acc_t_lgEnergy_x', 'acc_t_sma_x', 'acc_t_interq_x', 'acc_t_skew_x', 'acc_t_kurt_x', 'acc_t_rms_x', 'acc_f_amp_x', 'acc_f_mean_x', 'acc_f_max_x', 'acc_f_std_x', 'acc_f_var_x', 'acc_f_entr_x', 'acc_f_lgEnergy_x', 'acc_f_sma_x', 'acc_f_interq_x', 'acc_f_skew_x', 'acc_f_kurt_x', 'acc_f_rms_x', 'acc_p_amp_x', 'acc_p_mean_x', 'acc_p_max_x', 'acc_p_std_x', 'acc_p_var_x', 'acc_p_entr_x', 'acc_p_lgEnergy_x', 'acc_p_sma_x', 'acc_p_interq_x', 'acc_p_skew_x', 'acc_p_kurt_x', 'acc_p_rms_x', 'acc_a_amp_x', 'acc_a_mean_x', 'acc_a_max_x', 'acc_a_std_x', 'acc_a_var_x', 'acc_a_entr_x', 'acc_a_lgEnergy_x', 'acc_a_sma_x', 'acc_a_interq_x', 'acc_a_skew_x', 'acc_a_kurt_x', 'acc_a_rms_x', 'acc_a_mainY_x', 'acc_a_subY_x', 'acc_a_cftor_x', 'acc_t_amp_y', 'acc_t_mean_y', 'acc_t_max_y', 'acc_t_std_y', 'acc_t_var_y', 'acc_t_entr_y', 'acc_t_lgEnergy_y', 'acc_t_sma_y', 'acc_t_interq_y', 'acc_t_skew_y', 'acc_t_kurt_y', 'acc_t_rms_y', 'acc_f_amp_y', 'acc_f_mean_y', 'acc_f_max_y', 'acc_f_std_y', 'acc_f_var_y', 'acc_f_entr_y', 'acc_f_lgEnergy_y', 'acc_f_sma_y', 'acc_f_interq_y', 'acc_f_skew_y', 'acc_f_kurt_y', 'acc_f_rms_y', 'acc_p_amp_y', 'acc_p_mean_y', 'acc_p_max_y', 'acc_p_std_y', 'acc_p_var_y', 'acc_p_entr_y', 'acc_p_lgEnergy_y', 'acc_p_sma_y', 'acc_p_interq_y', 'acc_p_skew_y', 'acc_p_kurt_y', 'acc_p_rms_y', 'acc_a_amp_y', 'acc_a_mean_y', 'acc_a_max_y', 'acc_a_std_y', 'acc_a_var_y', 'acc_a_entr_y', 'acc_a_lgEnergy_y', 'acc_a_sma_y', 'acc_a_interq_y', 'acc_a_skew_y', 'acc_a_kurt_y', 'acc_a_rms_y', 'acc_a_mainY_y', 'acc_a_subY_y', 'acc_a_cftor_y', 'acc_t_amp_z', 'acc_t_mean_z', 'acc_t_max_z', 'acc_t_std_z', 'acc_t_var_z', 'acc_t_entr_z', 'acc_t_lgEnergy_z', 'acc_t_sma_z', 'acc_t_interq_z', 'acc_t_skew_z', 'acc_t_kurt_z', 'acc_t_rms_z', 'acc_f_amp_z', 'acc_f_mean_z', 'acc_f_max_z', 'acc_f_std_z', 'acc_f_var_z', 'acc_f_entr_z', 'acc_f_lgEnergy_z', 'acc_f_sma_z', 'acc_f_interq_z', 'acc_f_skew_z', 'acc_f_kurt_z', 'acc_f_rms_z', 'acc_p_amp_z', 'acc_p_mean_z', 'acc_p_max_z', 'acc_p_std_z', 'acc_p_var_z', 'acc_p_entr_z', 'acc_p_lgEnergy_z', 'acc_p_sma_z', 'acc_p_interq_z', 'acc_p_skew_z', 'acc_p_kurt_z', 'acc_p_rms_z', 'acc_a_amp_z', 'acc_a_mean_z', 'acc_a_max_z', 'acc_a_std_z', 'acc_a_var_z', 'acc_a_entr_z', 'acc_a_lgEnergy_z', 'acc_a_sma_z', 'acc_a_interq_z', 'acc_a_skew_z', 'acc_a_kurt_z', 'acc_a_rms_z', 'acc_a_mainY_z', 'acc_a_subY_z', 'acc_a_cftor_z', 'acc_t_amp_a', 'acc_t_mean_a', 'acc_t_max_a', 'acc_t_std_a', 'acc_t_var_a', 'acc_t_entr_a', 'acc_t_lgEnergy_a', 'acc_t_sma_a', 'acc_t_interq_a', 'acc_t_skew_a', 'acc_t_kurt_a', 'acc_t_rms_a', 'acc_f_amp_a', 'acc_f_mean_a', 'acc_f_max_a', 'acc_f_std_a', 'acc_f_var_a', 'acc_f_entr_a', 'acc_f_lgEnergy_a', 'acc_f_sma_a', 'acc_f_interq_a', 'acc_f_skew_a', 'acc_f_kurt_a', 'acc_f_rms_a', 'acc_p_amp_a', 'acc_p_mean_a', 'acc_p_max_a', 'acc_p_std_a', 'acc_p_var_a', 'acc_p_entr_a', 'acc_p_lgEnergy_a', 'acc_p_sma_a', 'acc_p_interq_a', 'acc_p_skew_a', 'acc_p_kurt_a', 'acc_p_rms_a', 'acc_a_amp_a', 'acc_a_mean_a', 'acc_a_max_a', 'acc_a_std_a', 'acc_a_var_a', 'acc_a_entr_a', 'acc_a_lgEnergy_a', 'acc_a_sma_a', 'acc_a_interq_a', 'acc_a_skew_a', 'acc_a_kurt_a', 'acc_a_rms_a', 'acc_a_mainY_a', 'acc_a_subY_a', 'acc_a_cftor_a']

    # 特征提取的参数设置
    window_size=300
    overlapping_rate = 0.5
    frequency = 200
    pd_num = 2
    activity_num = 14
    side_l = "wristl"
    side_r = "wristr"

    Feature = pd_utils.FeatureExtractWithProcess(pd_num, activity_num, data_path, side_r, fea_column, window_size, overlapping_rate, frequency)

    label_data = pd.read_csv(data_path + "Information_Sheet_Version_1.csv")
    label_table = label_data[["PatientID","Severity_Level"]]
    label_table = label_table.drop(0)

    Feature['PatientID'] = Feature['PatientID'].astype(int)
    label_table['PatientID'] = label_table['PatientID'].astype(int)

    feature_label = pd.merge(Feature,label_table,on = 'PatientID')
    feature_label.to_csv(data_path + side_r +"_feature_label.csv", index=False, float_format="%.6f")

    elapsed = time.perf_counter() - __start_time
    logger.info(
        f"Started {get_module_name()}@{setup_cfg['metadata']['version']} in {timedelta(seconds=elapsed)}"
    )


if __name__ == "__main__":
    main()
