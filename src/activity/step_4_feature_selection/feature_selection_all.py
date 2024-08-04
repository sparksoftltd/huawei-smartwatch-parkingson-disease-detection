import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import os
import sys


from src.utils.PDDataLoader import PDDataLoader

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

import pandas as pd
import numpy as np


class FeatureSelector:
    def __init__(self, data):
        self.data = data
        self.correlation_threshold = None
        self.one_hot_correlated = False
        self.base_features = data.columns.tolist()
        self.one_hot_features = []
        self.corr_matrix = None
        self.record_collinear = pd.DataFrame()
        self.ops = {'collinear': []}

    def identify_collinear(self, correlation_threshold, one_hot=False):
        """
        Finds collinear features based on the correlation coefficient between features.
        For each pair of features with a correlation coefficient greater than `correlation_threshold`,
        only one of the pair is identified for removal.

        Parameters
        --------
        correlation_threshold : float between 0 and 1
            Value of the Pearson correlation coefficient for identifying correlation features

        one_hot : boolean, default = False
            Whether to one-hot encode the features before calculating the correlation coefficients
        """

        self.correlation_threshold = correlation_threshold
        self.one_hot_correlated = one_hot

        # Calculate the correlations between every column
        if one_hot:
            # One hot encoding
            features = pd.get_dummies(self.data)
            self.one_hot_features = [column for column in features.columns if column not in self.base_features]

            # Add one hot encoded data to original data
            self.data_all = pd.concat([features[self.one_hot_features], self.data], axis=1)

            corr_matrix = features.corr()

        else:
            corr_matrix = self.data.corr()

        self.corr_matrix = corr_matrix

        # Extract the upper triangle of the correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Select the features with correlations above the threshold
        to_drop = [column for column in upper.columns if any(upper[column].abs() > correlation_threshold)]

        # Dataframe to hold correlated pairs
        record_collinear = []

        # Iterate through the columns to drop to record pairs of correlated features
        for column in to_drop:
            # Find the correlated features
            corr_features = list(upper.index[upper[column].abs() > correlation_threshold])

            # Find the correlated values
            corr_values = list(upper[column][upper[column].abs() > correlation_threshold])
            drop_features = [column for _ in range(len(corr_features))]

            # Record the information (need a temp df for now)
            temp_df = pd.DataFrame.from_dict({'drop_feature': drop_features,
                                              'corr_feature': corr_features,
                                              'corr_value': corr_values})

            # Add to list if temp_df is not empty and has no all-NA columns
            if not temp_df.empty and temp_df.notna().any().any():
                record_collinear.append(temp_df)

        # Concatenate all the dataframes in the list
        if record_collinear:
            self.record_collinear = pd.concat(record_collinear, ignore_index=True)
        else:
            self.record_collinear = pd.DataFrame(columns=['drop_feature', 'corr_feature', 'corr_value'])

        self.ops['collinear'] = to_drop

        print('%d features with a correlation magnitude greater than %0.2f.\n' % (
        len(self.ops['collinear']), self.correlation_threshold))

        return self.ops['collinear']

def check_features(activity_id):
    data_path = "../../../output/activity/step_2_select_sensors"
    data_name = "acc_data.csv"
    fold_groups_path = "../../../input/activity/step_3_output_feature_importance"
    fold_groups_name = "fold_groups_new_with_combinations.csv"
    severity_mapping = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 3}

    classifier = PDDataLoader(activity_id, os.path.join(data_path, data_name),
                              os.path.join(fold_groups_path, fold_groups_name), severity_mapping=severity_mapping)
    target = classifier.PD_data['Severity_Level']  # Replace 'target' with your actual target column name

    # Create the feature selector
    fs = FeatureSelector(classifier.PD_data[classifier.feature_name], target)

    # Identify and remove unwanted features
    missing_features = fs.identify_missing()
    collinear_features = fs.identify_collinear()
    zero_importance_features = fs.identify_zero_importance()
    low_importance_features = fs.identify_low_importance()
    single_unique_features = fs.identify_single_unique()


if __name__ == '__main__':
    data_path = "../../../output/activity/step_2_select_sensors"
    data_name = "acc_data.csv"
    fold_groups_path = "../../../input/activity/step_3_output_feature_importance"
    fold_groups_name = "fold_groups_new_with_combinations.csv"
    severity_mapping = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 3}

    classifier = PDDataLoader(3, os.path.join(data_path, data_name),
                              os.path.join(fold_groups_path, fold_groups_name), severity_mapping=severity_mapping)
    target = classifier.PD_data['Severity_Level']  # Replace 'target' with your actual target column name

    # Create the feature selector
    fs = FeatureSelector(classifier.PD_data[classifier.feature_name])
    fs.identify_collinear(correlation_threshold=0.9, one_hot=False)
    correlated_features = fs.ops['collinear']
    print(correlated_features)
