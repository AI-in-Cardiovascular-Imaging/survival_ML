import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from loguru import logger
from sksurv.datasets import load_veterans_lung_cancer, load_flchain
from sklearn.model_selection import train_test_split


class Preprocessing:
    def __init__(self, config) -> None:
        self.in_file = config.meta.in_file
        self.event_column = config.meta.events
        self.time_column = config.meta.times
        self.corr_threshold = config.preprocessing.corr_threshold
        self.test_size = config.preprocessing.test_size
        self.replace_zero_time_with = config.preprocessing.replace_zero_time_with

    def __call__(self, seed):
        self.seed = seed
        self.load_data()
        self.remove_highly_correlated_features()
        self.split_data()

        return self.data_x_train, self.data_x_test, self.data_y_train, self.data_y_test

    def load_data(self):
        if self.in_file == 'veterans':
            self.data_x, self.data_y = load_veterans_lung_cancer()
        elif self.in_file == 'flchain':
            self.data_x, self.data_y = load_flchain()
        else:
            try:
                data = pd.read_excel(self.in_file)
                self.data_x = data.drop(columns=[self.time_column, self.event_column])
                self.data_y = data[[self.event_column, self.time_column]]
                self.data_y[self.time_column] = self.data_y[self.time_column].replace(
                    0, self.replace_zero_time_with
                )  # some models do not accept t <= 0 -> set to small value > 0
            except FileNotFoundError:
                logger.error(f'File {self.in_file} not found, check the path in the config.yaml file.')
                raise

        self.data_y = self.to_structured_array(self.data_y)

    def to_structured_array(self, df):
        return np.array(
            list(zip(df[self.event_column], df[self.time_column])),
            dtype=[(self.event_column, '?'), (self.time_column, '<f8')],
        )

    def remove_highly_correlated_features(self):
        columns_to_exclude = [col for col in [self.time_column, self.event_column] if col in self.data_x.columns]
        feature_data = self.data_x.drop(columns=columns_to_exclude)
        corr_matrix = feature_data.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > self.corr_threshold)]
        reduced_data = self.data_x.drop(columns=to_drop)
        for col in columns_to_exclude:
            reduced_data[col] = self.data_x[col]

        return reduced_data

    def split_data(self):
        self.data_x_train, self.data_x_test, self.data_y_train, self.data_y_test = train_test_split(
            self.data_x, self.data_y, test_size=self.test_size, random_state=self.seed
        )
        # Ensure Test Set's Survival Times are Contained Within Training Set's Survival Times
        train_min, train_max = self.data_y_train[self.time_column].min(), self.data_y_train[self.time_column].max()
        # Finding violating indices
        lower_bound_violations = np.where(self.data_y_test[self.time_column] < train_min)
        upper_bound_violations = np.where(self.data_y_test[self.time_column] > train_max)
        violating_indices = np.hstack([lower_bound_violations[0], upper_bound_violations[0]])
        # Moving violating test set entries to the training set
        if len(violating_indices) > 0:
            self.data_x_train = pd.concat([self.data_x_train, self.data_x_test.iloc[violating_indices]])
            self.data_y_train = np.hstack((self.data_y_train, self.data_y_test[violating_indices]))
            self.data_x_test.drop(self.data_x_test.index[violating_indices], inplace=True)
            self.data_y_test = np.delete(self.data_y_test, violating_indices)
        # Verify that test set's survival times are now within the training set's range
        y_events_test = self.data_y_test[self.data_y_test[self.event_column]]
        test_min, test_max = y_events_test[self.time_column].min(), y_events_test[self.time_column].max()
        assert (
            train_min <= test_min and test_max <= train_max
        ), "Test data time range is not within training data time range."
