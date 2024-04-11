import os
import pickle

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None
from loguru import logger
from sksurv.datasets import load_veterans_lung_cancer, load_flchain
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer  # required for IterativeImputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler


class Preprocessing:
    def __init__(self, config) -> None:
        self.in_file = config.meta.in_file
        self.event_column = config.meta.events
        self.time_column = config.meta.times
        self.save_as_pickle = config.preprocessing.save_as_pickle
        self.corr_threshold = config.preprocessing.corr_threshold
        self.test_size = config.preprocessing.test_size
        self.replace_zero_time_with = config.preprocessing.replace_zero_time_with
        self.impute_strategy = config.preprocessing.impute_strategy
        self.normalisation = config.preprocessing.normalisation

    def __call__(self, seed):
        self.seed = seed
        self.load_data()
        self.split_data()
        self.impute_data()
        self.normalise_data()
        self.remove_highly_correlated_features()

        if self.save_as_pickle:
            data_dict = {
                "data_x_train": self.data_x_train,
                "data_x_test": self.data_x_test,
                "data_y_train": self.data_y_train,
                "data_y_test": self.data_y_test,
            }
            data_out_file = f'{os.path.splitext(self.in_file)[0]}_data_split_seed_{self.seed}.pkl'
            with open(data_out_file, 'wb') as f:
                pickle.dump(data_dict, f)
            logger.info(f'Saved data split to {data_out_file}')

        self.data_y_train = self.to_structured_array(self.data_y_train)  # scikit-survival requires structured array
        self.data_y_test = self.to_structured_array(self.data_y_test)

        return self.data_x_train, self.data_x_test, self.data_y_train, self.data_y_test

    def load_data(self):
        if self.in_file == 'veterans':
            self.data_x, self.data_y = load_veterans_lung_cancer()
        elif self.in_file == 'flchain':
            self.data_x, self.data_y = load_flchain()
        else:
            try:
                data = pd.read_excel(self.in_file)
                data = data.apply(pd.to_numeric, errors='coerce')  # replace non-numeric entries with NaN
                data = data.dropna(how='all', axis=1)  # drop columns with all NaN
                self.data_x = data.drop(columns=[self.time_column, self.event_column])
                self.data_y = data[[self.event_column, self.time_column]]
                self.data_y[self.time_column] = self.data_y[self.time_column].replace(
                    0, self.replace_zero_time_with
                )  # some models do not accept t <= 0 -> set to small value > 0
            except FileNotFoundError:
                logger.error(f'File {self.in_file} not found, check the path in the config.yaml file.')
                raise

    def split_data(self):
        self.data_x_train, self.data_x_test, self.data_y_train, self.data_y_test = train_test_split(
            self.data_x,
            self.data_y,
            test_size=self.test_size,
            stratify=self.data_y[self.event_column],
            random_state=self.seed,
        )
        # Ensure Test Set's Survival Times are Contained Within Training Set's Survival Times
        train_min, train_max = self.data_y_train[self.time_column].min(), self.data_y_train[self.time_column].max()
        lower_bound_violations = np.where(self.data_y_test[self.time_column] < train_min)
        upper_bound_violations = np.where(self.data_y_test[self.time_column] > train_max)
        violating_indices = np.hstack([lower_bound_violations[0], upper_bound_violations[0]])
        # Move violating test set entries to the training set
        if len(violating_indices) > 0:
            self.data_x_train = pd.concat([self.data_x_train, self.data_x_test.iloc[violating_indices]])
            self.data_y_train = pd.concat([self.data_y_train, self.data_y_test.iloc[violating_indices]])
            self.data_x_test.drop(self.data_x_test.index[violating_indices], inplace=True)
            self.data_y_test.drop(self.data_y_test.index[violating_indices], inplace=True)
        # Verify that test set's survival times are now within the training set's range
        y_events_test = self.data_y_test[self.data_y_test[self.event_column] == 1]
        test_min, test_max = y_events_test[self.time_column].min(), y_events_test[self.time_column].max()
        assert (
            train_min <= test_min and test_max <= train_max
        ), "Test data time range is not within training data time range."

    def impute_data(self):
        if self.impute_strategy in ['mean', 'median', 'constant']:
            self.imputer = SimpleImputer(strategy=self.impute_strategy)
        elif self.impute_strategy == 'mode':
            # imputed_value = subset[column].mode().iloc[0]
            pass
        elif self.impute_strategy == 'iterative':
            self.imputer = IterativeImputer(random_state=self.seed, max_iter=100, keep_empty_features=True)
        else:
            raise ValueError(f"Unknown imputation {self.impute_strategy}")

        imp_train = self.imputer.fit_transform(self.data_x_train)
        self.data_x_train = pd.DataFrame(imp_train, index=self.data_x_train.index, columns=self.data_x_train.columns)
        imp_test = self.imputer.transform(self.data_x_test)
        self.data_x_test = pd.DataFrame(imp_test, index=self.data_x_test.index, columns=self.data_x_test.columns)

    def normalise_data(self):
        if self.normalisation == 'z-score':
            self.scaler = StandardScaler()

        nunique = self.data_x_train.nunique()
        non_categorical = list(nunique[nunique > 5].index)
        if self.event_column in non_categorical:
            non_categorical.remove(self.event_column)
        if self.time_column in non_categorical:
            non_categorical.remove(self.time_column)

        self.data_x_train[non_categorical] = self.scaler.fit_transform(self.data_x_train[non_categorical])
        self.data_x_test[non_categorical] = self.scaler.transform(self.data_x_test[non_categorical])

    def remove_highly_correlated_features(self):
        corr_matrix = self.data_x_train.corr()
        importances = self.data_x_train.corrwith(self.data_y_train, axis=0).abs()
        importances = importances.sort_values(ascending=False)
        corr_matrix = corr_matrix.reindex(index=importances.index, columns=importances.index).abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > self.corr_threshold)]

        self.data_x_train = self.data_x_train.drop(columns=to_drop)
        self.data_x_test = self.data_x_test.drop(columns=to_drop)

    def to_structured_array(self, df):
        return np.array(
            list(zip(df[self.event_column], df[self.time_column])),
            dtype=[(self.event_column, '?'), (self.time_column, '<f8')],
        )
