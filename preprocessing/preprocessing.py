import os

import numpy as np
import pandas as pd
from loguru import logger
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.datasets import load_veterans_lung_cancer, load_flchain, load_gbsg2
from sklearn.model_selection import train_test_split


def load_data_from_excel(filepath, time_column, event_column):
    df = pd.read_excel(filepath)
    data_x = df.drop(columns=[time_column, event_column])
    data_y = df[[event_column, time_column]]
    return data_x, data_y

def load_and_preprocess_data(source='veterans', filepath=None, time_column=None, event_column=None):
    if source == 'veterans':
        data_x, data_y = load_veterans_lung_cancer()
    elif source == 'flchain':
        data_x, data_y = load_flchain()
    elif source == 'excel':
        if filepath is None:
            raise ValueError("filepath must be provided when loading data from Excel!")
        data_x, data_y = load_data_from_excel(filepath, time_column, event_column)
    else:
        raise ValueError(f"Unknown data source: {source}")
    # Remove highly correlated features
    data_x = remove_highly_correlated_features(data_x, time_column, event_column)

    # Convert DataFrame to Structured Array
    def to_structured_array(df):
        return np.array(list(zip(df[event_column], df[time_column])), dtype=[(event_column, '?'), (time_column, '<f8')])
    data_y = to_structured_array(data_y)
    # Split Data
    data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=42)
    # Ensure Test Set's Survival Times are Contained Within Training Set's Survival Times
    train_min, train_max = data_y_train[time_column].min(), data_y_train[time_column].max()
    # Finding violating indices
    lower_bound_violations = np.where(data_y_test[time_column] < train_min)
    upper_bound_violations = np.where(data_y_test[time_column] > train_max)
    violating_indices = np.hstack([lower_bound_violations[0], upper_bound_violations[0]])
    # Moving violating test set entries to the training set
    if len(violating_indices) > 0:
        data_x_train = pd.concat([data_x_train, data_x_test.iloc[violating_indices]])
        data_y_train = np.hstack((data_y_train, data_y_test[violating_indices]))
        data_x_test.drop(data_x_test.index[violating_indices], inplace=True)
        data_y_test = np.delete(data_y_test, violating_indices)
    # Verify that test set's survival times are now within the training set's range
    y_events_test = data_y_test[data_y_test[event_column]]
    test_min, test_max = y_events_test[time_column].min(), y_events_test[time_column].max()
    assert train_min <= test_min and test_max <= train_max, "Test data time range is not within training data time range."
    return data_x_train, data_x_test, data_y_train, data_y_test

def fit_and_score_features(X, y):
    n_features = X.shape[1]
    scores = np.empty(n_features)
    # m =RandomSurvivalForest()
    m = CoxPHSurvivalAnalysis(alpha=0.1)
    # m =CoxnetSurvivalAnalysis(fit_baseline_model=True,l1_ratio=0.9, alpha_min_ratio=0.001)
    # m =CoxnetSurvivalAnalysis(fit_baseline_model=True, l1_ratio=0.5, alpha_min_ratio='auto')
    for j in range(n_features):
        Xj = X[:, j : j + 1]
        m.fit(Xj, y)
        scores[j] = m.score(Xj, y)
    return scores


def remove_highly_correlated_features(data, time_column, event_column, threshold=0.90):
    columns_to_exclude = [col for col in [time_column, event_column] if col in data.columns]
    feature_data = data.drop(columns=columns_to_exclude)
    corr_matrix = feature_data.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    reduced_data = data.drop(columns=to_drop)
    for col in columns_to_exclude:
        reduced_data[col] = data[col]

    return reduced_data

