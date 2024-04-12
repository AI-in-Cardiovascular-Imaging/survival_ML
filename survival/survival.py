import os
import sys
import pickle

import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sksurv.preprocessing import OneHotEncoder
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
import sksurv.metrics as sksurv_metrics

from survival.init_estimators import init_estimators
from helpers.nested_dict import NestedDefaultDict


class Survival:
    def __init__(self, config, progress_manager) -> None:
        self.progress_manager = progress_manager
        self.encoder = OneHotEncoder()
        self.overwrite = config.meta.overwrite
        self.out_dir = config.meta.out_dir
        self.table_file = os.path.join(self.out_dir, 'results_table.xlsx')
        self.results_file = os.path.join(self.out_dir, 'results.pkl')
        self.event_column = config.meta.events
        self.time_column = config.meta.times
        self.n_seeds = config.meta.n_seeds
        self.n_workers = config.meta.n_workers
        self.scoring = config.survival.scoring
        self.scalers_dict = config.survival.scalers
        self.selectors_dict = config.survival.feature_selectors
        self.selector_params = config.survival.feature_selector_params
        self.models_dict = config.survival.models
        self.model_params = config.survival.model_params
        self.total_combinations = (
            self.n_seeds
            * sum(self.scalers_dict.values())
            * sum(self.selectors_dict.values())
            * sum(self.models_dict.values())
        )
        self.result_cols = [
            "Seed",
            "Scaler",
            "Selector",
            "Model",
            "c_index",
            "c_index_ipcw",
            "auc",
            "brier_score",
        ]

        try:  # load results if file exists
            if self.overwrite:
                raise FileNotFoundError  # force same behaviour as if file didn't exist

            self.results_table = pd.read_excel(self.table_file)  # human-readable results
            self.row_to_write = self.results_table.shape[0]
            to_concat = pd.DataFrame(
                index=range(self.total_combinations - self.row_to_write),
                columns=self.result_cols,
            )
            self.results_table = pd.concat([self.results_table, to_concat], ignore_index=True)

            with open(self.results_file, 'rb') as file:
                self.results = pickle.load(file)  # results for report
        except FileNotFoundError:
            self.results_table = pd.DataFrame(
                index=range(self.total_combinations),
                columns=self.result_cols,
            )
            self.row_to_write = 0

            self.results = NestedDefaultDict()

    def __call__(self, seed, x_train, y_train, x_test, y_test):
        self.seed = seed
        self.scalers, self.selectors, self.models = init_estimators(
            self.seed, self.n_workers, self.scalers_dict, self.selectors_dict, self.models_dict, self.scoring
        )
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.results[self.seed]['x_train'] = x_train
        self.results[self.seed]['y_train'] = y_train
        self.results[self.seed]['x_test'] = x_test
        self.results[self.seed]['y_test'] = y_test
        self.fit_and_evaluate_pipeline()

        return self.results_table

    def fit_and_evaluate_pipeline(self):
        pbar = self.progress_manager.counter(
            total=self.total_combinations, desc="Training and evaluating all combinations", unit='it', leave=False
        )
        for scaler_name, scaler in self.scalers.items():
            for selector_name, selector in self.selectors.items():
                for model_name, model in self.models.items():
                    if (  # skip if already evaluated
                        (self.results_table["Seed"] == self.seed)
                        & (self.results_table["Scaler"] == scaler_name)
                        & (self.results_table["Selector"] == selector_name)
                        & (self.results_table["Model"] == model_name)
                    ).any():
                        logger.info(f"Skipping {scaler_name} - {selector_name} - {model_name}")
                        pbar.update()
                        continue
                    logger.info(f"Training {scaler_name} - {selector_name} - {model_name}")
                    row = {"Seed": self.seed, "Scaler": scaler_name, "Selector": selector_name, "Model": model_name}
                    # Create pipeline and parameter grid
                    estimator = getattr(sksurv_metrics, self.scoring)(model)  # attach scoring function
                    pipe = Pipeline(
                        [
                            ("encoder", self.encoder),
                            ('scaler', scaler),
                            ("selector", selector),
                            ("model", estimator),
                        ]
                    )
                    model_params = OmegaConf.to_container(self.model_params[model_name], resolve=True)
                    for param in model_params:
                        if 'alphas' in param:  # alphas need to be a list for some reason
                            model_params[param] = [model_params[param]]
                    param_grid = {**self.selector_params[selector_name], **model_params}
                    # Grid search and evaluate model
                    # Larger n_splits may lead to ValueError: time must be smaller than largest observed time point
                    cv = KFold(n_splits=3, random_state=self.seed, shuffle=True)
                    gcv = GridSearchCV(
                        pipe,
                        param_grid,
                        return_train_score=True,
                        cv=cv,
                        n_jobs=self.n_workers,
                        error_score='raise',
                    )
                    gcv.fit(self.x_train, self.y_train)
                    logger.info(f'Evaluating {scaler_name} - {selector_name} - {model_name}')
                    metrics = self.evaluate_model(gcv, scaler_name, selector_name, model_name)
                    row.update(metrics)
                    self.results_table.loc[self.row_to_write] = row
                    self.row_to_write += 1
                    self.results_table = self.results_table.sort_values(["Seed", "Scaler", "Selector", "Model"])
                    logger.info(f'Saving results to {self.out_dir}')
                    try:  # ensure that intermediate results are not corrupted by KeyboardInterrupt
                        self.save_results()
                    except KeyboardInterrupt:
                        logger.warning('Keyboard interrupt detected, saving results before exiting...')
                        self.save_results()
                        sys.exit(130)
                    pbar.update()

        pbar.close()

    def evaluate_model(self, gcv, scaler_name, selector_name, model_name):
        risk_scores = gcv.predict(self.x_test)
        c_index = concordance_index_censored(
            self.y_test[self.event_column], self.y_test[self.time_column], risk_scores)[0]
        c_index_ipcw = concordance_index_ipcw(self.y_train, self.y_test, risk_scores,
                                              tau=self.y_train[self.time_column].max() - 1)[0]
        times = np.percentile(self.y_test[self.time_column], np.linspace(5, 91, 15))
        _, mean_auc = cumulative_dynamic_auc(self.y_train, self.y_test, risk_scores, times)

        best_estimator = gcv.best_estimator_
        self.results[self.seed][scaler_name][selector_name][model_name]['best_estimator'] = best_estimator

        # Check if the model has the 'predict_survival_function' method
        if hasattr(best_estimator["model"], "predict_survival_function"):
            surv_func = best_estimator.predict_survival_function(self.x_test)
            estimates = np.array([[func(t) for t in times] for func in surv_func])
            brier_score = integrated_brier_score(self.y_train, self.y_test, estimates, times)
        else:
            brier_score = None
        metrics_dict = {
            'c_index': c_index,
            'c_index_ipcw': c_index_ipcw,
            'auc': mean_auc,
            'brier_score': brier_score,
        }

        return metrics_dict

    def save_results(self):
        self.results_table.to_excel(self.table_file, index=False)
        with open(self.results_file, 'wb') as file:
            pickle.dump(self.results, file)
