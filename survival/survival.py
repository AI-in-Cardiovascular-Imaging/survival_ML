import os
import sys
import pickle

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sksurv.preprocessing import OneHotEncoder
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
import sksurv.metrics as sksurv_metrics
from skopt import BayesSearchCV

from survival.init_estimators import init_estimators, set_params_search_space
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
        self.models_dict = config.survival.models
        self.n_cv_splits = config.survival.n_cv_splits
        self.n_iter_search = config.survival.n_iter_search
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
            "mean_val_cindex",
            "std_val_cindex",
            "c_index_ipcw",
            "brier_score",
            "auc_mean",
            "auc",
            'evaluation_times',
            'truncation_time'
        ]

        self.model_params, self.selector_params = set_params_search_space()

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
                    try:
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
                        cv = StratifiedKFold(n_splits=self.n_cv_splits, random_state=self.seed, shuffle=True)
                        stratified_folds = [x for x in cv.split(self.x_train, self.y_train[self.event_column])]
                        self.tau = np.min(  # truncation time
                            [np.max(self.y_train[self.time_column][train_idx]) for train_idx, _ in stratified_folds]) - 1
                        estimator = getattr(sksurv_metrics, self.scoring)(model, tau=self.tau)
                        pipe = Pipeline(
                            [
                                ("encoder", self.encoder),
                                ('scaler', scaler),
                                ("selector", selector),
                                ("model", estimator),
                            ]
                        )
                        param_grid = {**self.selector_params[selector_name], **self.model_params[model_name]}
                        # Grid search
                        gcv = BayesSearchCV(
                            pipe,
                            param_grid,
                            n_iter=self.n_iter_search,
                            n_points=20,
                            return_train_score=True,
                            cv=stratified_folds,
                            n_jobs=self.n_workers,
                            error_score='raise',
                        )
                        gcv.fit(self.x_train, self.y_train)
                        # Evaluate model
                        logger.info(f'Evaluating {scaler_name} - {selector_name} - {model_name}')
                        metrics_cv = {
                            "mean_val_cindex": gcv.cv_results_["mean_test_score"][gcv.best_index_],
                            "std_val_cindex": gcv.cv_results_["std_test_score"][gcv.best_index_]
                        }
                        row.update(metrics_cv)
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
                    except Exception as e:
                        print(
                            f"Error encountered for Scaler={scaler_name}, Selector={selector_name}, Model={model_name}. Error message: {str(e)}")

        pbar.close()

    def evaluate_model(self, gcv, scaler_name, selector_name, model_name):
        best_estimator = gcv.best_estimator_   # extract best estimator

        # To estimate IPCW, test survival times must lie within the range of train survival times. Sksurv docs claim
        # that this can be achieved specifying evaluation times accordingly, but it doesn't seem to work. Thus, I
        # explicitly truncate test data.
        tau = np.max(self.y_train[self.time_column]) - 1  # truncation time
        y_test_truncated = self.y_test.copy()
        mask = y_test_truncated[self.time_column] > tau
        y_test_truncated[self.time_column][mask] = tau
        y_test_truncated[self.event_column][mask] = 0
        times = np.percentile(y_test_truncated[self.time_column], np.linspace(5, 90, 15))

        # Risk scores for the test set (time-independent) and C-Index
        risk_scores = gcv.predict(self.x_test)
        c_index_ipcw = concordance_index_ipcw(self.y_train, self.y_test, risk_scores, tau=tau)[0]

        # CD-AUC, if possible for time-dependent predicted risk
        if hasattr(best_estimator["model"], "predict_cumulative_hazard_function"):
            rsf_chf_funcs = best_estimator.predict_cumulative_hazard_function(self.x_test)
            risk_scores = np.row_stack([chf(times) for chf in rsf_chf_funcs])
        auc, mean_auc = cumulative_dynamic_auc(self.y_train, y_test_truncated, risk_scores, times)

        # Adding the best estimator to output file
        self.results[self.seed][scaler_name][selector_name][model_name]['best_estimator'] = best_estimator

        # If the model has the 'predict_survival_function' method, compute Brier score
        if hasattr(best_estimator["model"], "predict_survival_function"):
            surv_func = best_estimator.predict_survival_function(self.x_test)
            estimates = np.array([[func(t) for t in times] for func in surv_func])
            brier_score = integrated_brier_score(self.y_train, y_test_truncated, estimates, times)
        else:
            brier_score = None
        metrics_dict = {
            'c_index_ipcw': c_index_ipcw,
            'brier_score': brier_score,
            'auc_mean': mean_auc,
            'auc': auc.tolist(),
            'evaluation_times': times.tolist(),
            'truncation_time': tau
        }

        return metrics_dict

    def save_results(self):
        os.makedirs(os.path.dirname(self.table_file), exist_ok=True)
        self.results_table.to_excel(self.table_file, index=False)
        with open(self.results_file, 'wb') as file:
            pickle.dump(self.results, file)
