import sys

import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf
from sklearn.model_selection import GridSearchCV, StratifiedKFold
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


class Survival:
    def __init__(self, config, progress_manager) -> None:
        self.progress_manager = progress_manager
        self.encoder = OneHotEncoder()
        self.overwrite = config.meta.overwrite
        self.out_file = config.meta.out_file
        self.event_column = config.meta.events
        self.time_column = config.meta.times
        self.n_seeds = config.meta.n_seeds
        self.n_workers = config.meta.n_workers
        self.scalers_dict = config.survival.scalers
        self.selectors_dict = config.survival.feature_selectors
        self.selector_params = config.survival.feature_selector_params
        self.scoring = config.survival.scoring
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
            "c-index",
            "c-index (IPCW)",
            "AUC",
            "Brier Score",
        ]

        try:  # load results if file exists
            self.results = pd.read_excel(self.out_file)
            self.row_to_write = self.results.shape[0]
            to_concat = pd.DataFrame(
                index=range(self.total_combinations - self.row_to_write),
                columns=self.result_cols,
            )
            self.results = pd.concat([self.results, to_concat], ignore_index=True)
            if self.overwrite:
                raise FileNotFoundError  # force same behaviour as if file didn't exist
        except FileNotFoundError:
            self.results = pd.DataFrame(
                index=range(self.total_combinations),
                columns=self.result_cols,
            )
            self.row_to_write = 0

    def __call__(self, seed, data_x_train, data_y_train, data_x_test, data_y_test):
        self.seed = seed
        self.scalers, self.selectors, self.models = init_estimators(
            self.seed, self.n_workers, self.scalers_dict, self.selectors_dict, self.models_dict, self.scoring
        )
        self.data_x_train = data_x_train
        self.data_y_train = data_y_train
        self.data_x_test = data_x_test
        self.data_y_test = data_y_test
        self.fit_and_evaluate_pipeline()

        return self.results

    def fit_and_evaluate_pipeline(self):
        pbar = self.progress_manager.counter(
            total=self.total_combinations, desc="Training and evaluating all combinations", unit='it', leave=False
        )
        for scaler_name, scaler in self.scalers.items():
            for selector_name, selector in self.selectors.items():
                for model_name, model in self.models.items():
                    if (  # skip if already evaluated
                        (self.results["Seed"] == self.seed)
                        & (self.results["Scaler"] == scaler_name)
                        & (self.results["Selector"] == selector_name)
                        & (self.results["Model"] == model_name)
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
                    cv = StratifiedKFold(n_splits=10, random_state=self.seed, shuffle=True)
                    gcv = GridSearchCV(
                        pipe,
                        param_grid,
                        return_train_score=True,
                        cv=cv,
                        n_jobs=self.n_workers,
                        error_score='raise',
                    )
                    gcv.fit(self.data_x_train, self.data_y_train)
                    logger.info(f'Evaluating {scaler_name} - {selector_name} - {model_name}')
                    metrics = self.evaluate_model(gcv)
                    row.update(metrics)
                    self.results.loc[self.row_to_write] = row
                    self.row_to_write += 1
                    self.results = self.results.sort_values(["Seed", "Scaler", "Selector", "Model"])
                    logger.info(f'Saving results to {self.out_file}')
                    try:  # ensure that intermediate results are not corrupted by KeyboardInterrupt
                        self.results.to_excel(self.out_file, index=False)  # save results after each seed
                    except KeyboardInterrupt:
                        logger.warning('Keyboard interrupt detected, saving results before exiting...')
                        self.results.to_excel(self.out_file, index=False)
                        sys.exit(130)
                    pbar.update()

        pbar.close()

    def evaluate_model(self, gcv):
        # Predict risk scores
        risk_scores = gcv.predict(self.data_x_test)
        c_index = concordance_index_censored(
            self.data_y_test[self.event_column], self.data_y_test[self.time_column], risk_scores
        )[0]
        c_index_ipcw = concordance_index_ipcw(self.data_y_train, self.data_y_test, risk_scores)[0]
        times = np.percentile(self.data_y_test[self.time_column], np.linspace(5, 91, 15))
        _, mean_auc = cumulative_dynamic_auc(self.data_y_train, self.data_y_test, risk_scores, times)
        best_estimator = gcv.best_estimator_
        # Check if the model has the 'predict_survival_function' method
        if hasattr(best_estimator["model"], "predict_survival_function"):
            surv_functions = best_estimator.predict_survival_function(self.data_x_test)
            estimates = np.array([[func(t) for t in times] for func in surv_functions])
            int_brier_score = integrated_brier_score(self.data_y_train, self.data_y_test, estimates, times)
        else:
            int_brier_score = None  # Or any other default value
        metrics_dict = {
            "c-index": c_index,
            "c-index (IPCW)": c_index_ipcw,
            "AUC": mean_auc,
            "Brier Score": int_brier_score,
        }

        return metrics_dict
