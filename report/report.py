import os
import sys
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import rpy2.robjects as robjects
from loguru import logger
from pycox.utils import kaplan_meier
from cmprsk.rpy_utils import r_vector
from rpy2.robjects.packages import importr
from sksurv.metrics import cumulative_dynamic_auc

robjects.numpy2ri.activate()
robjects.pandas2ri.activate()

from helpers.nested_dict import NestedDefaultDict


class Report:
    def __init__(self, config) -> None:
        self.plot_format = config.meta.plot_format
        self.time_column = config.meta.times
        self.event_column = config.meta.events
        if config.meta.out_dir is None:
            self.experiment_dir = os.path.splitext(config.meta.in_file)[0]
        else:
            self.experiment_dir = config.meta.out_dir
        self.results_file = os.path.join(self.experiment_dir, 'results.pkl')
        np.random.seed(config.meta.init_seed)
        self.seeds = np.random.randint(low=0, high=2**32, size=config.meta.n_seeds)
        self.scalers = [scaler for scaler in config.survival.scalers if config.survival.scalers[scaler]]
        self.selectors = [sel for sel in config.survival.feature_selectors if config.survival.feature_selectors[sel]]
        self.models = [model for model in config.survival.models if config.survival.models[model]]

    def __call__(self):
        with open(self.results_file, 'rb') as f:
            self.results = pickle.load(f)

        # TODO: compute calibration only for models that predict absolute risk

        for seed in self.seeds:
            self.x_train = self.results[seed]['x_train']
            self.y_train = self.results[seed]['y_train']
            self.x_test = self.results[seed]['x_test']
            self.y_test = self.results[seed]['y_test']
            self.times = np.percentile(self.y_test[self.time_column], np.linspace(5, 91, 15))
            time_to_eval = self.times[len(self.times) // 2]  # TODO: which time(s) to use?
            for scaler in self.scalers:
                for selector in self.selectors:
                    for model in self.models:
                        try:
                            self.out_dir = os.path.join(self.experiment_dir, str(seed))
                            os.makedirs(self.out_dir, exist_ok=True)
                            self.best_estimator = self.results[seed][scaler][selector][model]['best_estimator']
                            self.risk_scores = self.best_estimator.predict(self.x_test)
                            self.plot_cumulative_dynamic_auc(scaler, selector, model)
                            self.survival_by_outcome(scaler, selector, model)
                            self.km_by_risk(scaler, selector, model)
                            self.calibration_plot_survival(scaler, selector, model, time_to_eval)
                        except Exception as e:
                            print(
                                f"Error encountered for Scaler={scaler}, Selector={selector}, Model={model}."
                                f"Error message: {str(e)}")

    def plot_cumulative_dynamic_auc(self, scaler, selector, model):
        auc, mean_auc = cumulative_dynamic_auc(self.y_train, self.y_test, self.risk_scores, self.times)

        plt.plot(self.times, auc, marker="o")
        plt.title(f"Cumulative Dynamic AUC for")
        plt.xlabel("Days from Enrollment")
        plt.ylabel("Time-dependent AUC")
        plt.axhline(mean_auc, linestyle="--")
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.out_dir, f"cumulative_dynamic_auc_{scaler}_{selector}_{model}.{self.plot_format}")
        )
        plt.close()

    def survival_by_outcome(self, scaler, selector, model):
        try:
            predicted_survival = self.best_estimator.predict_survival_function(self.x_test)
            estimates = np.array([f(self.times) for f in predicted_survival])
            predicted_survival = pd.DataFrame(estimates.T, index=self.times)
        except AttributeError:
            return

        plt.figure()
        for i in range(2):
            idx = self.y_test[self.event_column] == i
            predicted_survival.loc[:, idx].mean(axis=1).rename(i).plot()
        _ = plt.legend()
        plt.title(f"Mean predicted survival stratified by outcome")
        plt.xlabel("Time")
        plt.ylabel("Survival")
        plt.savefig(
            os.path.join(self.out_dir, f"mean_predicted_survival_{scaler}_{selector}_{model}.{self.plot_format}")
        )
        plt.close()

    def km_by_risk(self, scaler, selector, model):
        low_risk = self.risk_scores <= np.median(self.risk_scores)
        high_risk = self.risk_scores > np.median(self.risk_scores)
        plt.figure()
        kaplan_meier(
            durations=self.y_test[self.time_column][low_risk], events=self.y_test[self.event_column][low_risk]
        ).rename("Low predicted risk").plot()
        kaplan_meier(
            durations=self.y_test[self.time_column][high_risk], events=self.y_test[self.event_column][high_risk]
        ).rename("High predicted risk").plot()
        _ = plt.legend()
        plt.title("Kaplan Meier stratified by risk")
        plt.xlabel("Time")
        plt.ylabel("Survival probability")
        plt.savefig(os.path.join(self.out_dir, f"km_by_risk_{scaler}_{selector}_{model}.{self.plot_format}"))
        plt.close()

    def calibration_plot_survival(self, scaler, selector, model, time):
        """
        Plot clibration curve
        :param time: time instant at which calibration is evaluated (in days)
        """
        stats = importr("stats")
        polspline = importr("polspline")

        labels = robjects.FloatVector(self.y_test[self.event_column])
        durations = robjects.FloatVector(self.y_test[self.time_column])

        fig, ax = plt.subplots(1)

        # Creation of the grid for plotting
        predicted_survival = self.best_estimator.predict_survival_function(self.x_test)
        probas = np.array([f(time) for f in predicted_survival])
        risk = 1 - probas
        risk_r = r_vector(risk)
        grid = robjects.r.seq(stats.quantile(risk_r, probs=0.01), stats.quantile(risk_r, probs=0.99), length=100)
        grid_cll = robjects.FloatVector(np.log(-np.log(1 - np.array(grid))))  # complementary log-log grid

        _, calibration_model = self.ici_survival(labels, durations, risk, time)
        # Predicted probability for grid points
        predict_grid = polspline.phare(time, grid_cll, calibration_model)

        ax2 = ax.twinx()  # instantiate a second axis that shares the same x-axis

        ax.plot(grid, predict_grid, "-", linewidth=2, color="red")
        ax.set_xlim((-0.02, max(grid) + 0.1))
        ax.set_ylim((0, max(grid) + 0.1))
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed probability")
        # ax.set_title(f"{int(time/365)}-year calibration curve")
        ax.grid(alpha=0.3)
        ax.plot([0, 1], [0, 1], color='black')

        color = 'tab:blue'
        sns.kdeplot(self.risk_scores, ax=ax2)
        ax2.set_ylabel('Predicted probability density', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.grid(alpha=0.3)
        fig.tight_layout()
        plt.savefig(
            os.path.join(self.out_dir, f"calibration_survival_{scaler}_{selector}_{model}.{self.plot_format}")
        )
        plt.close()

    def ici_survival(self, labels, durations, risk, time):
        """Function to compute the integrated calibration index for time-to-event outcome, at a given time instant.
        To produce smooth calibration curves, the hazard of the outcome is regressed on the predicted outcome risk using a
        flexible regression model. Then the ICI is the weighted difference between smoothed observed proportions and
        predicted risks.

        Reference: Austin et al. (2020). https://doi.org/10.1002/sim.8570

        Input
        - time: time instant at which ICI has to be computed
        Output
        - ici: integrated calibration index for survival outcome
        - calibrate: calibration model
        """
        polspline = importr("polspline")

        np.set_printoptions(threshold=sys.maxsize)
        risk_cll = robjects.FloatVector(np.log(-np.log(1 - risk)))  # complementary log-log transformation
        calibrate = polspline.hare(
            data=durations, delta=labels, cov=risk_cll
        )

        predict_calibrate = np.array(polspline.phare(time, risk_cll, calibrate))
        ici = np.mean(np.abs(self.risk_scores - predict_calibrate))

        return ici, calibrate
