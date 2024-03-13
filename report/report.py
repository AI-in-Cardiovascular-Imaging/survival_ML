import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import rpy2.robjects as robjects
from pycox.utils import kaplan_meier
from cmprsk.rpy_utils import r_vector
from rpy2.robjects.packages import importr

robjects.numpy2ri.activate()
robjects.pandas2ri.activate()


class Report:
    def __init__(self, config) -> None:
        self.plot_format = config.meta.plot_format
        self.out_dir = config.meta.out_dir
        self.out_file = os.path.join(self.out_dir, 'aggregate_results.xlsx')

    def __call__(self, results):
        aggregate_results = self.aggregate_results(results)
        aggregate_results.to_excel(self.out_file, index=False, float_format='%.3f')

        return aggregate_results

    def aggregate_results(self, results):
        mean_results = results.groupby(["Scaler", "Selector", "Model"]).mean().drop('Seed', axis=1).reset_index()
        std_results = results.groupby(["Scaler", "Selector", "Model"]).std().drop('Seed', axis=1).reset_index()

        aggregate_results = mean_results.merge(
            std_results, on=["Scaler", "Selector", "Model"], suffixes=('_mean', '_std')
        )
        return aggregate_results

    def plot_cumulative_dynamic_auc(self, auc, mean_auc, times, label, color=None):
        plt.plot(times, auc, marker="o", color=color, label=label)
        plt.xlabel("Days from Enrollment")
        plt.ylabel("Time-dependent AUC")
        plt.axhline(mean_auc, color=color, linestyle="--")
        plt.legend()

    def survival_by_outcome(self, predicted_survival, events):
        """Plot predicted survival stratifed by outcome
        Input;
        :param predicted_survival: pandas dataframe with predicted survival. Each column represents a different patient,
        rows represent time. The dataframe's index is time.
        :param events: npy array with binary event labels.
        """

        plt.figure()
        for i in range(2):
            idx = events == i
            predicted_survival.loc[:, idx].mean(axis=1).rename(i).plot()
        _ = plt.legend()
        plt.title("Mean predicted survival stratified by outcome")
        plt.xlabel("Time")
        plt.ylabel("Survival")
        plt.savefig(os.path.join(self.out_dir, f"mean_predicted_survival.{self.plot_format}"))
        plt.close()

    def km_by_risk(self, risk, events, durations):
        """
        Plot
        :param risk: predicted risk at a given time point.
        :param events: binary event labels
        :param durations: time to event or to censor
        :return:
        """
        low_risk = risk <= np.median(risk)
        high_risk = risk > np.median(risk)
        plt.figure()
        kaplan_meier(durations=durations[low_risk], events=events[low_risk]).rename("Low predicted risk").plot()
        kaplan_meier(durations=durations[high_risk], events=events[high_risk]).rename("High predicted risk").plot()
        _ = plt.legend()
        plt.title("Kaplan Meier stratified by risk")
        plt.xlabel("Time")
        plt.ylabel("Survival probability")
        plt.savefig(os.path.join(self.out_dir, f"km_by_risk.{self.plot_format}"))
        plt.close()

    def ici_survival(self, durations, labels, risk, time):
        """Function to compute the integrated calibration index for time-to-event outcome, at a given time instant.
        To produce smooth calibration curves, the hazard of the outcome is regressed on the predicted outcome risk using a
        flexible regression model. Then the ICI is the weighted difference between smoothed observed proportions and
        predicted risks.

        Reference: Austin et al. (2020). https://doi.org/10.1002/sim.8570

        Input
        - durations {np array}: Event/censoring times
        - labels: array of event indicators
        - predictions: predicted risk at the given time
        - time: time instant at which ICI has to be computed
        - return_calib_model: bool, if True also the calibration model is returned
        Output
        - ici: integrated calibration index for survival outcome
        - calibrate: calibration model
        """
        polspline = importr("polspline")

        risk_cll = robjects.FloatVector(np.log(-np.log(1 - risk)))  # complementary log-log transformation
        calibrate = polspline.hare(data=durations, delta=labels, cov=risk_cll)

        predict_calibrate = np.array(polspline.phare(time, risk_cll, calibrate))
        ici = np.mean(np.abs(risk - predict_calibrate))

        return ici, calibrate

    def calibration_plot_survival(self, durations, events, risk, time):
        """
        Plot clibration curve
        :param durations: time to event or censor
        :param events: vent binary labels
        :param risk: predicted risk predicted at time instant "time"
        :param time: time instant at which calibration is evaluated (in days)
        :return:
        """
        stats = importr("stats")
        polspline = importr("polspline")

        labels = robjects.FloatVector(events)
        durations = robjects.FloatVector(durations)

        fig, ax = plt.subplots(1)

        # Creation of the grid for plotting
        risk_r = r_vector(risk)
        grid = r.seq(stats.quantile(risk_r, probs=0.01), stats.quantile(risk_r, probs=0.99), length=100)
        grid_cll = robjects.FloatVector(np.log(-np.log(1 - np.array(grid))))  # complementary log-log grid

        _, calibration_model = self.ici_survival(durations, labels, risk, time)
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
        sns.kdeplot(risk, ax=ax2)
        ax2.set_ylabel('Predicted probability density', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.grid(alpha=0.3)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped

        plt.close()

        return ax
