import os

import matplotlib.pyplot as plt
from sksurv.metrics import cumulative_dynamic_auc


class Report:
    def __init__(self, config) -> None:
        out_path = os.path.splitext(config.meta.out_file)[0]  # remove extension
        self.out_file = f'{out_path}_aggregate.xlsx'

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

    def plot_cumulative_dynamic_auc(self, data_y_train, data_y_test, risk_score, times, label, color=None):
        auc, mean_auc = cumulative_dynamic_auc(data_y_train, data_y_test, risk_score, times)
        plt.plot(times, auc, marker="o", color=color, label=label)
        plt.xlabel("Days from Enrollment")
        plt.ylabel("Time-dependent AUC")
        plt.axhline(mean_auc, color=color, linestyle="--")
        plt.legend()
