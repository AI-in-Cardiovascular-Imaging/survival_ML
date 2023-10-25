import os


import matplotlib.pyplot as plt
from sksurv.metrics import cumulative_dynamic_auc



def plot_cumulative_dynamic_auc(data_y_train, data_y_test, risk_score,times, label,  color=None):
    auc, mean_auc = cumulative_dynamic_auc(data_y_train, data_y_test, risk_score, times)
    plt.plot(times, auc, marker="o", color=color, label=label)
    plt.xlabel("Days from Enrollment")
    plt.ylabel("Time-dependent AUC")
    plt.axhline(mean_auc, color=color, linestyle="--")
    plt.legend()