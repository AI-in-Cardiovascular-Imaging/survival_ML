import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel, VarianceThreshold
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, 
                                   QuantileTransformer, Normalizer, Binarizer, PowerTransformer)
from sksurv.datasets import load_veterans_lung_cancer, load_flchain, load_gbsg2
from sksurv.functions import StepFunction
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.metrics import (concordance_index_censored, concordance_index_ipcw,
                            cumulative_dynamic_auc, integrated_brier_score)
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.preprocessing import OneHotEncoder, encode_categorical
from sksurv.util import Surv
from sksurv.ensemble import (RandomSurvivalForest, ComponentwiseGradientBoostingSurvivalAnalysis, 
                             GradientBoostingSurvivalAnalysis)
from sksurv.svm import FastSurvivalSVM, FastKernelSurvivalSVM
import openpyxl
import warnings
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD, FastICA, NMF, LatentDirichletAllocation
import inspect
import os


def numpy_range(start, stop, step):
    """helper function to load numpy range from yaml"""
    return np.arange(start, stop + step, step)


def init_estimators(seed, n_workers, scalers, selectors, models):
    scalers_dict = {
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler(),
        'MaxAbsScaler': MaxAbsScaler(),
        'QuantileTransformer': QuantileTransformer(),
        'Normalizer': Normalizer(),
        'Binarizer': Binarizer(),
        'PowerTransformer': PowerTransformer(),
    }
    scalers_dict = {scaler: scalers_dict[scaler] for scaler in scalers if scalers[scaler]}
    selectors_dict = {
        'SelectKBest': SelectKBest(),
        'VarianceThreshold': VarianceThreshold(),
        'FastICA': FastICA(),
        'PCA': PCA(),
        'KernelPCA': KernelPCA(),
        'TruncatedSVD': TruncatedSVD(),
    }
    selectors_dict = {selector: selectors_dict[selector] for selector in selectors if selectors[selector]}
    models_dict = {
        'CoxPH': CoxPHSurvivalAnalysis(),
        'Coxnet': CoxnetSurvivalAnalysis(fit_baseline_model=True),
        'RSF': RandomSurvivalForest(random_state=seed),
        'FastSVM': FastSurvivalSVM(random_state=seed),
        'FastKSVM': FastKernelSurvivalSVM(random_state=seed),
        'GBS': GradientBoostingSurvivalAnalysis(random_state=seed),
        'CGBS': ComponentwiseGradientBoostingSurvivalAnalysis(random_state=seed),
        'CoxLasso': CoxnetSurvivalAnalysis(fit_baseline_model=True, l1_ratio=1.0),
        'CoxElasticNet': CoxnetSurvivalAnalysis(fit_baseline_model=True),
    }
    models_dict = {model: models_dict[model] for model in models if models[model]}

    return scalers_dict, selectors_dict, models_dict
