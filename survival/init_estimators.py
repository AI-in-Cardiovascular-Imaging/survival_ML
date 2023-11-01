import numpy as np
from functools import partial
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
    QuantileTransformer,
    Normalizer,
    Binarizer,
    PowerTransformer,
)
import sksurv.metrics as sksurv_metrics
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import (
    RandomSurvivalForest,
    ComponentwiseGradientBoostingSurvivalAnalysis,
    GradientBoostingSurvivalAnalysis,
)
from sksurv.svm import FastSurvivalSVM, FastKernelSurvivalSVM
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD, FastICA


def init_estimators(seed, n_workers, scalers, selectors, models, scoring):
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
        'SelectKBest': SelectKBest(partial(fit_and_score_features, scoring=scoring)),
        'VarianceThreshold': VarianceThreshold(),
        'FastICA': FastICA(random_state=seed),
        'PCA': PCA(random_state=seed),
        'KernelPCA': KernelPCA(random_state=seed),
        'TruncatedSVD': TruncatedSVD(random_state=seed),
    }
    selectors_dict = {selector: selectors_dict[selector] for selector in selectors if selectors[selector]}
    models_dict = {
        'CoxPH': CoxPHSurvivalAnalysis(),
        'Coxnet': CoxnetSurvivalAnalysis(fit_baseline_model=True),
        'CoxLasso': CoxnetSurvivalAnalysis(fit_baseline_model=True, l1_ratio=1.0),
        'CoxElasticNet': CoxnetSurvivalAnalysis(fit_baseline_model=True),
        'RSF': RandomSurvivalForest(random_state=seed, n_jobs=n_workers),
        'FastSVM': FastSurvivalSVM(random_state=seed),
        'FastKSVM': FastKernelSurvivalSVM(random_state=seed),
        'GBS': GradientBoostingSurvivalAnalysis(random_state=seed),
        'CGBS': ComponentwiseGradientBoostingSurvivalAnalysis(random_state=seed),
    }
    models_dict = {model: models_dict[model] for model in models if models[model]}

    return scalers_dict, selectors_dict, models_dict 


def fit_and_score_features(X, y, scoring):
    n_features = X.shape[1]
    scores = np.empty(n_features)
    # model =RandomSurvivalForest()
    # model = CoxPHSurvivalAnalysis(alpha=0.1)
    model = CoxnetSurvivalAnalysis(fit_baseline_model=True, l1_ratio=1.0, n_alphas=100)
    # model =CoxnetSurvivalAnalysis(fit_baseline_model=True, l1_ratio=0.5, alpha_min_ratio='auto')
    estimator = getattr(sksurv_metrics, scoring)(model)  # attach scoring function
    for feature in range(n_features):
        X_feature = X[:, feature : feature + 1]
        estimator.fit(X_feature, y)
        scores[feature] = estimator.score(X_feature, y)
    return scores
