import numpy as np
from functools import partial
from sklearn.feature_selection import SelectKBest, VarianceThreshold, RFE
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
from skopt.space import Real, Categorical, Integer


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
        'SelectKBest': SelectKBest(partial(fit_and_score_features, scoring=None)),
        'VarianceThreshold': VarianceThreshold(),
        'FastICA': FastICA(max_iter=10000, random_state=seed),
        'PCA': PCA(n_components=0.9, random_state=seed),
        'KernelPCA': KernelPCA(random_state=seed),
        'TruncatedSVD': TruncatedSVD(random_state=seed),
        'RFE': RFE(CoxPHSurvivalAnalysis(n_iter=1000)),
    }
    selectors_dict = {selector: selectors_dict[selector] for selector in selectors if selectors[selector]}
    models_dict = {
        'CoxPH': CoxPHSurvivalAnalysis(n_iter=1000),
        'Coxnet': CoxnetSurvivalAnalysis(fit_baseline_model=True),
        'CoxLasso': CoxnetSurvivalAnalysis(fit_baseline_model=True, l1_ratio=1.0),
        'RSF': RandomSurvivalForest(random_state=seed, n_jobs=n_workers),
        'FastSVM': FastSurvivalSVM(random_state=seed, max_iter=1000, tol=0.00001),
        'FastKSVM': FastKernelSurvivalSVM(random_state=seed, max_iter=1000, tol=0.00001),
        'GBS': GradientBoostingSurvivalAnalysis(random_state=seed),
        'CGBS': ComponentwiseGradientBoostingSurvivalAnalysis(random_state=seed),
    }
    models_dict = {model: models_dict[model] for model in models if models[model]}

    return scalers_dict, selectors_dict, models_dict


def fit_and_score_features(X, y, scoring):
    n_features = X.shape[1]
    scores = np.empty(n_features)
    model = CoxPHSurvivalAnalysis(alpha=0.1)
    if scoring is not None:
        estimator = getattr(sksurv_metrics, scoring)(model)  # attach scoring function
    else:
        estimator = model
    for feature in range(n_features):
        X_feature = X[:, feature : feature + 1]
        estimator.fit(X_feature, y)
        scores[feature] = estimator.score(X_feature, y)
    return scores


def set_params_search_space():
    model_params = {
        "CoxPH": {
            "model__estimator__alpha": Real(low=0.0001, high=10, prior="log-uniform")
        },
        "Coxnet": {
            "model__estimator__n_alphas": Integer(low=10, high=200),
            "model__estimator__l1_ratio": Real(low=0.1, high=1)
        },
        "CoxLasso": {
            "model__estimator__n_alphas": Integer(low=10, high=200),
        },
        "RSF": {
            "model__estimator__n_estimators": Integer(low=50, high=200),
            "model__estimator__max_depth": Integer(low=5, high=50),
            "model__estimator__min_samples_split": Integer(low=3, high=10),
            "model__estimator__min_samples_leaf": Integer(low=2, high=10),
            "model__estimator__max_features": Categorical([None, "sqrt", "log2"])
        },
        "FastSVM": {
            "model__estimator__alpha": Real(low=0.0001, high=10, prior="log-uniform"),
            "model__estimator__rank_ratio": Real(low=0, high=1)
        },
        "FastKSVM": {
            "model__estimator__alpha":  Real(low=0.0001, high=10, prior="log-uniform"),
            "model__estimator__rank_ratio": Real(low=0, high=1),
            "model__estimator__kernel": Categorical(["linear", "rbf", "poly"])
        },
        "GBS": {
            "model__estimator__learning_rate": Real(low=0.001, high=0.5, prior="log-uniform"),
            "model__estimator__n_estimators": Integer(low=10, high=100),
            "model__estimator__dropout_rate": Real(low=0, high=0.7),
            "model__estimator__subsample": Real(low=0.5, high=1),
            "model__estimator__max_depth": Integer(low=3, high=10),
            "model__estimator__max_features": Categorical([None, "sqrt", "log2"])
        },
        "CGBS": {
            "model__estimator__learning_rate": Real(low=0.001, high=0.5, prior="log-uniform"),
            "model__estimator__n_estimators": Integer(low=10, high=100),
            "model__estimator__dropout_rate": Real(low=0, high=0.7),
            "model__estimator__subsample": Real(low=0.5, high=1),
        }
    }

    selector_params = {
        "SelectKBest": {
            "selector__k": Integer(low=10, high=30)
        },
        "VarianceThreshold": {
            "selector__threshold": Real(low=0.01, high=0.2)
        },
        "PCA": {},
        "KernelPCA": {
            "selector__kernel": Categorical(["rbf", "poly"]),
            "selector__n_components": Integer(low=10, high=30)
        },
        "RFE": {
            "selector__n_features_to_select": Real(low=0.5, high=1),
        },
        "FastICA": {},
        "TruncatedSVD": {}
    }

    return model_params, selector_params
