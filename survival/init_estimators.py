import numpy as np
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, 
                                   QuantileTransformer, Normalizer, Binarizer, PowerTransformer)
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import (RandomSurvivalForest, ComponentwiseGradientBoostingSurvivalAnalysis, 
                             GradientBoostingSurvivalAnalysis)
from sksurv.svm import FastSurvivalSVM, FastKernelSurvivalSVM
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD, FastICA


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
        'SelectKBest': SelectKBest(fit_and_score_features),
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
        'CoxLasso': CoxnetSurvivalAnalysis(fit_baseline_model=True, l1_ratio=1.0),
        'CoxElasticNet': CoxnetSurvivalAnalysis(fit_baseline_model=True),
        'RSF': RandomSurvivalForest(random_state=seed),
        'FastSVM': FastSurvivalSVM(random_state=seed),
        'FastKSVM': FastKernelSurvivalSVM(random_state=seed),
        'GBS': GradientBoostingSurvivalAnalysis(random_state=seed),
        'CGBS': ComponentwiseGradientBoostingSurvivalAnalysis(random_state=seed),
    }
    models_dict = {model: models_dict[model] for model in models if models[model]}

    return scalers_dict, selectors_dict, models_dict


def fit_and_score_features(X, y):
    n_features = X.shape[1]
    scores = np.empty(n_features)
    # m =RandomSurvivalForest()
    m = CoxPHSurvivalAnalysis(alpha=0.1)
    # m =CoxnetSurvivalAnalysis(fit_baseline_model=True,l1_ratio=0.9, alpha_min_ratio=0.001)
    # m =CoxnetSurvivalAnalysis(fit_baseline_model=True, l1_ratio=0.5, alpha_min_ratio='auto')
    for j in range(n_features):
        Xj = X[:, j : j + 1]
        m.fit(Xj, y)
        scores[j] = m.score(Xj, y)
    return scores