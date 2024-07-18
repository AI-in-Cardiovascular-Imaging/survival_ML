import numpy as np


def remove_features_with_many_nan(data, nan_threshold):
    # Remove features with more than % missing
    null_perc = data.isna().sum() / len(data)
    to_drop = null_perc[null_perc > nan_threshold].index.values.tolist()
    if len(to_drop) > 0:
        print(f"Removing {len(to_drop)} features (more than {nan_threshold*100}% of nans): {to_drop}")
    data = data.drop(columns=to_drop)
    return data


def remove_highly_correlated_features(data, outcome, corr_threshold=0.9):
    y = data[outcome]
    data = data.drop(columns=outcome)
    corr_matrix = data.corr(numeric_only=True)
    importances = data.corrwith(y, axis=0, numeric_only=True).abs()
    importances = importances.sort_values(ascending=False)
    corr_matrix = corr_matrix.reindex(index=importances.index, columns=importances.index).abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > corr_threshold)]
    if len(to_drop) > 0:
        print(f"Removing {len(to_drop)} highly correlated features (corr > {corr_threshold}): {to_drop}")
    data = data.drop(columns=to_drop)
    data[outcome] = y
    return data


def remove_binaries_not_populated(data, binary_threshold=0.01, return_drop_features=False, verbose=True):
    binary_features = [col for col in data.columns if set(data[col].dropna().unique()) in [{0, 1}, {0}, {1}]]
    binary_frac = data[binary_features].sum() / len(data)
    to_drop = binary_frac[(binary_frac < binary_threshold) | (binary_frac > 1-binary_threshold)]
    if len(to_drop) > 0 and verbose:
        print(f"Removing {len(to_drop)} low populated binary features (% < {binary_threshold*100} or % > "
              f"{100-binary_threshold*100}):\n{to_drop*100}")
    data = data.drop(columns=to_drop.index.values)
    if return_drop_features:
        return data, to_drop.index.values.tolist()
    else:
        return data


def remove_0_variance_features(data, return_drop_features=False, verbose=True):
    data_var = data.var(numeric_only=True)
    data_var_0 = data_var[data_var == 0].index.values
    if len(data_var_0) > 0 and verbose:
        print(f"Removing {len(data_var_0)} features with 0 variance: {data_var_0}")
    data = data.drop(columns=data_var_0)
    if return_drop_features:
        return data, data_var_0.tolist()
    else:
        return data


def remove_patients_without_outcome(data, time_column, event_column):
    indices_no_outcome = data[(data[time_column].isna()) | (data[event_column].isna())].index
    if len(indices_no_outcome) > 0:
        print(f"Removing {len(indices_no_outcome)} patients without time to event or event label")
    data = data.drop(indices_no_outcome)
    return data


def clean_strain_dataset(data_strain):
    # drop columns
    non_predictors = ["pat_id", "site", "date_cmr", "patient_year", "inclusion_criteria", "cad", "scanner", "km",
                      "contrast_dose",
                      "cad_excluded", "cad_exclusion_modality", "demographics_and_eligibility_complete",
                      "echo_type", "time_cmr_echo", "time_cmr_ct", "time_cmr_pet", "time_cmr_ergo", "days_ecg_mr",
                      "date_this_cmr", "cmr_feature_tracking_complete", "cmr_examination_complete",
                      "cmr_cardiac_function_complete", "cmr_feature_tracking_complete", "cmr_ca",
                      "cmr_mapping_complete", "time_cmr_cath", "time_cmr_ct", "time_cmr_pet", "time_cmr_ergo"]
    outcomes = ["ob_time_days", "time_to_mace_chf", "mace_chf", "time_to_mace_vt", "mace_vt", "time_to_mace_myo",
                "mace_myo", "time_to_death", "death", "ob_time_endfu"]
    fu_features = [col for col in data_strain.columns if "_endfu" in col or "_fu_" in col]
    to_drop = non_predictors + fu_features + outcomes
    data_strain = data_strain.drop(columns=to_drop)
    # drop patients without strain
    cmr_feature_tracking = ([c for c in data_strain.columns if "g_2d_" in c] +
                            [c for c in data_strain.columns if "g_3d_" in c[:3]] +
                            [c for c in data_strain.columns if "rv_2d_" in c] +
                            [c for c in data_strain.columns if "rv_3d" in c])
    patients_with_some_strain = data_strain[cmr_feature_tracking].notna().any(axis=1)
    data_strain = data_strain[patients_with_some_strain]
    return data_strain
