import os

import pandas as pd
import numpy as np
from cleaning_utils import (remove_patients_without_outcome, remove_0_variance_features, remove_binaries_not_populated,
                            remove_features_with_many_nan, remove_highly_correlated_features, clean_strain_dataset)


def read_radiomics_strain(config):
    # --- Radiomics ---
    data_radiomics = pd.read_csv(config["radiomics_path"])
    data_radiomics = data_radiomics[
        ["redcap_id", "time_to_censor", "mace", "dataset"] + [c for c in data_radiomics.columns.values if
                                                              "original" in c]]
    data_radiomics.loc[data_radiomics["time_to_censor"] < 0, "time_to_censor"] *= -1

    # --- Strain ---
    data_strain = pd.read_excel(config["strain_path"])
    # clean strain dataset
    data_strain["dataset"] = data_strain["site"].apply(lambda s: "Boston" if s == 1 else "Bern")
    data_strain.loc[data_strain["time_to_censor"] < 0, "time_to_censor"] *= -1
    data_strain = clean_strain_dataset(data_strain)

    # --- Together ---
    data_radiomics_strain = data_radiomics.drop(columns=["time_to_censor", "mace", "dataset"]).merge(data_strain,
                                                                                                     on="redcap_id",
                                                                                                     how="inner")
    # --- Save ---
    data_radiomics_strain = data_radiomics_strain.drop(columns="redcap_id")
    data_radiomics = data_radiomics.drop(columns="redcap_id")
    data_strain = data_strain.drop(columns="redcap_id")

    return data_radiomics, data_strain, data_radiomics_strain


def clean_dataset(config, data, name):
    print(f"Input dataset: {data.shape[0]} patients, {data.shape[1]} features")
    dataset_col = data["dataset"]
    data = data.apply(pd.to_numeric, errors='coerce')  # replace non-numeric entries with NaN
    data = data.dropna(how='all', axis=1)  # drop columns with all NaN
    data["dataset"] = dataset_col

    # Remove patients without time to event or mace
    data = remove_patients_without_outcome(data, config["time_column"], config["outcome"])
    data = remove_0_variance_features(data)
    data = remove_features_with_many_nan(data, config["nan_threshold"])  # Remove features with more than % missing
    data = remove_binaries_not_populated(data, config["binary_threshold"])

    # Multiply rv_ef and lv_ef by 100 if below 1 (they should be expressed in %)
    if "lv_ef" in data.columns:
        data["lv_ef"] = data["lv_ef"].apply(lambda x: 100 * x if x < 1 else x)
        data["rv_ef"] = data["rv_ef"].apply(lambda x: 100 * x if x < 1 else x)

    # Remove features with 0 variance or not populated in Boston (training set)
    print("\nFocusing on Boston....")
    _, binaries_empty_boston = remove_binaries_not_populated(data[data["dataset"] == "Boston"],
                                                             config["binary_threshold"], return_drop_features=True,
                                                             verbose=True)
    _, features_0var_boston = remove_0_variance_features(data[data["dataset"] == "Boston"], return_drop_features=True,
                                                         verbose=True)
    to_drop = list(set(binaries_empty_boston + features_0var_boston))
    if name == "strain":
        to_drop = to_drop + ['eligible', 'ecg_complete', 'laboratory_markers_complete', 'hxdm',
                             'mace_type_before_cmr___2', 'ergo_findings_4___6', 'medical_disorders___1',
                             'medical_disorders___10', 'medical_disorders___14']

    data = data.drop(columns=to_drop, errors='ignore')
    if len(to_drop) > 0:
        print(f"remove {len(to_drop)} columns with 0 variance in Boston dataset:\n{to_drop}")
    print()

    # Set outlier values to NaN in each feature
    numeric_columns = [col for col in data.columns if len(set(data[col].dropna().unique())) > 10]
    numeric_columns.remove(config["time_column"])
    q1 = data[numeric_columns].quantile(0.01)
    q3 = data[numeric_columns].quantile(0.99)
    iqr = q3 - q1
    mask = (data[numeric_columns] > q3 + 3 * iqr) | (data[numeric_columns] < q1 - 3 * iqr)
    # z_score = (data[numeric_columns] - data[numeric_columns].mean()).abs() / data[numeric_columns].std()
    # mask = z_score > 10
    for col in numeric_columns:
        if np.sum(mask[col]) > 0:
            print(f"Set {np.sum(mask[col])} nans for feature {col}, values {data.loc[mask[col], col].values}")
    data[mask] = np.nan

    data = remove_features_with_many_nan(data, config["nan_threshold"])  # Remove features with more than % missing
    if config["corr_threshold"] is not None:
        data = remove_highly_correlated_features(data, config["outcome"], corr_threshold=config["corr_threshold"])
    data = remove_binaries_not_populated(data, config["binary_threshold"])
    data = remove_0_variance_features(data)

    # Split in train and test based on center
    train = data[data["dataset"] == "Boston"].drop(columns=["dataset"])
    test = data[data["dataset"] == "Bern"].drop(columns=["dataset"])

    # Save
    out_folder = os.path.join(config["output_path"], name)
    os.makedirs(out_folder, exist_ok=True)
    train.to_excel(os.path.join(out_folder, "train.xlsx"), index=False)
    test.to_excel(os.path.join(out_folder, "test.xlsx"), index=False)
    data.to_excel(os.path.join(out_folder, "all.xlsx"), index=False)

    print(f"\nTrain (Boston): {len(train)} patients, {train.shape[1]} features")
    print(f"Test (Bern): {len(test)} patients, {test.shape[1]} features")
    print(f"Complete dataset: {len(data)} patients, {data.shape[1]} features")


def main():
    config = {
        "outcome": "mace",
        "time_column": "time_to_censor",
        "nan_threshold": .25,
        "corr_threshold": None,
        "binary_threshold": 0.005,
        "radiomics_path": "/home/aici/Myocarditis/Datasets/NewDataLGE_Out_3m_withOutcome.csv",
        "strain_path": "/home/aici/Myocarditis/Datasets/L1429FlamBeR_DATA_2024-04-10_1523_cleaned_original.xlsx",
        "output_path": "/home/aici/PycharmProjects/survival_analysis/datasets/flamber/"
    }
    data_radiomics, data_strain, data_radiomics_strain = read_radiomics_strain(config)
    names = ["radiomics", "strain", "radiomics_strain"]
    for data, name in zip([data_radiomics, data_strain, data_radiomics_strain], names):
        print(f"\n--- Cleaning dataset {name} ---")
        clean_dataset(config, data, name)


if __name__ == "__main__":
    main()
