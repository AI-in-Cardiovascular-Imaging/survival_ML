import os
import hydra

import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm

from helpers.helpers import numpy_range
from preprocessing.preprocessing import Preprocessing
from survival.survival import Survival
from report.report import Report


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config):
    OmegaConf.register_new_resolver("numpy_range", numpy_range)  # adds ability to use numpy_arange() in config file
    if config.meta.out_file is None:
            in_path = os.path.splitext(config.meta.in_file)[0]  # remove extension
            config.meta.out_file = f'{in_path}_results.xlsx'
    np.random.seed(config.meta.init_seed)
    seeds = np.random.randint(low=0, high=2**32, size=config.meta.n_seeds)  # generate desired number of random seeds
    preprocessing = Preprocessing(config)
    pipeline = Survival(config)
    report = Report(config)

    for seed in tqdm(seeds, desc='Seeds'):
        np.random.seed(seed)
        data_x_train, data_x_test, data_y_train, data_y_test = preprocessing(seed)
        results = pipeline(
            seed,
            data_x_train,
            data_y_train,
            data_x_test,
            data_y_test,
        )

    report(results)
    pd.options.display.float_format = '{:.3f}'.format
    logger.info(f'\n{results}')


if __name__ == "__main__":
    main()
