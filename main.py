import os
import hydra
import enlighten

import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf

from helpers.helpers import numpy_range
from preprocessing.preprocessing import Preprocessing
from survival.survival import Survival
from report.aggregate_results import aggregate_results


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config):
    OmegaConf.register_new_resolver("numpy_range", numpy_range)  # adds ability to use numpy_arange() in config file
    if config.meta.out_dir is None:
            config.meta.out_dir = os.path.splitext(config.meta.in_file)[0]
            os.makedirs(config.meta.out_dir, exist_ok=True)
    progress_manager = enlighten.get_manager()
    pbar = progress_manager.counter(total=config.meta.n_seeds, desc='Seeds', unit='seeds')
    np.random.seed(config.meta.init_seed)
    seeds = np.random.randint(low=0, high=2**32, size=config.meta.n_seeds)  # generate desired number of random seeds
    preprocessing = Preprocessing(config)
    pipeline = Survival(config, progress_manager)

    for seed in seeds:
        logger.info(f'Running seed {seed}')
        np.random.seed(seed)
        data_x_train, data_x_test, data_y_train, data_y_test = preprocessing(seed)
        if config.survival.active:
            results = pipeline(
                seed,
                data_x_train,
                data_y_train,
                data_x_test,
                data_y_test,
            )
        pbar.update()

    pbar.close()
    if config.survival.active:
        logger.info(f'Saving results to {config.meta.out_dir}')
        agg_results = aggregate_results(config, results)
        pd.options.display.float_format = '{:.3f}'.format
        logger.info(f'\n{agg_results}')


if __name__ == "__main__":
    main()
