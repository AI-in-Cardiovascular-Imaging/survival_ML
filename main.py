import hydra

import pandas as pd
from loguru import logger
from omegaconf import OmegaConf

from helpers.helpers import numpy_range
from preprocessing.preprocessing import Preprocessing
from survival.survival import Survival


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config):
    OmegaConf.register_new_resolver("numpy_range", numpy_range)

    preprocessing = Preprocessing(config)
    data_x_train, data_x_test, data_y_train, data_y_test = preprocessing()
    pipeline = Survival(config)
    results = pipeline(
        data_x_train,
        data_y_train,
        data_x_test,
        data_y_test,
    )

    pd.options.display.float_format = '{:.3f}'.format
    logger.info(f'\n{results}')


if __name__ == "__main__":
    main()
