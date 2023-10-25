import hydra

from loguru import logger
from omegaconf import OmegaConf

from helpers.helpers import numpy_range
from preprocessing.preprocessing import load_and_preprocess_data
from survival.survival import Survival


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config):
    OmegaConf.register_new_resolver("numpy_range", numpy_range)
    events = config.meta.events
    times = config.meta.times

    data_x_train, data_x_test, data_y_train, data_y_test = load_and_preprocess_data(
        source='excel', filepath=config.meta.in_file, time_column=times, event_column=events
    )
    pipeline = Survival(config)
    results = pipeline(
        data_x_train,
        data_y_train,
        data_x_test,
        data_y_test,
    )

    logger.info(f'\n{results}')


if __name__ == "__main__":
    main()
