import os
import hydra

from loguru import logger

from report.report import Report

@hydra.main(version_base=None, config_path=".", config_name="config")
def collect_report(config):
    logger.info(f"Collecting report for {config.meta.out_dir}")
    Report(config)()


if __name__ == "__main__":
    collect_report()