# Survival Analysis <!-- omit in toc -->

## Table of contents <!-- omit in toc -->

- [Installation](#installation)
- [Configuration](#configuration)
- [Run](#run)

## Installation

In the project directory, i.e. where the **pyproject.toml** file is located, use the following commands to set up a virtual environment and install all project dependencies:

```bash
    python3 -m venv env
    source env/bin/activate
    pip install poetry
    poetry install
```

## Configuration

Make sure to configure everything needed for your experiments in the **config.yaml** file.\
Most important is the path to your input_file and the names of your events and times columns.

## Run

After the config file is set up properly, you can run the pipeline using:

```bash
python3 main.py
```

Results are automatically saved after each iteration and will not be recomputed unless the meta.overwrite flag is set to True.
