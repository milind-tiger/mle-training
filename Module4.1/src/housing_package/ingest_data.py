# Assignment 2.1
"""VS Code Auto formatting
configured vs code to format the entire script
"""

import tarfile
import argparse
import os
import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    train_test_split,
)
import sys

sys.path.append("/home/deepak/practice_git_repo/assignment2/mle-training/")
from src.housing_package.log_config import LOGGING_DEFAULT_CONFIG, configure_logger


def fetch_housing_data(housing_url: str, housing_path: str):
    """Download the housing data

    Parameters
    ----------

    housing_url : str
        Url of the dataset

    housing_path : str
        Path where the dataset is stored
    """
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")

    r = requests.get(housing_url, timeout=300)
    with open(tgz_path, "wb") as f:
        f.write(r.content)

    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path: str):
    """Download the housing data

    Parameters
    ----------

    housing_path : str
        Path where the dataset is stored

    Returns
    -------
    df: pd.DataFrame
        It is a pandas Dataframe with rows x columns
    """
    csv_path = os.path.join(housing_path, "housing.csv")

    return pd.read_csv(csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p",
        "--path",
        type=str,
        required=False,
        help="add the path where you want to store the raw data",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        required=False,
        help="add the path where you want to store the processed data",
    )
    parser.add_argument(
        "-l",
        "--log-level",
        type=str,
        required=False,
        help="add the type of log level",
        choices=["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "-lp",
        "--log-path",
        type=str,
        required=False,
        help="add the path of log level",
    )
    parser.add_argument(
        "--no-console-log",
        action="store_true",
        required=False,
        help="add the path of log level",
    )
    args = parser.parse_args()

    if args.path:
        PATH = args.path
    else:
        PATH = "data/raw"

    if args.output_path:
        OUTPUT_PATH = args.output_path
    else:
        OUTPUT_PATH = "data/processed"

    if args.log_level:
        LOGGING_DEFAULT_CONFIG["root"]["level"] = args.log_level

    LOG_FILE = None
    if args.log_path:
        LOG_FILE = args.log_path

    CONSOLE_LOG = True
    if args.no_console_log:
        CONSOLE_LOG = False

    DOWNLOAD_ROOT = "http://raw.githubusercontent.com/ageron/handson-ml/master/"
    HOUSING_PATH = os.path.join(os.getcwd(), PATH)
    OUTPUT_PATH = os.path.join(os.getcwd(), OUTPUT_PATH)
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

    logger = configure_logger(
        log_file=os.path.join(LOG_FILE, "custom_config.log") if LOG_FILE else None,
        console=CONSOLE_LOG,
        log_level=LOGGING_DEFAULT_CONFIG["root"]["level"],
    )

    logger.warning("Logging - Start")
    logger.info("Running script!!")
    fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH)
    housing = load_housing_data(HOUSING_PATH)
    logger.info("Fetched housing data!!")

    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    logger.info("Splitted data into training and validation sets")
    strat_train_set.to_csv(
        os.path.join(OUTPUT_PATH, "strat_train_set.csv"), index=False
    )
    strat_test_set.to_csv(os.path.join(OUTPUT_PATH, "strat_test_set.csv"), index=False)

    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    train_set.to_csv(os.path.join(OUTPUT_PATH, "train_set.csv"), index=False)
    test_set.to_csv(os.path.join(OUTPUT_PATH, "test_set.csv"), index=False)
    logger.info("Exported the csv files")
    logger.info("ingest_data.py Logging - Completed\n")
