# Assignment 2.1
"""VS Code Auto formatting
configured vs code to format the entire script
"""
import os
import pickle
import argparse
import numpy as np
import pandas as pd
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, os.path.abspath("."))
from src.housing_package.log_config import LOGGING_DEFAULT_CONFIG, configure_logger


def lin_reg_pred(
    MODEL_PATH: str, housing_prepared: pd.DataFrame, housing_labels: pd.DataFrame
):
    """Make model prediction
    Note: Model should exist in the given path

    Parameters
    ----------

    MODEL_PATH : str
        Path of the model
    housing_prepared : csv
        Pre-processed housing dataset with features
    housing_labels : csv
        Target labels of the housing dataset

    Returns
    -----------
    rmse: float
        A non-negative floating point value (the best value is 0.0)

    mae: float
        A non-negative floating point. The best value is 0.0
    """

    LIN_REG = pickle.load(open(os.path.join(MODEL_PATH, "lin_reg.pkl"), "rb"))
    housing_predictions = LIN_REG.predict(housing_prepared)

    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    lin_mae = mean_absolute_error(housing_labels, housing_predictions)

    return lin_rmse, lin_mae


def dtree_pred(
    MODEL_PATH: str, housing_prepared: pd.DataFrame, housing_labels: pd.DataFrame
):
    """Make model prediction
    Note: Model should exist in the given path

    Parameters
    ----------

    MODEL_PATH : str
        Path of the model

    housing_prepared : csv
        Pre-processed housing dataset with features

    housing_labels : csv
        Target labels of the housing dataset

    Returns
    -----------
    mse: float
        A non-negative floating point. The best value is 0.0

    rmse: float
        It is the square root of mse (the best value is 0.0)
    """

    D_TREE = pickle.load(open(os.path.join(MODEL_PATH, "d_tree.pkl"), "rb"))
    housing_predictions = D_TREE.predict(housing_prepared)

    tree_mse = mean_squared_error(housing_labels, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    return tree_mse, tree_rmse


def rforest_pred(MODEL_PATH: str, X_test_prepared: pd.DataFrame, y_test: pd.DataFrame):
    """Make model prediction
    Note: Model should exist in the given path

    Parameters
    ----------

    MODEL_PATH : str
        Path of the model
    X_test_prepared : csv
        Pre-processed housing dataset with features
    y_test : csv
        Target labels of the housing dataset

    Returns
    -----------
    mse: float
        A non-negative floating point. The best value is 0.0
    """

    RFOREST_REG = pickle.load(open(os.path.join(MODEL_PATH, "random_forest.pkl"), "rb"))
    final_predictions = RFOREST_REG.predict(X_test_prepared)
    final_mse = mean_squared_error(y_test, final_predictions)
    return final_mse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=False,
        help="add the path where you have the models stored",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        required=False,
        help="provide the path where you have the processed data",
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

    if args.model:
        MODEL_PATH = args.model
    else:
        MODEL_PATH = os.path.join(os.getcwd(), "artifacts")

    if args.data:
        DATA_PATH = args.data
    else:
        DATA_PATH = os.path.join(os.getcwd(), "data/processed")

    if args.log_level:
        LOGGING_DEFAULT_CONFIG["root"]["level"] = args.log_level

    LOG_FILE = None
    if args.log_path:
        LOG_FILE = args.log_path

    CONSOLE_LOG = True
    if args.no_console_log:
        CONSOLE_LOG = False

    logger = configure_logger(
        log_file=os.path.join(LOG_FILE, "custom_config.log") if LOG_FILE else None,
        console=CONSOLE_LOG,
        log_level=LOGGING_DEFAULT_CONFIG["root"]["level"],
    )
    logger.warning("Logging - Start")
    logger.info("Running script!!")

    housing = pd.read_csv(os.path.join(DATA_PATH, "preprocessed_housing.csv"))
    housing_prepared = pd.read_csv(os.path.join(DATA_PATH, "housing_prepared.csv"))
    housing_labels = pd.read_csv(os.path.join(DATA_PATH, "housing_labels.csv"))
    logger.info("Fetched housing data!!")

    X_test_prepared = pd.read_csv(
        os.path.join(DATA_PATH, "RandomForestRegressor_TestFeatures.csv")
    )
    y_test = pd.read_csv(
        os.path.join(DATA_PATH, "RandomForestRegressor_TestGroundTruth.csv")
    )

    lin_rmse, lin_mae = lin_reg_pred(MODEL_PATH, housing_prepared, housing_labels)
    logger.info(f"Linear Regression\nRMSE: {lin_rmse}\nMAE: {lin_mae}\n\n")

    tree_mse, tree_rmse = dtree_pred(MODEL_PATH, housing_prepared, housing_labels)
    logger.info(f"Decision Tree Regressor\nMSE: {tree_mse}\nRMSE: {tree_rmse}\n\n")

    final_mse = rforest_pred(MODEL_PATH, X_test_prepared, y_test)
    logger.info(f"Random Forest Regressor\nMSE: {final_mse}")
    logger.info("score.py Logging - Completed\n")
