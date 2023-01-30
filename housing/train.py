""" This module is for traing the data"""
import argparse
import os
import sys
import warnings
from urllib.parse import urlparse

import logger
import mlflow
import mlflow.sklearn
import pandas as pd
from methods import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet

warnings.filterwarnings("ignore")

# import the data
import ingest_data

# define the logger
LOGGER = logger.get_logger(__name__)

# Parsing the command line inputs

DEFAULT_HOUSING_TRAIN_PATH = os.path.join("datasets", "train.csv")
DEFAULT_HOUSING_TEST_PATH = os.path.join("datasets", "test.csv")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_path", help="Path to training data", default=DEFAULT_HOUSING_TRAIN_PATH
)
parser.add_argument("--test_path", help="Path to testing data", default=DEFAULT_HOUSING_TEST_PATH)

args = parser.parse_args()

HOUSING_TRAIN_PATH = args.train_path
HOUSING_TEST_PATH = args.test_path

# *********Model building and training

# Load the trainig data
train_data = pd.read_csv(HOUSING_TRAIN_PATH)
test_data = pd.read_csv(HOUSING_TEST_PATH)

# Load preprocessed X and y

train_x, train_y = preprocess(train_data)
test_x, test_y = preprocess(test_data)

with mlflow.start_run(experiment_id="1") as run:

    LOGGER.info("Elastic Net Training Started")
    #  Model1 -- Elastic Net
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)

    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    # Model registry does not work with file store
    if tracking_url_type_store != "file":
        mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetWineModel")
    else:
        mlflow.sklearn.log_model(lr, "model")

with mlflow.start_run(experiment_id="2") as run:

    LOGGER.info("Random Forest Training Started")
    #  Model1 -- Elastic Net
    n_estimators = 100
    max_depth = 3
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    rf.fit(train_x, train_y)

    predicted_qualities = rf.predict(test_x)

    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Random Forest model (n_estimators=%f, max_depth=%f):" % (n_estimators, max_depth))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    mlflow.log_param("alpha", n_estimators)
    mlflow.log_param("l1_ratio", max_depth)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    # Model registry does not work with file store
    if tracking_url_type_store != "file":
        mlflow.sklearn.log_model(lr, "model", registered_model_name="RandomForestModel")
    else:
        mlflow.sklearn.log_model(lr, "model")
