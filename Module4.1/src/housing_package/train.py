# Assignment 2.1
"""Practice
to train models
"""
import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, os.path.abspath("."))
from src.housing_package.log_config import LOGGING_DEFAULT_CONFIG, configure_logger


def income_cat_proportions(data: str):
    """Find the income proportion per category

    Parameters
    ----------
    data : str
        Path of the dataset

    Returns
    -----------
    income_prop: pd.Series
        proportion of income category
    """
    income_prop = data["income_cat"].value_counts() / len(data)
    return income_prop


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        required=False,
        help="add the path where you stored the processed data",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=False,
        help="provide the path where you want to store the models",
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
        PATH = os.path.join(os.getcwd(), "data/processed")

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        OUTPUT_PATH = args.output
    else:
        OUTPUT_PATH = os.path.join(os.getcwd(), "artifacts")

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
    logger.info("Started logging train.py")

    train_set = pd.read_csv(os.path.join(PATH, "train_set.csv"))
    test_set = pd.read_csv(os.path.join(PATH, "test_set.csv"))
    strat_train_set = pd.read_csv(os.path.join(PATH, "strat_train_set.csv"))
    strat_test_set = pd.read_csv(os.path.join(PATH, "strat_test_set.csv"))
    logger.info("Imported the csv files")

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing = strat_train_set.copy()

    corr_matrix = housing.corr()
    corr_matrix["median_house_value"].sort_values(ascending=False)
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]

    housing = strat_train_set.drop(
        "median_house_value", axis=1
    )  # drop labels for training set
    housing.to_csv(os.path.join(PATH, "preprocessed_housing.csv"), index=False)

    housing_labels = strat_train_set["median_house_value"].copy()

    imputer = SimpleImputer(strategy="median")

    housing_num = housing.drop("ocean_proximity", axis=1)

    imputer.fit(housing_num)
    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
    housing_tr["rooms_per_household"] = (
        housing_tr["total_rooms"] / housing_tr["households"]
    )
    housing_tr["bedrooms_per_room"] = (
        housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    )
    housing_tr["population_per_household"] = (
        housing_tr["population"] / housing_tr["households"]
    )

    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))

    housing_prepared.to_csv(os.path.join(PATH, "housing_prepared.csv"), index=False)
    housing_labels.to_csv(os.path.join(PATH, "housing_labels.csv"), index=False)
    logger.info("Pre-processing done!")

    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)

    FNAME = "lin_reg.pkl"
    pickle.dump(lin_reg, open(os.path.join(OUTPUT_PATH, FNAME), "wb"))
    logger.info("Dumped Linear Regression")

    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(housing_prepared, housing_labels)

    FNAME = "d_tree.pkl"
    pickle.dump(tree_reg, open(os.path.join(OUTPUT_PATH, FNAME), "wb"))
    logger.info("Dumped Decision Tree")

    param_distribs = {
        "n_estimators": randint(low=1, high=200),
        "max_features": randint(low=1, high=8),
    }

    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(
        forest_reg,
        param_distributions=param_distribs,
        n_iter=10,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
    )
    rnd_search.fit(housing_prepared, housing_labels)
    cvres = rnd_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]

    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(housing_prepared, housing_labels)

    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    feature_importances = grid_search.best_estimator_.feature_importances_
    sorted(zip(feature_importances, housing_prepared.columns), reverse=True)

    final_model = grid_search.best_estimator_

    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()
    y_test.to_csv(
        os.path.join(PATH, "RandomForestRegressor_TestGroundTruth.csv"), index=False
    )

    X_test_num = X_test.drop("ocean_proximity", axis=1)
    X_test_prepared = imputer.transform(X_test_num)
    X_test_prepared = pd.DataFrame(
        X_test_prepared, columns=X_test_num.columns, index=X_test.index
    )
    X_test_prepared["rooms_per_household"] = (
        X_test_prepared["total_rooms"] / X_test_prepared["households"]
    )
    X_test_prepared["bedrooms_per_room"] = (
        X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
    )
    X_test_prepared["population_per_household"] = (
        X_test_prepared["population"] / X_test_prepared["households"]
    )

    X_test_cat = X_test[["ocean_proximity"]]
    X_test_prepared = X_test_prepared.join(pd.get_dummies(X_test_cat, drop_first=True))
    X_test_prepared.to_csv(
        os.path.join(PATH, "RandomForestRegressor_TestFeatures.csv"), index=False
    )

    FNAME = "random_forest.pkl"
    pickle.dump(final_model, open(os.path.join(OUTPUT_PATH, FNAME), "wb"))
    logger.info("Dumped Random Forest")

    logger.info("Exported all the models")
    logger.info("train.py Logging - Completed\n")
