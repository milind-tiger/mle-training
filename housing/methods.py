""" This fiule contains all functions to be used"""


import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def eval_metrics(actual, pred):
    """Evaluation Metric for calculating Accuracy metrics.

    Args:
        actual (int): Actual Data
        pred (int): Predicted Data

    Returns:
        tuple: (rmse, mae, r2)
    """
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def preprocess(data):
    """_summary_

    Args:
        data (DataFrame): Input data to be processed

    Returns:
        tuple: input data X and labels y
    """    
    # Create some additional features
    housing = data.copy()
    housing = data.drop("median_house_value", axis=1)  # drop labels for training set
    housing_labels = data["median_house_value"].copy()

    # Impute the missing
    imputer = SimpleImputer(strategy="median")
    housing_num = housing.drop("ocean_proximity", axis=1)
    imputer.fit(housing_num)
    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
    housing_tr["rooms_per_household"] = housing_tr["total_rooms"] / housing_tr["households"]
    housing_tr["bedrooms_per_room"] = housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    housing_tr["population_per_household"] = housing_tr["population"] / housing_tr["households"]

    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))

    return housing_prepared,housing_labels
