import os
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


def test_lin_reg_train():
    column_names = [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
        "rooms_per_household",
        "bedrooms_per_room",
        "population_per_household",
        "ocean_proximity_INLAND",
        "ocean_proximity_ISLAND",
        "ocean_proximity_NEAR BAY",
        "ocean_proximity_NEAR OCEAN",
    ]
    values = [
        -121.46,
        38.52,
        29.0,
        3873.0,
        797.0,
        2237.0,
        706.0,
        2.1736,
        5.485835694050992,
        0.20578363026077975,
        3.168555240793201,
        1,
        0,
        0,
        0,
    ]
    housing_prepared = pd.DataFrame(dict(zip(column_names, values)), index=[0])
    housing_labels = [72100.0]
    assert len(housing_labels) == len(housing_prepared)
    assert isinstance(housing_prepared.values, np.ndarray)

    assert sorted(housing_prepared.columns.to_list()) == sorted(column_names)
