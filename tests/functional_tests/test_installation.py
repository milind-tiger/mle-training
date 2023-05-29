def test_pkg_installation():
    try:
        import pickle
        import sys
        import os
        import pandas as pd
        import numpy as np
        import tarfile
        import argparse
        import requests
        from sklearn.model_selection import (
            StratifiedShuffleSplit,
            train_test_split,
        )

        from scipy.stats import randint
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LinearRegression

        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
        from sklearn.tree import DecisionTreeRegressor

        from src.housing_package.log_config import (
            LOGGING_DEFAULT_CONFIG,
            configure_logger,
        )

        from sklearn.metrics import mean_absolute_error, mean_squared_error

        import logging
        import logging.config

    except Exception as e:
        assert False
    else:
        assert True
