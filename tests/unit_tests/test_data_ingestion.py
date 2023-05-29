import pickle
import sys
import os
import pandas as pd
import numpy as np
import requests
import tarfile


DOWNLOAD_ROOT = "http://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

sys.path.append(".")


def test_fetch_tgz(housing_url=HOUSING_URL):
    """fetches to the dataset and makes assertion

    Parameters
    ----------

    housing_url: str
        url for fetching the dataset

    Returns
    -----------
    assertion
        True if successful, False otherwise

    """
    tgz_path = os.path.join(os.getcwd(), "housing.tgz")

    r = requests.get(housing_url, timeout=300)
    with open(tgz_path, "wb") as f:
        f.write(r.content)

    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=os.getcwd())
    housing_tgz.close()

    files = os.listdir()
    assert "housing.csv" in files

    column_names = [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
        "median_house_value",
        "ocean_proximity",
    ]
    df = pd.read_csv("housing.csv")
    os.remove("housing.csv")
    assert sorted(df.columns.to_list()) == sorted(column_names)
