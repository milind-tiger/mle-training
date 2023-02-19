import argparse
import logging
import os
import tarfile

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from config import *

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="Path to store the data files", default=DEFAULT_HOUSING_PATH)
args = parser.parse_args()

HOUSING_PATH = args.path

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


with requests.get(HOUSING_URL, stream=True) as rx, tarfile.open(
    fileobj=rx.raw, mode="r:gz"
) as tarobj:
    tarobj.extractall(HOUSING_PATH)


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


housing = load_housing_data()

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

train_set.to_csv(os.path.join(HOUSING_PATH, "train.csv"))
test_set.to_csv(os.path.join(HOUSING_PATH, "test.csv"))
