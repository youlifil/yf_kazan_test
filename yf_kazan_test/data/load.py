import gdown
import pandas as pd
from sklearn.model_selection import train_test_split

from .types import TheDatasets, Datapack

def download_datasets():
    gdown.download(id="1IsTN19WU00pcLAGf52Vp9_-xC-6c-49J")
    gdown.download(id="11r85SCXCJrfZ_vShAcyr8AJmjZd7lQsz")
    gdown.download(id="1u2TqxNY9U9KGchC77_s2PGbOPHrCvoFf")

    TheDatasets.train = pd.read_parquet("train.parquet")
    TheDatasets.test = pd.read_parquet("test.parquet")


def download_preprocessed_train_dataset():
    dataset_filename = "preprocessed_train.parquet"
   
    gdown.download(id="1P9exIDwqTYNa-kvLgFhRA7_1BhXAINWv", output=dataset_filename)
    TheDatasets.train = pd.read_parquet(dataset_filename)


def split_datasets(target_column):
    def check_datasets():
        if TheDatasets.train is None:
            TheDatasets.train = pd.read_parquet("preprocessed_train.parquet")
        if TheDatasets.test is None:
            TheDatasets.test = pd.read_parquet("test.parquet")

    check_datasets()

    data_columns = ["about", "rating", "feedback_quantity"]

    X_train, X_test, y_train, y_test = train_test_split(
        TheDatasets.train[data_columns],
        TheDatasets.train[target_column], 
        test_size=0.25, 
        random_state=47, 
        stratify=TheDatasets.train[target_column])

    return Datapack(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
