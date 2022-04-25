import gdown
import pandas as pd

CLASS_NUM = None

def download_datasets():
    gdown.download(id="1IsTN19WU00pcLAGf52Vp9_-xC-6c-49J")
    gdown.download(id="11r85SCXCJrfZ_vShAcyr8AJmjZd7lQsz")
    gdown.download(id="1u2TqxNY9U9KGchC77_s2PGbOPHrCvoFf")

def preprocessed_train_dataset():
    dataset_filename = "preprocessed_train.parquet"
    gdown.download(id="1P9exIDwqTYNa-kvLgFhRA7_1BhXAINWv", output=dataset_filename)
    train_df = pd.read_parquet(dataset_filename)
    CLASS_NUM = len(set(train_df["category_id"]))
    return train_df

def test_dataset():
    return pd.read_parquet("test.parquet")
