import gdown
import pandas as pd
from collections import Counter
from yf_kazan_test.vectors import tokenizer
from sklearn.model_selection import train_test_split

class Datasets:
    train = None
    test = None

    @classmethod
    @property
    def class_num(cls):
        return len(set(cls.train["category_id"]))


class Datapack:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test


def download_datasets():
    gdown.download(id="1IsTN19WU00pcLAGf52Vp9_-xC-6c-49J")
    gdown.download(id="11r85SCXCJrfZ_vShAcyr8AJmjZd7lQsz")
    gdown.download(id="1u2TqxNY9U9KGchC77_s2PGbOPHrCvoFf")

    Datasets.train = pd.read_parquet("train.parquet")
    Datasets.test = pd.read_parquet("test.parquet")


def download_preprocessed_train_dataset():
    dataset_filename = "preprocessed_train.parquet"
    gdown.download(id="1P9exIDwqTYNa-kvLgFhRA7_1BhXAINWv", output=dataset_filename)
    Datasets.train = pd.read_parquet(dataset_filename)


def split_dataset(target_column):

    data_columns = ["about", "rating", "feedback_quantity"]

    X_train, X_test, y_train, y_test = train_test_split(
        Datasets.train[data_columns],
        Datasets.train[target_column], 
        test_size=0.25, 
        random_state=47, 
        stratify=Datasets.train[target_column])

    return Datapack(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)


def preprocess_dataset():
    train = Datasets.train
    
    train["short_description"].fillna("", inplace=True)
    train["name_value_characteristics"].fillna("", inplace=True)

    rare_rate = 2
    rares = [y for y, cnt in Counter(train["category_id"]).most_common() if cnt <= rare_rate]
    print("Rare classes (less than {} records per class): {}, corresponding records: {}".format(rare_rate, len(rares), len(train[train["category_id"].isin(rares)])))
    train.drop(train[train["category_id"].isin(rares)].index, inplace=True)

    # создание колонки общего текстового описания, нормализованного
    row_texter = lambda row: tokenizer(row["title"] + " " + row["short_description"] + " " + row["name_value_characteristics"])
    train["about"] = train.apply(row_texter, axis=1)
    token_count = sum(len(ab.split()) for ab in train["about"].values)
    print("train corpus total token count: {:,}".format(token_count))
    file_prefix = "/content/drive/MyDrive/projects-ds/kazan-test/train_df_preprocessed_"
    train.to_parquet("{}{}.parquet".format(
        file_prefix,
        f"{token_count:,}".replace(",","-")
    ))
