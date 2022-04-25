from collections import Counter

from .types import TheDatasets
from .tokenize import tokenizer

def preprocess_dataset():
    train = TheDatasets.train
    
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
