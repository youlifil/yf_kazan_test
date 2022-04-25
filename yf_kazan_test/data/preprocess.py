from collections import Counter

from ..vectors.tokenize import tokenizer

def preprocess_train(dataset):
    def drop_rare_categories():
        rare_rate = 2
        rares = [y for y, cnt in Counter(dataset["category_id"]).most_common() if cnt <= rare_rate]
        dataset.drop(dataset[dataset["category_id"].isin(rares)].index, inplace=True)

    preproces_features(dataset)
    drop_rare_categories()


def preproces_features(dataset):
    def resolve_nan():
        dataset["short_description"].fillna("", inplace=True)
        dataset["name_value_characteristics"].fillna("", inplace=True)
    
    def make_about_column():
        row_texter = lambda row: ' '.join(tokenizer(row["title"] + " " + row["short_description"] + " " + row["name_value_characteristics"]))
        dataset["about"] = dataset.apply(row_texter, axis=1)

    resolve_nan()
    make_about_column()
