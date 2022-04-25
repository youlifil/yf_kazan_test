from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

from yf_kazan_test.data.load import Datapack, TheDatasets

def to_tfidf(src, min_freq=1):
    this = to_tfidf

    def fit_vectorizer():
        if not hasattr(this, "vectorizer"):
            this.vectorizer = TfidfVectorizer(min_df=min_freq)
            this.vectorizer.fit(TheDatasets.train["about"])

    fit_vectorizer()

    if type(src) is Datapack:
        return Datapack(
            X_train = this.vectorizer.transform(src.X_train["about"]),
            y_train = src.y_train.values,
            X_test = this.vectorizer.transform(src.X_test["about"]),
            y_test = src.y_test.values
        )
    elif type(src) is pd.DataFrame:
        return this.vectorizer.transform(src)


def tfidf_vocab():
    return to_tfidf.vectorizer.get_feature_names_out()
