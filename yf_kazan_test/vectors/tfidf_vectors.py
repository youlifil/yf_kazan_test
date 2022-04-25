from sklearn.feature_extraction.text import TfidfVectorizer

from yf_kazan_test.data.load import Datapack, TheDatasets

class _this:
    vectorizer = None
    vocab = None


def to_tfidf(datapack):
    _this.vectorizer = TfidfVectorizer(min_df=1)
    _this.vectorizer.fit(TheDatasets.train["about"])

    _this.vocab = _this.vectorizer.get_feature_names_out()

    return Datapack(
        X_train = _this.vectorizer.transform(datapack.X_train["about"]),
        y_train = datapack.y_train.values,
        X_test = _this.vectorizer.transform(datapack.X_test["about"]),
        y_test = datapack.y_test.values
    )


def tfidf_vocab():
    return _this.vocab
