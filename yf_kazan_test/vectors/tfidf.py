from sklearn.feature_extraction.text import TfidfVectorizer

from yf_kazan_test.data.load import Datapack, TheDatasets

def to_tfidf(source, min_freq=1):
    def fit_vectorizer():
        to_tfidf.vectorizer = TfidfVectorizer(min_df=min_freq)
        to_tfidf.vectorizer.fit(TheDatasets.train["about"])

    fit_vectorizer()

    return Datapack(
        X_train = to_tfidf.vectorizer.transform(source.X_train["about"]),
        y_train = source.y_train.values,
        X_test = to_tfidf.vectorizer.transform(source.X_test["about"]),
        y_test = source.y_test.values
    )


def tfidf_vocab():
    return to_tfidf.vectorizer.get_feature_names_out()
