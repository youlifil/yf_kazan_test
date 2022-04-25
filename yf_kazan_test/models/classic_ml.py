import datetime

from yf_kazan_test.util import print_md, catch_time
from yf_kazan_test.models import print_score
from yf_kazan_test.vectors.tfidf import to_tfidf
from yf_kazan_test.data import TheDatasets

def run_classic_ml(classifier, datapack, title=None):
    if not title:
        title = classifier.__class__.__name__

    with catch_time() as time_:
        classifier.fit(datapack.X_train, datapack.y_train)

        pred_train = classifier.predict(datapack.X_train)
        pred_test = classifier.predict(datapack.X_test)

    print_md(f"**{title}**: {datetime.timedelta(seconds=round(time_()))}")
    print_md("✅ **train**")
    print_score(datapack.y_train, pred_train)
    print_md("✅ **test**")
    print_score(datapack.y_test, pred_test)

