import pandas as pd

from .load import TheDatasets

def save_submission(target, name):
    sub = pd.DataFrame()
    sub["id"] = TheDatasets.test["id"]
    sub["predicted_category_id"] = target
    sub.to_csv(name + ".csv")
    sub.to_parquet(name + ".parquet")

