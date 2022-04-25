from collections import Counter

from yf_kazan_test.category_tree import Category
from yf_kazan_test.vectors import tokenizer
from yf_kazan_test.score import print_score

def run_simple_matching(train_df):
    true, pred = [], []

    DUMMY_CATEGORY = Counter(train_df["category_id"]).most_common()[0][0]
    category_tokens = [(cat, set(tokenizer(' '.join(cat.path_name)).split())) for cat in map(Category, set(train_df["category_id"]))]
    data = zip(train_df["category_id"], train_df["about"])

    for cat_id, about in data:
        name_tokens = set(about.split())
        matches = [(cat, cat_tokens.intersection(name_tokens)) 
                    for cat, cat_tokens in category_tokens 
                    if cat_tokens.intersection(name_tokens)]
        if matches:
            best = sorted(matches, key=lambda m:len(m[1]), reverse=True)[0]
            pred_id = best[0].id
        else:
            pred_id = DUMMY_CATEGORY
        true.append(cat_id)
        pred.append(pred_id)

    print_score(true, pred)