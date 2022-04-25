from yf_kazan_test.data import download_datasets, preprocessed_train_dataset, test_dataset
from yf_kazan_test.util import print_md, catch_time
from yf_kazan_test.category_tree import init_category_tree

# load_datasets()

train_df = preprocessed_train_dataset()
test_df = test_dataset()

init_category_tree()
