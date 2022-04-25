from yf_kazan_test.data import load_datasets
from yf_kazan_test.util import print_md, catch_time

# load_datasets()

import time

with catch_time() as t:
    time.sleep(3)
print(t())

