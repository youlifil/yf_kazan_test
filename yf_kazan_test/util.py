from IPython.display import Markdown, display
import time
from contextlib import contextmanager

def print_md(string):
    display(Markdown(string))

@contextmanager
def catch_time() -> float:
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start
