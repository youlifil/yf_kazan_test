class TheDatasets:
    train = None
    test = None

    @classmethod
    @property
    def class_num(cls):
        return len(set(cls.train["category_id"]))

class Datapack:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
