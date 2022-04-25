import csv

class Category:
    @classmethod
    def load_from_json(cls, json_file):
        with open(json_file, "r", encoding="utf8") as cat_file:
            next(csv.reader(cat_file))
            cls.catlist = {int(line[0]):{"title":line[1], "parent_id":int(line[2])} for line in csv.reader(cat_file)}
            
            max_len = 0
            for id, cat in cls.catlist.items():
                cat["path"] = cls.__path(id)
                max_len = max(max_len, len(cat["path"]))
            
            for cat in cls.catlist.values():
                cat["path"] = cat["path"] + [0] * (max_len - len(cat["path"]))

    @classmethod
    def __path(cls, id):
        id, path = id, []
        while id != 1:
            path.append(id)
            id = cls.catlist[id]["parent_id"] if id in cls.catlist else 1
        return path[::-1]

    def __init__(self, id):
        self.id = id
        props = type(self).catlist[id]
        self.title = props["title"]
        self.path = props["path"]

    @property
    def path_name(self):
        return [type(self).catlist[id]["title"] for id in self.path if id]


def init_category_tree():
    Category.load_from_json("categories_tree.csv")
