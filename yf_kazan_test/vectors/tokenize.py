import re

from pymorphy2 import MorphAnalyzer
import nltk
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

stopwords_ru = stopwords.words("russian")
garbage_regex = "[!#$%&'*.<=>?@[\]^_`{}~—\"]+"
delimeter_regex = "[\s\-/:;,()+|]"
morph = MorphAnalyzer()

def tokenizer(text):
    text = text.lower()
    text = re.sub(garbage_regex, "", text)
    tokens = []
    for token in re.split(delimeter_regex, text):
        token = token.strip()
        if token and token not in stopwords_ru:
            token = morph.normal_forms(token)[0]
            tokens.append(token)    

    return ' '.join(tokens)