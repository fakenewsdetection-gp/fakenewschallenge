import re
import nltk

token_pattern = r"(?u)\b\w\w+\b"
stemmer = nltk.stem.PorterStemmer()
stopwords = set(nltk.corpus.stopwords.words('english'))

def preprocess_data(line, filter_stopwords=True, stem=True):
    tk_re = re.compile(token_pattern, flags = re.UNICODE | re.LOCALE)
    tokens = [x.lower() for x in tk_re.findall(line)]
    
    if stem:
        tokens = [stemmer.stem(tok) for tok in tokens]
    if filter_stopwords:
        tokens = [tok for tok in tokens if tok not in stopwords]
    
    return tokens