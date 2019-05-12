import string
import nltk
from nltk.corpus import stopwords
import unicodedata
import inflect
import contractions
import html


nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

nltk.download('punkt')
nltk.download('wordnet')



class TextCleaner:
  
  def __init__(self):
    self.lem_ = nltk.stem.WordNetLemmatizer()
    self.stem_ = nltk.stem.PorterStemmer()
  
  @staticmethod
  def fix_contractions(text):
    return contractions.fix(text)

  @staticmethod
  def unescape_html(text):
    return html.unescape(text)

  @staticmethod
  def lemmatize(token, lemmatizer):
    return lemmatizer.lemmatize(token, 'v')

  @staticmethod
  def stem(token, stemmer):
    return stemmer.stem(token)
  
  @staticmethod
  def remove_non_ascii(word):
    return unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')

  @staticmethod
  def lowercase(word):
    return word.lower()

  @staticmethod
  def remove_punct(word):
    for sym in string.punctuation:
      word = word.replace(sym, '')
    return word

  @staticmethod
  def replace_num(word):
    p = inflect.engine()
    return p.number_to_words(word) if word.isdigit() else word

  @staticmethod
  def filter(tokens, filterset):
    """
    Filters tokens by removing the tokens that appear in filterset.
    """

    return [t for t in tokens if t not in filterset]

  @staticmethod
  def transform(tokens, func):
    """
    Applies a transformation function on a token.
    """

    return [func(t) for t in tokens if func(t).strip()]

  @staticmethod
  def normalize(tokens, remove_nonascii,
                   to_lower, remove_punctuation, replace_num, remove_stopwords):
    
    if remove_nonascii:
      tokens = TextCleaner.transform(tokens, TextCleaner.remove_non_ascii)
      
    if to_lower:
      tokens = TextCleaner.transform(tokens, TextCleaner.lowercase)
      
    if remove_punctuation:
      tokens = TextCleaner.transform(tokens, TextCleaner.remove_punct)
      
    if replace_num:
      tokens = TextCleaner.transform(tokens, TextCleaner.replace_num)

    if remove_stopwords:
      tokens = TextCleaner.filter(tokens, stop_words)
      
    return tokens
  
  @staticmethod
  def tokenize(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    
    return tokens
  
  def cleanup_text(self, text, lemmatize=True, stem=False, remove_stopwords=True,
                   remove_punctuation=True, remove_nonascii=True,
                   to_lower=True, replace_num=True, fix_contractions=True, unescape_html=True,
                   remove_possessive=True, remove_puncwords=True):
    
    if unescape_html:
      text = TextCleaner.unescape_html(text)
      
    if remove_punctuation:
      for ch in string.punctuation:
        text = text.replace(ch, ' ')
    
    if fix_contractions:
      text = TextCleaner.fix_contractions(text)
    
    tokens = TextCleaner.tokenize(text)
    
    if remove_possessive:
      tokens = [token.replace("'s", "").replace("'", "").strip() for token in tokens]
    
    if remove_puncwords:
      char_tester = str.isalpha if replace_num else str.isalnum
      tokens = [token for token in tokens if token and char_tester(token[0])]
    
    tokens = TextCleaner.normalize(tokens, remove_nonascii, to_lower, remove_punctuation, replace_num, remove_stopwords)
        
    if lemmatize:
      tokens = TextCleaner.transform(tokens, lambda token: TextCleaner.lemmatize(token, self.lem_))
          
    if stem:
      tokens = TextCleaner.transform(tokens, lambda token: TextCleaner.stem(token, self.stem_))
 
    return ' '.join(tokens)