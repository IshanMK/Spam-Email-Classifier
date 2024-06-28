import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from helper import download_nltk_stopwords

# Download NLTK stopwords if not already downloaded
download_nltk_stopwords()    
stopwords = stopwords.words('english')
ps = PorterStemmer()

def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text

def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")), 3)*100