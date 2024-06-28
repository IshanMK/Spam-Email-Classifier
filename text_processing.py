import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import os

# Set NLTK data path to include the nltk_data directory
nltk.data.path.append(os.path.join(os.getcwd(), "nltk_data"))

# Determine the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Path to NLTK data for stopwords
stopwords_dir = os.path.join(script_dir, "nltk_data", "corpora", "stopwords", "stopwords")

# Set NLTK data path specifically for stopwords
nltk.data.path.append(stopwords_dir)

# Load NLTK stopwords from the specified directory
with open(os.path.join(stopwords_dir, 'english'), 'r') as stopwords_file:
    stopwords = stopwords_file.read().splitlines()

# stopwords = stopwords.words('english')
ps = PorterStemmer()

def clean_text(text):
    text = "".join([char.lower() for char in text if char not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text

def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count / (len(text) - text.count(" ")), 3) * 100
