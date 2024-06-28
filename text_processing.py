import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os
import sys
import nltk 

# Determine the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add script directory to Python path
sys.path.append(script_dir)

# Set NLTK data path to include the current script directory
nltk_data_dir = os.path.join(script_dir, "nltk_data")
nltk.data.path.append(nltk_data_dir)

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