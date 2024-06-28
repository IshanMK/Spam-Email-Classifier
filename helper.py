
import os
import requests
from io import BytesIO
from zipfile import ZipFile
import nltk

# Function to download NLTK stopwords corpus if not already downloaded
def download_nltk_stopwords():
    nltk_data_dir = "nltk_data"
    stopwords_path = os.path.join(nltk_data_dir, "corpora", "stopwords")
    
    # Check if stopwords directory exists
    if not os.path.exists(stopwords_path):
        os.makedirs(stopwords_path)

    # Define GitHub URLs for stopwords
    github_url = "https://github.com/nltk/nltk_data/raw/gh-pages/packages/corpora/stopwords.zip"
    
    # Download stopwords.zip and extract
    response = requests.get(github_url)
    with ZipFile(BytesIO(response.content)) as z:
        z.extractall(stopwords_path)

    # Add nltk_data directory to nltk.data.path
    nltk.data.path.append(nltk_data_dir)