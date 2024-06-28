import subprocess

def download_nltk_stopwords():
    try:
        subprocess.run(["python", "-m", "nltk.downloader", "stopwords"], check=True)
        print("NLTK stopwords downloaded successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading NLTK stopwords: {e}")

# Call the function to download NLTK stopwords
download_nltk_stopwords()