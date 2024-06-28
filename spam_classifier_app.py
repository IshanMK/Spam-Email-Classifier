import streamlit as st
import pickle
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
import os
import sys
from text_processing import clean_text, count_punct  # Adjust this import based on your file structure

# Set the page configuration
st.set_page_config(page_title="Spam Email Classifier")

# Determine the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Set NLTK data path to include the current script directory
nltk_data_dir = os.path.join(script_dir, "nltk_data")
nltk.data.path.append(nltk_data_dir)

# Path to the directory containing stopwords
stopwords_dir = os.path.join(script_dir, "nltk_data", "corpora", "stopwords" , "stopwords")

# Load NLTK stopwords and stemmer
ps = PorterStemmer()

# Check if NLTK stopwords are already available
if not os.path.exists(stopwords_dir):
    st.error("NLTK stopwords directory not found. Please ensure it exists and contains the necessary files.")
    st.stop()

# Set NLTK data path specifically for stopwords
nltk.data.path.append(stopwords_dir)

# Ensure NLTK resources are available
try:
    nltk.data.find('corpora/stopwords/stopwords')
except LookupError:
    st.warning("NLTK stopwords not found. Please download NLTK data or ensure it is correctly configured.")
    st.stop()

# Load NLTK stopwords from the specified directory
with open(os.path.join(stopwords_dir, 'english'), 'r') as stopwords_file:
    stopwords = stopwords_file.read().splitlines()

# Load the pre-trained model and vectorizer
model_path = os.path.join(script_dir, 'spam_classifier_model.pkl')
vectorizer_path = os.path.join(script_dir, 'tfidf_vectorizer.pkl')

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(vectorizer_path, 'rb') as vectorizer_file:
    tfidf_vect = pickle.load(vectorizer_file)

# Streamlit app
st.title("Spam Email Classifier")

st.write("""
This app classifies an email as spam or not spam.
""")

# User input
email_text = st.text_area("Enter the email text:", height=200)

if st.button("Classify"):
    if email_text:
        # Prepare input data
        body_len = len(email_text) - email_text.count(" ")
        punct_perc = count_punct(email_text)
        tfidf_input = tfidf_vect.transform([email_text])
        input_vect = pd.concat([pd.DataFrame([[body_len, punct_perc]], columns=['body_len', 'punct%']),
                                pd.DataFrame(tfidf_input.toarray(), columns=tfidf_vect.get_feature_names_out())], axis=1)
        
        # Predict
        prediction = model.predict(input_vect)
        
        # Display result
        if prediction[0] == 'spam':
            st.error("This email is classified as Spam.")
        else:
            st.success("This email is classified as Not Spam.")
    else:
        st.warning("Please enter some text to classify.")
