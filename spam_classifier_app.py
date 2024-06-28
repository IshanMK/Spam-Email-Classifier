import streamlit as st
import pickle
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
import os
import sys

# Set NLTK data path to include the nltk_data directory
nltk.data.path.append(os.path.join(os.getcwd(), "nltk_data"))

# Import functions from text_processing.py
from text_processing import clean_text, count_punct

# Set the page configuration
st.set_page_config(page_title="Spam Email Classifier")

# Determine the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Path to NLTK data for stopwords
stopwords_dir = os.path.join(script_dir, "nltk_data", "corpora", "stopwords", "stopwords")

# Load NLTK stopwords and stemmer
ps = PorterStemmer()

# Set NLTK data path specifically for stopwords
nltk.data.path.append(stopwords_dir)

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
