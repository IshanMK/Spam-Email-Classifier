import streamlit as st
import pickle
import pandas as pd
import re
import string
import nltk
from text_processing import clean_text, count_punct
import tempfile

nltk.download('stopwords', download_dir=tempfile.gettempdir())
nltk.data.path.append(tempfile.gettempdir())

# nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()

# Set the page configuration
st.set_page_config(page_title="Spam Email Classifier")


# Load the pre-trained model and vectorizer
with open('spam_classifier_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
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
