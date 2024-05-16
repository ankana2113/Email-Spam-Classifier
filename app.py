import streamlit as st
import sklearn
import pickle
import string
from nltk.corpus import stopwords
import nltk, re
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


def transform_message(text):
    text = text.lower()

    def clean_text(text):
        rgx = r"[^A-Za-z0-9\s\.]"
        return re.sub(rgx, '', text)

    def remove_stop_words(text):
        tokens = text.split()
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        return ' '.join(tokens)

    lemmatizer = WordNetLemmatizer()

    def lemmatize_text(text):
        return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

    return lemmatize_text(remove_stop_words(clean_text(text)))


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('Email Spam Classifier')

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_message(input_sms)
    # 2. vectorizer
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header('SPAM')
    else:
        st.header('NOT SPAM')
