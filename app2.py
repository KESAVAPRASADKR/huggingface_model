import streamlit as st
import sqlite3
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np
import mysql.connector
from datetime import datetime

connection = mysql.connector.connect(
  host = "",
  port = ,
  user = "",
  password = "",
  database = "",
  
)
mycursor = connection.cursor(buffered=True)

# Function to preprocess text
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Load the model and tokenizer
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Function to predict sentiment
def predict(text):
    preprocessed_text = preprocess(text)
    encoded_input = tokenizer(preprocessed_text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    ranking = np.argsort(scores)
    ranking = ranking[::-1]

    output_text = ""
    for i in range(scores.shape[0]):
        label = config.id2label[ranking[i]]
        score = np.round(float(scores[ranking[i]]), 4)
        output_text += f"{i+1}) {label}: {score}\n"

    return output_text

# Main app
def main():
    # Check if the name is in the session state
    if 'name' not in st.session_state:
        st.session_state['name'] = None

    # Page navigation
    if st.session_state['name'] is None:
        name_page()
    else:
        sentiment_analysis_page()

# Page 1: Collect user name
def name_page():
    st.title("Welcome!")
    st.write("Please enter your name:")
    name = st.text_input("Name")
    if st.button("Submit Name"):
        if name:
            timestamp = datetime.now()
            query = "INSERT INTO user_data (name, timestamp) VALUES (%s, %s)"
            mycursor.execute(query, (name, timestamp))
            connection.commit()
            st.session_state['name'] = name
            st.experimental_rerun()  # Rerun the app to move to the next page
        else:
            st.write("Please enter a valid name.")

# Page 2: Sentiment analysis
def sentiment_analysis_page():
    st.title("Sentiment Analysis with Hugging Face Transformers")
    st.write(f"Hello, {st.session_state['name']}!")
    st.write("Enter some text and see the predicted sentiment with confidence scores.")
    user_input = st.text_area("Enter your text here:")
    if st.button("Analyze"):
        if user_input:
            result = predict(user_input)
            st.text(result)
        else:
            st.write("Please enter some text for analysis.")

if __name__ == "__main__":
    main()
