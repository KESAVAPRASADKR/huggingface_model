Documentation for Sentiment Analysis Streamlit Application
Overview
This Streamlit application performs sentiment analysis on user-provided text using a Hugging Face Transformers model. It also records the user's name and the timestamp when they first use the app in a MySQL database.
Dependencies
Ensure you have the following Python packages installed:
•	streamlit
•	transformers
•	scipy
•	numpy
•	mysql-connector-python
You can install them using pip:
bash
Copy code
pip install streamlit transformers scipy numpy mysql-connector-python
Database Setup
The application uses a MySQL database to store user information. Configure the MySQL connection with the following parameters:
•	host: "your host"
•	port: your port
•	user: "username"
•	password: ""
•	database: ""
Ensure the user_data table exists with the following schema:
Sql query 

CREATE TABLE user_data (
    name VARCHAR(255) NOT NULL,
    timestamp DATETIME NOT NULL
);
Application Structure
Imports
The necessary libraries and modules are imported at the beginning of the script:
import streamlit as st
import mysql.connector
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np
from datetime import datetime
Database Connection
A connection to the MySQL database is established:
python
Copy code
connection = mysql.connector.connect(
•	host= "your host"
•	port= your port
•	user= "username"
•	password=""
•	database= ""
)
mycursor = connection.cursor(buffered=True)
Text Preprocessing
A function to preprocess text by replacing user mentions and URLs:
python
Copy code
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)
Model Loading
Loading the sentiment analysis model and tokenizer:
python
Copy code
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
Sentiment Prediction
A function to predict sentiment from the provided text:
python
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
Main Application
The main function checks if the user’s name is stored in the session state and navigates to the appropriate page:
python
Copy code
def main():
    if 'name' not in st.session_state:
        st.session_state['name'] = None

    if st.session_state['name'] is None:
        name_page()
    else:
        sentiment_analysis_page()
Name Page
A page to collect the user's name and store it in the database:
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
            st.experimental_rerun()
        else:
            st.write("Please enter a valid name.")
Sentiment Analysis Page
A page for performing sentiment analysis on user-provided text:

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
Running the Application
Run the Streamlit application by executing the following command in your terminal:
in your terminal 
streamlit run app.py
Replace app.py with the filename of your script.
Conclusion
This Streamlit application allows users to perform sentiment analysis on text input using a pre-trained model from Hugging Face. It also stores user information in a MySQL database. The application consists of two main pages: one for collecting user names and another for performing sentiment analysis.

