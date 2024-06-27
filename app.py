import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from huggingface_hub import login
from scipy.special import softmax
import numpy as np
import mysql.connector
from datetime import datetime

connection = mysql.connector.connect(
  host = "gateway01.ap-southeast-1.prod.aws.tidbcloud.com",
  port = 4000,
  user = "JLMYagiq919U7D8.root",
  password = "W5p0nq33F4qJEN28",
  database = "t1",
  
)
mycursor = connection.cursor(buffered=True)

# Main app
def main():
    # Check if the name is in the session state
    if 'name' not in st.session_state:
        st.session_state['name'] = None

    # Page navigation
    if st.session_state['name'] is None:
        name_page()
    else:
        gpt_2()
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
def gpt_2():
   # Function to load model and tokenizer

    def load_model(model_path):
        model = GPT2LMHeadModel.from_pretrained(model_path)
        return model
    def load_tokenizer(tokenizer_path):
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        return tokenizer
    # Function to generate text
    def generate_text(model, tokenizer, sequence, max_length):
        ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
        final_outputs = model.generate(
            ids,
            do_sample=True,
            max_length=max_length,
            pad_token_id=model.config.eos_token_id,
            top_k=50,
            top_p=0.95,
        )
        return tokenizer.decode(final_outputs[0], skip_special_tokens=True)

    # Streamlit app
    def main():
        st.title("Text Generation with GPT-2")

    model_path = r'C:\Users\kesav\Guvi\guvi main\c6\gpt'
    tokenizer_path = r'C:\Users\kesav\Guvi\guvi main\c6\gpt'

        # Load the model and tokenizer
    model = load_model(model_path)
    tokenizer = load_tokenizer(tokenizer_path)

        # User input
    sequence = st.text_input("Enter a sequence to start generating text:")
    max_length = st.slider("Select max length of generated text:", 10, 500, 100)

    if st.button("Generate"):
        with st.spinner("Generating text..."):
            generated_text = generate_text(model, tokenizer, sequence, max_length)
            st.write("Generated Text:")
            st.write(generated_text)




if __name__ == "__main__":
    main()