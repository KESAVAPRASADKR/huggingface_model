Documentation for GPT-2 Fine-Tuning and Text Generation Script
This documentation provides an overview of the Python script used for fine-tuning a GPT-2 language model and generating text. The script leverages the Hugging Face Transformers library and includes functions for reading text files, preparing datasets, training the model, and generating text.
Requirements
Before running the script, ensure you have the following libraries installed:
•	pandas
•	numpy
•	re
•	os
•	transformers
You can install the required libraries using:
pip install pandas numpy 
pip install pip install
pip install transformers
Code Overview
Import Statements
python
Copy code
import pandas as pd
import numpy as np
import re
import os
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
These import statements bring in the necessary libraries and modules for data handling, text processing, and model training.
Functions
1. read_txt(file_path)
Reads a text file and returns its content as a string.
python
Copy code
def read_txt(file_path):
    with open(file_path, "r", encoding='utf-8') as file:
        text = file.read()
    return text
2. load_dataset(file_path, tokenizer, block_size=128)
Loads the dataset for training using the provided tokenizer.
python
Copy code
def load_dataset(file_path, tokenizer, block_size=128):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )
    return dataset
3. load_data_collator(tokenizer, mlm=False)
Loads the data collator for language modeling.
python
Copy code
def load_data_collator(tokenizer, mlm=False):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=mlm,
    )
    return data_collator
4. train(train_file_path, model_name, output_dir, overwrite_output_dir, per_device_train_batch_size, num_train_epochs, save_steps)
Trains the GPT-2 model using the specified parameters.
python
Copy code
def train (train_file_path, model_name,
          output_dir,
          overwrite_output_dir,
          per_device_train_batch_size,
          num_train_epochs,
          save_steps):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    train_dataset = load_dataset(train_file_path, tokenizer)
    data_collator = load_data_collator(tokenizer)

    tokenizer.save_pretrained(output_dir)

    model = GPT2LMHeadModel.from_pretrained(model_name)

    model.save_pretrained(output_dir)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
    )

    trainer = Trainer (
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model()
5. load_model(model_path)
Loads a pretrained GPT-2 model from the specified path.
python
Copy code
def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model
6. load_tokenizer(tokenizer_path)
Loads a pretrained tokenizer from the specified path.
python
Copy code
def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer
7. generate_text(model_path, tokenizer_path, sequence, max_length)
Generates text using the trained GPT-2 model.
python
Copy code
def generate_text(model_path, tokenizer_path, sequence, max_length):
    model = load_model(model_path)
    tokenizer = load_tokenizer(tokenizer_path)
    ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id,
        top_k=50,
        top_p=0.95,
    )
    print(tokenizer.decode(final_outputs[0], skip_special_tokens=True))
model_path=’paste your model path’
text=input()
max_len=150
generate_text(model_path, text, max_len)

Main Script Execution
1.	Read and preprocess the training text file:
python
Copy code
train_directory = r"your path "
text_data = read_txt(train_directory)
text_data = re.sub(r'\n+', '\n', text_data).strip()
with open(r'path, "w", encoding='utf-8') as f:
    f.write(text_data)
2.	Define training parameters and execute the training function:
python
Copy code
train_file_path = r"path"
model_name = 'gpt2'
output_dir = r'C:\Users\kesav\Guvi\guvi main\c6'
overwrite_output_dir = False
per_device_train_batch_size = 8
num_train_epochs = 30
save_steps = 10000

# Train the model
train (
    train_file_path=train_file_path,
    model_name=model_name,
    output_dir=output_dir,
    overwrite_output_dir=overwrite_output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    num_train_epochs=num_train_epochs,
    save_steps=save_steps
)
This script reads a text file, preprocesses the text, trains a GPT-2 model with the preprocessed text, and saves the trained model and tokenizer. The script also includes functions for loading the trained model and tokenizer and generating text from the trained model.


