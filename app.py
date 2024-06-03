import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TextVectorization
import pandas as pd

# Load the trained model
model = load_model('model.h5')

# Set up the TextVectorization layer (ensure it's configured as per your training)
MAX_FEATURES = 300000
vectorizer = TextVectorization(max_tokens=MAX_FEATURES, output_sequence_length=2000, output_mode='int')

# Assuming you have the training data to adapt the vectorizer
file_path = 'train.csv'
df = pd.read_csv(file_path)
X = df['comment_text']
vectorizer.adapt(X.values)

# Define the labels
labels = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

# Define the Streamlit app
st.title('Toxic Comment Classifier')

user_input = st.text_area('Enter a comment to classify:')

if st.button('Classify'):
    # Vectorize the input
    input_text = vectorizer([user_input])

    # Predict
    prediction = model.predict(np.array(input_text))
    prediction = (prediction > 0.5).astype(int)[0]

    # Display the results
    result = {label: pred for label, pred in zip(labels, prediction)}
    st.write('Prediction:', result)
