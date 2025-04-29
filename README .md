
# SentRNN - Sentiment Analysis with RNN

This notebook implements a sentiment analysis model using a Recurrent Neural Network (RNN) trained on the Sentiment140 dataset. It showcases end-to-end NLP processing and model training using TensorFlow/Keras.

## Features

- Preprocessing of tweets: cleaning, tokenizing, stopword removal, and lemmatization.
- Deep learning model using Embedding + RNN/LSTM.
- Evaluation of model performance.
- Dataset handling using Dask (optional).

## Model Architecture (Keras)

```python
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding,Dropout
from tensorflow.keras.optimizers import Adam
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import dask.dataframe as dd
import kagglehub
import os
num_classes = 3  # Update this based on your use case

model = Sequential([
    Embedding(input_dim=max_features, output_dim=16),
    SimpleRNN(64, activation='tanh', return_sequences=True),  # Change return_sequences to True
    SimpleRNN(128, activation='tanh', return_sequences=True),
    SimpleRNN(64, activation='tanh', return_sequences=False),  # The last RNN layer does not return sequences
    Dense(num_classes, activation='softmax')  # Use softmax for multi-class classification
])

model.compile(
    loss='sparse_categorical_crossentropy', # Use categorical_crossentropy for multi-class classification
    optimizer=Adam(learning_rate = 0.001),
    metrics=['accuracy']
)
# Save the whole model (architecture + weights + optimizer state)
model.save('models/model_SentRnn.keras')

# To load it back:
model = load_model('models/model_SentRnn.keras')
```

## Evaluation Summary

```python
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=500,
    validation_data=(X_val, y_val),
    verbose=1
)

score = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {score[1]:.2f}")
```

## Requirements

- Python
- TensorFlow / Keras
- NLTK
- scikit-learn
- Pandas, NumPy
- Dask (optional)

## Dataset

- Dataset used: Sentiment140 (via `kagglehub`)
- Adjust `file_path` as needed after download.

## Usage

```bash
pip install -r requirements.txt
jupyter notebook SentRNN.ipynb
```
