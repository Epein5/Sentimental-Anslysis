import os
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Embedding, Bidirectional, Dense
from tensorflow.keras.layers import LSTM 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TextVectorization


MAX_FEATURES = 200000 # number of words in the vocab

df = pd.read_csv(os.path.join('jigsaw-toxic-comment-classification-challenge','train.csv', 'train.csv'))[:1000]
X = df['comment_text']
y = df[df.columns[2:]].values

vectorizer = TextVectorization(max_tokens=MAX_FEATURES,
                               output_sequence_length=1800,
                               output_mode='int')
vectorizer.adapt(X.values)

def get_model():
    model = Sequential()
    model.add(Embedding(MAX_FEATURES+1, 32))
    model.add(Bidirectional(LSTM(32, activation='tanh')))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(6, activation='sigmoid'))
    model.compile(loss='BinaryCrossentropy', optimizer='Adam')
    return model

model = get_model()

import tensorflow as tf


model = tf.keras.models.load_model(r'C:\Users\KNYpe\Desktop\Sentimental-Anslysis\Models\toxicity.h5')


def score_comment(comment):
    vectorized_comment = vectorizer([comment])
    results = model.predict(vectorized_comment)
    
    text = ''
    for idx, col in enumerate(df.columns[2:]):
        text += '{}: {}\n'.format(col, results[0][idx]>0.5)
    
    return text


print(score_comment('I hate you!'))