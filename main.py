import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization


MAX_FEATURES = 200000 # number of words in the vocab

df = pd.read_csv(os.path.join('Datasets/jigsaw-toxic-comment-classification-challenge','train.csv', 'train.csv'))
X = df['comment_text']
y = df[df.columns[2:]].values

vectorizer = TextVectorization(max_tokens=MAX_FEATURES,
                               output_sequence_length=1800,
                               output_mode='int')
vectorizer.adapt(X.values)

model = tf.keras.models.load_model(r'PATH/TO/MODEL')


def score_comment(comment):
    vectorized_comment = vectorizer([comment])
    results = model.predict(vectorized_comment)
    
    text = ''
    for idx, col in enumerate(df.columns[2:]):
        text += '{}: {}\n'.format(col, results[0][idx]>0.5)
    
    return text


print(score_comment('I hate you!'))