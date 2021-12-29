import numpy as np
import pandas as pd
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import resample
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import metrics
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_confusion_matrix
from nltk.corpus import stopwords
from nltk import word_tokenize
import nltk
import string
import re
import pickle

# How does it work?
# Tokenize and remove stop words
# Vectorize
# Apply Multinomial Naive Bayes
# Save model

stop_words = set(stopwords.words('english'))

train_orig=pd.read_csv('train.csv')
test_nolabel=pd.read_csv('test.csv')

stop_words = set(stopwords.words('english'))

train = train_orig

def remove_stopwords(line):
    word_tokens = word_tokenize(line)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return " ".join(filtered_sentence)

def preprocess(line):
    line = line.lower()
    line = re.sub(r'\d+', '', line)
    line = line.translate(line.maketrans("","", string.punctuation))
    line = remove_stopwords(line)
    return line
for i,line in enumerate(train.text):
    train.text[i] = preprocess(line)

X_train, X_test, y_train, y_test = train_test_split(train['text'], train['label'], test_size=0.5, stratify=train['label'])

trainp = train[train.label=='positive']
trainn = train[train.label=='negative']
trainn.info()

train_imbalanced = train

df_majority = train[train.label=='positive']
df_minority = train[train.label=='negative']

df_minority_upsampled = resample(df_minority,
                                 replace=True,
                                 n_samples=len(df_majority),
                                 random_state=123)

df_upsampled = pd.concat([df_majority, df_minority_upsampled])

print(train.label.value_counts())

print(df_upsampled.label.value_counts())

X_train, X_test, y_train, y_test = train_test_split(df_upsampled['text'], df_upsampled['label'], test_size=0.5, stratify=df_upsampled['label'])

model = MultinomialNB()

vect = CountVectorizer()
tf_train=vect.fit_transform(X_train)
tf_test=vect.transform(X_test)

tf_test_nolabel=vect.transform(test_nolabel.text)

model.fit(X=tf_train,y=y_train)

expected = y_test
predicted=model.predict(tf_test)

print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

plot_confusion_matrix(metrics.confusion_matrix(expected, predicted))

trainn.iloc[:10]

gg=X_test.reset_index(drop=True)
for i, p in enumerate(predicted):
    print (gg[i] + " - " + str(p))
    if i>5:
        break


predicted_nolabel=model.predict(tf_test_nolabel)
for i, p in enumerate(tf_test_nolabel):
    print (test_nolabel.text[i] + " - " + str(predicted_nolabel[i]))
    if i>5:
        break

test_custom=pd.DataFrame(['this stock sucks', 'amc is a buy', 'I love GME', 'gme calls for days'])
tf_custom = vect.transform(test_custom[0])
perdorming = model.predict(tf_custom)

pickle_filename = 'bayes_sentiment_ml.pkl'
pickle_filename_vector = 'vector.pkl'

with open(pickle_filename, 'wb') as file:
    pickle.dump(model, file)

with open(pickle_filename_vector, 'wb') as file:
    pickle.dump(vect, file)
