import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle

vect = pickle.load(open("vector.pkl", "rb"))

pickle_filename = 'bayes_sentiment_ml.pkl'

pickled_model = pickle.load(open(pickle_filename, 'rb'))

test_custom=pd.DataFrame(['this stock sucks', 'amc is a buy', 'I love GME', 'gme calls for days'])
tf_custom = vect.transform(test_custom[0])
print(pickled_model.predict(tf_custom))
