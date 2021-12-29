import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle

# Load vectorizer model
vect = pickle.load(open("vector.pkl", "rb"))

# Load multinomial naive bayes model
pickle_filename = 'bayes_sentiment_ml.pkl'
pickled_model = pickle.load(open(pickle_filename, 'rb'))

# Test data as DataFrame
test_data = pd.DataFrame(['this stock sucks', 'amc is a buy', 'I love GME', 'gme calls for days'])

# Apply CountVectroizer
data_vect = vect.transform(test_data[0])

# Print results
print(pickled_model.predict(data_vect))
