"""
script that creates a Mutinomial Naives Bayes model
"""

import pandas as pd
import ast
import time
import re
import scipy

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from preprocessing import prep_review
from evaluation_metrics import eval_predictions

# lets get the data from the file
df = pd.read_csv('data/phones_processed.csv')

# getting an overview of the dataset
print(df.shape)
print(df.head(3))

df['overall'].value_counts().plot(kind='bar', color='cornflowerblue')

# lets remove incomplete data
print("dataset :", len(df))
df = df[df['reviewText'].notnull()]
print("removing null reviewText", len(df))
df = df[df['overall'].notnull()]
print("removing null overall values", len(df))

# split the dataset into train / test
X_train, X_test, y_train, y_test = train_test_split(df['reviewText'], df['overall'], test_size=.2, random_state=1)

# instantiate the vectorizer
vect = CountVectorizer()

# tokenize train and test text data
X_train_dtm = vect.fit_transform(X_train)
print("number words in training corpus:", len(vect.get_feature_names()))
X_test_dtm = vect.transform(X_test)

# lets build our model !
nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)

# make class predictions
y_pred = nb.predict(X_test_dtm)

# and evaluate our model
eval_predictions(y_test, y_pred)

"""
# model optimisation purpose
# print message text for the first 3 false positives
print('False positives:')
print()
for x in X_test[y_test < y_pred][:2]:
    print(x)
    print()

# print message text for the first 3 false negatives
print('False negatives:')
print()
for x in X_test[y_test > y_pred][:2]:
    print(x[:500])
    print()
"""
