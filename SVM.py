"""
script that creates a Mutinomial Naives Bayes model
"""

# various imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import svm

from evaluation_metrics import eval_predictions
from preprocessing import prep_review

import matplotlib
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator
import numpy as np 

# lets read our data
df = pd.read_csv('data/phones_processed.csv')

# remove null values
print("dataset :", len(df))
df = df[df['reviewText'].notnull()]
print("removing null reviewText", len(df))
df = df[df['overall'].notnull()]
print("removing null overall values", len(df))

# section used to merge classes to obtain 5 / 3 / 2 classes problem
"""
df.overall.replace([5.0, 4.0, 3.0, 2.0, 1.0], [1, 1, 1, -1, -1], inplace=True)
df = df.sample(frac=1).reset_index(drop=True)

df2 = df[df['overall'] == 1]
df2 = df2[:2000]
df1 = df[df['overall'] == -1]
df1 = df1[:2000]
frames = [df1, df2]
df = pd.concat(frames)
"""

# SVM can't really go higher than 5000 obs on my computer
df = df[:5000]

# split as test / train 
X_train, X_test, y_train, y_test = train_test_split(df['reviewText'], df['overall'], test_size=.2, random_state=1997)

# tfidf preprocessing
tfidf_vectorizer_1 = TfidfVectorizer(min_df=5, max_df=0.8)
tfidf_train_1 = tfidf_vectorizer_1.fit_transform(X_train)
tfidf_test_1 = tfidf_vectorizer_1.transform(X_test)

# instantiate and train model, kernel=rbf 
print("kernel=kbf")
svm_rbf = svm.SVC(random_state=12345, gamma="auto")
svm_rbf.fit(tfidf_train_1, y_train)

# evaluate model
y_pred_1 = svm_rbf.predict(tfidf_test_1)
eval_predictions(y_test, y_pred_1)
print()

# instantiate and train model, kernel=linear
print("kernel=linear")
svm_rbf = svm.SVC(kernel='linear', random_state=12345, gamma="auto")
svm_rbf.fit(tfidf_train_1, y_train)

# evaluate model
y_pred_1 = svm_rbf.predict(tfidf_test_1)
eval_predictions(y_test, y_pred_1)
print()

tfidf_vectorizer_2 = TfidfVectorizer(tokenizer=prep_review, min_df=5, max_df=0.8)
tfidf_train_2 = tfidf_vectorizer_2.fit_transform(X_train)
tfidf_test_2 = tfidf_vectorizer_2.transform(X_test)

# kernel=rbf
print('kernel=rbf')
svm_rbf = svm.SVC(kernel="rbf", random_state=1, gamma="auto")
svm_rbf.fit(tfidf_train_2, y_train)
y_pred_2 = svm_rbf.predict(tfidf_test_2)
eval_predictions(y_test, y_pred_2)
print() 

# some tests without preprocessing to compare performances
print('kernel=linear')
svm_rbf = svm.SVC(kernel='linear', random_state=1, gamma="auto")
svm_rbf.fit(tfidf_train_2, y_train)
y_pred_2 = svm_rbf.predict(tfidf_test_2)
eval_predictions(y_test, y_pred_2)
print()

"""
section used to create plots with matplotlib 
has changed a lot to produce graphs for final report
"""

n_groups = 3

ngrams = (0.785, 0.577, 0.387)
precision = (0.734, 0.552, 0.34)

fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.40
opacity = 0.4
error_config = {'ecolor': '0.3'}

accuracy = ax.bar(index, ngrams, bar_width,
                alpha=opacity, color='b', error_kw=error_config,
                label='with ngram')

precision = ax.bar(index + bar_width, precision, bar_width,
                alpha=opacity, color='r', error_kw=error_config,
                label='without ngram')


ax.set_xlabel('Number of target classes')
ax.set_ylabel('Accuracy')
ax.set_title('Effect of the ngram preprocessing on accuracy (MultinomialNB)')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('2 : [-1, 1]', '3 : [-1, 0, 1]', '5 : [0, 1, 2, 3, 4, 5]'))
ax.legend()

fig.tight_layout()
fig.savefig("report/ngrams/ngram_comparaison.png")