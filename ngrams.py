"""
script used to determine which value of ngram
worked better in our problem
"""


import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import everygrams
from sklearn.model_selection import train_test_split
import pandas as pd 
from preprocessing import prep_review
from evaluation_metrics import eval_predictions

import matplotlib
import matplotlib.pyplot as plt

# generates ngrams preprocessing
def create_ngram_features(sentence, n=2):
    preprocessed = prep_review(sentence)
    ngram_vocab = everygrams(preprocessed, 1, n)
    ngram_vocab = everygrams(sentence.split(), 1, n)
    my_dict = dict([(ng, True) for ng in ngram_vocab])
    return my_dict

# lets read our data
df = pd.read_csv("data/phones_processed.csv")

# remove null obs
df = df[df['reviewText'].notnull()]
df = df[df['overall'].notnull()]
df = df.sample(frac=1).reset_index(drop=True)

# lets work with the 5000 first obs
df = df[:5000]

"""
section used to simplify the 5 classes problem to 2 or 3 
df.overall.replace([5.0, 4.0, 3.0, 2.0, 1.0], [1, 1, 0, -1, -1], inplace=True)

df5 = df[df['overall'] == 5.0]
df5 = df5[:1000]
df4 = df[df['overall'] == 4.0]
df4 = df4[:1000]
df3 = df[df['overall'] == 3.0]
df3 = df3[:1000]
df2 = df[df['overall'] == 2.0]
df2 = df2[:1000]
df1 = df[df['overall'] == 1.0]
df1 = df1[:1000]
"""

for n in range(1,6):
  data = []

  for index, row in df.iterrows():
    review = row["reviewText"]
    grade = str(row["overall"])
    ngrams = create_ngram_features(review, 3)
    data.append((ngrams, grade))

  train = data[:4000]
  test = data[4000:]
  
  """
  # some debug 
  print("data : ", len(data))
  print("train : ", len(train))
  print("test : ", len(test))
  """

  classifier = NaiveBayesClassifier.train(train)

  accuracy = nltk.classify.util.accuracy(classifier, test)
  print(str(n) + "-gram : accuracy={" + str(accuracy) + "}")

"""
# section to generate a graph where you can see which n has best result for ngram !
fig, ax = plt.subplots()
ax.plot(ns, accuracies)
plt.xticks(ns)
ax.set(xlabel='ngrams', ylabel='model accuracy',
       title='Model accuracy vs Ngrams')
ax.grid()

fig.savefig("report/multiNB/5class_2k_80_20_prepro.png")
"""