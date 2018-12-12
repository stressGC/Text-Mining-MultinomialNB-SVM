"""
utility class used to print metrics based on a model
"""

from sklearn import metrics

# calculate accuracy, precision, recall, and F-measure of class predictions
def eval_predictions(y_test, y_pred, name="deafult_name"):
    print('accuracy:', metrics.accuracy_score(y_test, y_pred))
    print('precision:', metrics.precision_score(y_test, y_pred, average='weighted'))
    print('recall:', metrics.recall_score(y_test, y_pred, average='weighted'))
    print('F-measure:', metrics.f1_score(y_test, y_pred, average='weighted'))