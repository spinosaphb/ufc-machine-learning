import numpy as np
from sklearn.metrics import confusion_matrix
"""
accuracy score
"""
def accuracy_score(y_true, y_pred) -> float:
    score = y_true == y_pred
    hits = np.sum(score)
    accuracy = hits / len(y_true)
    return accuracy
"""
weights
"""
def _get_weights_label(y_true, labels):
    hits = np.sum(y_true == labels[0])
    if len(labels) <= 1:
        return hits
    return np.append(hits, _get_weights_label(y_true, labels[1:]))

def _weighted_average(data, weights):
    data_sum = np.sum(data * weights)
    weights_sum = np.sum(weights)
    return data_sum / weights_sum
"""
precision score
"""
def precision_score(y_true, y_pred, labels=None, average='weighted') -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    if labels is None:
        labels = np.unique(y_true) 

    conf_matrix = confusion_matrix(
        y_true = y_true,
        y_pred = y_pred,
        labels = labels
    )

    true_positives = np.diag(conf_matrix)
    total = conf_matrix.sum(axis=0)
    total[total == 0] = 1  

    precisions = true_positives / total
    weights = _get_weights_label(y_true=y_true, labels=labels)
    if average == 'weighted':
        return _weighted_average(precisions, weights)
    return precisions
"""
recal score
"""
def recall_score(y_true, y_pred, labels=None, average='weighted'):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    if labels is None:
        labels = np.unique(y_true) 

    conf_matrix = confusion_matrix(
        y_true = y_true,
        y_pred = y_pred,
        labels = labels
    )

    true_positives = np.diag(conf_matrix)
    false_negatives = conf_matrix.sum(axis=1)
    false_negatives[false_negatives == 0] = 1  

    recals = true_positives / false_negatives
    weights = _get_weights_label(y_true=y_true, labels=labels)
    if average == 'weighted':
        return _weighted_average(recals, weights)
    return recals
"""
f1-score
"""
def f1_score(y_true, y_pred, average='weighted'):
    precision  = precision_score(y_true, y_pred, average=average) 
    recall = recall_score(y_true, y_pred, average=average)
    f1_score = 2 * precision * recall / (precision + recall)
    return f1_score