import numpy as np
from sklearn.metrics import confusion_matrix

from . import _classfication

class ClassificationMetrics():
    def __init__(self, y_true, y_pred, avarages='weighted') -> None:
        self.y_true = y_true
        self.y_pred = y_pred
        self.LABELS = np.unique(y_true)
        self.conf_matrix = confusion_matrix(
            y_true = y_true,
            y_pred = y_pred,
            labels = self.LABELS
        )
        self.accuracy_score = _classfication.accuracy_score(
            y_true, y_pred
        )
        self.precision_score = _classfication.precision_score(
            y_true, y_pred, 
            labels=self.LABELS,
            average=avarages
        )
        self.recall_score = _classfication.recall_score(
            y_true, y_pred, 
            labels=self.LABELS,
            average=avarages
        )
        self.f1_score = _classfication.f1_score(
            y_true, y_pred,
            average=avarages 
        )
        self.metrics_matrix = {
            "accuracy": self.accuracy_score,
            "precision": self.precision_score,
            "recall": self.recall_score,
            "f1": self.f1_score
        }



        @property
        def conf_matrix(self):
            return self.conf_matrix
        
        @property
        def metrics_matrix(self):
            return self.metrics_matrix
            
        @property
        def accuracy_score(self):
            return self.accuracy_score
        
        @property
        def precision_score(self):
            return self.precision_score
        
        @property
        def recall_score(self):
            return self.recall_score

        @property
        def f1_score(self):
            return self.f1_score
