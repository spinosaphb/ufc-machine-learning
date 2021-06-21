"""
libs.metrics module includes scores functions
"""
from ._classfication import accuracy_score
from ._classfication import precision_score
from ._classfication import recall_score
from ._classfication import f1_score

from ._regression import mean_absolute_error
from ._regression import mean_squared_error

__all__ = [
    "accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "mean_absolute_error",
    "mean_squared_error"
]