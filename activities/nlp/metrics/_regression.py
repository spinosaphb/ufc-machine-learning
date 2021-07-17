import numpy as np
"""
Mean absolute error
"""
def mean_absolute_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    diff = np.abs(y_true - y_pred)
    diff_sum = np.sum(diff)
    mean = diff_sum / y_true.shape[0]
    return mean
"""
Mean squared error
"""
def mean_squared_error(y_true, y_pred, squared=True):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    diff = (y_true - y_pred)**2
    squared_error = np.sum(diff)
    mean_s_e = squared_error / y_true.shape[0] 
    if not squared:
        return mean_s_e ** 0.5
    return mean_s_e
