import numpy as np

# Normalize
def normalize(X:np.ndarray, minValueCols, maxValueCols, inplace=False):
    return (X - minValueCols) / (np.array(maxValueCols) - minValueCols)
    
# Euclidian distance
def euclidean_distances(X, X_row):
    X_ = (X - X_row) ** 2
    return np.sum(X_, axis=1) ** 0.5

# Getting the k neighbors
def get_neighbors(X_train, test_row, k):
    distances = euclidean_distances(X_train, test_row)
    idx_sort = np.argsort(distances)
    return idx_sort[:k] 

# Classification
def predict_classification(X, y, test_row, k):
    idx_sort = get_neighbors(X, test_row, k)
    output_values = y[idx_sort]
    counts = np.unique(output_values, return_counts=True)
    idx_max = np.argmax(counts[1])
    prediction = counts[0][idx_max]
    return prediction
# Regression
def predict_regression(X_train, y_train, X_test_row, k):
    idx_sort = get_neighbors(X_train, X_test_row, k)
    output_values = y_train[idx_sort]
    regression = np.sum(output_values) / k
    return regression
    
