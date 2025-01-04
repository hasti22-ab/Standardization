import numpy as np

def calculate_variance_ratio(X):
    """
    Calculate variance ratio for the dataset X.

    Parameters:
    X: Input feature matrix (numpy array or pandas dataframe).

    Returns:
    float: The variance ratio of the dataset.
    """
    feature_variance = np.var(X, axis=0)
    variance_ratio = np.max(feature_variance) / np.min(feature_variance)
    return variance_ratio

