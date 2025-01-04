import numpy as np

def calculate_iqr_ratio(X):
    """
    Calculate interquartile range (IQR) ratio for the dataset X.

    Parameters:
    X: Input feature matrix (numpy array or pandas dataframe).

    Returns:
    float: The IQR ratio of the dataset.
    """
    feature_iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
    iqr_ratio = np.max(feature_iqr) / np.min(feature_iqr)
    return iqr_ratio
