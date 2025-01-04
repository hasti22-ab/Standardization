import numpy as np

def calculate_feature_range_ratio(X):
    """
    Calculate feature range (max - min) ratio for the dataset X.

    Parameters:
    X: Input feature matrix (numpy array or pandas dataframe).

    Returns:
    float: The feature range ratio of the dataset.
    """
    feature_range = np.max(X, axis=0) - np.min(X, axis=0)
    range_ratio = np.max(feature_range) / np.min(feature_range)
    return range_ratio

