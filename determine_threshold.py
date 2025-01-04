

import numpy as np
from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.append('/content/drive/MyDrive/Colab Notebooks')



from percentile_selector import choose_best_percentile_threshold  # Import the best threshold selector
from variance_ratio import calculate_variance_ratio  # Import variance ratio calculation
from iqr_ratio import calculate_iqr_ratio  # Import IQR calculation
from feature_range_ratio import calculate_feature_range_ratio  # Import feature range calculation

from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def determine_threshold(model, X, custom_thresholds=None):
    """
    Dynamically adjusts scaling thresholds based on the model type and data distribution, with flexibility for user customization.

    Parameters:
    model: A machine learning model (e.g., KNeighborsClassifier, SVC, etc.)
    X: Input feature matrix (numpy array or pandas dataframe).
    custom_thresholds (dict): Optional. Dictionary with custom thresholds for 'high', 'moderate', and 'low' sensitivity.

    Returns:
    float: The dynamically recommended threshold for determining if standardization is needed.
    """

    # Default thresholds for high, moderate, and low sensitivity
    default_thresholds = {
        'high': 0.05,
        'moderate': 0.1,
        'low': 0.2
    }

    # If custom thresholds are provided, update the default ones
    if custom_thresholds:
        default_thresholds.update(custom_thresholds)

    # Calculate variance ratio, IQR ratio, and feature range ratio
    variance_ratio = calculate_variance_ratio(X)
    iqr_ratio = calculate_iqr_ratio(X)
    range_ratio = calculate_feature_range_ratio(X)

    # Dynamically determine the best threshold for each metric
    best_percentile_variance = choose_best_percentile_threshold([variance_ratio])
    best_percentile_iqr = choose_best_percentile_threshold([iqr_ratio])
    best_percentile_range = choose_best_percentile_threshold([range_ratio])

    # Define models based on their sensitivity to scale
    highly_sensitive_models = [KNeighborsClassifier, SVC, PCA]
    moderately_sensitive_models = [LogisticRegression, Ridge, Lasso, ElasticNet,
                                   GradientBoostingClassifier, XGBClassifier, LGBMClassifier, CatBoostClassifier]
    low_sensitivity_models = [RandomForestClassifier, LinearRegression]

    # Select metric and threshold based on model type and data characteristics
    if isinstance(model, tuple(highly_sensitive_models)):
        if variance_ratio > best_percentile_variance:
            return "Variance Ratio", variance_ratio
        elif iqr_ratio > best_percentile_iqr:
            return "IQR", iqr_ratio
        else:
            return "Feature Range", range_ratio
    elif isinstance(model, tuple(moderately_sensitive_models)):
        if iqr_ratio > best_percentile_iqr:
            return "IQR", iqr_ratio
        else:
            return "Feature Range", range_ratio
    elif isinstance(model, tuple(low_sensitivity_models)):
        return "Feature Range", range_ratio
    else:
        return "Feature Range", range_ratio

# Example usage:
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.neighbors import KNeighborsClassifier

    # Load dataset
    X, y = load_iris(return_X_y=True)

    # Select a model
    model = KNeighborsClassifier()

    # Call the determine_threshold function
    selected_metric, computed_value = determine_threshold(model, X)

    # Output the selected metric and the computed value
    print(f"Recommended metric for {model.__class__.__name__}: {selected_metric}")
    print(f"Computed value of {selected_metric}: {computed_value}")











