# -*- coding: utf-8 -*-
"""check_need_for_standardization.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1E50zUI5bbYeIVOkTXiCT0Bpo6jF9gKu1
"""

!pip install catboost

!pip install hdbscan

!cat /content/drive/MyDrive/Colab\ Notebooks/determine_threshold.py

!ls "/content/drive/MyDrive/Colab Notebooks"

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from hdbscan import HDBSCAN

# Import required libraries for Random Effects, Fixed Effects, DID, and HLM models
from statsmodels.regression.mixed_linear_model import MixedLM
from linearmodels.panel import PanelOLS, RandomEffects
from linearmodels import PooledOLS
from statsmodels.formula.api import ols
from google.colab import drive
import sys

# Mount the drive (use force_remount=True to remount forcibly if needed)
drive.mount('/content/drive', force_remount=True)

# Add your .py file's path to sys.path
sys.path.append('/content/drive/MyDrive/Colab Notebooks')
from determine_threshold import determine_threshold
from detect_scale_difference import detect_scale_difference

# Function to check whether columns need standardization for pairwise comparisons and additional models
def check_pairwise_standardization_df(model, df, custom_thresholds=None):
    scaling_needed_models = [
        LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet,
        KNeighborsClassifier, SVC,
        PCA, LDA, TSNE,
        KMeans, GaussianMixture, HDBSCAN,
        MLPClassifier, MLPRegressor,
        GradientBoostingClassifier, GradientBoostingRegressor,
        XGBClassifier, XGBRegressor,
        LGBMClassifier, LGBMRegressor,
        CatBoostClassifier, CatBoostRegressor
    ]

    # Additional models that may require standardization
    panel_data_models = [PanelOLS, RandomEffects, PooledOLS]  # Panel data regression models
    random_effect_models = [MixedLM]  # Random effects model
    did_model = [ols]  # Difference-in-Differences (DID)
    hlm_model = [MixedLM]  # Hierarchical Linear Model (HLM)

    # List to collect results for each pair of columns
    results = []

    # Check if the model requires scaling (either from sklearn or custom panel, DID, HLM models)
    if (any(isinstance(model, m) for m in scaling_needed_models)
        or any(isinstance(model, m) for m in panel_data_models)
        or any(isinstance(model, m) for m in random_effect_models)
        or any(isinstance(model, m) for m in did_model)
        or any(isinstance(model, m) for m in hlm_model)):

        # Iterate over all pairs of columns in the dataframe
        for i, col1 in enumerate(df.columns):
            for j, col2 in enumerate(df.columns):
                if i >= j:
                    continue  # Avoid comparing the same or previously compared columns

                # Detect scale differences for each feature
                col1_result = detect_scale_difference(df[[col1]], model, custom_thresholds)
                col2_result = detect_scale_difference(df[[col2]], model, custom_thresholds)

                # Check if standardization is recommended for either column
                scale_diff_col1 = col1_result.get(col1)
                scale_diff_col2 = col2_result.get(col2)

                needs_standardization = (
                    "Different scale detected" in scale_diff_col1 or
                    "Different scale detected" in scale_diff_col2
                )

                # Append the result as a dictionary
                results.append({
                    'Column 1': col1,
                    'Column 2': col2,
                    'Scale Difference Column 1': scale_diff_col1,
                    'Scale Difference Column 2': scale_diff_col2,
                    'Standardization Needed': 'Yes' if needs_standardization else 'No'
                })

        # Convert the results to a DataFrame
        return pd.DataFrame(results)

    else:
        return pd.DataFrame(columns=['Column 1', 'Column 2', 'Scale Difference Column 1', 'Scale Difference Column 2', 'Standardization Needed'])

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from linearmodels import datasets

    # Load dataset (for example, panel data regression dataset)
    X, y = load_iris(return_X_y=True)
    df = pd.DataFrame(X, columns=["Feature1", "Feature2", "Feature3", "Feature4"])

    # Example for panel data regression
    panel_data = datasets.jobtraining.load()
    panel_data_df = panel_data.data.reset_index()

    # Select a model (example with Random Effects)
    model = RandomEffects.from_formula('ln_scrap ~ grant + C(year)', panel_data_df)

    # Check if standardization is needed for column pairs
    result_df = check_pairwise_standardization_df(model, df)

    # Output the DataFrame
    print(result_df)