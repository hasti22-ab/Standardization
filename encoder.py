# -*- coding: utf-8 -*-
"""encoder.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1xKp4DPpjyc33HMiUGa1rWUIbG_IjByd6
"""

from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# Function to encode categorical variables
def encode_categorical(df, threshold=10):
    """
    Encodes categorical variables in the DataFrame.
    - High-cardinality columns are label-encoded.
    - Low-cardinality columns are one-hot encoded.

    Parameters:
    df (DataFrame): The input DataFrame containing categorical columns.
    threshold (int): The threshold for the number of unique values to decide between label encoding and one-hot encoding.

    Returns:
    DataFrame: A new DataFrame with encoded categorical variables, including other non-encoded columns.
    """
    df_encoded = df.copy()

    # Ensure there are no NaN or Inf values in the DataFrame before processing
    print("Checking for NaN and Inf values before encoding:")
    print("NaN count:", df_encoded.isnull().sum().sum())
    print("Inf count:", np.isinf(df_encoded).sum().sum())

    # Clean the DataFrame by replacing Inf with NaN and dropping rows with NaN values
    df_encoded = df_encoded.replace([np.inf, -np.inf], np.nan).dropna()

    print("After cleaning, checking for NaN and Inf values again:")
    print("NaN count:", df_encoded.isnull().sum().sum())
    print("Inf count:", np.isinf(df_encoded).sum().sum())

    categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
    encoded_cols = pd.DataFrame()

    for col in categorical_cols:
        num_unique = df_encoded[col].nunique()
        if num_unique > threshold:
            # High-cardinality column: apply Label Encoding
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            print(f"Label Encoded column: {col}")
        else:
            # Low-cardinality column: apply One-Hot Encoding
            dummies = pd.get_dummies(df_encoded[col], prefix=col)
            encoded_cols = pd.concat([encoded_cols, dummies], axis=1)
            df_encoded.drop(columns=[col], inplace=True)
            print(f"One-Hot Encoded column: {col}")

    # Concatenate the encoded columns with the rest of the DataFrame
    df_encoded = pd.concat([df_encoded, encoded_cols], axis=1)

    return df_encoded