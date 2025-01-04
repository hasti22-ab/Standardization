# -*- coding: utf-8 -*-
"""Diag_dtype (1).ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1gSLGktsmb2AxdHuEsDom_zmFRNkvDjd6
"""

from google.colab import files

# Upload the file
uploaded = files.upload()

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from google.colab import drive
import sys
import os

# Mount the Google Drive
drive.mount('/content/drive', force_remount=True)

# Verify the files in the directory
print(os.listdir("/content/drive/MyDrive"))

# Assuming you have uploaded the file correctly
data = pd.read_csv("/content/drive/MyDrive/test_data.csv")
df_example = pd.DataFrame(data)

print(df_example.head())



# Global detect_type function
def detect_type(value):
    str_value = str(value).strip()

    types_found = []

    if str_value in ["", "nan", "NaN", "---", "N/A"]:
        types_found.append('missing')

    if str_value.lower() in ['true', 'false', 'yes', 'no'] or isinstance(value, bool):
        types_found.append('bool')

    if re.match(r'^\d+$', str_value):  # Integer
        types_found.append('int')
    elif re.match(r'^\d*\.\d+$', str_value):  # Float
        types_found.append('float')

    try:
        pd.to_datetime(str_value, errors='raise')
        types_found.append('datetime')
    except (ValueError, TypeError):
        pass

    if re.match(r'^[£€$]\d+(\.\d+)?$', str_value):
        types_found.append('currency')

    if re.match(r'^[a-zA-Z\s,.!?]+$', str_value) and len(str_value.split()) > 2:
        types_found.append('text')

    if len(types_found) > 1:
        return 'Mixed Types: ' + ', '.join(types_found)
    elif types_found:
        return types_found[0]
    return 'str'

def Diag_dtype(df, column_names):
    print("Starting Diagnosis")

    # Ensure column_names is treated as a list, even if it contains only one element
    if not isinstance(column_names, list):
        raise ValueError("The input should be a list of column names.")

    # Initialize an empty list to store the results for each column
    results = []

    for column_name in column_names:
        column = df[column_name]
        print(f"Processing column: {column_name}")

        # Check if the column is numeric
        is_numeric = pd.api.types.is_numeric_dtype(column)

        # Elbow method for threshold determination
        def get_elbow_threshold(column):
            unique_counts = column.value_counts().values
            diff = np.diff(unique_counts)
            elbow_point = np.argmax(diff < 0.1 * diff.max()) if len(diff) > 0 else 0
            return unique_counts[elbow_point] if len(unique_counts) > elbow_point else len(unique_counts)

        # Function to select threshold based on distribution
        def choose_thresholding_method(column):
            unique_values = column.nunique()
            print(f"Unique values for {column_name}: {unique_values}")

            if unique_values <= 5:
                threshold = unique_values
                method = 'Percentile-Based'
            elif is_numeric and column.std() > 5:
                threshold = column.mean() + 2 * column.std()
                method = 'Standard Deviation-Based'
            else:
                threshold = get_elbow_threshold(column)
                method = 'Elbow Method'

            return threshold, method

        threshold, method = choose_thresholding_method(column)

        # Detecting types for non-null values
        non_null_df = df.loc[df[column_name].notna()].copy()
        non_null_df['Detected_Type'] = non_null_df[column_name].apply(detect_type)

        type_summary = non_null_df['Detected_Type'].value_counts().to_dict()
        mixed = len(type_summary) > 1

        unique_values = non_null_df[column_name].nunique()
        is_high_cardinality = unique_values > threshold

        if is_numeric and unique_values <= 50:
            categorical_type = 'Binary/Boolean' if unique_values <= 2 else 'Numeric Identifier or Categorical'
        elif is_high_cardinality:
            categorical_type = 'High-Cardinality Categorical'
        elif 'str' in type_summary and unique_values <= 5:
            categorical_type = 'Nominal/Low-Cardinality'
        elif 'str' in type_summary and mixed:
            categorical_type = 'Mixed Types'
        elif 'bool' in type_summary:
            categorical_type = 'Binary/Boolean'
        elif 'datetime' in type_summary:
            categorical_type = 'Datetime'
        elif 'text' in type_summary:
            categorical_type = 'Non-Categorical Text'
        elif is_numeric:
            categorical_type = 'Non-Categorical Numeric'
        else:
            categorical_type = 'Non-High Cardinality'

        output = {
            'Column': column_name,
            'Types Detected': type_summary,
            'Mixed Types': mixed,
            'Categorical Type': categorical_type,
            'Unique Values': unique_values,
            'Threshold': threshold,
            'Method Used': method
        }

        results.append(output)

    return pd.DataFrame(results)

# Display settings for wide dataframes
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

# Analyze the dataframe
def analyze_dataframe(df):
    for column in df.columns:
        result_df = Diag_dtype(df, [column])
        print(result_df)

# Analyze each column with thresholding
analyze_dataframe(df_example)

# Additional Step to show type in front of each cell
df_copy = df_example.copy()

# Insert type-detection columns immediately to the right of each original column
column_order = []
for column_name in df_copy.columns:
    df_copy[f'{column_name}_type'] = df_copy[column_name].apply(detect_type)

    # Store the original and type column order
    column_order.append(column_name)
    column_order.append(f'{column_name}_type')

# Reorder the dataframe based on the original columns and their corresponding type columns
df_copy = df_copy[column_order]

# Display dataframe with types next to each column
print(df_copy.head())

# Save dataframe with types
df_copy.to_csv(r'c:\Users\asal\Downloads\output_with_types16.csv', index=False)