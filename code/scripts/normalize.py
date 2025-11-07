from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

def normalize_data(data_x, method='standard'):
    """
    Normalize the input data using the specified method.

    Parameters:
    data_x (array-like): The input data to be normalized.
    method (str): The normalization method to use ('standard' or 'minmax').

    Returns:
    array-like: The normalized data.
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Unsupported normalization method. Use 'standard' or 'minmax'.")

    normalized_data = scaler.fit_transform(data_x)
    return scaler, normalized_data