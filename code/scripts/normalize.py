import sklearn.preprocessing as preproc
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
        scaler = preproc.StandardScaler()
    elif method == 'minmax':
        scaler = preproc.MinMaxScaler()
    elif method == 'l2':
        scaler = preproc.Normalizer(norm='l2')
    else:
        raise ValueError("Unsupported normalization method. Use 'standard', 'minmax', or 'l2'.")

    normalized_data = scaler.fit_transform(data_x)
    return scaler, normalized_data