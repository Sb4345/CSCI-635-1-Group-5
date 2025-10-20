from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd

def sample_stratify(data, label_col, n_samples=500, rand_state=42):
    """
    Generate stratified samples from the dataset using
    provided column name as labels.
    """
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=n_samples, random_state=rand_state)
    for train_index, test_index in splitter.split(data, data[label_col]):
        strat_train_set = data.iloc[train_index]
        strat_test_set = data.iloc[test_index]

    return strat_train_set, strat_test_set
