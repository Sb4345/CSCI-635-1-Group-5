from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler



def sample_stratify(data, label_col, n_samples=500, rand_state=42):
    """
    Generate stratified samples from the dataset using
    provided column name as labels.
    """
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=n_samples, random_state=rand_state)
    for _, test_index in splitter.split(data, data[label_col]):
        strat_test_set = data.iloc[test_index]

    return strat_test_set


def top_pca(data, n_components=6):
    """
    Perform PCA on the dataset and return the top n_components.
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components)
    pca.fit(data)
    principal_components = pca.transform(data)
    pc_df = pd.DataFrame(data=principal_components,
                         columns=[f'PC_{i+1}' for i in range(n_components)])
    return pc_df, pca


def resample_data(x, y, target_over=None, target_under=None, rand_state=42):
    """
    Resample the dataset using SMOTE for oversampling and RandomUnderSampler for undersampling.
    """
    if target_over:
        smote = SMOTE(sampling_strategy=target_over, random_state=rand_state)
        x_resampled, y_resampled = smote.fit_resample(x, y)
    else:
        x_resampled, y_resampled = x, y

    if target_under:
        rus = RandomUnderSampler(sampling_strategy=target_under, random_state=rand_state)
        x_resampled, y_resampled = rus.fit_resample(x_resampled, y_resampled)

    return x_resampled, y_resampled


def main():
    from sklearn.datasets import load_iris
    iris = load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df['target'] = iris.target

    reduced = sample_stratify(iris_df, 'target', n_samples=30, rand_state=1)

    print("Stratified Reduced Set:\n", reduced['target'].value_counts())

    num_components = 3
    pc_df, pca_model = top_pca(reduced.drop(columns=['target']), n_components=num_components)
    print(f"\nTop {num_components} Principal Components:\n", pc_df.head())
    print("PCA Components:\n", pca_model.components_)

if __name__ == "__main__":
    main()
