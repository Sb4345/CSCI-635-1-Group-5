import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preproc
import scripts

DATADIR = '~/Classes/CSCI635/CSCI-635-1-Group-5/data/'

"""
Training counts before resampling:
Cover_Type
0    169472
1    226640
2     28603
3      2198
4      7594
5     13894
6     16408
"""

# Global random seed
rand_seed = 42

def load_data(dataPath, colNames, targetCol, nSamples=None):
    if colNames is None:
        data = pd.read_csv(dataPath)
    else:
        data = pd.read_csv(dataPath, header=None)
        data.columns = colNames
    
    # Optional stratified sampling
    if nSamples:
        data = scripts.sample_stratify(data, targetCol, n_samples=nSamples, rand_state=rand_seed)
    X = data.drop(columns=[targetCol])
    y = data[targetCol]
    return X, y


def main(path=DATADIR, dst_name='tree'):
    import os

    # Load the data
    tree_cols = ['Elevation', 'Aspect', 'Slope', 'Horizontal_To_Hydrology',
                'Vertical_To_Hydrology', 'Horizontal_To_Roadways',
                'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                'Horizontal_To_Fire'] + \
                [f'Wilderness_Area_{i}' for i in range(4)] + \
                [f'Soil_Type_{i}' for i in range(40)] + \
                ['Cover_Type']
    data_path = os.path.join(path, 'covtype.data')
    X_tree, y_tree = load_data(data_path, tree_cols, 'Cover_Type')

    y_tree -= 1  # make labels zero-indexed


    # Rescale features
    std_scaler, x_normalize = scripts.normalize_data(X_tree, method='standard')

    # split out testing data
    x_train, x_test, y_train, y_test = train_test_split(x_normalize, y_tree, test_size=0.1, stratify=y_tree, random_state=rand_seed)
    print("Training counts before resampling:")
    print(y_train.value_counts())

    # Resample training data to address class imbalance
    target_over = {}
    # target_over = {3: 6000}
    # target_under = {0: 35000, 1: 35000}
    target_under = {1: 35000}

    x_train_resampled, y_train_resampled = scripts.resample_data(x_train, y_train,
                                                                target_over=target_over,
                                                                target_under=target_under,
                                                                rand_state=rand_seed)

    print("Training counts after resampling:")
    print(y_train_resampled.value_counts())

    x_train, x_val, y_train, y_val = train_test_split(x_train_resampled, y_train_resampled, test_size=0.1,
                                                    stratify=y_train_resampled, random_state=rand_seed)

    print(f"Training set size: {x_train.shape[0]}")
    print(f"Validation set size: {x_val.shape[0]}")
    print(f"Test set size: {x_test.shape[0]}")

    # save processed data to csv files
    # create processed directory if it doesn't exist
    os.makedirs(f'{path}/processed/', exist_ok=True)
    train_data = pd.DataFrame(x_train, columns=tree_cols[:-1])
    train_data['Cover_Type'] = y_train.values
    train_data.to_csv(f'{path}/processed/{dst_name}_train.csv', index=False)

    val_data = pd.DataFrame(x_val, columns=tree_cols[:-1])
    val_data['Cover_Type'] = y_val.values
    val_data.to_csv(f'{path}/processed/{dst_name}_val.csv', index=False)
    test_data = pd.DataFrame(x_test, columns=tree_cols[:-1])
    test_data['Cover_Type'] = y_test.values
    test_data.to_csv(f'{path}/processed/{dst_name}_test.csv', index=False)

    print(f"Preprocessed data saved to {path}/processed/")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",
                        nargs="+",
                        default=DATADIR,
                        help="Path to data directory")
    parser.add_argument("--dst_name",
                        default='tree',
                        help="Destination prefix for processed files")
    args = parser.parse_args()
    main(path=str(args.data_path[0]), dst_name=args.dst_name)