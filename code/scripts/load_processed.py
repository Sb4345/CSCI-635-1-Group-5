import pandas as pd
import os

def load_processed_data(path, prefix='tree'):
    """
    Load processed training, validation, and test data from CSV files.

    Args:
        path (str): Path to the directory containing processed data.
        prefix (str): Prefix for the processed data files.
    Returns:
        Tuple of pd.DataFrame: (train_data, val_data, test_data)
    """
    processed_path = os.path.join(path, 'processed')
    train_data = pd.read_csv(os.path.join(processed_path, f'{prefix}_train.csv'))
    val_data = pd.read_csv(os.path.join(processed_path, f'{prefix}_val.csv'))
    test_data = pd.read_csv(os.path.join(processed_path, f'{prefix}_test.csv'))
    return train_data, val_data, test_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",
                        nargs="+",
                        default='data/',
                        help="Path to data directory")
    parser.add_argument("--prefix",
                        default='tree',
                        help="Prefix for processed files")
    args = parser.parse_args()

    train_data, val_data, test_data = load_processed_data(path=str(args.data_path[0]), prefix=args.prefix)
    print("Training Data:")
    print(train_data.head())
    print("\nValidation Data:")
    print(val_data.head())
    print("\nTest Data:")
    print(test_data.head())