import scripts
import os
import pickle


def load_model(model_path: str, model_name: str):
    """Load a trained model from disk.

    Args:
        model_path (str): Path to the directory containing the model.
        model_name (str): Name of the model file.   
    Returns:
        model: The loaded model object.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model path {model_path} does not exist.')

    full_path = os.path.join(model_path, f'{model_name}.pkl')
    model = pickle.load(open(full_path, 'rb'))
    return model


def save_model(model, model_path: str, model_name: str):
    """Save a trained model to disk.

    Args:
        model: The model object to save.
        model_path (str): Path to the directory to save the model.
        model_name (str): Name of the model file.
    """
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    full_path = os.path.join(model_path, f'{model_name}.pkl')
    pickle.dump(model, open(full_path, 'wb'))