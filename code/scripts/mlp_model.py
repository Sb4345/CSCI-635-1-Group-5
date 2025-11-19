"""MLP model wrapper for training and prediction.

Provides a class-based implementation of the perceptron (MLP) model that
was used in `mlp_model.ipynb`. The class accepts training, validation and
test datasets (as numpy arrays or similar), exposes setters to change them,
methods to build and train the model, evaluate on a dataset, and predict a
single label for an input feature vector.

Usage (example):
	m = MLPModel(input_dim=54, num_classes=7)
	m.set_train(x_train, y_train)
	m.set_val(x_val, y_val)
	m.build_model()
	history = m.train(epochs=50)
	label = m.predict_single(x_sample)

This module intentionally keeps the API small and compatible with the
notebook flow that performs scaling/reshaping outside this class.
"""

from typing import Optional, Tuple, Union

import numpy as np
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.callbacks import EarlyStopping


class MLPModel:
    """A simple MLP wrapper for classification.

    Constructor arguments set the network shape and optimizer hyperparameters.
    Datasets (train/val/test) may be provided at construction or later via
    setter methods.

    Inputs expected:
      - x arrays: 2D numeric arrays (n_samples, n_features)
      - y arrays: 1D integer labels (n_samples,) zero-indexed

    Methods:
      - set_train(x, y), set_val(x, y), set_test(x, y)
      - build_model(): constructs the Keras model
      - train(...): fits the model and returns the History
      - predict_single(x): returns predicted label for one sample
      - evaluate(x, y): returns (loss, accuracy)
    """

    def __init__(
        self,
        input_dim: int = 54,
        num_classes: int = 7,
        learning_rate: float = 1e-4,
    ) -> None:
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # datasets
        self.x_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.x_val: Optional[np.ndarray] = None
        self.y_val: Optional[np.ndarray] = None
        self.x_test: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

        self.model: Optional[keras.Model] = None
        self.history: Optional[keras.callbacks.History] = None

    def _check_features(self, x: np.ndarray) -> None:
        """Internal method to validate feature array shape."""
        x_arr = np.asarray(x)
        if x_arr.ndim != 2:
            raise ValueError(
                f"Features must be a 2D array (n_samples, n_features); got ndim={x_arr.ndim}"
            )
        if x_arr.shape[1] != self.input_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.input_dim} features, got {x_arr.shape[1]}"
            )
    
    def _check_labels(self, y: np.ndarray) -> None:
        """Internal method to validate label array shape."""
        y_arr = np.asarray(y)
        if y_arr.ndim != 1:
            raise ValueError(
                f"Labels must be a 1D array of length n_samples; got ndim={y_arr.ndim}"
            )
    
    def _check_labels_match(self, x_arr: np.ndarray, y_arr: np.ndarray) -> None:
        """Internal method to validate that features and labels have matching sample counts."""
        if x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"Number of samples in x and y must match; got {x_arr.shape[0]} and {y_arr.shape[0]}"
            )

    def _check_data_input(self, x: np.ndarray, y: np.ndarray) -> None:
        """Internal method to validate feature and label arrays."""
        self._check_features(x)
        self._check_labels(y)
        self._check_labels_match(x, y)

    # dataset setters
    def set_train(self, x: np.ndarray, y: np.ndarray) -> None:
        """Set training data and perform basic validation."""
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)

        self._check_data_input(x_arr, y_arr)

        self.x_train = x_arr
        self.y_train = y_arr

    def set_val(self, x: np.ndarray, y: np.ndarray) -> None:
        """Set validation data (basic validation performed)."""
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)

        self._check_data_input(x_arr, y_arr)

        self.x_val = x_arr
        self.y_val = y_arr

    def set_test(self, x: np.ndarray, y: np.ndarray) -> None:
        """Set test data (basic validation performed)."""
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)

        self._check_data_input(x_arr, y_arr)

        self.x_test = x_arr
        self.y_test = y_arr

    def build_model(self) -> keras.Model:
        """Constructs the Keras Sequential model and compiles it.

        The architecture mirrors the notebook: Input -> 256 relu -> 256 relu -> softmax
        """
        model = Sequential([
            InputLayer(input_shape=(self.input_dim,)),
            Dense(256, activation="relu"),
            Dense(256, activation="relu"),
            Dense(self.num_classes, activation="softmax"),
        ])

        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        self.model = model
        return model

    def overview(self) -> None:
        """Print a summary of the model architecture."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        self.model.summary()

    def train(
        self,
        epochs: int = 400,
        batch_size: int = 1024,
        patience: int = 5,
        verbose: int = 1,
        callbacks: Optional[list] = None,
    ) -> Optional[keras.callbacks.History]:
        """Train the model using stored training and validation sets.

        Returns Keras History.
        """
        if self.model is None:
            self.build_model()

        if self.x_train is None or self.y_train is None:
            raise ValueError("Training data not set. Call set_train(x, y) first.")

        if callbacks is None:
            callbacks = [
                EarlyStopping(monitor="val_loss", mode="min", patience=patience, verbose=1)
            ]

        val_data = None
        if self.x_val is not None and self.y_val is not None:
            val_data = (self.x_val, self.y_val)

        history = self.model.fit(
            self.x_train,
            self.y_train,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
        )
        self.history = history
        return history

    def predict(self, x: Union[np.ndarray, list, tuple]) -> Union[int, np.ndarray]:
        """Predict one or more samples and return class labels.

        Input may be:
          - a single sample: 1D array-like with shape (n_features,)
          - multiple samples: 2D array-like with shape (n_samples, n_features)

        Returns:
          - int: predicted label if a single sample provided
          - numpy.ndarray of ints: predicted labels for multiple samples
        """
        if self.model is None:
            raise ValueError("Model not built or loaded. Call build_model() before predict.")

        arr = np.asarray(x)
        # If user passed a single 1D sample, reshape to (1, n_features)
        single_input = False
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
            single_input = True

        preds = self.model.predict(arr)
        labels = np.argmax(preds, axis=1).astype(int)

        if single_input:
            return int(labels[0])
        return labels

    def predict_single(self, x: Union[np.ndarray, list, tuple]) -> int:
        """Backward-compatible wrapper around `predict` that returns a single int.

        This preserves the original API while encouraging use of `predict` which
        supports batch inputs.
        """
        result = self.predict(x)
        # predict returns int for single input; ensure we return int
        if isinstance(result, np.ndarray):
            return int(result[0])
        return int(result)

    def evaluate(self,
                 x: Optional[np.ndarray] = None,
                 y: Optional[np.ndarray] = None,
                 metrics: Optional[list] = None) -> Tuple[float, float]:
        """Evaluate model on provided (x,y) or stored test set.

        Returns (loss, accuracy) or a tuple of metric results.
        """
        if self.model is None:
            raise ValueError("Model not built or loaded. Call build_model() before evaluate.")

        if x is None or y is None:
            x = self.x_test
            y = self.y_test

        if x is None or y is None:
            raise ValueError(
                "No evaluation data available. Provide (x,y) or set test data with set_test()."
            )

        if metrics is None:
            loss, acc = self.model.evaluate(x, y, verbose=0)
            return float(loss), float(acc)
        else:
            y_pred = self.predict(x)
            results = []
            for metric in metrics:
                results.append(metric(y, y_pred))
            return tuple(results)


    def visualize_history(self, history: Optional[keras.callbacks.History] = None) -> None:
        """Visualize training history (loss and accuracy) using subplots."""
        import matplotlib.pyplot as plt

        if history is None:
            history = self.history

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot training & validation accuracy values
        ax1.plot(history.history["accuracy"], label="Train acc", color="blue")
        ax1.plot(history.history["val_accuracy"], label="Val acc", color="orange")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend(loc="upper left")
        ax1.set_title("Model Accuracy")

        # Plot training & validation loss values
        ax2.plot(history.history["loss"], label="Train loss", color="blue")
        ax2.plot(history.history["val_loss"], label="Val loss", color="orange")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend(loc="upper left")
        ax2.set_title("Model Loss")

        plt.tight_layout()
        plt.show()


def main():
    from sklearn.datasets import load_iris
    import pandas as pd
    from sample import sample_stratify
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import matthews_corrcoef, classification_report, confusion_matrix

    iris = load_iris()
    iris_x = iris.data
    iris_y = iris.target

    scaler = StandardScaler()
    iris_x_scaled = scaler.fit_transform(iris_x)

    x_train, x_test, y_train, y_test = train_test_split(iris_x_scaled, iris_y, test_size=0.2, random_state=1)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1)

    mlp = MLPModel(input_dim=iris_x.shape[1], num_classes=len(set(iris_y)), learning_rate=1e-3)
    mlp.set_train(x_train, y_train)
    mlp.set_val(x_val, y_val)
    mlp.set_test(x_test, y_test)

    mlp.build_model()
    history = mlp.train(epochs=100, verbose=0)
    mlp.visualize_history(history)

    loss, acc = mlp.evaluate()
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")
    mcc, conf, class_report = mlp.evaluate(
        x_test,
        y_test,
        metrics=[matthews_corrcoef, confusion_matrix, classification_report]
    )
    print(f"Matthews Correlation Coefficient: {mcc:.4f}")
    print("Confusion Matrix:\n", conf)
    print("Classification Report:\n", class_report)


if __name__ == "__main__":
    # test using iris dataset
    main()
