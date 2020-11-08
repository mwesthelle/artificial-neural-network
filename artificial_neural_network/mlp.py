import re
from pathlib import Path
from typing import Iterable, List

import numpy as np

from base_model import BaseModel
from sigmoid import sigmoid


class UninitializedNetworkError(Exception):
    pass


class IncompatibleNetworkStructureError(Exception):
    pass


class MLP(BaseModel):
    """
    This is a basic implementation of a MultiLayer Perceptron (MLP).

    Attributes
    ----------
    weights : dict[int, numpy_array]
        a dictionary where each key represents a layer and its value contains its
        respective weights

    layers : list[int]
        number of layers, represented by integers that tell how many
        neurons the layer has. The first layer is always the input layer, whilst
        the last layer is the output layer; all layers in-between are hidden layers

    lambda_ : float
        regularization factor; must be a non-negative float
    """

    def __init__(self, layers: List[int] = [], lambda_: float = 0):
        """
        Parameters
        ----------
        layers : list[int]
            number of layers, represented by integers that tell how many
            neurons the layer has. The first layer is always the input layer, whilst
            the last layer is the output layer; all layers in-between are hidden layers

        lambda_ : float
            regularization factor; must be a non-negative float

        """
        self._layers = layers
        self._lambda = lambda_
        self._weights = dict()
        layer2size = {idx: size for idx, size in enumerate(self.layers)}
        for idx in range(len(self.layers) - 1):
            rows = layer2size[idx + 1]
            # Add bias column
            cols = layer2size[idx] + 1
            self._weights[idx] = np.random.normal(size=(rows, cols))

    @property
    def layers(self):
        return self._layers

    @layers.setter
    def layers(self, values):
        for val in values:
            if val >= 0:
                self._layers = values
            else:
                raise AttributeError("layers must have a positive number of neurons")

    @property
    def lambda_(self):
        return self._lambda

    @lambda_.setter
    def lambda_(self, value):
        if value >= 0:
            self._lambda = value
        else:
            raise AttributeError("regularization must be a non-negative value")

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        self._weights = value

    def load_weights(self, weights_filename: str):
        weights_file = Path(weights_filename).resolve(strict=True)
        layer2size = {idx: size for idx, size in enumerate(self.layers)}
        if len(layer2size) == 0:
            raise UninitializedNetworkError(
                "cannot load weights for an uninitialized "
                "network; define the network's layers "
                "before loading weights"
            )
        self.weights = dict()
        with weights_file.open() as f:
            for idx, line in enumerate(f):
                rows = layer2size[idx + 1]
                # Add bias column
                cols = layer2size[idx] + 1
                layer_weights_text = re.split("[,;]", line.strip())
                layer_weights = np.array([float(val) for val in layer_weights_text])
                try:
                    layer_weights = layer_weights.reshape((rows, cols))
                except ValueError as err:
                    raise IncompatibleNetworkStructureError(
                        f"weights are incompatible with network structure; {err}"
                    )
                else:
                    self.weights[idx] = layer_weights

    def load_network_definition(self, network_def_filename: str):
        network_def_file = Path(network_def_filename).resolve(strict=True)
        with network_def_file.open() as f:
            reg_factor = f.readline().strip()
            self.regularization = float(reg_factor)
            layers = []
            for line in f:
                layer = int(line.strip())
                layers.append(layer)
            self.layers = layers

    def forward_pass(self, X):
        a = dict()
        z = dict()
        a[0] = np.ones((1, X.shape[0] + 1))
        a[0][:, 1:] = X
        for k in range(1, len(self.layers) - 1):
            z[k] = a[k - 1] @ self.weights[k - 1].T
            a[k] = np.ones((z[k].shape[0], z[k].shape[1] + 1))
            a[k][:, 1:] = sigmoid(z[k])
        last_layer = len(self.layers) - 1
        z[last_layer] = a[last_layer - 1] @ self.weights[last_layer - 1].T
        return sigmoid(z[last_layer])

    def backpropagation(self):
        pass

    def fit(self, data_iter: Iterable[List[str]], classes: List[str]):
        pass

    def predict(self, test_data: Iterable[List[str]]):
        pass

    def cost_function(self, X, y, num_labels: int):
        m = len(X)
        J = 0
        for x_, y_ in zip(X, y):
            h = self.forward_pass(x_)
            J += np.sum(np.nan_to_num(-y_ * np.log(h) - (1 - y) * np.log(1 - h)))
        J += (
            self.lambda_
            / (2 * m)
            * np.sum(np.linalg.norm(theta) ** 2 for theta in self.weights.values())
        )
        return J
