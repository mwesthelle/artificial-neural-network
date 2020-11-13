import re
from pathlib import Path
from typing import Iterable, List

import numpy as np

from base_model import BaseModel
from sigmoid import sigmoid, sigmoid_prime


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

    def __init__(
        self,
        layers: List[int] = [],
        lambda_: float = 0,
        weight_file: str = None,
        net_file: str = None,
    ):
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
        self.gradients = None
        self.learning_rate = 0.001
        self.learning_curve = []
        if net_file:
            self.load_network_definition(net_file)
        else:
            self._layers = layers
            self._lambda = lambda_
        layer2size = {idx: size for idx, size in enumerate(self.layers)}
        if weight_file:
            self.load_weights(weight_file)
        else:
            self.weights = dict()
            for idx in range(len(self.layers) - 1):
                rows = layer2size[idx + 1]
                # Add bias column
                cols = layer2size[idx] + 1
                self.weights[idx] = np.random.normal(size=(rows, cols))

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
            raise AttributeError("regularization factor must be a non-negative value")

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
                    self.weights[idx + 1] = layer_weights

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
        """
        Gets X as a 1-D np.array of inputs and returns an output prediction and the
        layers' activations for use in backpropagation
        """
        a = dict()
        z = dict()
        a[1] = np.ones(X.shape[0] + 1)
        a[1][1:] = X
        for k in range(2, len(self.layers)):
            z[k] = a[k - 1] @ self.weights[k - 1].T
            a[k] = np.ones(z[k].shape[0] + 1)
            a[k][1:] = sigmoid(z[k])
        last_layer = len(self.layers)
        z[last_layer] = a[last_layer - 1] @ self.weights[last_layer - 1].T
        a[last_layer] = sigmoid(z[last_layer])
        return a[last_layer], a, z

    def calculate_deltas(self, y_pred, y, z):
        deltas = dict()
        deltas[len(self.layers)] = np.array([y_pred - y]).T
        for i in range(len(self.layers) - 1, 1, -1):
            sig_prime = sigmoid_prime(np.array([z[i]]))
            deltas[i] = self.weights[i][:, 1:].T @ deltas[i + 1] * sig_prime.T
        return deltas

    def update_gradients(self, deltas, activations):
        if not self.gradients:
            self.gradients = dict()
            for i in range(len(self.layers) - 1, 0, -1):
                self.gradients[i] = np.zeros((self.weights[i].shape))
        for i in range(1, len(self.layers)):
            self.gradients[i] = (
                self.gradients[i] + deltas[i + 1] @ activations[i][None, :]
            )

    def update_weights(self):
        for i in self.weights:
            self.weights[i] -= self.learning_rate * self.gradients[i]

    def backpropagation(self, X, y):
        m = len(X)
        for x_, y_ in zip(X, y):
            h, a, z = self.forward_pass(x_)
            deltas = self.calculate_deltas(h, y_, z)
            self.update_gradients(deltas, a)
        P = dict()
        for i in self.gradients:
            P[i] = self.lambda_ * self.weights[i]
            P[i][:, 0] = 0
            self.gradients[i] = (1 / m) * (self.gradients[i] + P[i])
        self.update_weights()

    def fit_epoch(self, data_iter: List[str], classes: List[str]):
        self.backpropagation(data_iter, classes)
        current_cost = self.evaluate_costs(data_iter, classes)
        self.learning_curve.append(current_cost)
        return current_cost

    def evaluate_costs(self, current_cost, previous_cost):
        return current_cost >= previous_cost  # TODO implement better logic

    def get_learning_curve(self):
        return self.learning_curve

    def fit(self, data_iter: List[str], classes: List[str]):
        previous_cost = 0
        should_continue = True
        while should_continue:
            current_cost = self.fit_epoch(data_iter, classes)
            should_continue = self.evaluate_costs(current_cost, previous_cost)
            if should_continue:
                previous_cost = current_cost

    def predict(self, test_data: Iterable[List[str]]):
        pass

    def cost_function(self, X, y):
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
