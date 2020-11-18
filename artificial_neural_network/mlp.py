import re

from pathlib import Path
from typing import Iterable, List

import numpy as np

from base_model import BaseModel
from sigmoid import sigmoid, sigmoid_prime

from one_hot_encoder import OneHotEncoder


class UninitializedNetworkError(Exception):
    pass


class IncompatibleNetworkStructureError(Exception):
    pass


def shorten(val):
    return str(round(val, 5))


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
        classes_names: List[str] = [],
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
        self.weight_file = weight_file
        self.gradients = dict()
        self.learning_rate = 1e-2
        self._lambda = lambda_
        self.learning_curve = []
        self.one_hot_encoder = OneHotEncoder()
        self.one_hot_encoder.encode(classes_names)
        if net_file:
            self.load_network_definition(net_file)
        else:
            self._layers = layers
            self._lambda = lambda_
        self.initialize_weights()
        # used for momentum
        self.prev_grad_delta = dict()
        for theta in self.weights:
            self.prev_grad_delta[theta] = np.zeros((self.weights[theta].shape))

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

    def initialize_weights(self):
        layer2size = {idx: size for idx, size in enumerate(self.layers)}
        if self.weight_file:
            self.load_weights(self.weight_file)
        else:
            np.random.seed(171)
            self.weights = dict()
            for idx in range(len(self.layers) - 1):
                rows = layer2size[idx + 1]
                # Add bias column
                cols = layer2size[idx] + 1
                self.weights[idx + 1] = np.random.normal(size=(rows, cols))

    def one_hot_encode_y(self, Y):
        encoded_y = [self.one_hot_encoder.label_to_decode(y) for y in Y]
        return np.array(encoded_y)

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
            self._lambda = float(reg_factor)
            layers = []
            for line in f:
                layer = int(line.strip())
                layers.append(layer)
            self.layers = layers

    def forward_pass(self, X):
        """
        Given an input, return an output prediction and the layers' activations for use
        in backpropagation
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
        beta = 0.85
        delta = dict()
        for i in self.gradients:
            delta[i] = -self.learning_rate * self.gradients[i]
        for i in self.weights:
            self.weights[i] += delta[i] + (beta * self.prev_grad_delta[i])
        for i in self.prev_grad_delta:
            self.prev_grad_delta[i] = delta[i]

    def regularize_gradients(self, m):
        P = dict()
        for i in self.gradients:
            P[i] = self.lambda_ * self.weights[i]
            P[i][:, 0] = 0
            self.gradients[i] = (1 / m) * (self.gradients[i] + P[i])

    def backpropagation(self, data):
        m = len(data)
        mini_batch_size = 16

        mini_batches = (
            data[i : i + mini_batch_size] for i in range(0, m, mini_batch_size)
        )
        for mini_batch in mini_batches:
            for x_, y_ in zip(mini_batch[0], mini_batch[1]):
                h, a, z = self.forward_pass(x_)
                deltas = self.calculate_deltas(h, y_, z)
                self.update_gradients(deltas, a)
            self.regularize_gradients(m)
            self.update_weights()

    def fit(self, data_iter: List[str], labels: List[str]):
        self.initialize_weights()
        encoded_labels = self.one_hot_encode_y(labels)
        epochs = 15
        epsilon = 1e-1

        max_consecutive_epochs_without_improving = 10
        consecutive_epochs_without_improving = 0
        loss = 0
        for epoch in range(epochs):
            if epoch % 100 == 0 and epoch > 0:
                print(f"Epoch {epoch}   training loss: {loss}")
            self.backpropagation((data_iter, encoded_labels))
            loss = self.calculate_loss(data_iter, encoded_labels)
            self.learning_curve.append(loss)
            previous_loss = None
            try:
                previous_loss = self.learning_curve[-1]
            except IndexError:
                previous_loss = 0
            finally:
                if previous_loss > loss + epsilon or previous_loss < loss - epsilon:
                    consecutive_epochs_without_improving += 1
                else:
                    consecutive_epochs_without_improving = 0
            if (
                consecutive_epochs_without_improving
                >= max_consecutive_epochs_without_improving
            ):
                self.save_model()
                return
        self.save_model()

    def save_model(self):
        with open("saved_model.txt", "w") as file:
            for layer in sorted(self.gradients.keys()):
                layer_grads = [
                    list(neuron_grads) for neuron_grads in self.gradients[layer]
                ]
                layer_grads = [
                    ", ".join(map(shorten, neuron_grads))
                    for neuron_grads in layer_grads
                ]
                file.write("; ".join(layer_grads) + "\n")

    def get_predicted_class_by_probabilities(self, classes_probs):
        list_of_zeros = np.zeros(len(classes_probs))
        max_index = np.argmax(classes_probs)
        list_of_zeros[max_index] = 1
        return self.one_hot_encoder.decode(list_of_zeros)

    def predict(self, test_data: Iterable[List[str]]):
        classes_probabilities, _, _ = self.forward_pass(test_data)
        return self.get_predicted_class_by_probabilities(classes_probabilities)

    def calculate_loss(self, X, y):
        m = len(X)
        J = 0
        for x_, y_ in zip(X, y):
            h, _, _ = self.forward_pass(x_)
            J += self.cost_function(y_, h)
        J /= m
        J += self.lambda_ / (2 * m) * self.loss_regularization()
        return J

    def loss_regularization(self):
        return np.sum(
            np.linalg.norm(theta[:, 1:]) ** 2 for theta in self.weights.values()
        )

    def get_estimated_gradients(self, X, y):
        EPSILON = 1e-5
        estimated_gradients = dict()
        for theta in self.weights:
            estimated_gradients[theta] = np.zeros(self.weights[theta].shape)
        for theta in self.weights:
            for neuron_idx, _ in enumerate(self.weights[theta]):
                for idx, _ in enumerate(self.weights[theta][neuron_idx]):
                    self.weights[theta][neuron_idx][idx] += EPSILON
                    J_plus_eps = self.calculate_loss(X, y)
                    self.weights[theta][neuron_idx][idx] -= 2 * EPSILON
                    J_minus_eps = self.calculate_loss(X, y)
                    estimated_gradients[theta][neuron_idx][idx] = (
                        J_plus_eps - J_minus_eps
                    ) / (2 * EPSILON)
        return estimated_gradients

    @staticmethod
    def cost_function(y, y_pred):
        return np.sum(np.nan_to_num(-y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)))
