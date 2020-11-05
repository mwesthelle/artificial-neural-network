import re
from pathlib import Path
from typing import Iterable, List

import numpy as np

from base_model import BaseModel


class MLP(BaseModel):
    def __init__(self, layers: List[int] = [], regularization: float = 0):
        """
        This is a basic implementation of a MultiLayer Perceptron (MLP).

        Parameters
        ----------
        *layers : tuple of ints
            Variable number of layers, represented by integers that tell how many
            neurons the layer has. The first layer is always the input layer, whilst
            the last layer is the output layer; all layers in-between are hidden layers

        regularization : int
            Regularization factor; must be a non-negative integer

        Attributes
        ----------
        weights : dict int -> numpy array
            A dictionary where each key represents a layer and its value contains its
            respective weights
        """
        self.__layers = layers
        self.__regularization = regularization

    @property
    def layers(self):
        return self.__layers

    @layers.setter
    def layers(self, values):
        for val in values:
            if val <= 0:
                raise AttributeError("layers must have a positive number of neurons")
        self.__layers = values

    @property
    def regularization(self):
        return self.__regularization

    @regularization.setter
    def regularization(self, value):
        if value >= 0:
            self.__regularization = value
        else:
            raise AttributeError("regularization must be a non-negative value")

    def initialize_weights(self):
        if not hasattr(self, "_weights"):
            self.weights = dict()
            layer2size = {idx: size for idx, size in enumerate(self.layers, 1)}
            for idx in range(len(self.layers) - 1):
                rows = layer2size[idx + 1]
                # Add bias column
                cols = layer2size[idx] + 1
                self.weights[idx] = np.random.normal(size=(rows, cols))
        else:
            raise AttributeError("weights are already initialized")

    def load_weights(self, weights_filename: str):
        weights_file = Path(weights_filename).resolve(strict=True)
        layer2size = {idx: size for idx, size in enumerate(self.layers, 1)}
        self.weights = dict()
        with weights_file.open() as f:
            for idx, line in enumerate(f, 1):
                rows = layer2size[idx + 1]
                # Add bias column
                cols = layer2size[idx] + 1
                layer_weights_text = re.split("[,;]", line.strip())
                layer_weights = np.array([float(val) for val in layer_weights_text])
                layer_weights = layer_weights.reshape((rows, cols))
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

    def forward_pass(self):
        pass

    def backpropagation(self):
        pass

    def fit(self, data_iter: Iterable[List[str]], attribute_names: List[str]):
        pass

    def predict(self, test_data: Iterable[List[str]]):
        pass

    def cost_function(self):
        pass
