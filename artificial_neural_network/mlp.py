from pathlib import Path
from typing import Iterable, List

from base_model import BaseModel


class MLP(BaseModel):
    def __init__(
        self,
        layers: list[int] = [],
        weights: list[list[float]] = [],
        regularization: float = 0,
    ):
        self.layers = layers
        self.weights = weights
        self.regularization = regularization

    @property
    def layers(self):
        return self._layers

    @layers.setter
    def layers(self, value):
        self._layers = value

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        self._weights = value

    def load_weights(self, weights_file: Path):
        weights_file = Path(weights_file)

    def forward_pass(self):
        pass

    def backpropagation(self):
        pass

    def fit(self, data_iter: Iterable[List[str]], attribute_names: List[str]):
        pass

    def predict(self, test_data: Iterable[List[str]]):
        pass
