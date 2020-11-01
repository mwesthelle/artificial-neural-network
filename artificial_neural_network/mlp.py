from pathlib import Path


class MLP:
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
