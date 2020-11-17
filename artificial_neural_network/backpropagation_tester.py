import textwrap
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from typing import cast, Tuple

import numpy as np

from mlp import MLP


def shorten(val):
    return str(round(val, 5))


def generate_instance(dataset_file):
    """
    ad-hoc function to load data from  a file
    """
    with open(dataset_file) as f:
        for line in f:
            X, y = line.split(";")
            X = np.array([float(val.strip()) for val in X.split(",")])
            y = np.array([float(val.strip()) for val in y.split(",")])
            yield X, y


if __name__ == "__main__":
    parser = ArgumentParser(
        formatter_class=RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
            This script tests the backpropagation implementation, given a neural net
            specification file, an initial weights file, and a dataset. All arguments
            are positional and should be given in this order. Run it as below:
                python3 backpropagation_tester.py <network> <weights> <dataset>
            """
        ),
    )
    parser.add_argument("network_file", help="Neural network specification file")
    parser.add_argument("weights_file", help="File containing initial weights")
    parser.add_argument("dataset_file", help="Dataset to run backpropagation on")
    parser.add_argument("--check-gradients", action="store_true")
    args = parser.parse_args()

    model = MLP(net_file=args.network_file, weight_file=args.weights_file)
    data = list(generate_instance(args.dataset_file))
    X, y = list(zip(*data))
    X, y = map(np.array, (X, y))
    m = len(X)
    if args.check_gradients:
        estimated_gradients = model.get_estimated_gradients(X, y)
        model.backpropagation(X, y)
        backprop_gradients = model.gradients
        print("Estimated gradients:")
        print(sorted(estimated_gradients.items()))
        print("Backpropagation gradients:")
        print(sorted(cast(Tuple, backprop_gradients.items())))
        for grad in sorted(backprop_gradients.keys()):
            is_close = np.isclose(
                backprop_gradients[grad],
                estimated_gradients[grad],
                atol=1e-5,
                rtol=1e-5,
            )
            for booleans in is_close:
                if not all(booleans):
                    print("Gradients don't match ):")
        else:
            print("Gradients match!")
    else:
        model.backpropagation(X, y)
        for layer in sorted(model.gradients.keys()):
            layer_grads = [
                list(neuron_grads) for neuron_grads in model.gradients[layer]
            ]
            layer_grads = [
                ", ".join(map(shorten, neuron_grads)) for neuron_grads in layer_grads
            ]
            print("; ".join(layer_grads))
