from artificial_neural_network.mlp import MLP

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


def main():
    dataset = pd.read_csv('datasets/toy_normalized.csv', sep=';')
    model = MLP(dataset['target'].unique(), layers=[10, 2, 2], lambda_=0.5)
    labels = np.array(dataset['target'])
    features = np.array(dataset.drop('target', axis=1))
    model.fit(features, labels)
    print(model.learning_curve)
    for feature in features:
        print(model.predict(feature))
    ts = pd.Series(model.learning_curve)
    ts.plot()
    plt.show()


if __name__ == '__main__':
    main()
