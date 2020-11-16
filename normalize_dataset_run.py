from artificial_neural_network.normalization import normalize_dataset

import pandas as pd


def main(file_path, sep):
    dataset = pd.read_csv(file_path, sep=sep)
    dataset = normalize_dataset(dataset)
    dataset.to_csv(file_path[:-4] + '_normalized.csv', sep=sep)


if __name__ == '__main__':
    main('datasets/toy.csv', ';')
