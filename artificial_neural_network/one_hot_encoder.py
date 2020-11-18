import numpy as np


class OneHotEncoder:
    def __init__(self):
        self.int2label = dict()
        self.label2int = dict()
        self.one_hot_encoding = dict()

    def encode(self, labels):
        labels = [str(lab) for lab in labels]
        self.int2label = {idx: label for idx, label in enumerate(labels)}
        label2int = {label: idx for idx, label in enumerate(labels)}
        for label in labels:
            self.one_hot_encoding[label] = np.zeros(len(labels))
            self.one_hot_encoding[label][label2int[label]] = 1

    def label_to_decode(self, label):
        return self.one_hot_encoding[label]

    def decode(self, one_hot_array):
        return self.int2label[one_hot_array[1]]
