import numpy as np


def accuracy(y_predicted, y_outcomes) -> float:
    hits = 0
    for y_pred, y_o in zip(y_predicted, y_outcomes):
        hits += 1 if y_pred == y_o else 0
    accuracy = hits / len(y_predicted)
    return accuracy


def precision(y_predicted, y_outcomes) -> float:
    TP = np.sum(np.logical_and(y_predicted == 1, y_outcomes == 1))
    FP = np.sum(np.logical_and(y_predicted == 1, y_outcomes == 0))
    return TP / (TP + FP)


def recall(y_predicted, y_outcomes) -> float:
    TP = np.sum(np.logical_and(y_predicted == 1, y_outcomes == 1))
    FN = np.sum(np.logical_and(y_predicted == 0, y_outcomes == 1))
    return TP / (TP + FN)


def f_measure(y_predicted, y_outcomes, beta=1) -> float:
    prec = precision(y_predicted, y_outcomes)
    rec = recall(y_predicted, y_outcomes)
    beta_squared = beta ** 2
    return (1 + beta_squared) * (prec * rec) / ((beta_squared * prec) + rec)
