import random
from collections import defaultdict
from fractions import Fraction
from itertools import chain
from typing import Dict, List, NewType, cast

import numpy as np

from base_model import BaseModel
from metrics import accuracy

FoldType = NewType("FoldType", List[List[List[str]]])


class KFoldCrossValidation:
    """
    Evaluates a model using k-fold cross validation

    Methods
    -------
    _index_dataset(filename: str)
        Build a map from our classes to the indices they appear in, as well as a list
        of offsets for fast access

    generate_stratified_fold(k_folds: int)
        Given a `k_folds` number and our map of classes to indices, generate a fold by
        performing a weighted random sample from our indices, with the class
        proportions as weights

    k_fold_cross_validation(
        filename: str,
        k_folds: int,
        repetitions: int,
        ):
        Perform k-fold cross validation with `k_folds` and repeating it `repetitions`
        times, reading data from the `filename` dataset
    """

    def __init__(self, model: BaseModel, delimiter: str = ","):
        self.klass_idxes: Dict[str, List[int]] = defaultdict(
            list
        )  # Holds classes as keys and indices they occur on as values
        self.delimiter = delimiter
        self.model = model
        # TODO: use a better/faster data structure (probably some self-balanced BST, but
        # implementing the list interface for maximum code reuse)
        self._line_offsets: List[int] = []
        self.headers = []

    def index_dataset(self, file_handle):
        offset: int = 0
        self._line_offsets.clear()
        file_handle.seek(0)
        headers = next(file_handle)
        offset += len(headers)
        self.headers = headers.decode("utf-8").strip().split(self.delimiter)
        for idx, row in enumerate(file_handle):
            self._line_offsets.append(offset)
            offset += len(row)
            values = row.decode("utf-8").strip().split(self.delimiter)
            self.klass_idxes[values[-1]].append(idx)
        file_handle.seek(0)

    def generate_stratified_fold(self, k_folds: int) -> List[int]:
        """
        Generate a stratified fold by sampling our index map without repetition. The
        fold is represented by a list of indices.
        """
        klass_proportions = {}
        fold_size = len(self._line_offsets) // k_folds
        fold: List[int] = []
        for klass in self.klass_idxes:
            proportion = Fraction(
                numerator=len(self.klass_idxes[klass]),
                denominator=len(self._line_offsets),
            )
            klass_proportions[klass] = proportion
            random.shuffle(self.klass_idxes[klass])
        for _ in range(fold_size):
            # Choose a random class using the class proportions as weights for the
            # random draw
            chosen_klass = random.choices(
                list(klass_proportions.keys()),
                weights=list(klass_proportions.values()),
                k=1,
            )[0]
            chosen_idx = 0
            try:
                chosen_idx = self.klass_idxes[chosen_klass].pop()
            except IndexError:
                del self.klass_idxes[chosen_klass]
                del klass_proportions[chosen_klass]
                chosen_klass = random.choices(
                    list(klass_proportions.keys()),
                    weights=list(klass_proportions.values()),
                )[0]
                chosen_idx = self.klass_idxes[chosen_klass].pop()
            finally:
                fold.append(chosen_idx)
        return fold

    def create_k_folds(self, file_handle, k_folds, seed):
        random.seed(seed)
        self.index_dataset(file_handle)
        folds: FoldType = FoldType([])
        for _ in range(k_folds):
            fold_rows: List[List[str]] = []
            for idx in self.generate_stratified_fold(k_folds):
                file_handle.seek(self._line_offsets[idx])
                line = file_handle.readline().decode("utf-8").strip()
                data = line.split(self.delimiter)
                fold_rows.append(data)
            folds.append(fold_rows)

        remaining_idxs = []
        for klass in self.klass_idxes:
            if len(idxes := self.klass_idxes[klass]) > 0:
                remaining_idxs.extend(idxes)
        self.klass_idxes.clear()
        remaining_data = []
        for idx in remaining_idxs:
            file_handle.seek(self._line_offsets[idx])
            line = file_handle.readline().decode("utf-8").strip()
            data = line.split(self.delimiter)
            remaining_data.append(data)
        folds[-1].extend(remaining_data)
        file_handle.seek(0)
        return folds

    def kfold_cross_validation(
        self, filename: str, k_folds: int = 10, repetitions: int = 1,
    ):
        results = []
        for i_repetition in range(repetitions):
            with open(filename, "rb") as f:
                seed = i_repetition * 3 + 2
                folds = self.create_k_folds(f, k_folds, seed)
            fold_idxes: List[int] = list(range(len(cast(FoldType, folds))))
            random.shuffle(fold_idxes)
            all_folds_results = []
            for i in range(k_folds):
                test_fold_idx = fold_idxes.pop()
                test_outcomes = [t[-1] for t in folds[test_fold_idx]]
                train_folds = list(
                    chain(*(folds[:test_fold_idx] + folds[test_fold_idx + 1 :]))
                )
                self.model.fit(train_folds, attribute_names=self.headers[:-1])
                predictions = self.model.predict(folds[test_fold_idx])
                acc = accuracy(predictions, test_outcomes)
                print(f"Fold {i + 1} accuracy: {100 * acc:.2f}%")
                all_folds_results.append(acc)

            results.append(all_folds_results)
        print(f"Mean accuracy: {100 * np.mean(results):.2f}%")
        return np.mean(results)
