import pandas as pd
import numpy as np


def normalize_dataset(dataset: pd.DataFrame):
    return DatasetNormalizer(dataset).normalize()


class DatasetNormalizer:
    MAX_COLS_FOR_CATEGORICAL_DATA = 10

    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset

    def check_for_numerical_categorical_cols(self):
        for column in self.dataset.columns:
            if self.has_few_possible_values(column) \
                    and self.is_numerical_data(column):
                self.dataset[column] = self.dataset[column].apply(str)
                self.dataset[column] = self.dataset[column].astype(str)

    def has_few_possible_values(self, column):
        return len(self.dataset[column].unique()) < self.MAX_COLS_FOR_CATEGORICAL_DATA

    def is_categorical_data(self, column):
        return not self.is_numerical_data(column)

    def normalize(self):
        self.check_for_numerical_categorical_cols()
        self.normalize_categorical_cols()
        self.normalize_numerical_cols()
        return self.dataset

    def normalize_categorical_cols(self):
        for column in self.dataset.columns:
            if self.is_categorical_data(column):
                self.crate_one_hot_representation_for_column(column)

    def is_numerical_data(self, column):
        return self.dataset[column].dtype == np.float64 \
            or self.dataset[column].dtype == np.int64

    def crate_one_hot_representation_for_column(self, column):
        for distinct_value in self.dataset[column].unique():
            self.dataset[f'{column}_{distinct_value}'] = \
                self.dataset[column].apply(lambda x: 0 if x != distinct_value else 1)

    def get_dataset(self):
        return self.dataset

    def normalize_numerical_cols(self):
        for column in self.dataset.columns:
            if self.is_numerical_data(column):
                self.normalize_numerical_column(column)

    def normalize_numerical_column(self, column):
        max_ = max(self.dataset[column])
        min_ = min(self.dataset[column])
        self.dataset[column] = self.dataset[column]\
            .apply(lambda x: (x - min_) / (max_ - min_))

