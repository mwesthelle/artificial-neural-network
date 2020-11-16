import pandas as pd
import pytest

from artificial_neural_network.normalization import DatasetNormalizer


class TestDatasetNormalizer:
    @pytest.fixture()
    def simple_dataset_normalizer(self):
        return DatasetNormalizer(
            pd.DataFrame.from_dict({
                'num_categ': [3, 2, 1, 0],
                'categ': ['a', 'b', 'c', 'd'],
                'numerical': [1.0, 2.0, 2.0, 3.0],
                'target': [1, 0, 1, 0]
            })
        )

    def test_is_categorical_data(self, simple_dataset_normalizer):
        assert not simple_dataset_normalizer.is_categorical_data('num_categ')
        assert simple_dataset_normalizer.is_categorical_data('categ')

    def test_check_for_numerical_categorical_cols(self, simple_dataset_normalizer):
        simple_dataset_normalizer.check_for_numerical_categorical_cols()
        assert simple_dataset_normalizer.is_categorical_data('num_categ')

    def test_separate_column(self, simple_dataset_normalizer):
        simple_dataset_normalizer.create_one_hot_representation_for_column('categ')
        dataset = simple_dataset_normalizer.get_dataset()
        assert 'categ_a' in dataset.columns
        assert 'categ_b' in dataset.columns
        assert 'categ_c' in dataset.columns
        assert 'categ_d' in dataset.columns
        assert dataset['categ_a'].equals(pd.Series([1, 0, 0, 0]))
        assert dataset['categ_b'].equals(pd.Series([0, 1, 0, 0]))

    def test_normalize(self, simple_dataset_normalizer):
        dataset = simple_dataset_normalizer.normalize()
        assert 'categ_a' in dataset.columns
        assert 'num_categ_0' in dataset.columns
        assert 'categ' not in dataset.columns
        assert 'target' in dataset.columns

    def test_normalize_numerical_cols(self, simple_dataset_normalizer):
        simple_dataset_normalizer.normalize_numerical_cols()
        dataset = simple_dataset_normalizer.get_dataset()
        assert dataset['numerical'].equals(pd.Series([0.0, 0.5, 0.5, 1.0]))
