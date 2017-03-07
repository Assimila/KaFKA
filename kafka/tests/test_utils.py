from kafka.utils import matrix_squeeze
import numpy as np


def test_squeeze_with_mask():
    test_matrix = np.array([[1, 0, 3], [0, 0, 0], [7, 0, 9]])
    expected_result = np.array([[1, 3], [7, 9]])
    mask = np.array([True, False, True], dtype=bool)
    squeeze_matrix = matrix_squeeze(test_matrix, mask=mask)
    assert np.array_equal(squeeze_matrix, expected_result)

def test_squeeze_with_mask_all_true():
    test_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    mask = np.array([True, True, True], dtype=bool)
    squeeze_matrix = matrix_squeeze(test_matrix, mask=mask)
    assert np.array_equal(squeeze_matrix, test_matrix)


def test_squeeze_just_matrix():
    test_matrix = np.array([[1, 0, 3], [0, 0, 0], [7, 0, 9]])
    expected_result = np.array([[1, 3], [7, 9]])
    squeeze_matrix = matrix_squeeze(test_matrix)
    assert np.array_equal(squeeze_matrix, expected_result)


def test_squeeze_just_matrix_no_zeros():
    test_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    squeeze_matrix = matrix_squeeze(test_matrix)
    assert np.array_equal(squeeze_matrix, test_matrix)

def __main__():
    test_squeeze_with_mask()
    test_squeeze_just_matrix()
