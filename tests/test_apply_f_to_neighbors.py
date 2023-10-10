import pytest
import numpy as np
import numpy as cp  # import cupy as cp
from boo import GhostArray
from tests.tools import GhostArray_is_consistent

cupy = [False]


@pytest.mark.parametrize("cupy", cupy)
def test_1d_min(cupy):
    array1d = np.asarray([3, 8, 7])
    expected_output = np.array([3, 3, 7])
    if cupy:
        array1d = cp.asarray(array1d)
        expected_output = cp.asarray(expected_output)
    ga_1d = GhostArray(
        interior=array1d, pad_width=1, mode="dirichlet", constant_values=10
    )
    out = ga_1d.apply_f_to_neighbors(f=np.min)
    assert out.dtype == "int"
    assert GhostArray_is_consistent(out)
    assert np.all(out.to_numpy() == expected_output)


@pytest.mark.parametrize("cupy", cupy)
def test_1d_max(cupy):
    array1d = np.asarray([3, 8, 7])
    expected_output = np.array([10, 8, 10])
    if cupy:
        array1d = cp.asarray(array1d)
        expected_output = cp.asarray(expected_output)
    ga_1d = GhostArray(
        interior=array1d, pad_width=1, mode="dirichlet", constant_values=10
    )
    out = ga_1d.apply_f_to_neighbors(f=np.max)
    assert out.dtype == "int"
    assert GhostArray_is_consistent(out)
    assert np.all(out.to_numpy() == expected_output)


@pytest.mark.parametrize("cupy", cupy)
def test_2d_von_neumann_min(cupy):
    array2d = np.asarray([[3, 8, 7], [1, 7, 9], [4, 4, 0]])
    expected_output = np.asarray([[1, 3, 7], [1, 1, 0], [1, 0, 0]])
    if cupy:
        array2d = cp.asarray(array2d)
        expected_output = cp.asarray(expected_output)
    ga_2d = GhostArray(
        interior=array2d, pad_width=1, mode="dirichlet", constant_values=10
    )
    out = ga_2d.apply_f_to_neighbors(f=np.min, mode="neumann")
    assert out.dtype == "int"
    assert GhostArray_is_consistent(out)
    assert np.all(out.to_numpy() == expected_output)


@pytest.mark.parametrize("cupy", cupy)
def test_2d_von_neumann_max(cupy):
    array2d = np.asarray([[3, 8, 7], [1, 7, 9], [4, 4, 0]])
    expected_output = np.asarray([[10, 10, 10], [10, 9, 10], [10, 10, 10]])
    if cupy:
        array2d = cp.asarray(array2d)
        expected_output = cp.asarray(expected_output)
    ga_2d = GhostArray(
        interior=array2d, pad_width=1, mode="dirichlet", constant_values=10
    )
    out = ga_2d.apply_f_to_neighbors(f=np.max, mode="neumann")
    assert out.dtype == "int"
    assert GhostArray_is_consistent(out)
    assert np.all(out.to_numpy() == expected_output)


def test_2d_moore_min():
    array2d = np.asarray([[3, 8, 7], [1, 7, 9], [4, 4, 0]])
    expected_output = np.asarray([[1, 1, 7], [1, 0, 0], [1, 0, 0]])
    if cupy:
        array2d = cp.asarray(array2d)
        expected_output = cp.asarray(expected_output)
    ga_2d = GhostArray(
        interior=array2d, pad_width=1, mode="dirichlet", constant_values=10
    )
    out = ga_2d.apply_f_to_neighbors(f=np.min, mode="moore")
    assert out.dtype == "int"
    assert GhostArray_is_consistent(out)
    assert np.all(out.to_numpy() == expected_output)


def test_2d_moore_max():
    array2d = np.asarray([[3, 8, 7], [1, 7, 9], [4, 4, 0]])
    expected_output = np.asarray([[10, 10, 10], [10, 9, 10], [10, 10, 10]])
    if cupy:
        array2d = cp.asarray(array2d)
        expected_output = cp.asarray(expected_output)
    ga_2d = GhostArray(
        interior=array2d, pad_width=1, mode="dirichlet", constant_values=10
    )
    out = ga_2d.apply_f_to_neighbors(f=np.max, mode="moore")
    assert out.dtype == "int"
    assert GhostArray_is_consistent(out)
    assert np.all(out.to_numpy() == expected_output)
