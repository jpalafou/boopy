import numpy as np
from boo import GhostArray
from tests.tools import GhostArray_is_consistent

array1d = np.asarray([3, 8, 7])
ga_1d = GhostArray(interior=array1d, pad_width=1, mode="dirichlet", constant_values=10)

array2d = np.asarray([[3, 8, 7], [1, 7, 9], [4, 4, 0]])
ga_2d = GhostArray(interior=array2d, pad_width=1, mode="dirichlet", constant_values=10)


def test_1d_min():
    expected_output = np.array([3, 3, 7])
    out = ga_1d.apply_f_to_neighbors(f=np.min)
    assert out.dtype == "int"
    assert GhostArray_is_consistent(out)
    assert np.all(out.to_numpy() == expected_output)


def test_1d_max():
    expected_output = np.array([10, 8, 10])
    out = ga_1d.apply_f_to_neighbors(f=np.max)
    assert out.dtype == "int"
    assert GhostArray_is_consistent(out)
    assert np.all(out.to_numpy() == expected_output)


def test_2d_von_neumann_min():
    expected_output = np.asarray([[1, 3, 7], [1, 1, 0], [1, 0, 0]])
    out = ga_2d.apply_f_to_neighbors(f=np.min, mode="neumann")
    assert out.dtype == "int"
    assert GhostArray_is_consistent(out)
    assert np.all(out.to_numpy() == expected_output)


def test_2d_von_neumann_max():
    expected_output = np.asarray([[10, 10, 10], [10, 9, 10], [10, 10, 10]])
    out = ga_2d.apply_f_to_neighbors(f=np.max, mode="neumann")
    assert out.dtype == "int"
    assert GhostArray_is_consistent(out)
    assert np.all(out.to_numpy() == expected_output)


def test_2d_moore_min():
    expected_output = np.asarray([[1, 1, 7], [1, 0, 0], [1, 0, 0]])
    out = ga_2d.apply_f_to_neighbors(f=np.min, mode="moore")
    assert out.dtype == "int"
    assert GhostArray_is_consistent(out)
    assert np.all(out.to_numpy() == expected_output)


def test_2d_moore_max():
    expected_output = np.asarray([[10, 10, 10], [10, 9, 10], [10, 10, 10]])
    out = ga_2d.apply_f_to_neighbors(f=np.max, mode="moore")
    assert out.dtype == "int"
    assert GhostArray_is_consistent(out)
    assert np.all(out.to_numpy() == expected_output)
