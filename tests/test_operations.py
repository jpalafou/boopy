import pytest
import random
import numpy as np
from boo import GhostArray
from tests.tools import (
    create_random_array,
    create_random_pad_width,
    GhostArray_is_consistent,
    pad_widths_are_equal,
)

n_tests = 10
max_gw = 3
dtype = ["int", "float"]
N = [1, 2, 3, 5, 10]
dimensions = [1, 2, 3, 4, 5]
mode = ["periodic", "dirichlet"]


@pytest.mark.parametrize("dtype", dtype)
@pytest.mark.parametrize("N", N)
@pytest.mark.parametrize("dimensions", dimensions)
@pytest.mark.parametrize("mode", mode)
@pytest.mark.parametrize("n_test", range(n_tests))
def test_negative(n_test, dtype, N, dimensions, mode):
    a = create_random_array(nmax=N, ndim=dimensions, dtype=dtype)
    pad_width = create_random_pad_width(ndim=dimensions, max=max_gw)
    constant_values = create_random_pad_width(ndim=dimensions, max=max_gw, min=-max_gw)
    a_gw = GhostArray(
        interior=a, pad_width=pad_width, mode=mode, constant_values=constant_values
    )
    original_ghost_Array = a_gw.ghost_array.copy()
    n_a_gw = -a_gw
    assert n_a_gw.dtype == dtype
    assert GhostArray_is_consistent(n_a_gw)
    assert np.all(n_a_gw.interior == -a)
    assert np.all(n_a_gw.ghost_array == -original_ghost_Array)
    assert pad_widths_are_equal(a_gw.pad_width, n_a_gw.pad_width)


@pytest.mark.parametrize("dtype", dtype)
@pytest.mark.parametrize("N", N)
@pytest.mark.parametrize("dimensions", dimensions)
@pytest.mark.parametrize("mode", mode)
@pytest.mark.parametrize("n_test", range(n_tests))
def test_multiplication(n_test, dtype, N, dimensions, mode):
    a = create_random_array(nmax=N, ndim=dimensions, dtype=dtype)
    pad_width = create_random_pad_width(ndim=dimensions, max=max_gw)
    constant_values = create_random_pad_width(ndim=dimensions, max=max_gw, min=-max_gw)
    a_gw = GhostArray(
        interior=a, pad_width=pad_width, mode=mode, constant_values=constant_values
    )
    original_ghost_Array = a_gw.ghost_array.copy()
    multiplier = random.randint(-max_gw, max_gw)
    scaled_a_gw = multiplier * a_gw
    assert scaled_a_gw.dtype == dtype
    assert GhostArray_is_consistent(scaled_a_gw)
    assert np.all(scaled_a_gw.interior == multiplier * a)
    assert np.all(scaled_a_gw.ghost_array == multiplier * original_ghost_Array)
    assert pad_widths_are_equal(a_gw.pad_width, scaled_a_gw.pad_width)
