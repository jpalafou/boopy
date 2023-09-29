import pytest
import numpy as np
import random
from boo import GhostArray
from tests.tools import (
    create_random_array,
    create_random_pad_width,
    GhostArray_is_consistent,
)

n_tests = 10
max_gw = 3
N = [1, 10]
dimensions = [1, 2, 3, 4, 5]
mode = ["periodic", "dirichlet"]
dtype = ["int", "float"]


@pytest.mark.parametrize("dtype", dtype)
@pytest.mark.parametrize("N", N)
@pytest.mark.parametrize("dimensions", dimensions)
@pytest.mark.parametrize("mode", mode)
@pytest.mark.parametrize("n_test", range(n_tests))
def test_add_padding(n_test, dtype, N, dimensions, mode):
    """
    when using add_along_axis method
        initial GhostArray should be consistent
        interior should not change
        new pad_width should be consistent with old pad_width + changes
        new constant_values should be consistent with old constant_valuse + change
        new GhostArray should have correct data type
        new GhostArray should be consistent
    """
    # initialize random GhostArray
    a = create_random_array(nmax=N, ndim=dimensions, dtype=dtype)
    pad_width = create_random_pad_width(ndim=dimensions, max=max_gw)
    constant_values = create_random_pad_width(ndim=dimensions, max=max_gw, min=-max_gw)
    a_gw = GhostArray(
        interior=a, pad_width=pad_width, mode=mode, constant_values=constant_values
    )
    assert GhostArray_is_consistent(a_gw)
    original_pad_width = a_gw.pad_width.copy()
    original_constant_values = a_gw.constant_values.copy()
    # determine what modification will be made
    axis_of_modification = sorted(
        random.sample(range(0, dimensions), random.randint(1, dimensions))
    )
    new_pad_width = create_random_pad_width(
        ndim=len(axis_of_modification), max=max_gw, type="list"
    )
    new_constant_values = create_random_pad_width(
        ndim=len(axis_of_modification), max=max_gw, min=-max_gw, type="list"
    )
    # modify GhostArray
    a_gw.add_along_axis(
        axis=axis_of_modification,
        pad_width=new_pad_width,
        constant_values=new_constant_values,
    )
    assert np.all(a_gw.interior == a)
    solution_pad_width = np.asarray(original_pad_width)
    solution_pad_width[axis_of_modification] = np.asarray(new_pad_width)
    assert np.all(np.array(a_gw.pad_width) == solution_pad_width)
    if mode == "dirichlet":
        solution_constant_values = np.asarray(original_constant_values)
        solution_constant_values[axis_of_modification] = np.asarray(new_constant_values)
        assert np.all(np.array(a_gw.constant_values) == solution_constant_values)
    assert a_gw.dtype == dtype
    assert GhostArray_is_consistent(a_gw)


@pytest.mark.parametrize("dtype", dtype)
@pytest.mark.parametrize("N", N)
@pytest.mark.parametrize("dimensions", dimensions)
@pytest.mark.parametrize("mode", mode)
@pytest.mark.parametrize("n_test", range(n_tests))
def test_remove_all_padding(n_test, dtype, N, dimensions, mode):
    """
    when using remove_along_axis method with axis=None
        interior should not change
        new pad_width should be (0,0)
        new constant_values should be nan
        new Ghost array should have correct dtype
        new GhostArray should be consistent
    """
    # initialize random GhostArray
    a = create_random_array(nmax=N, ndim=dimensions, dtype=dtype)
    pad_width = create_random_pad_width(ndim=dimensions, max=max_gw)
    constant_values = create_random_pad_width(ndim=dimensions, max=max_gw, min=-max_gw)
    a_gw = GhostArray(
        interior=a, pad_width=pad_width, mode=mode, constant_values=constant_values
    )
    GhostArray_is_consistent(a_gw)
    a_gw.remove_along_axis(axis=None)
    assert np.all(a_gw.interior == a)
    assert np.all(np.asarray(a_gw.pad_width) == np.zeros_like(a_gw.pad_width))
    if mode == "dirichlet":
        assert np.all(
            np.asarray(a_gw.constant_values) == np.zeros_like(a_gw.constant_values)
        )
    assert GhostArray_is_consistent(a_gw)
    assert a_gw.dtype == dtype


@pytest.mark.parametrize("dtype", dtype)
@pytest.mark.parametrize("N", N)
@pytest.mark.parametrize("dimensions", dimensions)
@pytest.mark.parametrize("mode", mode)
@pytest.mark.parametrize("n_test", range(n_tests))
def test_remove_one_axis_of_padding(n_test, dtype, N, dimensions, mode):
    """
    when using remove_along_axis method with axis=int
        interior should not change
        new pad_width should be (0,0)
        new constant_values should be nan
        new GhostArray should have correct dtype
        new GhostArray should be consistent
    """
    # initialize random GhostArray
    a = create_random_array(nmax=N, ndim=dimensions, dtype=dtype)
    pad_width = create_random_pad_width(ndim=dimensions, max=max_gw)
    constant_values = create_random_pad_width(ndim=dimensions, max=max_gw, min=-max_gw)
    a_gw = GhostArray(
        interior=a, pad_width=pad_width, mode=mode, constant_values=constant_values
    )
    original_pad_width = a_gw.pad_width.copy()
    original_constant_values = a_gw.constant_values.copy()
    # determine what modification will be made
    axis_of_modification = random.randint(0, dimensions - 1)
    axis_of_fixation = [i for i in range(a_gw.ndim) if i not in [axis_of_modification]]
    # modify GhostArray
    a_gw.remove_along_axis(axis=axis_of_modification)
    assert np.all(a_gw.interior == a)
    solution_pad_width = np.asarray(original_pad_width)
    solution_pad_width[axis_of_modification] = np.asarray([[0, 0]])
    assert np.all(np.array(a_gw.pad_width) == solution_pad_width)
    if mode == "dirichlet":
        assert np.all(
            np.asarray(a_gw.constant_values)[axis_of_modification]
            == np.zeros_like(a_gw.constant_values)[axis_of_modification]
        )
        assert np.all(
            np.asarray(a_gw.constant_values)[axis_of_fixation]
            == np.asarray(original_constant_values)[axis_of_fixation]
        )
    assert a_gw.dtype == dtype
    assert GhostArray_is_consistent(a_gw)


@pytest.mark.parametrize("dtype", dtype)
@pytest.mark.parametrize("N", N)
@pytest.mark.parametrize("dimensions", dimensions)
@pytest.mark.parametrize("mode", mode)
@pytest.mark.parametrize("n_test", range(n_tests))
def test_remove_multiple_axes_of_padding(n_test, dtype, N, dimensions, mode):
    """
    when using remove_along_axis method with axis=list
        initial GhostArray should be consistent
        interior should not change
        new pad_width should be (0,0)
        new constant_values should be nan
        new GhostArray should be have correct dtype
        new GhostArray should be consistent
    """
    # initialize random GhostArray
    a = create_random_array(nmax=N, ndim=dimensions, dtype=dtype)
    pad_width = create_random_pad_width(ndim=dimensions, max=max_gw)
    constant_values = create_random_pad_width(ndim=dimensions, max=max_gw, min=-max_gw)
    a_gw = GhostArray(
        interior=a, pad_width=pad_width, mode=mode, constant_values=constant_values
    )
    original_pad_width = a_gw.pad_width.copy()
    original_constant_values = a_gw.constant_values.copy()
    # determine what modification will be made
    axis_of_modification = sorted(
        random.sample(range(0, dimensions), random.randint(1, dimensions))
    )
    axis_of_fixation = [i for i in range(a_gw.ndim) if i not in axis_of_modification]
    nsub = len(axis_of_modification)
    # modify GhostArray
    a_gw.remove_along_axis(axis=axis_of_modification)
    assert np.all(a_gw.interior == a)
    solution_pad_width = np.asarray(original_pad_width)
    solution_pad_width[axis_of_modification] = np.asarray([[0, 0]] * nsub)
    assert np.all(np.array(a_gw.pad_width) == solution_pad_width)
    if mode == "dirichlet":
        assert np.all(
            np.asarray(a_gw.constant_values)[axis_of_modification]
            == np.zeros_like(a_gw.constant_values)[axis_of_modification]
        )
        assert np.all(
            np.asarray(a_gw.constant_values)[axis_of_fixation]
            == np.asarray(original_constant_values)[axis_of_fixation]
        )
    assert a_gw.dtype == dtype
    assert GhostArray_is_consistent(a_gw)
