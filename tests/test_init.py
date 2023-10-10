import pytest
import numpy as np
import cupy as cp
from boo import GhostArray
from tests.tools import GhostArray_is_consistent

cupy = [True, False]
dtype = ["int", "float"]
N = [1, 5]
dimensions = [1, 2, 3, 4, 5]
axis = [0, 1, 2, 3, 4]
mode = ["periodic", "dirichlet"]
gw = [0, 1, 2]


@pytest.mark.parametrize("cupy", cupy)
@pytest.mark.parametrize("dtype", dtype)
@pytest.mark.parametrize("N", N)
@pytest.mark.parametrize("dimensions", dimensions)
@pytest.mark.parametrize("mode", mode)
@pytest.mark.parametrize("left_gw", gw)
@pytest.mark.parametrize("right_gw", gw)
def test_interior_init(cupy, dtype, N, dimensions, mode, left_gw, right_gw):
    """
    initialize GhostArray with an interior array
    """
    convert = 1 if dtype == "int" else 1.0
    a = convert * np.random.randint(0, 10, tuple([N] * dimensions))
    if cupy:
        a = cp.asarray(a)
    ghost_width = [(left_gw, right_gw)] * dimensions
    ga = GhostArray(interior=a, pad_width=ghost_width, mode=mode)
    assert GhostArray_is_consistent(ga)
    assert ga.dtype == dtype


@pytest.mark.parametrize("cupy", cupy)
@pytest.mark.parametrize("dtype", dtype)
@pytest.mark.parametrize("N", N)
@pytest.mark.parametrize("dimensions", dimensions)
@pytest.mark.parametrize("mode", mode)
@pytest.mark.parametrize("left_gw", gw)
@pytest.mark.parametrize("right_gw", gw)
def test_padded_init(cupy, dtype, N, dimensions, mode, left_gw, right_gw):
    """
    initialize GhostArray with a padded array
    """
    convert = 1 if dtype == "int" else 1.0
    ghost_array = convert * np.random.randint(
        0, 10, ([left_gw + N + right_gw] * dimensions)
    )
    if cupy:
        ghost_array = cp.asarray(ghost_array)
    ghost_width = [(left_gw, right_gw)] * dimensions
    ga = GhostArray(
        interior=None,
        ghost_array=ghost_array,
        pad_width=ghost_width,
        mode=mode,
    )
    assert GhostArray_is_consistent(ga)
    assert ga.dtype == dtype


@pytest.mark.parametrize("cupy", cupy)
@pytest.mark.parametrize("dtype", dtype)
@pytest.mark.parametrize("N", N)
@pytest.mark.parametrize("dimensions", dimensions)
@pytest.mark.parametrize("axis", axis)
@pytest.mark.parametrize("mode", mode)
def test_to_numpy(cupy, dtype, N, dimensions, axis, mode):
    """
    initialize GhostArray with a padded array, remove the padding
    """
    if axis > dimensions - 1:
        return None
    pad_width = [(0, 0)] * dimensions
    pad_width[axis] = (2, 2)
    convert = 1 if dtype == "int" else 1.0
    a = convert * np.random.randint(0, 10, tuple([N] * dimensions))
    if cupy:
        a = cp.asarray(a)
    if mode == "dirichlet":
        a_with_pad = np.pad(array=a, pad_width=pad_width, mode="constant")
    if mode == "periodic":
        a_with_pad = np.pad(array=a, pad_width=pad_width, mode="wrap")
    ghost_array = GhostArray(
        interior=None, ghost_array=a_with_pad, pad_width=pad_width, mode=mode
    )
    interior = ghost_array.to_numpy()
    assert np.all(a == interior)
    assert GhostArray_is_consistent(ghost_array)
    assert ghost_array.dtype == dtype


@pytest.mark.parametrize("cupy", cupy)
@pytest.mark.parametrize("dtype", dtype)
@pytest.mark.parametrize("N", N)
@pytest.mark.parametrize(
    "pad_width_format", ["int", "tuple", "list of tuples", "list", "list of lists"]
)
@pytest.mark.parametrize("gw_left", gw)
@pytest.mark.parametrize("gw_right", gw)
@pytest.mark.parametrize("dimensions", dimensions)
@pytest.mark.parametrize("mode", mode)
def test_as_pairs(
    cupy, dtype, N, pad_width_format, gw_left, gw_right, dimensions, mode
):
    """
    pad_width should always be broadcast to a list of tuple or list of lists
    of shape (dimensions, 2)
    """
    if pad_width_format == "int":
        pad_width = gw_left
    if pad_width_format == "tuple":
        pad_width = (gw_left, gw_right)
    if pad_width_format == "list of tuples":
        pad_width = [(gw_left, gw_right)] * dimensions
    if pad_width_format == "list":
        pad_width = [gw_left, gw_right]
    if pad_width_format == "list of lists":
        pad_width = [[gw_left, gw_right]] * dimensions
    convert = 1 if dtype == "int" else 1.0
    a = convert * np.random.randint(0, 10, list([N] * dimensions))
    if cupy:
        a = cp.asarray(a)
    ghost_array = GhostArray(interior=a, pad_width=pad_width, mode=mode)
    if pad_width_format == "int":
        assert (
            ghost_array.pad_width == [[gw_left, gw_left]] * dimensions
            or ghost_array.pad_width == [(gw_left, gw_left)] * dimensions
        )
    else:
        assert (
            ghost_array.pad_width == [[gw_left, gw_right]] * dimensions
            or ghost_array.pad_width == [(gw_left, gw_right)] * dimensions
        )
    assert GhostArray_is_consistent(ghost_array)
    assert ghost_array.dtype == dtype
