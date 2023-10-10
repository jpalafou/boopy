from boo import GhostArray
import numpy as np
import numpy as cp  # import cupy as cp
import random


def create_random_array(
    nmax: int,
    ndim: int,
    upper: int = 9,
    lower: int = 0,
    cupy: bool = False,
    dtype: str = "int",
):
    """
    args:
        nmax    maximum possible length along a dimension
        ndim    number of dimensions
        upper   max possible value
        lower   min possible value
        cupy    whether to use cupy
        dtype   data type
    returns
        np.ndarray
    """

    dims = []
    for _ in range(ndim):
        dims.append(random.randint(1, nmax))
    if dtype == "int":
        convert = 1
    if dtype == "float":
        convert = 1.0
    xpy = np
    if cupy:
        xpy = cp
    return convert * xpy.random.randint(lower, upper + 1, dims)


def create_random_pad_width(ndim: int, max: int, min: int = 0, type: "str" = None):
    """
    args:
        ndim    number of dimensions
        upper   max possible value
        lower   min possible value
        type    "int", "tuple", "list"
    returns
        int, tuple, or list formatting of pad_width
    """
    roll = random.randint(1, 5) if type is None else -1
    if roll == 1 or type == "int":
        return random.randint(min, max)
    if roll == 2 or type == "tuple":
        return (random.randint(min, max), random.randint(0, max))
    pad_width = []
    for _ in range(ndim):
        pad_width.append((random.randint(0, max), random.randint(0, max)))
    return pad_width


def GhostArray_is_consistent(ga: GhostArray):
    """
    returns true if interior, pad_width, and ghost_array dimensions are consistent
    returns False otherwise
    """
    interior = ga.to_numpy()
    length = []
    for i in range(ga.ndim):
        length.append(ga.pad_width[i][0] + interior.shape[i] + ga.pad_width[i][1])
    return all([length[i] == ga.ghost_array.shape[i] for i in range(ga.ndim)])


def pad_widths_are_equal(pad_width1, pad_width2):
    return np.all(np.asarray(pad_width1) == np.asarray(pad_width2))
