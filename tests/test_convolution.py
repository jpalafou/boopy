import pytest
import numpy as np
import cupy as cp
from boo import GhostArray
from tests.tools import (
    create_random_array,
    create_random_pad_width,
    GhostArray_is_consistent,
)

n_tests = 1
max_gw = 3
N = [1, 10]
dimensions = [1, 2, 3, 4, 5]
axis = [0, 1, 2, 3, 4]
mode = ["dirichlet", "periodic"]
dtype = ["float", "int"]
cupy = [True, False]


@pytest.mark.parametrize("cupy", cupy)
@pytest.mark.parametrize("dtype", dtype)
@pytest.mark.parametrize("N", N)
@pytest.mark.parametrize("dimensions", dimensions)
@pytest.mark.parametrize("axis", axis)
@pytest.mark.parametrize("mode", mode)
@pytest.mark.parametrize(
    "kernel_config",
    [{"kernel": [0], "gw": (0, 0)}, {"kernel": [0, 0, 0], "gw": (1, 1)}],
)
@pytest.mark.parametrize("n_test", range(n_tests))
def test_zero_convolution(
    n_test, cupy, dtype, N, dimensions, axis, mode, kernel_config
):
    """
    convolution with a kernel of 0s
        data type of convolved array should be correct
        convolved ghost array should be consistent
        all elements of convolved array should be 0
    """
    if axis > dimensions - 1:
        return
    kernel = kernel_config["kernel"]
    a = create_random_array(nmax=N, ndim=dimensions, dtype=dtype, cupy=cupy)
    pad_width = create_random_pad_width(ndim=dimensions, max=max_gw, type="list")
    pad_width[axis] = kernel_config["gw"]
    constant_values = create_random_pad_width(
        ndim=dimensions, max=max_gw, min=-max_gw, type="list"
    )
    a_gw = GhostArray(
        interior=a,
        pad_width=pad_width,
        mode=mode,
        constant_values=constant_values,
    )
    a_gw = a_gw.convolve(kernel, axis=axis)
    assert a_gw.dtype == dtype
    assert GhostArray_is_consistent(a_gw)
    a_np = a_gw.to_numpy()
    assert np.all(a_np == np.zeros_like(a_np))


@pytest.mark.parametrize("cupy", cupy)
@pytest.mark.parametrize("dtype", dtype)
@pytest.mark.parametrize("N", N)
@pytest.mark.parametrize("dimensions", dimensions)
@pytest.mark.parametrize("axis", axis)
@pytest.mark.parametrize("mode", mode)
@pytest.mark.parametrize(
    "kernel_config",
    [{"kernel": [1], "gw": (0, 0)}, {"kernel": [0, 1, 0], "gw": (1, 1)}],
)
@pytest.mark.parametrize("n_test", range(n_tests))
def test_unit_convolution(
    n_test, cupy, dtype, N, dimensions, axis, mode, kernel_config
):
    """
    convolution with a unit kernet
        data type of convolved array should be correct
        convolved ghost array should be consistent
        all elements of convolved array should be equal to original array
    """
    if axis > dimensions - 1:
        return
    kernel = kernel_config["kernel"]
    a = create_random_array(nmax=N, ndim=dimensions, dtype=dtype, cupy=cupy)
    pad_width = create_random_pad_width(ndim=dimensions, max=max_gw, type="list")
    pad_width[axis] = kernel_config["gw"]
    constant_values = create_random_pad_width(
        ndim=dimensions, max=max_gw, min=-max_gw, type="list"
    )
    a_gw = GhostArray(
        interior=a,
        pad_width=pad_width,
        mode=mode,
        constant_values=constant_values,
    )
    a_gw = a_gw.convolve(kernel, axis=axis)
    assert a_gw.dtype == dtype
    assert GhostArray_is_consistent(a_gw)
    a_np = a_gw.to_numpy()
    assert np.all(a_np == a)


@pytest.mark.parametrize("cupy", cupy)
@pytest.mark.parametrize("dtype", dtype)
@pytest.mark.parametrize("N", N)
@pytest.mark.parametrize("dimensions", dimensions)
@pytest.mark.parametrize("axis", axis)
@pytest.mark.parametrize("mode", mode)
@pytest.mark.parametrize("n_test", range(n_tests))
def test_multiconvolution_01(n_test, cupy, dtype, N, dimensions, axis, mode):
    """
    convolution with a list of kernels: one unit, one zero, and another unit
        data type of convolved array should be correct
        convolved ghost array should be consistent
        shape of output array should be the same as a with a new first axis of length 3
        first slice of output array should equal a
        second slice of output array should equal zeros
        third slice of output array should equal a
    """
    if axis > dimensions - 1:
        return
    kernels = [[0, 1, 0], [0, 0, 0], [0, 1, 0]]
    a = create_random_array(nmax=N, ndim=dimensions, dtype=dtype, cupy=cupy)
    pad_width = create_random_pad_width(ndim=dimensions, max=max_gw, type="list")
    pad_width[axis] = (1, 1)
    constant_values = create_random_pad_width(
        ndim=dimensions, max=max_gw, min=-max_gw, type="list"
    )
    a_gw = GhostArray(
        interior=a,
        pad_width=pad_width,
        mode=mode,
        constant_values=constant_values,
    )
    a_gw = a_gw.multiconvolve(kernels, axis=axis)
    assert a_gw.dtype == dtype
    assert GhostArray_is_consistent(a_gw)
    a_np = a_gw.to_numpy()
    assert a_np.shape == tuple([3] + list(a.shape))
    assert np.all(a_np[0] == a)
    assert np.all(a_np[1] == np.zeros_like(a))
    assert np.all(a_np[2] == a)


@pytest.mark.parametrize("cupy", cupy)
@pytest.mark.parametrize("dtype", dtype)
@pytest.mark.parametrize("N", N)
@pytest.mark.parametrize("dimensions", dimensions)
@pytest.mark.parametrize("axis", axis)
@pytest.mark.parametrize("mode", mode)
@pytest.mark.parametrize("n_test", range(n_tests))
def test_multiconvolution_duplicates(n_test, cupy, dtype, N, dimensions, axis, mode):
    """
    convolution with a list of kernels: one unit, one zero, and another unit
        data type of convolved array should be correct
        convolved ghost array should be consistent
        shape of output array should be the same as a with a new first axis of length 3
        second slice of output array first slice
        third slice of output array should equal first slice
    """
    if axis > dimensions - 1:
        return
    kernels = [[1, 2, 1], [1, 2, 1], [1, 2, 1]]
    a = create_random_array(nmax=N, ndim=dimensions, dtype=dtype, cupy=cupy)
    pad_width = create_random_pad_width(ndim=dimensions, max=max_gw, type="list")
    pad_width[axis] = (1, 1)
    constant_values = create_random_pad_width(
        ndim=dimensions, max=max_gw, min=-max_gw, type="list"
    )
    a_gw = GhostArray(
        interior=a,
        pad_width=pad_width,
        mode=mode,
        constant_values=constant_values,
    )
    a_gw = a_gw.multiconvolve(kernels, axis=axis)
    assert a_gw.dtype == dtype
    assert GhostArray_is_consistent(a_gw)
    a_np = a_gw.to_numpy()
    assert a_np.shape == tuple([3] + list(a.shape))
    assert np.all(a_np[1] == a_np[0])
    assert np.all(a_np[2] == a_np[0])


@pytest.mark.parametrize("cupy", cupy)
def test_convolve_1d(cupy):
    """
    compute the second derivative of g(x) = exp(-x^2) by applying a second order
    central derivative approximation twice
    """

    def g(x):
        return np.exp(-np.square(x))

    def gdotdot(x):
        return 2 * (2 * np.square(x) - 1) * np.exp(-np.square(x))

    kernel = [-1, 0, 1]

    previous_l2norm = np.inf
    for N in [1024, 2048, 4096]:
        x = np.linspace(-5, 5, N)
        h = 10 / (N - 1)

        expected_values = gdotdot(x[2:-2])
        if cupy:
            expected_values = cp.asarray(expected_values)

        u = g(x)
        if cupy:
            u = cp.asarray(u)
        u_ghost = GhostArray(
            ghost_array=u, interior=None, pad_width=2, mode="dirichlet"
        )
        dudx = u_ghost.convolve(kernel) / (2 * h)
        ddudxx = (dudx.convolve(kernel) / (2 * h)).to_numpy()
        l2norm = np.sqrt(np.sum(np.square(ddudxx - expected_values)) / N)

        assert l2norm < previous_l2norm
        previous_l2norm = l2norm


@pytest.mark.parametrize("cupy", cupy)
@pytest.mark.parametrize("n_vars", [1, 5, 10])
def test_convole_1d_multivar(cupy, n_vars):
    """
    compute the second derivative of g(x) = exp(-x^2) by applying a second order
    central derivative approximation twice for an array of size (n_vars, N)
    the results should be indential across the first axis
    """

    def g(x):
        return np.array([np.exp(-np.square(x))] * n_vars)

    def gdotdot(x):
        return np.array([2 * (2 * np.square(x) - 1) * np.exp(-np.square(x))])

    kernel = [-1, 0, 1]

    previous_l2norm = [np.inf] * n_vars
    for N in [1024, 2048, 4096]:
        x = np.linspace(-5, 5, N)
        h = 10 / (N - 1)

        expected_values = gdotdot(x[2:-2])
        if cupy:
            expected_values = cp.asarray(expected_values)

        u = g(x)
        if cupy:
            u = cp.asarray(u)
        u_ghost = GhostArray(
            ghost_array=u,
            interior=None,
            pad_width=[(0, 0), (2, 2)],
            mode="dirichlet",
        )
        dudx = u_ghost.convolve(kernel, axis=1) / (2 * h)
        ddudxx = (dudx.convolve(kernel, axis=1) / (2 * h)).to_numpy()
        l2norm = [
            np.sqrt(np.sum(np.square((ddudxx - expected_values)[i])) / N)
            for i in range(n_vars)
        ]

        assert all([l2norm[i] < previous_l2norm[i] for i in range(n_vars)])
        assert all([l2norm[0] == l2norm[i + 1] for i in range(n_vars - 1)])
        previous_l2norm = l2norm


@pytest.mark.parametrize("cupy", cupy)
def test_convolve_2d(cupy):
    """
    solve 2D advection du/dt + v_x du/dx + v_y du/dy = 0
    """

    # problem config
    courant = 0.5
    v_x = 2
    v_y = 1
    stencil = np.array(
        [
            -1 / 560,
            17 / 840,
            -97 / 840,
            449 / 840,
            319 / 420,
            -223 / 840,
            71 / 840,
            -1 / 56,
            1 / 560,
        ]
    )

    def g(x, y):
        return np.sin(2 * np.pi * (x + y[:, np.newaxis]))

    previous_l2norm = np.inf
    for N in [16, 32, 64]:
        # set up domain
        x = np.linspace(0, 1, N)
        y = x
        h = 1 / (N - 1)
        dt = courant / (v_x / h + v_y / h)
        T = 1

        # initialize
        u0 = g(x, y)
        if cupy:
            u0 = cp.asarray(u0)
        u = u0.copy()
        t = 0
        while t < T:
            uga = GhostArray(
                interior=u, pad_width=len(stencil) // 2 + 1, mode="periodic"
            )
            right_flux = (
                v_x
                * uga.convolve(kernel=np.flip(stencil), axis=0)
                .remove_along_axis(1)
                .ghost_array
            )
            top_flux = (
                v_y
                * uga.convolve(kernel=np.flip(stencil), axis=1)
                .remove_along_axis(0)
                .ghost_array
            )
            dudt = -(1 / h) * (right_flux[1:-1, :] - right_flux[:-2, :]) - (1 / h) * (
                top_flux[:, 1:-1] - top_flux[:, :-2]
            )
            u += dt * dudt
            t += dt
            if t + dt > T:
                dt = T - t
        l2norm = np.sqrt(np.sum(np.square(u - u0)) * h * h)
        assert l2norm < previous_l2norm
        previous_l2norm = l2norm


def test_multiconvolve_2d():
    """
    solve 2D advection du/dt + v_x du/dx + v_y du/dy = 0
    reconstruct left and right of cell faces simulaneously with multiconvolve
    """

    # problem config
    courant = 0.5
    v_x = 2
    v_y = 1
    stencil = np.array(
        [
            -1 / 560,
            17 / 840,
            -97 / 840,
            449 / 840,
            319 / 420,
            -223 / 840,
            71 / 840,
            -1 / 56,
            1 / 560,
        ]
    )
    stencils = np.asarray([stencil, np.flip(stencil)])  # left, right

    def g(x, y):
        return np.sin(2 * np.pi * (x + y[:, np.newaxis]))

    previous_l2norm = np.inf
    for N in [16, 32, 64]:
        # set up domain
        x = np.linspace(0, 1, N)
        y = x
        h = 1 / (N - 1)
        dt = courant / (v_x / h + v_y / h)
        T = 1

        # initialize
        u0 = g(x, y)
        if cupy:
            u0 = cp.asarray(u0)
        u = u0.copy()
        t = 0
        while t < T:
            uga = GhostArray(
                interior=u, pad_width=len(stencil) // 2 + 1, mode="periodic"
            )
            left_right_flux = (
                v_x
                * uga.multiconvolve(kernels=stencils, axis=0)
                .remove_along_axis(2)
                .ghost_array
            )
            bottom_top_flux = (
                v_y
                * uga.multiconvolve(kernels=stencils, axis=1)
                .remove_along_axis(1)
                .ghost_array
            )
            right_flux = left_right_flux[1]
            top_flux = bottom_top_flux[1]
            dudt = -(1 / h) * (right_flux[1:-1, :] - right_flux[:-2, :]) - (1 / h) * (
                top_flux[:, 1:-1] - top_flux[:, :-2]
            )
            u += dt * dudt
            t += dt
            if t + dt > T:
                dt = T - t
        l2norm = np.sqrt(np.sum(np.square(u - u0)) * h * h)
        assert l2norm < previous_l2norm
        previous_l2norm = l2norm
