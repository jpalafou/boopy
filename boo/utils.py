import numpy as np


def _as_pairs(x, ndim, as_index=False):
    """
    copy and pasted from numpy by author, modified to return a list

    Broadcast `x` to an array with the shape [`ndim`, 2].

    A helper function for `pad` that prepares and validates arguments like
    `pad_width` for iteration in pairs.

    Parameters
    ----------
    x : {None, scalar, array-like}
        The object to broadcast to the shape (`ndim`, 2).
    ndim : int
        Number of pairs the broadcasted `x` will have.
    as_index : bool, optional
        If `x` is not None, try to round each element of `x` to an integer
        (dtype `np.intp`) and ensure every element is positive.

    Returns
    -------
    pairs : nested iterables, shape [`ndim`, 2]
        The broadcasted version of `x`.

    Raises
    ------
    ValueError
        If `as_index` is True and `x` contains negative elements.
        Or if `x` is not broadcastable to the shape [`ndim`, 2].
    """
    if x is None:
        # Pass through None as a special case, otherwise np.round(x) fails
        # with an AttributeError
        return [
            (None, None),
        ] * ndim

    x = np.array(x)
    if as_index:
        x = np.round(x).astype(np.intp, copy=False)

    if x.ndim < 3:
        # Optimization: Possibly use faster paths for cases where `x` has
        # only 1 or 2 elements. `np.broadcast_to` could handle these as well
        # but is currently slower

        if x.size == 1:
            # x was supplied as a single value
            x = x.ravel()  # Ensure x[0] works for x.ndim == 0, 1, 2
            if as_index and x < 0:
                raise ValueError("index can't contain negative values")
            return [(x[0], x[0])] * ndim

        if x.size == 2 and x.shape != (2, 1):
            # x was supplied with a single value for each side
            # but except case when each dimension has a single value
            # which should be broadcasted to a pair,
            # e.g. [[1], [2]] -> [[1, 1], [2, 2]] not [[1, 2], [1, 2]]
            x = x.ravel()  # Ensure x[0], x[1] works
            if as_index and (x[0] < 0 or x[1] < 0):
                raise ValueError("index can't contain negative values")
            return [(x[0], x[1])] * ndim

    if as_index and x.min() < 0:
        raise ValueError("index can't contain negative values")

    # Converting the array with `tolist` seems to improve performance
    # when iterating and indexing the result (see usage in `pad`)
    return np.broadcast_to(x, (ndim, 2)).tolist()


def _chop_off_ends(array: np.ndarray, chop_size: tuple, axis: int) -> np.ndarray:
    """
    helper function for Ghost array
    remove ends of an array along axis
    args:
        array       arbitrary size
        chop_size   tuple (remove left elements, remove right elements)
        axis        int
    returns:
        out         array reduced by sum(chop_size) along axis
    """
    index = [slice(None)] * array.ndim
    index[axis] = slice(chop_size[0], -chop_size[1] or None)
    out = array[tuple(index)]
    return out


def _convolve(array: np.ndarray, kernel: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    array of arbitrary size convolved with 1d kernel
    args:
        array       array of arbitrary shape
        kernels     array of shape (n_kernel,)
        axis
    returns:
        out         array with axis dimension reduced by $n_kernel - 1$
    """
    # reshape kernels with two new axes (N_kernels, n_kernels, ...)
    n_kernel = kernel.size
    new_kernel_shape = [n_kernel] + [1] * array.ndim
    reshaped_kernel = np.flip(kernel).reshape(new_kernel_shape)
    # reshape array similarly
    array_stack = _stack(array, stacks=n_kernel, axis=axis)
    out = np.sum(reshaped_kernel * array_stack, axis=0)
    return out


def _gather_list_of_1D_neighbors(array: np.ndarray, axis: int) -> np.ndarray:
    """
    args:
        array   arbitrary shape
        axis    spatial dimension of array
    returns:
        out     new first dimension of 3 neighbors, reduces length along $axis + 1$
                dimension
    """
    idx = [slice(None)] * (array.ndim + 1)
    expanded_array = np.expand_dims(array, axis=0)
    # center
    idx[axis + 1] = slice(1, -1)
    neighbors = expanded_array[tuple(idx)]
    # left neighbor
    idx[axis + 1] = slice(None, -2)
    neighbors = np.concatenate((neighbors, expanded_array[tuple(idx)]))
    # right neighbor
    idx[axis + 1] = slice(2, None)
    out = np.concatenate((neighbors, expanded_array[tuple(idx)]))
    return out


def _gather_list_of_von_neumann_neighbors(array: np.ndarray, axes: list) -> np.ndarray:
    """
    args:
        array   arbitrary shape
        axis    spatial dimensions of array
    returns:
        out     new first dimension of $2*ndim + 1$ neighbors, reduces length along
                $axis + 1$ dimensions
    """
    idx = [slice(None)] * (array.ndim + 1)
    expanded_array = np.expand_dims(array, axis=0)
    if len(axes) == 2:
        # center
        idx[axes[0] + 1] = slice(1, -1)
        idx[axes[1] + 1] = slice(1, -1)
        neighbors = expanded_array[tuple(idx)]
        # left
        idx[axes[0] + 1] = slice(None, -2)
        neighbors = np.concatenate((neighbors, expanded_array[tuple(idx)]))
        # right
        idx[axes[0] + 1] = slice(2, None)
        neighbors = np.concatenate((neighbors, expanded_array[tuple(idx)]))
        # bottom
        idx[axes[0] + 1] = slice(1, -1)  # reset
        idx[axes[1] + 1] = slice(None, -2)
        neighbors = np.concatenate((neighbors, expanded_array[tuple(idx)]))
        # top
        idx[axes[1] + 1] = slice(2, None)
        out = np.concatenate((neighbors, expanded_array[tuple(idx)]))
        return out
    raise NotImplementedError()


def _gather_list_of_moore_neighbors(array: np.ndarray, axes: list):
    idx = [slice(None)] * (array.ndim + 1)
    expanded_array = np.expand_dims(array, axis=0)
    neighbors = _gather_list_of_von_neumann_neighbors(array=array, axes=axes)
    """
    args:
        array   arbitrary shape
        axes    spatial dimensions of array
    returns:
        out     new first dimension of $2*ndim + 1$ neighbors, reduces length along
                $axis + 1$ dimensions
    """
    if len(axes) == 2:
        # bottom left
        idx[axes[0] + 1] = slice(None, -2)
        idx[axes[1] + 1] = slice(None, -2)
        neighbors = np.concatenate((neighbors, expanded_array[tuple(idx)]))
        # top left
        idx[axes[1] + 1] = slice(2, None)
        neighbors = np.concatenate((neighbors, expanded_array[tuple(idx)]))
        # bottom right
        idx[axes[0] + 1] = slice(2, None)
        idx[axes[1] + 1] = slice(None, -2)
        neighbors = np.concatenate((neighbors, expanded_array[tuple(idx)]))
        # top right
        idx[axes[1] + 1] = slice(2, None)
        out = np.concatenate((neighbors, expanded_array[tuple(idx)]))
        return out
    raise NotImplementedError()


def _multiconvolve(array: np.ndarray, kernels: np.ndarray, axis: int) -> np.ndarray:
    """
    array of arbitrary size convolved with multiple 1d kernels
    args:
        array       array of arbitrary shape
        kernels     array of shape (N_kernels, n_kernel)
        axis
    returns:
        out         array with new first dimension of length N_kernels and $axis + 1$
                    dimension reduced by $n_kernel - 1$
    """
    # reshape kernels with two new axes (N_kernels, n_kernels, ...)
    N_kernels, n_kernel = kernels.shape
    new_kernel_shape = [N_kernels, n_kernel] + [1] * array.ndim
    reshaped_kernels = np.fliplr(kernels).reshape(new_kernel_shape)
    # reshape array similarly
    array_stack = _stack(array, stacks=n_kernel, axis=axis)
    repeated_array_stack = np.repeat(
        np.expand_dims(array_stack, axis=0), repeats=N_kernels, axis=0
    )
    out = np.sum(reshaped_kernels * repeated_array_stack, axis=1)
    return out


def _stack(array: np.ndarray, stacks: int, axis: int = 0) -> np.ndarray:
    """
    helper function of _multiconvolve
    args:
        array   arbitrary shape
        stacks  number of stacks to form
        axis
    returns:
        array with a new first dimension of length stacks and $stacks - 1$ reduced
        length along $axis + 1$ dimension
    """
    shape = list(array.shape)
    shape[axis] -= stacks - 1
    out = np.concatenate(
        [
            np.expand_dims(array.take(range(i, i + shape[axis]), axis=axis), axis=0)
            for i in range(stacks)
        ],
        axis=0,
    )
    return out
