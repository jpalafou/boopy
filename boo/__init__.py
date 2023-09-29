import numpy as np
from boo.utils import _as_pairs, _chop_off_ends, _multiconvolve


class GhostArray:
    def __init__(
        self,
        interior: np.ndarray,
        pad_width: list,
        mode: str = "dirichlet",
        constant_values: list = 0,
        ghost_array: np.ndarray = None,
    ):
        """
        args:
            interior    numpy or cupy array, arbitrary size, no ghost cells
            ghost_array numpy or cupy array, arbitrary size, includes ghost cells
                        overwritten if interior is defined
            pad_width   int a
                            applies a ghost width of a around the entire array
                        tuple (l,r)
                            applies a ghost width of l on the left of each dimension,
                            r on the right of each dimension
                        iterable of tuples  ((l0, r0), ...)
                            applies a ghost width of l1 on the left of axis 0, ...,
                            r1 on the right of axis 0, ...
                        iterable of tuples  ((s0,), ...)
                            applies a ghost width of s1 on the left and right
                            of axis 0, ...
            mode        'dirichlet', 'periodic'
            constant_values     same format as pad_width, for 'dirichlet' mode
        """
        self.interior = interior
        self.ndim = interior.ndim if self.interior is not None else ghost_array.ndim

        # configure arguments for np.pad()
        self.mode = mode
        pad_width_np = np.asarray(pad_width)
        if np.any(pad_width_np < 0):
            raise BaseException("Negative pad width.")
        self.pad_width = _as_pairs(pad_width_np, self.ndim, as_index=True)
        constant_values_np = np.asarray(constant_values)
        self.constant_values = _as_pairs(constant_values_np, self.ndim, as_index=False)

        # define ghost array
        if self.interior is not None:
            self.shape = self.interior.shape
            self._compute_ghost_zone()
        else:
            self.ghost_array = ghost_array
            self.dtype = ghost_array.dtype
        self.slices = None  # compute slices only as needed

    def _config_np_pad(self):
        """
        helper method for _compute_ghost_zone
        args:
            self.mode
            self.pad_width
            self.constant_values
        returns:
            self._np_pad_config     argument dictionary for np.pad()
        """
        np_pad_configs = {
            "periodic": {"pad_width": self.pad_width, "mode": "wrap"},
            "dirichlet": {
                "pad_width": self.pad_width,
                "mode": "constant",
                "constant_values": self.constant_values,
            },
        }
        self._np_pad_config = np_pad_configs[self.mode]

    def _compute_ghost_zone(self):
        """
        args:
            self.interior
            self._np_pad_config
        returns:
            self.ghost_array    same shape as self.interior, includes ghost zones
            self.dtype
        """
        self._config_np_pad()
        self.ghost_array = np.pad(array=self.interior, **self._np_pad_config)
        self.dtype = self.ghost_array.dtype

    def _compute_slices(self):
        """
        helper method for _compute_interior
        args:
            self.pad_width
        returns:
            self.slices     list of slices that index ghost_array into interior
        """
        if self.slices is None:
            self.slices = [
                slice(left, -right or None) for left, right, in self.pad_width
            ]

    def _compute_interior(self):
        """
        args:
            self.ghost_array
            self.slices
        returns:
            self.interior       defined region of ghost_array with no padding
        """
        if self.interior is None:
            self._compute_slices()
            self.interior = self.ghost_array[tuple(self.slices)]
            self.shape = self.interior.shape

    def to_numpy(self, region: str = "interior"):
        """
        return interior arra as np.ndarray
        args:
            region  "interior" or "ghost_array"
        returns
            self.interior
        """
        self._compute_interior()
        return self.interior

    def add_along_axis(self, axis: list, pad_width: list, constant_values: list = 0):
        """
        modify the padding along specified axes
        args:
            axis
            pad_width
            constant_values
        returns:
            self.ghost_array
            self.pad_width
            self.constant_values
            self.slices
        """
        # configure pad_width and constant_values
        axes = [axis] if isinstance(axis, int) else axis
        nadd = len(axes)
        pad_width_np = np.asarray(pad_width)
        delta_pad_width = _as_pairs(pad_width_np, nadd, as_index=True)
        constant_values_np = np.asarray(constant_values)
        delta_constant_values = _as_pairs(constant_values_np, nadd, as_index=False)
        # new attributes and arguments for np.pad()
        new_pad_width = self.pad_width.copy()
        new_constant_values = self.constant_values.copy()
        np_pad_argument = [(0, 0)] * self.ndim
        constant_values_argument = [(0, 0)] * self.ndim
        for i, ax in enumerate(axes):
            l_old, r_old = self.pad_width[ax]
            if l_old != 0 or r_old != 0:
                self.remove_along_axis(axis=ax)
            new_pad_width[ax] = [delta_pad_width[i][0], delta_pad_width[i][1]]
            np_pad_argument[ax] = [delta_pad_width[i][0], delta_pad_width[i][1]]
            if self.mode == "dirichlet":
                new_constant_values[ax] = [
                    delta_constant_values[i][0],
                    delta_constant_values[i][1],
                ]
                constant_values_argument[ax] = [
                    delta_constant_values[i][0],
                    delta_constant_values[i][1],
                ]
        # temporarily set pad_width and constant_values to deltas to trick pad_config
        self.pad_width = np_pad_argument
        if self.mode == "dirichlet":
            self.constant_values = constant_values_argument
        self._config_np_pad()
        self.ghost_array = np.pad(array=self.ghost_array, **self._np_pad_config)
        # reset pad_width and constant_values to true values
        self.pad_width = new_pad_width
        if self.mode == "dirichlet":
            self.constant_values = new_constant_values
        self.slices = None  # slices are no longer accurate
        return self

    def remove_along_axis(self, axis: list = None):
        """
        remove the padding along specified axes
        args:
            axis        removes all padding if 'None'
        returns:
            self.ghost_array
            self.pad_width
            self.constant_values
            self.slices
        """
        if axis is None:
            self._compute_slices()
            self.ghost_array = self.ghost_array[tuple(self.slices)]
            self.pad_width = [(0, 0)] * self.ndim
            self.constant_values = [(0, 0)] * self.ndim
            self.slices = None
            return self
        if isinstance(axis, int):
            self.ghost_array = _chop_off_ends(
                array=self.ghost_array, chop_size=self.pad_width[axis], axis=axis
            )
            self.pad_width[axis] = [0, 0]
            self.constant_values[axis] = [0, 0]
            self.slices = None
            return self
        rm_slice = [slice(None)] * self.ndim
        new_pad_width = self.pad_width.copy()
        new_constant_values = self.constant_values.copy()
        for i in axis:
            rm_slice[i] = slice(self.pad_width[i][0], -self.pad_width[i][1] or None)
            new_pad_width[i] = [0, 0]
            new_constant_values[i] = [0, 0]
        self.ghost_array = self.ghost_array[tuple(rm_slice)]
        self.pad_width = new_pad_width
        self.constant_values = new_constant_values
        self.slices = None
        return self

    def convolve(self, kernel: np.ndarray, axis: int = 0, bias_shift: int = 0):
        """
        1d convolution of a kernel on an array
        args:
            kernel  1d array of weights
            axis    int
            bias_shift  int, defaults to 0
                            [X, _, _, _, X] for even kernels
                            [X, _, _, _, X + 1] for odd kernels
        returns:
            new GhostArray instance with less padding in the axis dimension
        """
        # find kernel dimensions
        kernel_np = np.asarray(kernel)
        n_kernel = kernel_np.size
        # perform convolutions
        convolved_array = _multiconvolve(
            array=self.ghost_array, kernels=kernel_np[np.newaxis], axis=axis
        )[0]
        # find amount to chop off either end of array
        # if n_kernel is even and bias_shift = 0, right_rm > left_rm by 1
        left_rm = n_kernel // 2 + n_kernel % 2 - 1 - bias_shift
        right_rm = n_kernel // 2 + bias_shift
        # remove spent pad width
        new_pad_width = self.pad_width.copy()
        new_pad_width[axis] = [
            self.pad_width[axis][0] - left_rm,
            self.pad_width[axis][1] - right_rm,
        ]
        # set constant value to 0 if the padding has been depleted
        new_constant_values = np.asarray(self.constant_values)
        if new_pad_width[axis][0] == 0:
            new_constant_values[axis][0] = 0
        if new_pad_width[axis][1] == 0:
            new_constant_values[axis][1] = 0
        out = self.__class__(
            ghost_array=convolved_array,
            interior=None,
            pad_width=new_pad_width,
            mode=self.mode,
            constant_values=new_constant_values,
        )
        return out

    def multiconvolve(self, kernels: np.ndarray, axis: int = 0, bias_shift: int = 0):
        """
        1d convolution of a multiple kernels on an array
        args:
            kernels  2d array of weights, shape (N_kernels, n_kernel)
            axis    int
            bias_shift  int, defaults to 0
                            [X, _, _, _, X] for even kernels
                            [X, _, _, _, X + 1] for odd kernels
        returns:
            new GhostArray instance with a new first dimension of length N_kernels and
            an axis + 1 dimension with reduced length
        """
        # find kernel dimensions
        kernels_np = np.asarray(kernels)
        N_kernels, n_kernel = kernels_np.shape
        # perform convolutions
        convolved_array = _multiconvolve(
            array=self.ghost_array, kernels=kernels_np, axis=axis
        )
        # find amount to chop off either end of array
        # if n_kernel is even and bias_shift = 0, right_rm > left_rm by 1
        left_rm = n_kernel // 2 + n_kernel % 2 - 1 - bias_shift
        right_rm = n_kernel // 2 + bias_shift
        # remove spent pad width
        new_pad_width = [[0, 0]] + self.pad_width.copy()
        new_pad_width[axis + 1] = [
            self.pad_width[axis][0] - left_rm,
            self.pad_width[axis][1] - right_rm,
        ]
        # set constant value to 0 if the padding has been depleted
        new_constant_values = [[0, 0]] + self.constant_values
        if new_pad_width[axis + 1][0] == 0:
            new_constant_values[axis + 1] = [0, new_constant_values[axis + 1][1]]
        if new_pad_width[axis + 1][1] == 0:
            new_constant_values[axis + 1] = [new_constant_values[axis + 1][0], 0]
        out = self.__class__(
            ghost_array=convolved_array,
            interior=None,
            pad_width=new_pad_width,
            mode=self.mode,
            constant_values=new_constant_values,
        )
        return out

    def gather_neighbors(self, type: str = "moore"):
        new_pad_width = self.pad_width.copy()
        if self.ndim == 1:
            list_of_neighbors = [
                self.ghost_array[:-2],
                self.ghost_array[1:-1],
                self.ghost_array[2:],
            ]
            new_pad_width = (new_pad_width[0][0] - 1, new_pad_width[0][1] - 1)
            return list_of_neighbors
        if self.ndim == 2:
            list_of_neighbors = [
                self.ghost_array[:-2, 1:-1],
                self.ghost_array[1:-1, :-2],
                self.ghost_array[1:-1, 1:-1],
                self.ghost_array[2:, 1:-1],
                self.ghost_array[1:-1, 2:],
            ]

            new_pad_width = [
                (new_pad_width[0][0] - 1, new_pad_width[0][1] - 1),
                (new_pad_width[1][0] - 1, new_pad_width[1][1] - 1),
            ]
            return list_of_neighbors
        raise NotImplementedError(
            f"Gathering neighbors of a {self.ndim}-dimensional array."
        )

    def __repr__(self):
        return (
            f"GhostArray(interior = {self.interior}"
            + f", pad_width = {self.pad_width}, mode = {self.mode})"
        )

    def __eq__(self, other):
        interior_is_eq = np.all(self.interior == other.interior)
        ghost_array_is_eq = np.all(self.ghost_array == other.ghost_array)
        return interior_is_eq and ghost_array_is_eq

    def __neg__(self):
        self._compute_interior()
        if self.mode == "dirichlet":
            return self.__class__(
                interior=-self.interior,
                pad_width=self.pad_width,
                mode=self.mode,
                constant_values=-np.asarray(self.constant_values),
            )
        return self.__class__(
            interior=-self.interior,
            pad_width=self.pad_width,
            mode=self.mode,
            constant_values=self.constant_values,
        )

    def __add__(self, other):
        raise NotImplementedError(f"{self.__class__} + {type(other)}")

    def __radd__(self, other):
        raise NotImplementedError(f"{type(other)} + {self.__class__}")

    def __sub__(self, other):
        raise NotImplementedError(f"{self.__class__} - {type(other)}")

    def __rsub__(self, other):
        raise NotImplementedError(f"{type(other)} - {self.__class__}")

    def __mul__(self, other):
        self._compute_interior()
        if isinstance(other, int) or isinstance(other, float):
            if self.mode == "dirichlet":
                return self.__class__(
                    interior=self.interior * other,
                    pad_width=self.pad_width,
                    mode=self.mode,
                    constant_values=np.asarray(self.constant_values) * other,
                )
            return self.__class__(
                interior=self.interior * other,
                pad_width=self.pad_width,
                mode=self.mode,
                constant_values=self.constant_values,
            )
        raise NotImplementedError(f"{self.__class__} * {type(other)}")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __floordiv__(self, other):
        raise NotImplementedError(f"{self.__class__} // {type(other)}")

    def __rfloordiv__(other, self):
        raise NotImplementedError(f"{type(other)} // {self.__class__}")

    def __truediv__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return self.__mul__(1 / other)
        raise NotImplementedError(f"{type(other)} / {self.__class__}")

    def __rtruediv__(self, other):
        raise NotImplementedError(f"{self.__class__} / {type(other)}")
