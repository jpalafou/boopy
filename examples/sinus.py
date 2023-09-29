import time
import numpy as np
import matplotlib.pyplot as plt
from boo import GhostArray

# problem config
courant = 0.5
v_x = 2
v_y = 1
# 8th order polynomial reconstruction of left cell face
stencil = np.asarray(
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
line_stencils = np.stack([stencil, np.flip(stencil), stencil, np.flip(stencil)])
pointwise_stencils = np.stack([stencil, stencil])


def g(x, y):
    return 0.5 * np.sin(2 * np.pi * (x + y[:, np.newaxis])) + 0.5


# set up domain
N = 128
x = np.linspace(0, 1, N)
y = x
h = 1 / (N - 1)
dt = courant / (v_x / h + v_y / h)
T = 1

# initialize
u0 = g(x, y)
u = u0.copy()
t = 0
step_count = 0
starting_time = time.time()
while t < T:
    if step_count % 10 == 0:
        print(f"t = {t:.2f}, elapsed time = {time.time() - starting_time:.2f} s")
    volume_averages = GhostArray(
        interior=u, pad_width=len(stencil) // 2 + 1, mode="periodic"
    )
    horizontal_line_averages = volume_averages.multiconvolve(line_stencils, axis=1)
    vertical_line_averages = volume_averages.multiconvolve(line_stencils, axis=0)
    # compute unused datapoints
    horizontal_points = horizontal_line_averages.multiconvolve(
        pointwise_stencils, axis=1
    )
    vertical_points = vertical_line_averages.multiconvolve(pointwise_stencils, axis=2)

    # gather linear fluxes
    right_flux = v_x * vertical_line_averages.remove_along_axis(2).ghost_array[1]
    top_flux = v_y * horizontal_line_averages.remove_along_axis(1).ghost_array[1]
    dudt = -(1 / h) * (right_flux[1:-1, :] - right_flux[:-2, :]) - (1 / h) * (
        top_flux[:, 1:-1] - top_flux[:, :-2]
    )
    u += dt * dudt
    t += dt
    if t + dt > T:
        dt = T - t
    step_count += 1
stopping_time = time.time()
l2norm = np.sqrt(np.sum(np.square(u - u0)) * h * h)
print(f"Computed {step_count} steps in {stopping_time - starting_time:.2f} s")

plt.imshow(np.flipud(u.T), extent=(x[0], x[-1], x[0], x[-1]))
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar()
plt.show()
