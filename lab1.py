import matplotlib.pyplot as plt
import numpy as np
from lib.ray import Ray
from lib.trace import trace
from lib.surface import SphereSurface, FlatSurface

SURFACES = [
    SphereSurface(center = [40.0, 0], radius = 50.0, n_after = 1.517),    # air -> glass
    SphereSurface(center = [-40.0, 0], radius = -50.0, n_after = 1.000),  # glass -> air
    FlatSurface(center = [100.0, 0], norm = [-1, 0], n_after = 10.0),     # screen
    ]

heights = np.linspace(-11, 11, 21)

traced_rays = []
for h in heights:
    ray = Ray(pos = [-100, h], dir = [1, 0.])
    path, _ = trace(ray, SURFACES)
    traced_rays.append(path)

fig, ax = plt.subplots(figsize=(10, 4))
for path, h in zip(traced_rays, heights):
    for i in range(1, len(path)):
        ax.plot([path[i-1].pos[0], path[i].pos[0]],
                [path[i-1].pos[1], path[i].pos[1]], color=f'C{i-1}')

ax.axhline(0, lw=0.5, ls="--", color="gray")
ax.set_ylim([-20, 20])
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]");
plt.show()
