import matplotlib.pyplot as plt
import numpy as np
from lib.ray import Ray
from lib.trace import trace
from lib.surface import SphereSurface, FlatSurface
from lib.materials import BK7

# visible wavelengths [um], traced together in one multi-wavelength ray
WAVES = np.linspace(0.40, 0.70, 7)
NW = len(WAVES)

SURFACES = [
    SphereSurface(center = [40.0, 0], radius = 50.0, n_after = BK7(WAVES)),     # air -> glass
    SphereSurface(center = [-40.0, 0], radius = -50.0, n_after = np.ones(NW)),  # glass -> air
    FlatSurface(center = [100.0, 0], norm = [-1, 0], n_after = np.full(NW, 10.0)),  # screen
    ]

def wave_color(lam):
    """Approximate RGB color of a visible wavelength [um]."""
    nm = lam * 1000.
    if nm < 440:   r, g, b = (440 - nm) / 60, 0., 1.
    elif nm < 490: r, g, b = 0., (nm - 440) / 50, 1.
    elif nm < 510: r, g, b = 0., 1., (510 - nm) / 20
    elif nm < 580: r, g, b = (nm - 510) / 70, 1., 0.
    elif nm < 645: r, g, b = 1., (645 - nm) / 65, 0.
    else:          r, g, b = 1., 0., 0.
    return (r, g, b)

heights = np.linspace(-11, 11, 21)

traced_rays = []
for h in heights:
    ray = Ray(pos = [-100, h], dir = [1, 0.], n_in = np.ones(NW))
    path, _ = trace(ray, SURFACES)
    traced_rays.append(path)

def plot_rays(ax, paths):
    for path in paths:
        pos = [np.broadcast_to(node.pos, (2, NW)) for node in path]
        # all wavelengths share the incoming segment: white light
        ax.plot([pos[0][0, 0], pos[1][0, 0]], [pos[0][1, 0], pos[1][1, 0]],
                color = "black", lw = 0.6)
        for i in range(2, len(pos)):
            for k, lam in enumerate(WAVES):
                ax.plot([pos[i-1][0, k], pos[i][0, k]],
                        [pos[i-1][1, k], pos[i][1, k]],
                        color = wave_color(lam), lw = 0.6)
    ax.axhline(0, lw=0.5, ls="--", color="gray")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")

# where each wavelength's marginal ray crosses the axis (longitudinal chromatic aberration)
last = traced_rays[-1][-2]      # ray leaving the lens at the top height
x_axis = last.pos[0] - last.pos[1] * last.dir[0] / last.dir[1]
for lam, n, x in zip(WAVES, BK7(WAVES), x_axis):
    print(f"lambda = {lam*1000:.0f} nm, n = {n:.4f}, axis crossing x = {x:.2f} mm")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plot_rays(ax1, traced_rays)
ax1.set_ylim([-20, 20])
ax1.set_title("BK7 singlet: dispersion of white light")

plot_rays(ax2, [traced_rays[0], traced_rays[-1]])     # marginal rays only
ax2.set_xlim([x_axis.min() - 2, x_axis.max() + 2])
ax2.set_ylim([-0.5, 0.5])
ax2.set_title("zoom near focus: blue focuses before red")

for lam in WAVES:
    ax1.plot([], [], color = wave_color(lam), label = f"{lam*1000:.0f} nm")
ax1.legend(loc = "upper left", fontsize = 8)

plt.tight_layout()
plt.show()
