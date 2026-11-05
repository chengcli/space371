"""Polarized Fresnel reflection at an air–BK7 interface (lengths in mm)."""
import matplotlib.pyplot as plt
import numpy as np
from lib.materials import BK7
from lib.ray import Ray
from lib.refraction import calc_reflectivity
from lib.surface import FlatSurface
from lib.trace import trace

n_glass = float(BK7(0.55))
brewster = np.arctan(n_glass)
normal = np.array([[-1.], [0.]])
surface = FlatSurface(center=[0, 0], norm=[-1, 0], n_after=n_glass)
direction = np.array([np.cos(brewster), np.sin(brewster)])
ray = Ray(pos=-30 * direction, dir=direction)  # unpolarized: equal s/p power
nodes, parents = trace(ray, [surface])

fig, (ax, curve) = plt.subplots(1, 2, figsize=(12, 5))
ax.axvspan(0, 35, color='lightblue', alpha=0.3)
ax.axvline(0, color='gray', label='air / BK7')
ax.axhline(0, color='gray', ls=':', lw=0.7)
# Parent links identify incoming segments; draw shared segments only once.
segments = set()
for i, parent in enumerate(parents):
    if parent < 0:
        continue
    start, end = nodes[parent].pos[:, 0], nodes[i].pos[:, 0]
    key = tuple(np.concatenate([start, end]))
    if key not in segments:
        ax.plot([start[0], end[0]], [start[1], end[1]], color='black', label='incident: I=1, s/p=50/50')
        segments.add(key)
    node = nodes[i]
    if i in parents or not np.any(node.intensity > 0):
        continue
    reflected = node.dir[0, 0] < 0
    label = 'reflected' if reflected else 'transmitted'
    color = 'tab:red' if reflected else 'tab:blue'
    end = node.pos[:, 0] + 30 * node.dir[:, 0]
    intensity = float(node.intensity[0])
    s, p = node.polarization[:, 0]
    ax.annotate('', xy=end, xytext=node.pos[:, 0],
                arrowprops=dict(arrowstyle='->', color=color, lw=2))
    ax.plot([], [], color=color, label=f'{label}: I={intensity:.3f}, s/p={s:.0%}/{p:.0%}')
    print(f'{label}: intensity={intensity:.6f}, s={s:.6f}, p={p:.6f}')

angles = np.linspace(0, np.pi / 2, 361)
probe = Ray(pos=[0, 0], dir=np.array([np.cos(angles), np.sin(angles)]))
rs, rp = calc_reflectivity(probe, normal, n_glass)
curve.plot(np.degrees(angles), rs, label='s polarization')
curve.plot(np.degrees(angles), rp, label='p polarization')
curve.plot(np.degrees(angles), (rs + rp) / 2, '--', label='unpolarized')
curve.axvline(np.degrees(brewster), color='gray', ls=':', label='Brewster angle')
curve.set(xlabel='Incidence angle [degrees]', ylabel='Power reflectivity', xlim=(0, 90), ylim=(0, 1))
curve.legend()
ax.set(xlabel='x [mm]', ylabel='y [mm]', xlim=(-35, 35), ylim=(-30, 35),
       title=f'Unpolarized light at Brewster angle ({np.degrees(brewster):.1f}°)')
ax.set_aspect('equal')
ax.legend(fontsize=8, loc='lower right')
print(f'BK7 at 550 nm: n={n_glass:.6f}, Brewster angle={np.degrees(brewster):.3f} degrees')
fig.tight_layout()
plt.show()
