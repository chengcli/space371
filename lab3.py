"""Polarized Fresnel reflection at an interface and a BK7 lens (mm)."""
import matplotlib.pyplot as plt
import numpy as np
from lib.materials import BK7
from lib.ray import Ray
from lib.refraction import calc_reflectivity
from lib.surface import FlatSurface, SphereSurface
from lib.trace import trace

# Ray line width is proportional to intensity: unit power is 6 points.
def ray_width(ray):
    return 6. * float(np.asarray(ray.intensity).ravel()[0])


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
        ax.plot([start[0], end[0]], [start[1], end[1]], color='black', lw=ray_width(nodes[parent]), label='incident: I=1, s/p=50/50')
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
                arrowprops=dict(arrowstyle='->', color=color, lw=ray_width(node)))
    ax.plot([], [], color=color, lw=ray_width(node), label=f'{label}: I={intensity:.3f}, s/p={s:.0%}/{p:.0%}')
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

# Same singlet geometry as lab2, now showing reflections at both faces.
lens = [
    SphereSurface(center=[40., 0], radius=50., n_after=n_glass),
    SphereSurface(center=[-40., 0], radius=-50., n_after=1.),
]
lens_fig, lens_ax = plt.subplots(figsize=(12, 5))
y = np.linspace(-16, 16, 200)
front = 40 - np.sqrt(50**2 - y**2)
back = -40 + np.sqrt(50**2 - y**2)
lens_ax.fill_betweenx(y, front, back, color='lightblue', alpha=0.3)
lens_ax.plot(front, y, color='gray', lw=0.8)
lens_ax.plot(back, y, color='gray', lw=0.8)

for height in [-12., -6., 0., 6., 12.]:
    path, parents = trace(Ray(pos=[-45, height], dir=[1, 0]), lens)
    segments = set()
    reflected_path = [False] * len(path)
    for i, parent in enumerate(parents):
        if parent < 0:
            continue
        node, source = path[i], path[parent]
        reflected_path[i] = reflected_path[parent] or node.dir[0, 0] * source.dir[0, 0] < 0
        start, end = source.pos[:, 0], node.pos[:, 0]
        key = tuple(np.concatenate([start, end]))
        if key not in segments:
            color = 'black' if parent == 0 else ('tab:red' if reflected_path[parent] else 'tab:blue')
            lens_ax.plot([start[0], end[0]], [start[1], end[1]],
                         color=color, lw=ray_width(source))
            segments.add(key)
        if i in parents or not np.any(node.intensity > 0):
            continue
        # Only extend rays that escaped into air. Dim internal branches stop
        # at their last interaction rather than appearing to leave the lens.
        if not np.isclose(node.n_in, 1.):
            continue
        reflected = reflected_path[i]
        length = 35. if node.dir[0, 0] < 0 else 70.
        end = node.pos[:, 0] + length * node.dir[:, 0]
        lens_ax.plot([node.pos[0, 0], end[0]], [node.pos[1, 0], end[1]],
                     color='tab:red' if reflected else 'tab:blue', lw=ray_width(node))

for color, label in [('black', 'incident'), ('tab:blue', 'refracted'), ('tab:red', 'paths with reflections')]:
    lens_ax.plot([], [], color=color, label=label)
for intensity in [1., 0.1, 0.04]:
    lens_ax.plot([], [], color='gray', lw=6 * intensity, label=f'I = {intensity:g}')
lens_ax.axhline(0, color='gray', ls=':', lw=0.5)
lens_ax.set(xlabel='x [mm]', ylabel='y [mm]', xlim=(-50, 80), ylim=(-22, 22),
            title='BK7 lens at 550 nm: line thickness ∝ intensity\n'
                  'Repeated internal reflections; tracing stops below I = 0.001')
lens_ax.set_aspect('equal')
lens_ax.legend(loc='upper right', fontsize=8, ncol=2)
lens_fig.tight_layout()
plt.show()
