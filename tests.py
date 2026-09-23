import numpy as np
from lib.ray import Ray
from lib.surface import FlatSurface, SphereSurface
from lib.trace import trace

# check 1: a flat ray keeps its height
def test1():
    ray = Ray(pos = [0, 2], dir =[1, 0])
    assert ray.advance(50) == Ray(pos = [50., 2.], dir = [1, 0])

# check 2: a tilted ray gains 2.5 mm over 50 mm of x
def test2():
    ray = Ray(pos = [0, 2], dir = [1, 0.05])
    step = 50. / ray.dir[0]      # travel until x = 50 mm
    assert ray.advance(step) == Ray(pos = [50., 4.5], dir = [1, 0.05])

# check 3: a normal incidence keeps its height
def test3():
    ray = Ray(pos = [-10, 1], dir =[1, 0])
    surf = FlatSurface(center = [0, 0], norm = [-1, 0], n_after = np.array([1.517]))
    ray_in = surf.hit(ray)
    assert ray_in == Ray(pos = [0., 1.], dir = [1., 0.])

    ray_out = surf.refract(ray_in)
    assert ray_out == Ray(pos = [0., 1.], dir = [1., 0.], n_in = 1.517)

# check 4: a ray impinging on a surface at 45 degree angle will refract
def test4():
    ray = Ray(pos = [-10, 1], dir =[1, 0])
    surf = FlatSurface(center = [0, 0], norm = [-1, 1], n_after = np.array([1.517]))
    ray_in = surf.hit(ray)
    assert ray_in == Ray(pos = [1., 1.], dir = [1., 0.])
    
    ray_out = surf.refract(ray_in)
    assert np.allclose(ray_out.pos, ray_in.pos)
    assert ray_out.dir[0] < ray_in.dir[0]  # x component decreases
    assert ray_out.dir[1] < ray_in.dir[1]  # y component decreases
    assert ray_out.n_in == 1.517

# check 5: a grazing incidence at y=50. will refract twice
def test5():
    ray = Ray(pos = [-100, 50], dir = [1, 0])
    surf = SphereSurface(center = np.array([0.0, 0.0]), radius = -50.,
                         n_after = np.array([1.517]))
    path, _ = trace(ray, [surf])
    assert len(path) == 3
    assert path[0] == Ray(pos=[-100.,   50.], dir=[1., 0.], n_in=1.0)
    assert path[1] == Ray(pos=[ 0., 50.], dir=[ 0.65919578, -0.75197136], n_in=1.517)
    assert path[2] == Ray(pos=[49.56963462, -6.54609221], dir=[ 0.65919578, -0.75197136], n_in=1.517)

# ---------------------------------------------------------------------------
# wavelength-dependent refractive index
# ---------------------------------------------------------------------------
from lib.materials import BK7

WAVES = np.array([0.4861, 0.5876, 0.6563])   # F, d, C lines [um]

def lens(glass, air):
    return [
        SphereSurface(center = [40.0, 0], radius = 50.0, n_after = glass),
        SphereSurface(center = [-40.0, 0], radius = -50.0, n_after = air),
    ]

# check 6: BK7 shows normal dispersion
def test6():
    n = BK7(WAVES)
    assert np.isclose(n[1], 1.5168, atol = 1e-4)
    assert np.all(np.diff(n) < 0)

# check 7: each wavelength follows its own monochromatic path
def test7():
    ray = Ray(pos = [-100, 10], dir = [1, 0], n_in = np.ones(3))
    path, _ = trace(ray, lens(BK7(WAVES), np.ones(3)))
    assert len(path) == 3
    assert np.allclose(path[1].n_in, BK7(WAVES))

    for i, lam in enumerate(WAVES):
        mono, _ = trace(Ray(pos = [-100, 10], dir = [1, 0]),
                        lens(np.array([BK7(lam)]), np.ones(1)))
        for node, ref in zip(path, mono):
            assert np.allclose(np.broadcast_to(node.pos, (2, 3))[:, i], ref.pos[:, 0])
            assert np.allclose(np.broadcast_to(node.dir, (2, 3))[:, i], ref.dir[:, 0])

    # longitudinal chromatic aberration: blue crosses the axis first
    last = path[-1]
    x_axis = last.pos[0] - last.pos[1] * last.dir[0] / last.dir[1]
    assert x_axis[0] < x_axis[1] < x_axis[2]

# check 8: total internal reflection removes only the affected wavelength
def test8():
    theta = np.deg2rad(44.)     # critical angle: 41.8 deg (n=1.5), 45.6 deg (n=1.4)
    ray = Ray(pos = [-1, 0], dir = [np.cos(theta), np.sin(theta)], n_in = np.array([1.5, 1.4]))
    surf = FlatSurface(center = [0, 0], norm = [-1, 0], n_after = np.ones(2))
    ray_out = surf.refract(surf.hit(ray))
    assert np.all(np.isnan(ray_out.dir[:, 0]))
    assert np.isclose(ray_out.dir[1, 1], 1.4 * np.sin(theta))   # Snell's law

    ray = Ray(pos = [-1, 0], dir = [np.cos(theta), np.sin(theta)], n_in = np.array([1.5, 1.6]))
    assert surf.refract(surf.hit(ray)) is None
