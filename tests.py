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
    surf = FlatSurface(center = [0, 0], norm = [-1, 0], n_after = 1.517)
    ray_in = surf.hit(ray)
    assert ray_in == Ray(pos = [0., 1.], dir = [1., 0.])

    ray_out = surf.refract(ray_in)
    assert ray_out == Ray(pos = [0., 1.], dir = [1., 0.], n_in = 1.517)

# check 4: a ray impinging on a surface at 45 degree angle will refract
def test4():
    ray = Ray(pos = [-10, 1], dir =[1, 0])
    surf = FlatSurface(center = [0, 0], norm = [-1, 1], n_after = 1.517)
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
                         n_after = 1.517)
    path, _ = trace(ray, [surf])
    assert len(path) == 3
    assert path[0] == Ray(pos=[-100.,   50.], dir=[1., 0.], n_in=1.0)
    assert path[1] == Ray(pos=[ 0., 50.], dir=[ 0.65919578, -0.75197136], n_in=1.517)
    assert path[2] == Ray(pos=[49.56963462, -6.54609221], dir=[ 0.65919578, -0.75197136], n_in=1.517)

# ---------------------------------------------------------------------------
# wavelength-dependent refractive index
# ---------------------------------------------------------------------------
from lib.instrument import Instrument
from lib.materials import BK7

WAVES = np.array([0.4861, 0.5876, 0.6563])   # F, d, C lines [um]

def lens(glass):
    return [
        SphereSurface(center = [40.0, 0], radius = 50.0, n_after = glass),
        SphereSurface(center = [-40.0, 0], radius = -50.0, n_after = 1.0),
    ]

# check 6: the instrument holds a refractive index look-up table
def test6():
    inst = Instrument(wavelengths = WAVES,
                      materials = {"air": 1.0, "BK7": BK7,
                                   "tab": ([0.4, 0.7], [1.6, 1.5])})
    assert inst.n_table.shape == (3, 3)
    assert np.allclose(inst.refractive_index("air"), 1.0)
    assert np.isclose(inst.refractive_index("BK7")[1], 1.5168, atol = 1e-4)
    assert np.all(np.diff(inst.refractive_index("BK7")) < 0)   # normal dispersion
    assert np.allclose(inst.refractive_index("tab"), 1.6 - (WAVES - 0.4) / 3.)
    assert np.allclose(inst.weights, 1. / 3.)

# check 7: a ray is expanded onto the instrument wavelengths
def test7():
    inst = Instrument(wavelengths = WAVES, materials = {"BK7": BK7})
    ray = inst.make_ray(pos = [-100, 5], dir = [1, 0])
    assert ray.pos.shape == (2, 3)
    assert ray.dir.shape == (2, 3)
    assert ray.n_in.shape == (3,)
    assert np.allclose(ray.pos[:, 2], [-100, 5])

    ray = inst.make_ray(pos = [-100, 5], dir = [1, 0], medium = "BK7")
    assert np.allclose(ray.n_in, inst.refractive_index("BK7"))

# check 8: each wavelength follows its own monochromatic path, and n_in is
# recomputed from the look-up table at every node
def test8():
    inst = Instrument(wavelengths = WAVES, materials = {"BK7": BK7})
    path, _ = trace(Ray(pos = [-100, 10], dir = [1, 0]), lens("BK7"),
                    instrument = inst)
    assert len(path) == 3
    assert np.allclose(path[1].n_in, BK7(WAVES))
    assert np.allclose(path[2].n_in, 1.0)

    for i, lam in enumerate(WAVES):
        mono, _ = trace(Ray(pos = [-100, 10], dir = [1, 0]), lens(float(BK7(lam))))
        for node, ref in zip(path, mono):
            assert np.allclose(node.pos[:, i], ref.pos[:, 0])
            assert np.allclose(node.dir[:, i], ref.dir[:, 0])

    # longitudinal chromatic aberration: blue crosses the axis first
    last = path[-1]
    x_axis = last.pos[0] - last.pos[1] * last.dir[0] / last.dir[1]
    assert x_axis[0] < x_axis[1] < x_axis[2]

# check 9: total internal reflection removes only the affected wavelength
def test9():
    theta = np.deg2rad(44.)     # critical angle: 41.8 deg (n=1.5), 45.6 deg (n=1.4)
    ray = Ray(pos = [-1, 0], dir = [np.cos(theta), np.sin(theta)], n_in = [1.5, 1.4])
    surf = FlatSurface(center = [0, 0], norm = [-1, 0], n_after = 1.0)
    ray_out = surf.refract(surf.hit(ray))
    assert np.array_equal(ray_out.alive, [False, True])
    assert np.isclose(ray_out.dir[1, 1], 1.4 * np.sin(theta))   # Snell's law

    ray = Ray(pos = [-1, 0], dir = [np.cos(theta), np.sin(theta)], n_in = [1.5, 1.6])
    assert surf.refract(surf.hit(ray)) is None

# check 10: response-weighted average over the wavelength dimension
def test10():
    inst = Instrument(wavelengths = WAVES, response = [1., 2., 1.])
    assert np.allclose(inst.weights, [0.25, 0.5, 0.25])
    assert np.isclose(inst.weighted_mean(np.array([1., 2., 3.])), 2.)
    assert np.isclose(inst.weighted_mean(np.array([1., 2., np.nan])), 5. / 3.)
