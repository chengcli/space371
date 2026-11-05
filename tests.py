import numpy as np
from lib.ray import Ray
from lib.surface import FlatSurface, SphereSurface
from functools import partial
from lib.trace import trace
from lib.refraction import calc_reflectivity

trace_branches = partial(trace, reflection=True)

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
    assert ray_out == Ray(pos = [0., 1.], dir = [1., 0.], n_in = 1.517,
                          intensity = 1 - ((1 - 1.517)/(1 + 1.517))**2)

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

# check 5: grazing incidence carries no transmitted power
def test5():
    ray = Ray(pos=[-100, 50], dir=[1, 0])
    surf = SphereSurface(center=[0, 0], radius=-50., n_after=np.array([1.517]))
    path, parents = trace_branches(ray, [surf])
    assert parents == [-1, 0, 0]
    assert np.allclose(path[1].intensity, 0)
    assert np.allclose(path[2].intensity, 1)

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


def test_fresnel_limits():
    normal = np.array([[-1.], [0.]])
    ray = Ray(pos=[0, 0], dir=[1, 0])
    assert np.allclose(calc_reflectivity(ray, normal, 1.5), 0.04)
    theta = np.arctan(1.5)
    ray = Ray(pos=[0, 0], dir=[np.cos(theta), np.sin(theta)])
    assert np.isclose(calc_reflectivity(ray, normal, 1.5)[1, 0], 0)
    grazing = Ray(pos=[0, 0], dir=[0, 1])
    assert np.allclose(calc_reflectivity(grazing, normal, 1.5), 1)
    assert np.allclose(calc_reflectivity(grazing, normal, 1.), 0)


def test_polarized_power_and_reflection():
    ray = Ray(pos=[-1, 0], dir=[1, 0.5], intensity=0.8, polarization=(0.2, 0.8))
    surf = FlatSurface(center=[0, 0], norm=[-1, 0], n_after=1.5)
    hit = surf.hit(ray)
    reflected, transmitted = surf.reflect(hit), surf.refract(hit)
    assert np.allclose(reflected.dir[:, 0], [-ray.dir[0, 0], ray.dir[1, 0]])
    assert np.allclose(reflected.n_in, ray.n_in)
    assert np.allclose(reflected.intensity + transmitted.intensity, ray.intensity)
    assert np.allclose(reflected.intensity * reflected.polarization +
                       transmitted.intensity * transmitted.polarization,
                       ray.intensity * ray.polarization)
    for node in (hit, reflected, transmitted):
        assert np.allclose(node.polarization.sum(axis=0), 1)
    assert np.allclose(hit.intensity, ray.intensity)


def test_tir_branches():
    theta = np.deg2rad(44.)
    surf = FlatSurface(center=[0, 0], norm=[-1, 0], n_after=np.ones(2))
    for indices in ([1.5, 1.4], [1.5, 1.6]):
        ray = Ray(pos=[-1, 0], dir=[np.cos(theta), np.sin(theta)], n_in=indices)
        path, parents = trace_branches(ray, [surf])
        reflected = path[-1]
        assert np.isclose(reflected.intensity[0], 1)
        assert np.allclose(reflected.polarization[:, 0], 0.5)
        assert parents[-1] == 0
        if indices[1] == 1.6:
            assert len(path) == 2
            assert np.allclose(reflected.intensity, 1)
        else:
            assert np.allclose(path[1].intensity + reflected.intensity, 1)


def test_cutoff_and_parents():
    surfaces = [FlatSurface(center=[x, 0], norm=[-1, 0], n_after=n)
                for x, n in [(0, 1.5), (1, 1.)]]
    ray = Ray(pos=[-1, 0], dir=[1, 0])
    path, parents = trace_branches(ray, surfaces)
    assert parents == [-1, 0, 1, 1, 3, 3, 5, 5, 0]
    path, parents = trace_branches(ray, surfaces, min_intensity=0.98)
    assert parents == [-1, 0, 0]  # 96% transmission stops before second surface
    assert len(trace_branches(Ray(pos=[-1, 0], dir=[1, 0], intensity=0), surfaces)[0]) == 1
    same = FlatSurface(center=[0, 0], norm=[-1, 0], n_after=1.)
    reflected = same.reflect(same.hit(ray))
    assert np.allclose(reflected.intensity, 0)
    assert np.all(np.isfinite(reflected.polarization))


def test_repeated_reflections():
    # A normal-incidence slab has R=0.04 at either face, in either direction.
    surfaces = [FlatSurface(center=[x, 0], norm=[-1, 0], n_after=n)
                for x, n in [(0, 1.5), (1, 1.)]]
    path, parents = trace_branches(Ray(pos=[-1, 0], dir=[1, 0]), surfaces)
    assert np.allclose(path[3].intensity, 0.96 * 0.04)
    assert np.allclose(path[4].intensity, 0.96**2 * 0.04)
    assert np.allclose(path[4].n_in, 1.)  # escapes backward into air
    assert path[4].dir[0, 0] < 0
    assert np.allclose(path[5].intensity, 0.96 * 0.04**2)
    assert np.allclose(path[5].n_in, 1.5)  # second internal reflection
    assert path[5].dir[0, 0] > 0
    leaves = [node for i, node in enumerate(path) if i not in parents]
    assert np.allclose(sum(node.intensity for node in leaves), 1.)
    for i, parent in enumerate(parents):
        assert -1 <= parent < i
    # With the intensity cutoff disabled, the depth cap still terminates.
    path, parents = trace_branches(Ray(pos=[-1, 0], dir=[1, 0]), surfaces, min_intensity=0)
    from lib.trace import MAX_TRACE_DEPTH
    depths = []
    for parent in parents:
        depths.append(0 if parent == -1 else depths[parent] + 1)
    assert max(depths) == MAX_TRACE_DEPTH


def test_lens_returning_rays():
    path, parents = trace_branches(Ray(pos=[-100, 6], dir=[1, 0], n_in=np.ones(3)),
                                  lens(BK7(WAVES), np.ones(3)))
    # Rear-face reflection reaches the front face, then splits again.
    assert parents[4] == 3
    assert np.all(path[3].dir[0] < 0)
    assert np.all(path[4].pos[0] < path[3].pos[0])
    assert np.allclose(path[4].n_in, 1.)
    assert np.allclose(np.linalg.norm(path[4].pos - np.array([[40.], [0.]]), axis=0), 50.)
    assert np.allclose(path[5].n_in, BK7(WAVES))
    leaves = [node for i, node in enumerate(path) if i not in parents]
    assert np.allclose(sum(node.intensity for node in leaves), 1.)
    for k, wavelength in enumerate(WAVES):
        mono, mono_parents = trace_branches(Ray(pos=[-100, 6], dir=[1, 0]),
                                           lens(BK7(wavelength), 1.))
        assert parents == mono_parents
        for node, single in zip(path, mono):
            assert np.allclose(np.broadcast_to(node.pos, (2, 3))[:, k], single.pos[:, 0])
            assert np.allclose(np.broadcast_to(node.intensity, (3,))[k], single.intensity)
