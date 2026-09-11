import numpy as np
from lib.ray import Ray
from lib.surface import FlatSurface

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
