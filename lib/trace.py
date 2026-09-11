import numpy as np
from copy import deepcopy
from typing import List, Any
from .ray import Ray

MAX_TRACE_DEPTH = 10

def trace(ray: Ray, surfaces: List[Any], count = 0) -> List[Ray]:
    path = [deepcopy(ray)]
    if len(surfaces) == 0 or count > MAX_TRACE_DEPTH:
        return path
    surf = surfaces[0]
    ray_in = surf.hit(ray)
    if ray_in is None: # missing
        return path + trace(ray, surfaces[1:])
    ray_out = surf.refract(ray_in)
    if ray_out is None: # TODO: total internal reflection
        return []
    # check if the refracted ray hits the same surface again
    ray_next = surf.hit(ray_out)
    if ray_next is not None:
        return path + trace(ray_out, surfaces, count + 1)
    # move on to next surface
    return path + trace(ray_out, surfaces[1:], count + 1)
