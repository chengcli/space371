import numpy as np
from typing import List, Tuple, Any
from .ray import Ray

MAX_TRACE_DEPTH = 10

def trace(ray: Ray, surfaces: List[Any], parent = -1, count = 0) -> Tuple[List[Ray], List[int]]:
    if len(surfaces) == 0 or count > MAX_TRACE_DEPTH:
        return [ray], [parent]
    surf = surfaces[0]
    ray_in = surf.hit(ray)

    if ray_in is None:      # miss
        return trace(ray, surfaces[1:], parent, count)

    ray_out1 = surf.refract(ray_in)
    if ray_out1 is None:    # TODO: total internal reflection
        return [], []

    # check if the refracted ray hits the same surface again
    ray_next = surf.hit(ray_out1)
    if ray_next is not None:
        node, prev = trace(ray_out1, surfaces, parent + 1, count + 1)
        return [ray] + node, [parent] + prev

    # move on to the next surface
    node, prev = trace(ray_out1, surfaces[1:], parent + 1, count + 1)
    return [ray] + node, [parent] + prev
