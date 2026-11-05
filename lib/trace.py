import numpy as np
from typing import List, Tuple, Any
from .ray import Ray

MAX_TRACE_DEPTH = 10


def trace(ray: Ray, surfaces: List[Any], parent = -1, count = 0,
          reflection = False, min_intensity = 0.001,
          surface_index = 0, step = 1, n_before = None) -> Tuple[List[Ray], List[int]]:
    """Trace through ordered surfaces, reversing the order on reflection.

    surface_index and step track the next interface and travel order during
    recursion. n_before is the medium before the first surface.
    Dim terminal rays are retained but are not traced further.
    """
    if n_before is None:
        n_before = ray.n_in

    valid = np.all(np.isfinite(ray.dir), axis = 0) & np.all(np.isfinite(ray.pos), axis = 0)
    if (surface_index < 0 or surface_index >= len(surfaces) or count >= MAX_TRACE_DEPTH or
            not np.any(valid & (ray.intensity > 0) & (ray.intensity >= min_intensity))):
        return [ray], [parent]

    surf = surfaces[surface_index]
    ray_in = surf.hit(ray)
    if ray_in is None:      # miss: try the next surface without adding a node
        return trace(ray, surfaces, parent, count, reflection, min_intensity,
                     surface_index + step, step, n_before)

    n_after = surf.n_after
    if step < 0:           # crossing backward into the preceding medium
        n_after = surfaces[surface_index - 1].n_after if surface_index > 0 else n_before

    ray_out1 = surf.refract(ray_in, n_after)
    ray_out2 = surf.reflect(ray_in, n_after) if reflection else None
    nodes, parents = [ray], [parent]
    for ray_out, next_step in [(ray_out1, step), (ray_out2, -step)]:
        if ray_out is None:
            continue
        node, prev = trace(ray_out, surfaces, 0, count + 1, reflection,
                           min_intensity, surface_index + next_step, next_step, n_before)
        # The branch root attaches to this ray; other indices shift when appended.
        offset = len(nodes)
        nodes += node
        parents += [0] + [p + offset for p in prev[1:]]
    return nodes, parents
