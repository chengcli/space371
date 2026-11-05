import numpy as np
from typing import List, Tuple, Any
from .ray import Ray

MAX_TRACE_DEPTH = 10


def trace(ray: Ray, surfaces: List[Any], parent=-1, count=0,
          include_reflection=True, min_intensity=0.001) -> Tuple[List[Ray], List[int]]:
    """Trace transmission and repeated reflections through ordered interfaces.

    Surfaces are listed in initial propagation order. The initial ray's index
    defines the first medium; each n_after defines the next medium. Reflection
    reverses traversal through this sequence (a simple sequential lens model).
    parents indexes the returned ray list (-1 for the root). Dim terminal nodes
    are retained but stop; intensity is relative to unit incident power.
    """
    nodes, parents = [], []
    media = [ray.n_in] + [surf.n_after for surf in surfaces]

    def visit(current, surface_index, step, previous, depth):
        index = len(nodes)
        nodes.append(current)
        parents.append(previous)
        valid = np.all(np.isfinite(current.dir), axis=0) & np.all(np.isfinite(current.pos), axis=0)
        if (depth >= MAX_TRACE_DEPTH or
                not np.any(valid & (current.intensity > 0) & (current.intensity >= min_intensity))):
            return
        while 0 <= surface_index < len(surfaces):
            surf = surfaces[surface_index]
            hit = surf.hit(current)
            if hit is None:
                surface_index += step
                continue
            n_after = media[surface_index + 1] if step == 1 else media[surface_index]
            transmitted = surf.refract(hit, n_after)
            if transmitted is not None:
                visit(transmitted, surface_index + step, step, index, depth + 1)
            if include_reflection:
                reflected = surf.reflect(hit, n_after)
                visit(reflected, surface_index - step, -step, index, depth + 1)
            return

    visit(ray, 0, 1, parent, count)
    return nodes, parents
