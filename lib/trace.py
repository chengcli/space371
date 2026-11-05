import numpy as np
from typing import List, Tuple, Any
from .ray import Ray

MAX_TRACE_DEPTH = 10


def trace(ray: Ray, surfaces: List[Any], parent=-1, count=0,
          include_reflection=True, min_intensity=0.001) -> Tuple[List[Ray], List[int]]:
    """Trace forward transmission, retaining reflections as terminal branches.

    parents indexes the returned ray list (-1 for the root). Intensities are
    relative to unit incident power. Dim terminal nodes are retained but stop.
    """
    nodes, parents = [], []

    def visit(current, remaining, previous, depth):
        index = len(nodes)
        nodes.append(current)
        parents.append(previous)
        valid = np.all(np.isfinite(current.dir), axis=0) & np.all(np.isfinite(current.pos), axis=0)
        if (depth > MAX_TRACE_DEPTH or
                not np.any(valid & (current.intensity > 0) & (current.intensity >= min_intensity))):
            return
        for i, surf in enumerate(remaining):
            hit = surf.hit(current)
            if hit is None:
                continue
            transmitted = surf.refract(hit)
            if transmitted is not None:
                following = remaining[i:] if surf.hit(transmitted) is not None else remaining[i+1:]
                visit(transmitted, following, index, depth + 1)
            if include_reflection:
                reflected = surf.reflect(hit)
                # Reflected rays leave this simple sequential optical system.
                visit(reflected, [], index, depth + 1)
            return

    visit(ray, surfaces, parent, count)
    return nodes, parents
