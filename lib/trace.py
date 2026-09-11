import numpy as np
from copy import deepcopy
from typing import List, Any
from .ray import Ray

def trace(ray: Ray, surfaces: List[Any]) -> List[Ray]:
  path = [deepcopy(ray)]
  if len(surfaces) == 0: return path
  surf = surfaces[0]
  ray_in = surf.hit(ray)
  if ray_in is None: # missing
    return path + trace(ray, surfaces[1:])
  ray_out = surf.refract(ray_in)
  if ray_out is None: # TODO: total internal reflection
    return []
  return path + trace(ray_out, surfaces[1:])
