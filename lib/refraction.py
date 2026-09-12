import numpy as np
from .ray import Ray

def calc_refraction(ray_in: Ray, norm: np.ndarray, n_after: float) -> None | Ray:
  mu = ray_in.n_in / n_after
  c1 = -(norm @ ray_in.dir)
  k = 1 - mu * mu * (1 - c1 * c1)
  if k < 0: # total internal reflection
      return None
  return Ray(pos = ray_in.pos, dir = mu * ray_in.dir + (mu * c1 - np.sqrt(k)) * norm,
             n_in = n_after)

def calc_reflection(ray_in: Ray, norm: np.ndarray, n_after: float) -> None | Ray:
  c1 = -(norm @ ray_in.dir)
  return Ray(pos = ray_in.pos, dir = ray_in.dir + 2. * mu * norm,
             n_in = n_after)
