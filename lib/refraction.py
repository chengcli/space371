import numpy as np
from .ray import Ray

def calc_refraction(ray_in: Ray, norm: np.ndarray, n_after: np.ndarray) -> None | Ray:
  mu = ray_in.n_in / n_after
  c1 = -np.sum(norm * ray_in.dir, axis = 0)
  k = 1 - mu * mu * (1 - c1 * c1)
  k[k < 0] = np.nan   # total internal reflection, the wavelength is lost
  if np.all(np.isnan(k)):
      return None
  return Ray(pos = ray_in.pos, dir = mu * ray_in.dir + (mu * c1 - np.sqrt(k)) * norm,
             n_in = n_after)
