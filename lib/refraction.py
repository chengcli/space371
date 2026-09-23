import numpy as np
from .ray import Ray

def calc_refraction(ray_in: Ray, norm: np.ndarray, n_after) -> None | Ray:
  """Snell's law in vector form, applied independently at each wavelength.

  ray_in:  incoming ray with pos/dir of shape (ndim, nwave)
  norm:    unit surface normal facing the incoming ray, (ndim, nwave) or (ndim,)
  n_after: refractive index after the surface, scalar or (nwave,)

  Wavelengths that undergo total internal reflection are marked as lost
  (NaN position). Returns None if no wavelength is transmitted.
  """
  norm = np.asarray(norm, float)
  if norm.ndim == 1:
    norm = norm[:, None]
  n_after = np.broadcast_to(np.asarray(n_after, float), (ray_in.nwave,))

  mu = ray_in.n_in / n_after                  # (nwave,)
  c1 = -np.sum(norm * ray_in.dir, axis = 0)   # (nwave,)
  k = 1 - mu * mu * (1 - c1 * c1)

  tir = k < 0   # total internal reflection
  if not np.any(ray_in.alive & ~tir):
      return None

  dir = mu * ray_in.dir + (mu * c1 - np.sqrt(np.where(tir, np.nan, k))) * norm
  pos = ray_in.pos.copy()
  pos[:, tir] = np.nan
  return Ray(pos = pos, dir = dir, n_in = n_after)
