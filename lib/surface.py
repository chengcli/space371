import numpy as np
from dataclasses import dataclass
from .ray import Ray
from .refraction import calc_refraction

EPSILON = 1e-6

def _advance_to_hit(ray: Ray, t: np.ndarray) -> None | Ray:
  """Advance each wavelength by its own distance t (nwave,).

  Wavelengths without a valid forward intersection are marked lost (NaN).
  Returns None if no wavelength hits the surface.
  """
  hit = np.isfinite(t) & (t >= EPSILON)   # False for misses or lost rays
  if not np.any(hit):
    return None               # behind or at the ray, or a miss
  return ray.advance(np.where(hit, t, np.nan))

def _check_index(n_after) -> np.ndarray:
  """n_after must be the refractive index sampled at the ray wavelengths,
  a 1D array of shape (nwave,)."""
  if not isinstance(n_after, np.ndarray):
    raise TypeError(f"n_after should be a np.ndarray, got {type(n_after).__name__}")
  if n_after.ndim != 1:
    raise ValueError("n_after should be a 1D array of shape (nwave,)")
  return n_after.astype(float)

def _refract(surf, ray_in: Ray) -> None | Ray:
  if surf.n_after.shape != (ray_in.nwave,):
    raise ValueError(f"n_after has {len(surf.n_after)} wavelengths, "
                     f"ray has {ray_in.nwave}")
  norm = surf.surface_norm(ray_in.pos)
  flip = np.sum(norm * ray_in.dir, axis = 0) > 0
  norm = np.where(flip, -norm, norm)
  return calc_refraction(ray_in, norm, surf.n_after)

@dataclass
class FlatSurface:
  """n_after: refractive index behind the surface sampled at the ray
  wavelengths, an (nwave,) array (see Instrument.refractive_index)."""
  center: np.ndarray
  norm: np.ndarray
  n_after: np.ndarray

  def __post_init__(self):
    self.center = np.asarray(self.center, float)
    self.n_after = _check_index(self.n_after)
    dir = np.asarray(self.norm, float)
    self.norm = dir / np.linalg.norm(dir)

  def surface_norm(self, pos: np.ndarray) -> np.ndarray:
    return np.broadcast_to(self.norm[:, None], pos.shape)

  def hit(self, ray: Ray) -> None | Ray:
    m = ray.pos - self.center[:, None]
    b = self.norm @ ray.dir
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
      t = - (self.norm @ m) / b
    return _advance_to_hit(ray, t)

  def refract(self, ray_in: Ray) -> None | Ray:
    return _refract(self, ray_in)

@dataclass
class SphereSurface:
  """n_after: refractive index behind the surface sampled at the ray
  wavelengths, an (nwave,) array (see Instrument.refractive_index)."""
  center: np.ndarray
  radius: float
  n_after: np.ndarray

  def __post_init__(self):
    self.center = np.asarray(self.center, float)
    self.n_after = _check_index(self.n_after)

  def surface_norm(self, pos: np.ndarray) -> np.ndarray:
    d = pos - self.center[:, None]
    return d / np.linalg.norm(d, axis = 0)

  def hit(self, ray: Ray) -> None | Ray:
    m = ray.pos - self.center[:, None]
    b = np.sum(ray.dir * m, axis = 0)
    disc = b * b - (np.sum(m * m, axis = 0) - self.radius * self.radius)
    s = np.sqrt(np.where(disc < 0, np.nan, disc))   # NaN: the ray misses
    sgn = self.radius / abs(self.radius)
    t = - b - sgn * s
    return _advance_to_hit(ray, t)

  def refract(self, ray_in: Ray) -> None | Ray:
    return _refract(self, ray_in)
