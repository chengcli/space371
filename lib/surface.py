import numpy as np
from dataclasses import dataclass
from .ray import Ray
from .refraction import calc_refraction

EPSILON = 1e-6

@dataclass
class FlatSurface:
  center: np.ndarray
  norm: np.ndarray
  n_after: np.ndarray

  def __post_init__(self):
    self.center = np.asarray(self.center, float)
    dir = np.asarray(self.norm, float)
    self.norm = dir / np.linalg.norm(dir)

  def surface_norm(self, pos: np.ndarray) -> np.ndarray:
    return self.norm[:, None]

  def hit(self, ray: Ray) -> None | Ray:
    m = ray.pos - self.center[:, None]
    b = self.norm @ ray.dir
    t = - (self.norm @ m) / b
    t[t < EPSILON] = np.nan   # behind or at the ray
    if np.all(np.isnan(t)):
      return None
    return ray.advance(t)

  def refract(self, ray_in: Ray) -> None | Ray:
    norm = self.surface_norm(ray_in.pos)
    norm = np.where(np.sum(norm * ray_in.dir, axis = 0) > 0, -norm, norm)
    return calc_refraction(ray_in, norm, self.n_after)

@dataclass
class SphereSurface:
  center: np.ndarray
  radius: np.ndarray
  n_after: np.ndarray

  def __post_init__(self):
    self.center = np.asarray(self.center, float)

  def surface_norm(self, pos: np.ndarray) -> np.ndarray:
    d = pos - self.center[:, None]
    return d / np.linalg.norm(d, axis = 0)

  def hit(self, ray: Ray) -> None | Ray:
    m = ray.pos - self.center[:, None]
    b = np.sum(ray.dir * m, axis = 0)
    disc = b * b - (np.sum(m * m, axis = 0) - self.radius * self.radius)
    disc[disc < 0] = np.nan  # the ray misses
    s = np.sqrt(disc)
    sgn = self.radius / abs(self.radius)
    t = - b - sgn * s
    t[t < EPSILON] = np.nan  # behind or at the ray
    if np.all(np.isnan(t)):
        return None
    return ray.advance(t)

  def refract(self, ray_in: Ray) -> None | Ray:
      norm = self.surface_norm(ray_in.pos)
      norm = np.where(np.sum(norm * ray_in.dir, axis = 0) > 0, -norm, norm)
      return calc_refraction(ray_in, norm, self.n_after)
