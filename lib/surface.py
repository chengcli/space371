import numpy as np
from dataclasses import dataclass
from .ray import Ray
from .refraction import calc_refraction

@dataclass
class FlatSurface:
  center: np.ndarray
  norm: np.ndarray
  n_after: float

  def __post_init__(self):
    self.center = np.asarray(self.center, float)
    dir = np.asarray(self.norm, float)
    self.norm = dir / np.linalg.norm(dir)

  def surface_norm(self, pos: np.ndarray) -> np.ndarray:
    return self.norm

  def hit(self, ray: Ray) -> None | Ray:
    m = ray.pos - self.center
    b = ray.dir @ self.norm
    t = - (m @ self.norm) / b
    if t < 0:
      return None   # behind the ray
    return ray.advance(t)

  def refract(self, ray_in: Ray) -> None | Ray:
    norm = self.surface_norm(ray_in.pos)
    if norm @ ray_in.dir > 0:
        norm = -norm
    return calc_refraction(ray_in, norm, self.n_after)

@dataclass
class SphereSurface:
  center: np.ndarray
  radius: np.ndarray
  n_after: float

  def surface_norm(self, pos: np.ndarray) -> np.ndarray:
    return (pos - self.center) / np.linalg.norm(pos - self.center)

  def hit(self, ray: Ray) -> None | Ray:
    m = ray.pos - self.center
    b = ray.dir @ m
    disc = b * b - (m @ m - self.radius * self.radius)
    if disc < 0:    # the ray misses
        return None
    s = np.sqrt(disc)
    sgn = self.radius / abs(self.radius)
    t = - b - sgn * s
    if t < 0:       # behind the ray
        return None
    return ray.advance(t)

  def refract(self, ray_in: Ray) -> None | Ray:
      norm = self.surface_norm(ray_in.pos)
      if norm @ ray_in.dir > 0:
          norm = -norm
      return calc_refraction(ray_in, norm, self.n_after)
